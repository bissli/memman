"""Host-side e2e tests for the OpenClaw integration.

Covers the install / register / uninstall surface in
`src/memman/setup/openclaw.py` and the shipped Node hook at
`src/memman/setup/assets/openclaw/hooks/memman-prime/handler.js`.

These tests catch:
- symlink wiring against `importlib.resources` package paths
- non-destructive merge of the memman entry into a pre-existing
  `openclaw.json` plugin manifest
- uninstall stripping the plugin entry without touching siblings
- handler.js execution (requires `node` on PATH): real bootstrap-file
  push with a shimmed `memman` binary, plus the catch-block fallback
  when memman is absent
- shipped SKILL.md substantive content + canonical section headers
"""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from memman.setup.openclaw import (
    openclaw_register_plugin,
    openclaw_uninstall,
    openclaw_write_hook,
    openclaw_write_plugin,
    openclaw_write_skill,
)

pytestmark = [pytest.mark.e2e_cli]

HANDLER_REL = (
    'src/memman/setup/assets/openclaw/hooks/memman-prime/handler.js')


def _resolve_handler_path() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent.parent / HANDLER_REL


# ---------------------------------------------------------------------
# Install path: individual writers
# ---------------------------------------------------------------------

class TestInstallPaths:

    def test_install_writes_expected_paths(self, tmp_path: Path):
        """Each writer creates a symlink that resolves to a real asset.
        """
        config_dir = str(tmp_path / 'openclaw')

        skill_path = Path(openclaw_write_skill(config_dir))
        hook_dir = Path(openclaw_write_hook(config_dir))
        plugin_dir = Path(openclaw_write_plugin(config_dir))

        skill_link = skill_path
        hook_md = hook_dir / 'HOOK.md'
        handler_js = hook_dir / 'handler.js'
        plugin_pkg = plugin_dir / 'package.json'
        plugin_manifest = plugin_dir / 'openclaw.plugin.json'
        plugin_index = plugin_dir / 'index.js'

        for link in (skill_link, hook_md, handler_js,
                     plugin_pkg, plugin_manifest, plugin_index):
            assert link.is_symlink(), f'{link} is not a symlink'
            target = link.resolve()
            assert target.is_file(), f'{link} target {target} missing'
            assert 'memman/setup/assets/openclaw' in str(target), (
                f'{link} resolved outside the package: {target}')


# ---------------------------------------------------------------------
# Plugin registration: JSON merge logic
# ---------------------------------------------------------------------

class TestRegisterPlugin:

    def test_register_plugin_writes_valid_json(self, tmp_path: Path):
        """Default register call writes enabled+remind+nudge=True.
        """
        config_dir = str(tmp_path / 'openclaw')
        Path(config_dir).mkdir(parents=True)

        cfg_path = Path(openclaw_register_plugin(config_dir))
        cfg = json.loads(cfg_path.read_text())

        entry = cfg['plugins']['entries']['memman']
        assert entry['enabled'] is True
        assert entry['config']['remind'] is True
        assert entry['config']['nudge'] is True

    def test_register_plugin_preserves_other_entries(self, tmp_path: Path):
        """Existing unrelated plugins survive the merge intact.
        """
        config_dir = Path(tmp_path / 'openclaw')
        config_dir.mkdir(parents=True)

        seed = {
            'plugins': {
                'entries': {
                    'other-plugin': {
                        'enabled': True,
                        'config': {'foo': 'bar'},
                        },
                    },
                },
            'unrelated_top_level': {'keep': 'me'},
            }
        (config_dir / 'openclaw.json').write_text(
            json.dumps(seed, indent=2))

        openclaw_register_plugin(str(config_dir))

        cfg = json.loads((config_dir / 'openclaw.json').read_text())
        entries = cfg['plugins']['entries']
        assert entries['other-plugin'] == seed['plugins']['entries'][
            'other-plugin']
        assert 'memman' in entries
        assert cfg['unrelated_top_level'] == {'keep': 'me'}

    def test_register_plugin_recovers_from_corrupt_openclaw_json(
            self, tmp_path: Path):
        """Corrupt openclaw.json is overwritten with a valid file.

        Exercises the bare-except fallback at openclaw.py:54 that
        otherwise would only run in production accidents.
        """
        config_dir = Path(tmp_path / 'openclaw')
        config_dir.mkdir(parents=True)
        (config_dir / 'openclaw.json').write_text(
            '\x00\x01garbage{{not-json')

        openclaw_register_plugin(str(config_dir))

        cfg = json.loads((config_dir / 'openclaw.json').read_text())
        assert cfg['plugins']['entries']['memman']['enabled'] is True


# ---------------------------------------------------------------------
# Uninstall: strips entry, removes dirs
# ---------------------------------------------------------------------

class TestUninstall:

    def test_uninstall_strips_plugin_entry(self, tmp_path: Path):
        """Uninstall removes the plugin entry and the three target dirs.
        """
        config_dir = Path(tmp_path / 'openclaw')

        openclaw_write_skill(str(config_dir))
        openclaw_write_hook(str(config_dir))
        openclaw_write_plugin(str(config_dir))
        openclaw_register_plugin(str(config_dir))

        errs = openclaw_uninstall(str(config_dir))
        assert errs == []

        for sub in ('skills', 'hooks', 'extensions'):
            assert not (config_dir / sub).exists(), (
                f'{sub}/ not removed by uninstall')

        cfg_path = config_dir / 'openclaw.json'
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            entries = cfg.get('plugins', {}).get('entries', {})
            assert 'memman' not in entries


# ---------------------------------------------------------------------
# SKILL.md content sanity
# ---------------------------------------------------------------------

class TestSkillContent:

    def test_openclaw_skill_md_has_canonical_sections(self):
        """Shipped SKILL.md is substantive and carries the topic sections.
        """
        skill = (Path(__file__).resolve().parent.parent.parent
                 / 'src/memman/setup/assets/openclaw/SKILL.md')
        content = skill.read_text()
        assert len(content) > 500
        assert '## Storing what you learn' in content
        assert '## Guardrails' in content


# ---------------------------------------------------------------------
# handler.js execution (requires Node on PATH)
# ---------------------------------------------------------------------

class TestHandlerJs:

    def _run_handler(self, handler_path: Path, env_path: str
                     ) -> list[dict]:
        """Drive handler.js once and return the bootstrapFiles array."""
        driver = textwrap.dedent(f"""
            import handler from '{handler_path}';
            const event = {{
              type: 'agent',
              action: 'bootstrap',
              context: {{ bootstrapFiles: [] }},
            }};
            await handler(event);
            process.stdout.write(
              JSON.stringify(event.context.bootstrapFiles));
            """).strip()

        env = {**os.environ, 'PATH': env_path}
        result = subprocess.run(
            ['node', '--no-warnings', '--input-type=module', '-e', driver],
            capture_output=True, text=True, env=env, check=True)
        return json.loads(result.stdout)

    def test_handler_pushes_bootstrap_with_status_and_guide(
            self, tmp_path: Path, node_available):
        """With a shim memman, handler captures status counts + guide."""
        shim_dir = tmp_path / 'shim'
        shim_dir.mkdir()
        shim = shim_dir / 'memman'
        shim.write_text(textwrap.dedent("""
            #!/bin/sh
            case "$1" in
              status)
                printf '{"total_insights": 7, "edge_count": 13}'
                ;;
              guide)
                printf 'Guide body for tests'
                ;;
            esac
            """).lstrip())
        shim.chmod(0o755)

        node_path = shutil.which('node')
        assert node_path, 'node_available fixture should have skipped'
        node_dir = str(Path(node_path).parent)
        env_path = f'{shim_dir}:{node_dir}:/usr/bin:/bin'

        files = self._run_handler(_resolve_handler_path(), env_path)
        assert len(files) == 1
        f = files[0]
        assert f['name'] == 'MEMMAN-GUIDE.md'
        assert '7 insights' in f['content']
        assert '13 edges' in f['content']
        assert 'Guide body for tests' in f['content']

    def test_handler_falls_back_when_memman_missing(
            self, tmp_path: Path, node_available):
        """Without memman on PATH, the catch block pushes the fallback."""
        node_path = shutil.which('node')
        assert node_path, 'node_available fixture should have skipped'
        node_dir = str(Path(node_path).parent)
        env_path = f'{node_dir}:/usr/bin:/bin'

        files = self._run_handler(_resolve_handler_path(), env_path)
        assert len(files) == 1
        f = files[0]
        assert f['name'] == 'MEMMAN-GUIDE.md'
        assert f['content'] == '[memman] Memory active.'
