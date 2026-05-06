"""Tests for memman.setup — settings, markdown, detection."""

import json
import os
import pathlib
import subprocess

import pytest
from click.testing import CliRunner
from memman.cli import cli
from memman.setup.claude import claude_register_hooks
from memman.setup.markdown import remove_memory_block
from memman.setup.settings import add_claude_hooks_selective
from memman.setup.settings import add_memman_permission, read_json_file
from memman.setup.settings import remove_claude_hooks, remove_if_empty
from memman.setup.settings import remove_memman_permission, strip_json5
from memman.setup.settings import write_json_file


def _stop_script():
    """Return path to stop.sh asset."""
    from importlib.resources import files as pkg_files
    return str(
        pkg_files('memman.setup.assets')
        .joinpath('claude/stop.sh'))


def _prompt_script():
    """Return path to user_prompt.sh asset."""
    from importlib.resources import files as pkg_files
    return str(
        pkg_files('memman.setup.assets')
        .joinpath('claude/user_prompt.sh'))


def _run_hook(script: str, input_json: str,
              tmp_home: pathlib.Path) -> subprocess.CompletedProcess:
    """Run a hook script with HOME overridden."""
    return subprocess.run(
        ['bash', script],
        check=False, input=input_json,
        capture_output=True, text=True,
        env={**os.environ, 'HOME': str(tmp_home)})


class TestStripJson5:
    """JSON5 stripping primitive (comments, trailing commas)."""

    def test_strip_json5_line_comments(self):
        """Remove // line comments."""
        s = '{"key": "value" // comment\n}'
        assert json.loads(strip_json5(s)) == {'key': 'value'}

    def test_strip_json5_comment_in_string(self):
        """// inside quotes is preserved."""
        s = '{"url": "https://example.com"}'
        assert json.loads(strip_json5(s)) == {'url': 'https://example.com'}

    def test_strip_json5_trailing_comma(self):
        """Trailing commas before closing brackets are removed."""
        s = '{"a": 1, "b": 2,}'
        assert json.loads(strip_json5(s)) == {'a': 1, 'b': 2}

    def test_strip_json5_trailing_comma_array(self):
        """Trailing commas in arrays are removed."""
        s = '[1, 2, 3,]'
        assert json.loads(strip_json5(s)) == [1, 2, 3]

    def test_strip_json5_block_comment(self):
        """/* ... */ block comments are removed."""
        s = '{"a": 1 /* this is a block\n comment */, "b": 2}'
        assert json.loads(strip_json5(s)) == {'a': 1, 'b': 2}

    def test_strip_json5_block_comment_inline(self):
        """Block comments on a single line are removed."""
        s = '{/* inline */"x": 42}'
        assert json.loads(strip_json5(s)) == {'x': 42}

    def test_strip_json5_single_quoted_passthrough(self):
        """Single-quoted strings are preserved verbatim so callers can
        normalize them downstream; the stripper must not mistake `//`
        inside a single-quoted string for a comment.
        """
        s = "{\"url\": 'https://example.com'}"
        stripped = strip_json5(s)
        assert "'https://example.com'" in stripped

    def test_strip_json5_block_comment_inside_string(self):
        """`/* */` inside a double-quoted string is NOT stripped."""
        s = '{"note": "not /* a */ comment"}'
        assert json.loads(strip_json5(s)) == {'note': 'not /* a */ comment'}

    def test_strip_json5_escape_in_single_quoted(self):
        """An escaped single-quote inside a single-quoted string does
        not terminate the string.
        """
        s = "{\"msg\": 'it\\'s fine'}"
        stripped = strip_json5(s)
        assert "'it\\'s fine'" in stripped


class TestFileOps:
    """`remove_if_empty`, `read_json_file`, `write_json_file`."""

    def test_remove_if_empty_allows_known_leaf(self, tmp_path):
        """remove_if_empty deletes an empty 'hooks' dir (known leaf name)."""
        target = tmp_path / 'hooks'
        target.mkdir()
        remove_if_empty(str(target))
        assert not target.exists()

    def test_remove_if_empty_allows_config_root(self, tmp_path):
        """remove_if_empty accepts a directory named '.claude'."""
        target = tmp_path / '.claude'
        target.mkdir()
        remove_if_empty(str(target))
        assert not target.exists()

    def test_remove_if_empty_rejects_outside_allowlist(self, tmp_path):
        """remove_if_empty raises when basename is not in the allowlist."""
        target = tmp_path / 'arbitrary'
        target.mkdir()
        with pytest.raises(ValueError, match='refused'):
            remove_if_empty(str(target))
        assert target.exists()

    def test_remove_if_empty_rejects_root(self):
        """remove_if_empty refuses to operate on '/' (basename is empty)."""
        with pytest.raises(ValueError, match='refused'):
            remove_if_empty('/')

    def test_remove_if_empty_noop_on_non_empty_dir(self, tmp_path):
        """remove_if_empty leaves a non-empty allowed dir intact."""
        target = tmp_path / 'hooks'
        target.mkdir()
        (target / 'keep.json').write_text('{}')
        remove_if_empty(str(target))
        assert target.exists()
        assert (target / 'keep.json').exists()

    def test_read_json_missing_file(self, tmp_path):
        """Missing file returns empty dict."""
        result = read_json_file(str(tmp_path / 'nope.json'))
        assert result == {}

    def test_read_json_with_comments(self, tmp_path):
        """JSON5 with comments parses correctly."""
        p = tmp_path / 'test.json'
        p.write_text('{\n  "key": "val" // comment\n}')
        result = read_json_file(str(p))
        assert result == {'key': 'val'}

    def test_write_json_atomic(self, tmp_path):
        """Write uses .tmp + rename pattern."""
        p = str(tmp_path / 'out.json')
        write_json_file(p, {'hello': 'world'})
        assert pathlib.Path(p).exists()
        assert not pathlib.Path(p + '.tmp').exists()
        data = json.loads(pathlib.Path(p).open().read())
        assert data == {'hello': 'world'}


class TestHookManagement:
    """`add_claude_hooks_selective` and `remove_claude_hooks`."""

    def test_remove_claude_hooks(self):
        """Remove memman hooks from settings dict."""
        data = {
            'hooks': {
                'SessionStart': [
                    {'hooks': [{'type': 'command', 'command': '/path/to/memman/prime.sh'}]},
                    {'hooks': [{'type': 'command', 'command': '/other/tool.sh'}]},
                ],
            },
        }
        remove_claude_hooks(data)
        assert len(data['hooks']['SessionStart']) == 1
        assert 'memman' not in str(data['hooks']['SessionStart'][0])

    def test_add_claude_hooks_selective(self):
        """Add hooks idempotently with selective options."""
        data = {}
        add_claude_hooks_selective(data, '/hooks/dir', remind=True, nudge=False)
        hooks = data['hooks']
        assert 'SessionStart' in hooks
        assert 'UserPromptSubmit' in hooks
        assert 'Stop' not in hooks

    def test_remove_memory_block(self, tmp_path):
        """Remove markers and content between them."""
        p = tmp_path / 'test.md'
        p.write_text('before\n<!-- memman:start -->\nstuff\n<!-- memman:end -->\nafter\n')
        assert remove_memory_block(str(p)) is True
        content = p.read_text()
        assert 'memman' not in content
        assert 'before' in content
        assert 'after' in content

    def test_remove_memory_block_empty_file(self, tmp_path):
        """File deleted if empty after marker removal."""
        p = tmp_path / 'test.md'
        p.write_text('<!-- memman:start -->\nstuff\n<!-- memman:end -->\n')
        assert remove_memory_block(str(p)) is True
        assert not p.exists()

    def test_remove_memory_block_no_markers(self, tmp_path):
        """No markers returns False."""
        p = tmp_path / 'test.md'
        p.write_text('no markers here')
        assert remove_memory_block(str(p)) is False

    def test_add_claude_hooks_with_task_recall(self):
        """task_recall=True produces PreToolUse entry with Task matcher."""
        data = {}
        add_claude_hooks_selective(
            data, '/hooks/dir', task_recall=True)
        hooks = data['hooks']
        assert 'PreToolUse' in hooks
        entries = hooks['PreToolUse']
        assert len(entries) == 1
        assert entries[0]['matcher'] == 'Task'
        assert entries[0]['hooks'][0]['command'].endswith(
            'task_recall.sh')

    def test_add_claude_hooks_task_recall_default_false(self):
        """Default (no task_recall) does NOT create PreToolUse."""
        data = {}
        add_claude_hooks_selective(data, '/hooks/dir')
        hooks = data['hooks']
        assert 'PreToolUse' not in hooks

    def test_remove_claude_hooks_cleans_pretooluse(self):
        """MemMan PreToolUse entries removed, non-memman preserved."""
        data = {
            'hooks': {
                'PreToolUse': [
                    {
                        'hooks': [{'type': 'command',
                                   'command': '/memman/task_recall.sh'}],
                        'matcher': 'Task',
                        },
                    {
                        'hooks': [{'type': 'command',
                                   'command': '/other/enforce.py'}],
                        'matcher': 'Bash',
                        },
                    ],
                },
            }
        remove_claude_hooks(data)
        entries = data['hooks']['PreToolUse']
        assert len(entries) == 1
        assert entries[0]['matcher'] == 'Bash'

    def test_remove_claude_hooks_preserves_non_memman_pretooluse(self):
        """PreToolUse with only non-memman entries is untouched."""
        data = {
            'hooks': {
                'PreToolUse': [
                    {
                        'hooks': [{'type': 'command',
                                   'command': '/other/lint.sh'}],
                        'matcher': 'Bash',
                        },
                    ],
                },
            }
        remove_claude_hooks(data)
        entries = data['hooks']['PreToolUse']
        assert len(entries) == 1
        assert entries[0]['matcher'] == 'Bash'

    def test_add_claude_hooks_appends_to_existing_pretooluse(self):
        """task_recall appends to existing PreToolUse array."""
        data = {
            'hooks': {
                'PreToolUse': [
                    {
                        'hooks': [{'type': 'command',
                                   'command': '/other/enforce.py'}],
                        'matcher': 'Bash',
                        },
                    ],
                },
            }
        add_claude_hooks_selective(
            data, '/hooks/dir', task_recall=True)
        entries = data['hooks']['PreToolUse']
        assert len(entries) == 2
        matchers = {e['matcher'] for e in entries}
        assert matchers == {'Bash', 'Task'}

    def test_add_claude_hooks_with_compact(self):
        """compact=True produces PreCompact entry."""
        data = {}
        add_claude_hooks_selective(
            data, '/hooks/dir', compact=True)
        hooks = data['hooks']
        assert 'PreCompact' in hooks
        entries = hooks['PreCompact']
        assert len(entries) == 1
        assert entries[0]['hooks'][0]['command'].endswith(
            'compact.sh')

    def test_add_claude_hooks_compact_default_false(self):
        """Default (no compact) does NOT create PreCompact."""
        data = {}
        add_claude_hooks_selective(data, '/hooks/dir')
        hooks = data['hooks']
        assert 'PreCompact' not in hooks

    def test_add_claude_hooks_with_exit_plan(self):
        """exit_plan=True produces PreToolUse entry with ExitPlanMode matcher."""
        data = {}
        add_claude_hooks_selective(
            data, '/hooks/dir', exit_plan=True)
        hooks = data['hooks']
        assert 'PreToolUse' in hooks
        entries = hooks['PreToolUse']
        assert len(entries) == 1
        assert entries[0]['matcher'] == 'ExitPlanMode'
        assert entries[0]['hooks'][0]['command'].endswith(
            'exit_plan.sh')

    def test_add_claude_hooks_task_recall_and_exit_plan(self):
        """Both task_recall and exit_plan produce two PreToolUse entries."""
        data = {}
        add_claude_hooks_selective(
            data, '/hooks/dir', task_recall=True, exit_plan=True)
        hooks = data['hooks']
        entries = hooks['PreToolUse']
        assert len(entries) == 2
        matchers = {e['matcher'] for e in entries}
        assert matchers == {'Task', 'ExitPlanMode'}


class TestPermissions:
    """`add_memman_permission`, `remove_memman_permission`."""

    def test_add_memman_permission(self):
        """Adds Bash(memman:*) to allow list. Idempotent."""
        data = {}
        add_memman_permission(data)
        assert 'Bash(memman:*)' in data['permissions']['allow']
        add_memman_permission(data)
        assert data['permissions']['allow'].count(
            'Bash(memman:*)') == 1

    def test_add_memman_permission_existing_allow(self):
        """Appends without disturbing existing entries."""
        data = {'permissions': {'allow': ['Bash(git:*)']}}
        add_memman_permission(data)
        allow = data['permissions']['allow']
        assert allow == ['Bash(git:*)', 'Bash(memman:*)']

    def test_remove_memman_permission(self):
        """Removes Bash(memman:*), preserves others."""
        data = {
            'permissions': {
                'allow': ['Bash(git:*)', 'Bash(memman:*)'],
                },
            }
        remove_memman_permission(data)
        assert data['permissions']['allow'] == ['Bash(git:*)']

    def test_remove_memman_permission_missing(self):
        """No-op when Bash(memman:*) not present."""
        data = {'permissions': {'allow': ['Bash(git:*)']}}
        remove_memman_permission(data)
        assert data['permissions']['allow'] == ['Bash(git:*)']

    def test_register_hooks_no_permission(self, tmp_path):
        """claude_register_hooks() does not add Bash(memman:*) to settings."""
        config_dir = str(tmp_path / '.claude')
        hooks_dir = os.path.join(config_dir, 'hooks', 'memman')
        pathlib.Path(hooks_dir).mkdir(parents=True)
        claude_register_hooks(config_dir, remind=True, nudge=True,
                              task_recall=True)
        data = read_json_file(os.path.join(config_dir, 'settings.json'))
        allow = data.get('permissions', {}).get('allow', [])
        assert 'Bash(memman:*)' not in allow


class TestPrimeAndCompactHooks:
    """Prime / compact hook script execution."""

    def test_compact_hook_script(self, tmp_path):
        """Compact hook writes flag file with session info."""
        from importlib.resources import files as pkg_files
        script = str(
            pkg_files('memman.setup.assets')
            .joinpath('claude/compact.sh'))

        result = subprocess.run(
            ['bash', script],
            check=False, input='{"session_id": "test-abc-123", "trigger": "manual"}',
            capture_output=True, text=True,
            env={**os.environ, 'HOME': str(tmp_path)})
        assert result.returncode == 0

        flag = tmp_path / '.memman' / 'compact' / 'test-abc-123.json'
        assert flag.exists()
        data = json.loads(flag.read_text())
        assert data['trigger'] == 'manual'
        assert 'ts' in data

    def test_compact_hook_script_no_session(self, tmp_path):
        """Compact hook writes no flag when session_id is missing."""
        from importlib.resources import files as pkg_files
        script = str(
            pkg_files('memman.setup.assets')
            .joinpath('claude/compact.sh'))

        result = subprocess.run(
            ['bash', script],
            check=False, input='{"trigger": "auto"}',
            capture_output=True, text=True,
            env={**os.environ, 'HOME': str(tmp_path)})
        assert result.returncode == 0

        compact_dir = tmp_path / '.memman' / 'compact'
        assert not compact_dir.exists()

    def test_prime_hook_compact_source(self, tmp_path):
        """Prime hook outputs recall instruction on compact source."""
        from importlib.resources import files as pkg_files
        script = str(
            pkg_files('memman.setup.assets')
            .joinpath('claude/prime.sh'))

        compact_dir = tmp_path / '.memman' / 'compact'
        compact_dir.mkdir(parents=True)
        flag = compact_dir / 'sess-42.json'
        flag.write_text('{"trigger":"manual","ts":"2026-01-01T00:00:00Z"}')

        result = subprocess.run(
            ['bash', script],
            check=False, input='{"source": "compact", "session_id": "sess-42"}',
            capture_output=True, text=True,
            env={**os.environ, 'HOME': str(tmp_path)})
        assert result.returncode == 0
        assert 'compacted' in result.stdout
        assert 'manual' in result.stdout
        assert 'recall' in result.stdout.lower()
        assert flag.exists()

    def test_prime_hook_compact_no_flag(self, tmp_path):
        """Prime hook outputs recall instruction even without flag file."""
        from importlib.resources import files as pkg_files
        script = str(
            pkg_files('memman.setup.assets')
            .joinpath('claude/prime.sh'))

        result = subprocess.run(
            ['bash', script],
            check=False, input='{"source": "compact", "session_id": "no-flag"}',
            capture_output=True, text=True,
            env={**os.environ, 'HOME': str(tmp_path)})
        assert result.returncode == 0
        assert 'compacted' in result.stdout
        assert 'auto' in result.stdout

    def test_prime_hook_normal_source(self, tmp_path):
        """Prime hook does NOT output recall instruction on normal startup."""
        from importlib.resources import files as pkg_files
        script = str(
            pkg_files('memman.setup.assets')
            .joinpath('claude/prime.sh'))

        result = subprocess.run(
            ['bash', script],
            check=False, input='{"source": "startup"}',
            capture_output=True, text=True,
            env={**os.environ, 'HOME': str(tmp_path)})
        assert result.returncode == 0
        assert 'compacted' not in result.stdout

    def test_exit_plan_hook_advisory(self, tmp_path):
        """Exit plan hook passes with advisory message (non-blocking)."""
        from importlib.resources import files as pkg_files
        script = str(
            pkg_files('memman.setup.assets')
            .joinpath('claude/exit_plan.sh'))

        result = subprocess.run(
            ['bash', script],
            check=False, input='{"session_id": "test-plan-123"}',
            capture_output=True, text=True,
            env={**os.environ, 'HOME': str(tmp_path)})
        assert result.returncode == 0
        assert 'memman' in result.stdout.lower()
        assert 'plan' in result.stdout.lower()

        flag_dir = tmp_path / '.memman' / 'exit_plan'
        assert not flag_dir.exists()

    def _stop_script():
        """Return path to stop.sh asset."""
        from importlib.resources import files as pkg_files
        return str(
            pkg_files('memman.setup.assets')
            .joinpath('claude/stop.sh'))

    def _prompt_script():
        """Return path to user_prompt.sh asset."""
        from importlib.resources import files as pkg_files
        return str(
            pkg_files('memman.setup.assets')
            .joinpath('claude/user_prompt.sh'))

    def _run_hook(self: str, input_json: str,
                  tmp_home: pathlib.Path) -> subprocess.CompletedProcess:
        """Run a hook script with HOME overridden."""
        return subprocess.run(
            ['bash', self],
            check=False, input=input_json,
            capture_output=True, text=True,
            env={**os.environ, 'HOME': str(tmp_home)})

    def test_prime_hook_emits_guide_content(self, tmp_path):
        """prime.sh with memman on PATH emits guide content via `memman prime`.
        """
        from importlib.resources import files as pkg_files
        script = str(pkg_files('memman.setup.assets')
                     .joinpath('claude/prime.sh'))

        shim_dir = tmp_path / 'shim-bin'
        shim_dir.mkdir()
        shim = shim_dir / 'memman'
        shim.write_text(
            '#!/bin/bash\n'
            'cat <<EOF\n'
            '[memman] Memory active.\n'
            'SHIM-GUIDE-MARKER\n'
            'EOF\n'
            )
        shim.chmod(0o755)

        env = {
            **os.environ,
            'HOME': str(tmp_path),
            'PATH': f'{shim_dir}:{os.environ.get("PATH", "")}',
            }
        result = subprocess.run(
            ['bash', script],
            check=False, input='{}',
            capture_output=True, text=True,
            env=env)
        assert result.returncode == 0
        assert 'SHIM-GUIDE-MARKER' in result.stdout

    def test_prime_hook_warns_when_memman_missing(self, tmp_path):
        """prime.sh emits a warning and exits cleanly when memman is not on PATH.
        """
        from importlib.resources import files as pkg_files
        script = str(pkg_files('memman.setup.assets')
                     .joinpath('claude/prime.sh'))

        env = {'HOME': str(tmp_path), 'PATH': '/usr/bin:/bin'}
        result = subprocess.run(
            ['/bin/bash', script],
            check=False, input='{}',
            capture_output=True, text=True,
            env=env)
        assert result.returncode == 0
        assert 'not on PATH' in result.stdout


class TestStopHook:
    """`stop.sh` exit/block behavior across session resets."""

    def test_stop_hook_first_stop_blocks(self, tmp_path):
        """First stop with session_id blocks for memory eval."""
        result = _run_hook(
            _stop_script(),
            '{"stop_hook_active": false, "session_id": "sess-1"}',
            tmp_path)
        assert result.returncode == 0
        output = json.loads(result.stdout.strip())
        assert output['decision'] == 'block'
        assert 'memman' in output['reason'].lower()

    def test_stop_hook_second_stop_silent(self, tmp_path):
        """Second stop in same turn is silent (flag dir exists)."""
        _run_hook(
            _stop_script(),
            '{"stop_hook_active": false, "session_id": "sess-2"}',
            tmp_path)
        result = _run_hook(
            _stop_script(),
            '{"stop_hook_active": false, "session_id": "sess-2"}',
            tmp_path)
        assert result.returncode == 0
        assert result.stdout.strip() == ''

    def test_stop_hook_blocks_again_after_reset(self, tmp_path):
        """After rmdir reset, stop blocks again."""
        _run_hook(
            _stop_script(),
            '{"stop_hook_active": false, "session_id": "sess-3"}',
            tmp_path)
        flag_dir = tmp_path / '.memman' / 'stop_fired' / 'sess-3'
        assert flag_dir.is_dir()
        flag_dir.rmdir()

        result = _run_hook(
            _stop_script(),
            '{"stop_hook_active": false, "session_id": "sess-3"}',
            tmp_path)
        output = json.loads(result.stdout.strip())
        assert output['decision'] == 'block'

    def test_stop_hook_active_silent(self, tmp_path):
        """stop_hook_active=true bypasses gate entirely."""
        result = _run_hook(
            _stop_script(),
            '{"stop_hook_active": true, "session_id": "sess-4"}',
            tmp_path)
        assert result.returncode == 0
        assert result.stdout.strip() == ''

    def test_stop_hook_no_session_id_blocks(self, tmp_path):
        """Missing session_id falls back to always blocking."""
        result = _run_hook(
            _stop_script(),
            '{"stop_hook_active": false}',
            tmp_path)
        assert result.returncode == 0
        output = json.loads(result.stdout.strip())
        assert output['decision'] == 'block'


class TestUserPromptHook:
    """`user_prompt.sh` flag-clearing semantics."""

    def test_user_prompt_clears_stop_flag(self, tmp_path):
        """user_prompt.sh removes stop_fired flag dir."""
        flag_dir = tmp_path / '.memman' / 'stop_fired' / 'sess-5'
        flag_dir.mkdir(parents=True)

        result = _run_hook(
            _prompt_script(),
            '{"session_id": "sess-5"}',
            tmp_path)
        assert result.returncode == 0
        assert not flag_dir.exists()
        assert 'recall' in result.stdout.lower()

    def test_user_prompt_no_flag_dir(self, tmp_path):
        """user_prompt.sh exits cleanly when no flag dir exists."""
        result = _run_hook(
            _prompt_script(),
            '{"session_id": "sess-6"}',
            tmp_path)
        assert result.returncode == 0
        assert 'recall' in result.stdout.lower()

    def test_user_prompt_no_session_id(self, tmp_path):
        """user_prompt.sh exits cleanly when session_id is missing."""
        result = _run_hook(
            _prompt_script(),
            '{}',
            tmp_path)
        assert result.returncode == 0
        assert 'recall' in result.stdout.lower()


class TestSetupCli:
    """`memman guide` and `memman prime` CLI commands."""

    def test_guide_command_prints_shipped_content(self):
        """`memman guide` prints the shipped guide.md from the package."""
        from importlib.resources import files as pkg_files
        shipped = (pkg_files('memman.setup.assets')
                   .joinpath('claude/guide.md').read_text())
        runner = CliRunner()
        result = runner.invoke(cli, ['guide'])
        assert result.exit_code == 0
        assert shipped.strip() in result.output

    def test_guide_command_ignores_any_local_override_file(self, tmp_path, monkeypatch):
        """`memman guide` must NOT read ~/.memman/prompt/guide.local.md.

        Confirms the override mechanism is gone; any leftover file at the
        old path has zero effect on output.
        """
        monkeypatch.setattr(pathlib.Path, 'home', lambda: tmp_path)
        prompt_dir = tmp_path / '.memman' / 'prompt'
        prompt_dir.mkdir(parents=True)
        override = prompt_dir / 'guide.local.md'
        override.write_text('USER-OVERRIDE-MARKER-SHOULD-NOT-APPEAR\n')
        runner = CliRunner()
        result = runner.invoke(cli, ['guide'])
        assert result.exit_code == 0
        assert 'USER-OVERRIDE-MARKER-SHOULD-NOT-APPEAR' not in result.output
        assert '<!-- user overrides -->' not in result.output

    def test_prime_command_emits_status_and_guide(self, tmp_path, monkeypatch):
        """`memman prime` emits a status line and the guide content."""
        monkeypatch.setattr(pathlib.Path, 'home', lambda: tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ['prime'], input='{}')
        assert result.exit_code == 0
        assert '[memman] Memory active' in result.output
        from importlib.resources import files as pkg_files
        shipped = (pkg_files('memman.setup.assets')
                   .joinpath('claude/guide.md').read_text())
        assert shipped.strip() in result.output

    def test_prime_honors_memman_store_env(self, tmp_path, monkeypatch):
        """`memman prime` targets MEMMAN_STORE when set, not just the
        active-store file.
        """
        from memman.store.db import default_data_dir, open_db, store_dir

        monkeypatch.setattr(pathlib.Path, 'home', lambda: tmp_path)
        data_dir = default_data_dir()
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(store_dir(data_dir, 'default')).mkdir(
            parents=True, exist_ok=True)
        db = open_db(store_dir(data_dir, 'default'))
        db.close()
        # Create a different store that will be targeted via MEMMAN_STORE.
        other = store_dir(data_dir, 'work')
        pathlib.Path(other).mkdir(parents=True, exist_ok=True)
        db = open_db(other)
        db.close()

        monkeypatch.setenv('MEMMAN_STORE', 'work')
        runner = CliRunner()
        result = runner.invoke(cli, ['prime'], input='{}')
        assert result.exit_code == 0
        assert '[memman] Memory active' in result.output

    def test_prime_command_emits_compact_hint(self, tmp_path, monkeypatch):
        """`memman prime` emits the compact-recall hint when source=compact."""
        monkeypatch.setattr(pathlib.Path, 'home', lambda: tmp_path)
        runner = CliRunner()
        payload = json.dumps({'source': 'compact', 'session_id': 'sess-x'})
        result = runner.invoke(cli, ['prime'], input=payload)
        assert result.exit_code == 0
        assert 'Context was just compacted' in result.output

    def test_prime_command_reads_compact_flag_trigger(self, tmp_path, monkeypatch):
        """`memman prime` picks up trigger from compact flag file."""
        monkeypatch.setattr(pathlib.Path, 'home', lambda: tmp_path)
        compact_dir = tmp_path / '.memman' / 'compact'
        compact_dir.mkdir(parents=True)
        (compact_dir / 'sess-y.json').write_text(
            json.dumps({'trigger': 'manual'}))
        runner = CliRunner()
        payload = json.dumps({'source': 'compact', 'session_id': 'sess-y'})
        result = runner.invoke(cli, ['prime'], input=payload)
        assert result.exit_code == 0
        assert 'compacted (manual)' in result.output


class TestSymlinks:
    """`claude_write_skill` / `claude_write_hook` symlink behavior."""

    def test_claude_write_skill_creates_symlink(self, tmp_path):
        """claude_write_skill creates a symlink to the shipped SKILL.md."""
        from importlib.resources import files as pkg_files

        from memman.setup.claude import claude_write_skill
        config = tmp_path / 'claude'
        link_path = claude_write_skill(str(config))
        link = pathlib.Path(link_path)
        assert link.is_symlink()
        target = pathlib.Path(str(pkg_files('memman.setup.assets')
                                  .joinpath('claude/SKILL.md'))).resolve()
        assert link.resolve() == target

    def test_claude_write_hook_creates_symlink(self, tmp_path):
        """claude_write_hook creates a symlink to the shipped hook script."""
        from importlib.resources import files as pkg_files

        from memman.setup.claude import claude_write_hook
        config = tmp_path / 'claude'
        link_path = claude_write_hook(str(config), 'prime.sh')
        link = pathlib.Path(link_path)
        assert link.is_symlink()
        target = pathlib.Path(str(pkg_files('memman.setup.assets')
                                  .joinpath('claude/prime.sh'))).resolve()
        assert link.resolve() == target

    def test_symlink_replaces_stale_symlink(self, tmp_path):
        """Re-install replaces a dangling symlink with a live one."""
        from memman.setup.claude import claude_write_skill
        config = tmp_path / 'claude'
        link = config / 'skills' / 'memman' / 'SKILL.md'
        link.parent.mkdir(parents=True)
        link.symlink_to('/nonexistent/path')
        assert link.is_symlink()
        assert not link.exists()
        claude_write_skill(str(config))
        assert link.is_symlink()
        assert link.exists()

    def test_symlink_replaces_regular_file(self, tmp_path):
        """Re-install replaces a pre-existing regular file with a symlink."""
        from memman.setup.claude import claude_write_skill
        config = tmp_path / 'claude'
        link = config / 'skills' / 'memman' / 'SKILL.md'
        link.parent.mkdir(parents=True)
        link.write_text('stale pre-symlink content')
        assert not link.is_symlink()
        claude_write_skill(str(config))
        assert link.is_symlink()

    def test_uninstall_removes_symlink_not_target(self, tmp_path):
        """claude_uninstall removes the symlink without touching the target."""
        from importlib.resources import files as pkg_files

        from memman.setup.claude import claude_uninstall, claude_write_skill
        config = tmp_path / 'claude'
        claude_write_skill(str(config))
        target = pathlib.Path(str(pkg_files('memman.setup.assets')
                                  .joinpath('claude/SKILL.md'))).resolve()
        target_bytes = target.read_bytes()
        claude_uninstall(str(config))
        assert not (config / 'skills' / 'memman' / 'SKILL.md').exists()
        assert target.exists()
        assert target.read_bytes() == target_bytes
