"""Tests for memman.setup - settings, markdown, detection."""

import html
import json
import os
import pathlib
import re
import subprocess

import click
import pytest
from click.testing import CliRunner
from memman.cli import cli, list_claude_permissions
from memman.setup.markdown import remove_memory_block
from memman.setup.settings import add_claude_hooks_selective
from memman.setup.settings import add_memman_permission, read_json_file
from memman.setup.settings import remove_claude_hooks, remove_if_empty
from memman.setup.settings import remove_memman_permission, strip_json5
from memman.setup.settings import write_json_file


# The host truncates hook stdout above this many bytes and persists the
# remainder to a file it never reads back.
HOOK_STDOUT_LIMIT = 10_000


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
        add_claude_hooks_selective(data, '/hooks/dir', remind=True)
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
        """Verify the pre-delegation entry matches the delegation tool.

        Mutation: a matcher naming no delegation tool - left empty, or
            pointed at a tool that spawns nothing - so the reminder
            never reaches the agent before a sub-agent launch.
        Oracle: the tool name Claude Code canonicalizes a delegation to,
            Agent, required among the matcher's tokens.
        """
        data = {}
        add_claude_hooks_selective(
            data, '/hooks/dir', task_recall=True)
        entries = data['hooks']['PreToolUse']
        assert len(entries) == 1
        matched = set(entries[0]['matcher'].split('|'))
        assert 'Agent' in matched
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
        assert matchers == {'Bash', 'Agent|Task'}

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
        assert matchers == {'Agent|Task', 'ExitPlanMode'}


class TestPermissions:
    """`add_memman_permission`, `remove_memman_permission`."""

    def test_add_memman_permission(self):
        """Adds every curated entry to allow. Idempotent."""
        data = {}
        add_memman_permission(data, list_claude_permissions())
        allow = data['permissions']['allow']
        for entry in list_claude_permissions():
            assert allow.count(entry) == 1
        add_memman_permission(data, list_claude_permissions())
        for entry in list_claude_permissions():
            assert data['permissions']['allow'].count(entry) == 1

    def test_add_memman_permission_existing_allow(self):
        """Appends curated entries without disturbing existing ones."""
        data = {'permissions': {'allow': ['Bash(git:*)']}}
        add_memman_permission(data, list_claude_permissions())
        allow = data['permissions']['allow']
        assert allow[0] == 'Bash(git:*)'
        assert allow[1:] == list(list_claude_permissions())

    def test_remove_memman_permission(self):
        """Drops every memman-containing entry from allow; preserves others."""
        data = {
            'permissions': {
                'allow': ['Bash(git:*)', *list_claude_permissions()],
                },
            }
        remove_memman_permission(data)
        assert data['permissions']['allow'] == ['Bash(git:*)']

    def test_remove_memman_permission_missing(self):
        """No-op when no memman entries present."""
        data = {'permissions': {'allow': ['Bash(git:*)']}}
        remove_memman_permission(data)
        assert data['permissions']['allow'] == ['Bash(git:*)']

    def test_remove_memman_permission_sweeps_user_added(self):
        """Removes hypothetical user-added Bash(memman ...) entries too."""
        data = {
            'permissions': {
                'allow': [
                    'Bash(git:*)',
                    'Bash(memman recall:*)',
                    'Bash(memman:*)',
                    'Bash(memman foo)',
                    ],
                },
            }
        remove_memman_permission(data)
        assert data['permissions']['allow'] == ['Bash(git:*)']

    def test_remove_memman_permission_sweeps_deny_and_ask(self):
        """Sweeps deny and ask sections as well as allow."""
        data = {
            'permissions': {
                'allow': ['Bash(memman recall:*)'],
                'deny': ['Bash(memman uninstall:*)', 'Bash(rm:*)'],
                'ask': ['Bash(memman scheduler stop)'],
                },
            }
        remove_memman_permission(data)
        assert data['permissions'] == {'deny': ['Bash(rm:*)']}

    def test_remove_memman_permission_drops_empty_permissions(self):
        """Empty permissions dict is removed entirely."""
        data = {'permissions': {'allow': list(list_claude_permissions())}}
        remove_memman_permission(data)
        assert 'permissions' not in data

    def test_install_uninstall_roundtrip(self):
        """Add then remove returns dict to starting state."""
        before = {'permissions': {'allow': ['Bash(git:*)']}}
        data = json.loads(json.dumps(before))
        add_memman_permission(data, list_claude_permissions())
        remove_memman_permission(data)
        assert data == before

    def test_hook_wiring_adds_no_permission(self):
        """Verify wiring hooks never grants a memman Bash permission.

        Mutation: a permission write folded into the hook wiring, so a
            caller asking only for hooks silently widens what memman
            may run.
        Oracle: the curated permission list, none of whose entries may
            appear in allow after the call.
        """
        data: dict = {}
        add_claude_hooks_selective(
            data, '/hooks/dir', remind=True, compact=True,
            task_recall=True, exit_plan=True)
        allow = data.get('permissions', {}).get('allow', [])
        for entry in list_claude_permissions():
            assert entry not in allow


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


    def test_prime_payload_reaches_the_model_whole(self, tmp_path):
        """Verify the SessionStart payload fits the host's stdout limit.

        Mutation: guide.md grown past the limit, or another line added
            to prime, either of which truncates the payload to a
            preview and drops the rest without erroring.
        Oracle: HOOK_STDOUT_LIMIT, the byte count above which the host
            replaces the payload with a preview, checked against the
            emitted bytes; plus both halves of the contract, which sit
            past the cut today.
        """
        result = CliRunner().invoke(
            cli, ['prime'], input='{}',
            env={'MEMMAN_DATA_DIR': str(tmp_path)})

        assert result.exit_code == 0
        assert len(result.output.encode()) < HOOK_STDOUT_LIMIT
        assert 'memman recall' in result.output
        assert 'memman remember' in result.output


class TestUserPromptHook:
    """`user_prompt.sh` recall reminder and session-id hint."""

    def test_session_id_reaches_the_reminder(self, tmp_path):
        """Verify the hook relays the session id it was handed.

        Mutation: dropping the SESSION_HINT line, so the agent is never
            told which --session to stamp its recalls and writes with.
        Oracle: hand-written pair straddling the hook's one branch - the
            exact id when the payload carries one, no --session at all
            when it does not.
        """
        with_id = _run_hook(
            _prompt_script(),
            '{"session_id": "sess-6"}',
            tmp_path)
        assert with_id.returncode == 0
        assert 'recall' in with_id.stdout.lower()
        assert '--session sess-6' in with_id.stdout

        without_id = _run_hook(
            _prompt_script(),
            '{}',
            tmp_path)
        assert without_id.returncode == 0
        assert 'recall' in without_id.stdout.lower()
        assert '--session' not in without_id.stdout

    def test_carries_no_write_instruction(self, tmp_path):
        """Verify the reminder asks for a recall and never for a write.

        Mutation: restoring the trailing "After responding, evaluate:
            remember needed?" clause, which UserPromptSubmit delivers
            before the turn whose end it asks about.
        Oracle: the two phrases that made up the deleted clause, checked
            against a payload that still carries a session id, so the
            flag-stamping mention of remember in the hint stays allowed.
        """
        out = _run_hook(
            _prompt_script(),
            '{"session_id": "sess-7"}',
            tmp_path)
        assert out.returncode == 0
        assert 'recall' in out.stdout.lower()
        assert 'after responding' not in out.stdout.lower()
        assert 'remember needed' not in out.stdout.lower()


class TestNoBlockingHook:
    """No shipped hook re-invokes the model after a turn has ended."""

    def test_no_shipped_hook_emits_a_block_decision(self, tmp_path):
        """Verify every shipped hook emits plain text, never a block decision.

        Mutation: a hook re-emitting {"decision": "block"}, which restarts
            the model after the turn already ended.
        Oracle: hand-written expectation that the shipped hook set emits
            text only - stdout parsed as JSON carries no decision key.
        """
        from importlib.resources import files as pkg_files
        assets = pkg_files('memman.setup.assets')
        payload = json.dumps({
            'stop_hook_active': False,
            'session_id': 'sess-block-probe',
            'trigger': 'auto',
            })
        trees = (
            ('claude', assets.joinpath('claude')),
            ('nanoclaw/hooks', assets.joinpath('nanoclaw').joinpath('hooks')),
            )
        offenders = []
        checked = 0
        for label, tree in trees:
            for entry in tree.iterdir():
                if not entry.name.endswith('.sh'):
                    continue
                checked += 1
                result = subprocess.run(
                    ['bash', str(entry)],
                    check=False, input=payload,
                    capture_output=True, text=True,
                    env={'HOME': str(tmp_path), 'PATH': '/usr/bin:/bin'})
                out = result.stdout.strip()
                if not out.startswith('{'):
                    continue
                try:
                    parsed = json.loads(out)
                except ValueError:
                    continue
                if isinstance(parsed, dict) and 'decision' in parsed:
                    offenders.append(f'{label}/{entry.name}')
        assert checked > 0
        assert offenders == []

    def test_hook_wiring_offers_no_stop_switch(self):
        """Verify no argument combination registers a Claude Code Stop hook.

        Mutation: a resurrected nudge branch putting a script back under
            the Stop event.
        Oracle: hand-enumerated event set, with every boolean switch the
            function offers turned on.
        """
        import inspect
        switches = {
            name: True
            for name, param in
            inspect.signature(add_claude_hooks_selective).parameters.items()
            if isinstance(param.default, bool)
            }
        data: dict = {}
        add_claude_hooks_selective(data, '/hooks/dir', **switches)
        assert 'Stop' not in data['hooks']
        assert set(data['hooks']) == {
            'SessionStart',
            'UserPromptSubmit',
            'PreCompact',
            'PreToolUse',
            }

    def test_shipped_assets_name_no_stop_fired_directory(self):
        """Verify the stop_fired turn gate is absent from every shipped asset.

        Mutation: a partial deletion leaving prime.sh's stale sweep or
            user_prompt.sh's rmdir reading a directory no hook writes.
        Oracle: hand-counted zero occurrences across the asset tree.
        """
        root = (pathlib.Path(__file__).resolve().parents[1]
                / 'src' / 'memman' / 'setup' / 'assets')
        hits = [
            str(path.relative_to(root))
            for path in root.rglob('*')
            if path.is_file() and 'stop_fired' in path.read_text(errors='replace')
            ]
        assert hits == []

    def test_nanoclaw_ships_no_stop_hook(self):
        """Verify the nanoclaw surface ships no Stop hook and names none.

        Mutation: a single-surface ship that deletes the Claude Code hook
            and leaves nanoclaw's ungated block in place.
        Oracle: hand-written two-part expectation - no stop.sh asset, and
            no Stop registration in the integrator instructions.
        """
        from importlib.resources import files as pkg_files
        nanoclaw = pkg_files('memman.setup.assets').joinpath('nanoclaw')
        names = sorted(
            entry.name for entry in nanoclaw.joinpath('hooks').iterdir())
        assert 'stop.sh' not in names
        skill = nanoclaw.joinpath('SKILL.md').read_text()
        assert 'stop.sh' not in skill
        assert 'Stop:' not in skill


class TestPreToolUseHooksReachTheModel:
    """PreToolUse reminders ride the one channel Claude Code reads."""

    def test_pretooluse_hooks_emit_additional_context(self, tmp_path):
        """Verify every PreToolUse hook speaks additionalContext.

        Mutation: a PreToolUse hook echoing plain text, which Claude
            Code discards for every event but SessionStart and
            UserPromptSubmit, so the reminder never reaches the model.
        Oracle: the registered event set - every script the installer
            files under PreToolUse, each required to emit JSON naming
            the event and carrying a memman line.
        """
        from importlib.resources import files as pkg_files
        registered: dict = {}
        add_claude_hooks_selective(
            registered, '/hooks/dir', remind=True, compact=True,
            task_recall=True, exit_plan=True)
        scripts = [
            pathlib.Path(hook['command']).name
            for entry in registered['hooks']['PreToolUse']
            for hook in entry['hooks']
            ]
        assert scripts
        for name in scripts:
            script = str(
                pkg_files('memman.setup.assets')
                .joinpath(f'claude/{name}'))
            result = _run_hook(
                script, '{"session_id": "sess-ctx"}', tmp_path)
            assert result.returncode == 0, name
            emitted = json.loads(result.stdout)['hookSpecificOutput']
            assert emitted['hookEventName'] == 'PreToolUse', name
            assert '[memman]' in emitted['additionalContext'], name


class TestDocsMatchShippedHooks:
    """Prose and diagram counts track the shipped hook set."""

    def test_doc_hook_counts_match_the_asset_tree(self):
        """Verify every doc site naming a hook count names the real one.

        Mutation: a hook added or deleted on some doc sites only - the
            deletion that corrected three README counts and left the
            fourth reading six, and regenerated one diagram of two.
        Oracle: the shipped asset tree counted directly - the .sh files
            under assets/claude and under assets/nanoclaw/hooks.
        """
        assets = (pathlib.Path(__file__).resolve().parents[1]
                  / 'src' / 'memman' / 'setup' / 'assets')
        shipped = {
            len(list(assets.glob('claude/*.sh'))),
            len(list(assets.glob('nanoclaw/hooks/*.sh'))),
            }
        words = {
            'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            }
        readme = (pathlib.Path(__file__).resolve().parents[1]
                  / 'README.md').read_text()
        counted = {
            words[match.group(1).lower()]
            for match in re.finditer(
                r'\b(\w+)\s+(?:lifecycle\s+)?hooks?\b', readme, re.I)
            if match.group(1).lower() in words
            }
        assert counted == shipped

    def test_architecture_diagram_names_the_shipped_hooks(self):
        """Verify the architecture diagram lists the shipped hook roles.

        Mutation: a hook deleted from the installer and left standing in
            the diagram, which is the PNG the design docs embed.
        Oracle: the shipped asset tree - one role per .sh file under
            assets/claude, counted independently of the diagram.
        """
        root = pathlib.Path(__file__).resolve().parents[1]
        shipped = len(list(
            (root / 'src' / 'memman' / 'setup' / 'assets')
            .glob('claude/*.sh')))
        diagram = (root / 'docs' / 'diagrams'
                   / '01-system-architecture.drawio').read_text()
        label = re.search(r'id="a_hooks" value="([^"]*)"', diagram)
        assert label is not None
        text = re.sub(r'<[^>]+>', '/', html.unescape(label.group(1)))
        roles = [
            token.strip() for token in text.split('/')
            if token.strip() and token.strip().lower() != 'hooks'
            ]
        assert len(roles) == shipped


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

    def test_guide_names_no_host_tool(self):
        """Verify the shared guide.md names no host-specific tool.

        Mutation: the shared guide naming Bash, so the OpenClaw
            bootstrap names a tool that host does not expose.
        Oracle: the exact tokens 'via Bash' and 'the `exec` tool' read
            from the two SKILL files confirm they disagree, so the shared
            text must name neither; the emitted guide carries no 'Bash'
            and no 'exec' at all.
        """
        from importlib.resources import files as pkg_files
        assets = pkg_files('memman.setup.assets')
        claude_skill = assets.joinpath('claude/SKILL.md').read_text()
        openclaw_skill = assets.joinpath('openclaw/SKILL.md').read_text()
        assert 'via Bash' in claude_skill
        assert 'the `exec` tool' in openclaw_skill
        runner = CliRunner()
        result = runner.invoke(cli, ['guide'])
        assert result.exit_code == 0
        assert 'Bash' not in result.output
        assert 'exec' not in result.output

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


class TestInstallLoopErrorSurfacing:
    """Per-env install errors are surfaced with detail (F.4)."""

    def test_install_loop_surfaces_per_env_errors(
            self, tmp_path, monkeypatch):
        """A failing `_install_env` is logged and rolled into the
        ClickException detail string.

        Pre-F.4 the loop counted errors and raised an opaque
        '%d error(s)' message. The new shape includes per-env name +
        exception text.
        """
        from memman.setup import claude as claude_setup

        def _boom(env, data_dir, no_wizard=False):
            raise RuntimeError(f'boom-{env["name"]}')

        monkeypatch.setattr(claude_setup, '_install_env', _boom)
        monkeypatch.setattr(
            claude_setup, 'install_scheduler',
            lambda data_dir, knobs: {
                'platform': 'noop', 'env_actions': [], 'actions': []})

        envs = [
            {'name': 'envA', 'detected': True, 'display': 'envA',
             'version': '1.0', 'config_dir': '/tmp/a'},
            {'name': 'envB', 'detected': True, 'display': 'envB',
             'version': '1.0', 'config_dir': '/tmp/b'},
            ]

        with pytest.raises(click.ClickException) as exc:
            claude_setup._run_install_flow(
                envs, target='', data_dir=str(tmp_path / 'memman'),
                knobs={})
        msg = str(exc.value.message)
        assert 'envA' in msg
        assert 'envB' in msg
        assert 'boom-envA' in msg
        assert 'boom-envB' in msg


class TestInstallConsent:
    """`_install_claude_code` TTY consent flow for permissions."""

    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        """Set up a clean Claude Code config dir and stub heavy deps."""
        config_dir = tmp_path / '.claude'
        config_dir.mkdir()
        monkeypatch.setenv('HOME', str(tmp_path))
        from memman.setup import claude as claude_setup
        monkeypatch.setattr(claude_setup, '_init_default_store',
                            lambda data_dir: None)
        return {'name': 'claude-code', 'config_dir': str(config_dir)}

    def _allow(self, config_dir: str) -> list:
        path = os.path.join(config_dir, 'settings.json')
        data = read_json_file(path)
        return data.get('permissions', {}).get('allow', [])

    def test_tty_consent_accept_writes_permissions(
            self, env, tmp_path, monkeypatch):
        """Interactive accept writes all curated entries."""
        from memman.setup import claude as claude_setup
        monkeypatch.setattr('sys.stdin.isatty', lambda: True)
        monkeypatch.setattr(
            'memman.setup.claude.click.confirm',
            lambda *a, **kw: True)
        claude_setup._install_claude_code(
            env, data_dir=str(tmp_path / 'memman'))
        allow = self._allow(env['config_dir'])
        for entry in list_claude_permissions():
            assert entry in allow

    def test_tty_consent_decline_skips_permissions(
            self, env, tmp_path, monkeypatch):
        """Interactive decline leaves permissions untouched; hooks present."""
        from memman.setup import claude as claude_setup
        monkeypatch.setattr('sys.stdin.isatty', lambda: True)
        monkeypatch.setattr(
            'memman.setup.claude.click.confirm',
            lambda *a, **kw: False)
        claude_setup._install_claude_code(
            env, data_dir=str(tmp_path / 'memman'))
        allow = self._allow(env['config_dir'])
        for entry in list_claude_permissions():
            assert entry not in allow
        data = read_json_file(
            os.path.join(env['config_dir'], 'settings.json'))
        assert 'hooks' in data
        assert data['hooks']

    def test_no_wizard_skips_prompt_and_writes(
            self, env, tmp_path, monkeypatch):
        """`no_wizard=True` writes silently with no confirm call."""
        from memman.setup import claude as claude_setup
        monkeypatch.setattr('sys.stdin.isatty', lambda: True)
        confirm_calls: list = []
        monkeypatch.setattr(
            'memman.setup.claude.click.confirm',
            lambda *a, **kw: confirm_calls.append(True) or True)
        claude_setup._install_claude_code(
            env, data_dir=str(tmp_path / 'memman'), no_wizard=True)
        assert confirm_calls == []
        allow = self._allow(env['config_dir'])
        for entry in list_claude_permissions():
            assert entry in allow

    def test_non_tty_skips_prompt_and_writes(
            self, env, tmp_path, monkeypatch):
        """Non-TTY writes silently with no confirm call."""
        from memman.setup import claude as claude_setup
        monkeypatch.setattr('sys.stdin.isatty', lambda: False)
        confirm_calls: list = []
        monkeypatch.setattr(
            'memman.setup.claude.click.confirm',
            lambda *a, **kw: confirm_calls.append(True) or True)
        claude_setup._install_claude_code(
            env, data_dir=str(tmp_path / 'memman'))
        assert confirm_calls == []
        allow = self._allow(env['config_dir'])
        for entry in list_claude_permissions():
            assert entry in allow
