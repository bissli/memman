"""Tests for memman.setup - settings, markdown, detection.
"""

import copy
import html
import inspect
import json
import os
import pathlib
import re
import subprocess
from importlib.resources import files

import click
import pytest
from click.testing import CliRunner
from memman import config
from memman.cli import cli, list_claude_permissions
from memman.setup import claude as claude_setup
from memman.setup.claude import claude_uninstall, claude_write_hook
from memman.setup.claude import claude_write_skill
from memman.setup.settings import add_claude_hooks_selective
from memman.setup.settings import add_memman_permission, read_json_file
from memman.setup.settings import remove_claude_hooks, remove_if_empty
from memman.setup.settings import remove_memman_permission, strip_json5
from memman.setup.settings import write_json_file
from memman.store.db import open_db, store_dir, write_active
from memman.store.node import insert_insight
from tests.conftest import _create_seeded_store, make_insight

# The host truncates hook stdout above this many bytes and persists the
# remainder to a file it never reads back.
HOOK_STDOUT_LIMIT = 10_000


def _prompt_script() -> str:
    """Return path to user_prompt.sh asset.
    """
    return str(
        files('memman.setup.assets')
        .joinpath('claude/user_prompt.sh'))


def _run_hook(script: str, input_json: str,
              tmp_home: pathlib.Path) -> subprocess.CompletedProcess:
    """Run a hook script with HOME overridden.
    """
    return subprocess.run(
        ['bash', script],
        check=False, input=input_json,
        capture_output=True, text=True,
        env={**os.environ, 'HOME': str(tmp_home)})


class TestStripJson5:
    """JSON5 stripping primitive (comments, trailing commas).
    """

    def test_strip_json5_line_comments(self):
        """Verify strip_json5 removes // line comments.

        Mutation: a stripper that leaves the comment text in place, so the
            JSON parse fails.
        Oracle: the parsed dict of a hand-written document.
        """
        s = '{"key": "value" // comment\n}'
        assert json.loads(strip_json5(s)) == {'key': 'value'}

    def test_strip_json5_comment_in_string(self):
        """Verify // inside a double-quoted string survives stripping.

        Mutation: treating the // of a URL as a comment start and cutting
            the string short.
        Oracle: the literal URL, round-tripped through json.loads.
        """
        s = '{"url": "https://example.com"}'
        assert json.loads(strip_json5(s)) == {'url': 'https://example.com'}

    def test_strip_json5_trailing_comma(self):
        """Verify a trailing comma before } is removed.

        Mutation: leaving the comma in place, which json.loads rejects.
        Oracle: the parsed dict of a hand-written document.
        """
        s = '{"a": 1, "b": 2,}'
        assert json.loads(strip_json5(s)) == {'a': 1, 'b': 2}

    def test_strip_json5_trailing_comma_array(self):
        """Verify a trailing comma before ] is removed.

        Mutation: handling only object commas, so arrays stay unparseable.
        Oracle: the parsed list of a hand-written document.
        """
        s = '[1, 2, 3,]'
        assert json.loads(strip_json5(s)) == [1, 2, 3]

    def test_strip_json5_block_comment(self):
        """Verify a multi-line block comment is removed.

        Mutation: a stripper that ends the comment at the first newline, or
            never removes it.
        Oracle: the parsed dict of a hand-written document.
        """
        s = '{"a": 1 /* this is a block\n comment */, "b": 2}'
        assert json.loads(strip_json5(s)) == {'a': 1, 'b': 2}

    def test_strip_json5_block_comment_inline(self):
        """Verify a block comment on one line is removed.

        Mutation: requiring whitespace around the comment delimiters.
        Oracle: the parsed dict of a hand-written document.
        """
        s = '{/* inline */"x": 42}'
        assert json.loads(strip_json5(s)) == {'x': 42}

    def test_strip_json5_single_quoted_passthrough(self):
        """Verify // inside a single-quoted string is not read as a comment.

        Mutation: tracking only double quotes, so the // of a single-quoted
            URL truncates the string.
        Oracle: the URL literal present in the stripped text.
        """
        s = "{\"url\": 'https://example.com'}"
        stripped = strip_json5(s)
        assert "'https://example.com'" in stripped

    def test_strip_json5_block_comment_inside_string(self):
        """Verify /* */ inside a double-quoted string is kept.

        Mutation: stripping block comments without tracking string state.
        Oracle: the literal note text, round-tripped through json.loads.
        """
        s = '{"note": "not /* a */ comment"}'
        assert json.loads(strip_json5(s)) == {'note': 'not /* a */ comment'}

    def test_strip_json5_escape_in_single_quoted(self):
        """Verify an escaped quote does not end a single-quoted string.

        Mutation: ending the string at a backslash-escaped quote.
        Oracle: the escaped literal present in the stripped text.
        """
        s = "{\"msg\": 'it\\'s fine'}"
        stripped = strip_json5(s)
        assert "'it\\'s fine'" in stripped


class TestFileOps:
    """`remove_if_empty`, `read_json_file`, `write_json_file`.
    """

    def test_remove_if_empty_allows_known_leaf(self, tmp_path):
        """Verify remove_if_empty deletes an empty 'hooks' directory.

        Mutation: an allowlist missing 'hooks', so uninstall leaves the empty
            directory behind.
        Oracle: the directory no longer exists.
        """
        target = tmp_path / 'hooks'
        target.mkdir()
        remove_if_empty(str(target))
        assert not target.exists()

    def test_remove_if_empty_allows_config_root(self, tmp_path):
        """Verify remove_if_empty deletes an empty '.claude' directory.

        Mutation: an allowlist missing '.claude'.
        Oracle: the directory no longer exists.
        """
        target = tmp_path / '.claude'
        target.mkdir()
        remove_if_empty(str(target))
        assert not target.exists()

    def test_remove_if_empty_rejects_outside_allowlist(self, tmp_path):
        """Verify remove_if_empty refuses a directory outside the allowlist.

        Mutation: dropping the allowlist check, so any empty directory is
            removed.
        Oracle: a ValueError naming the refusal, and the directory still
            present.
        """
        target = tmp_path / 'arbitrary'
        target.mkdir()
        with pytest.raises(ValueError, match='refused'):
            remove_if_empty(str(target))
        assert target.exists()

    def test_remove_if_empty_rejects_root(self):
        """Verify remove_if_empty refuses '/'.

        Mutation: an allowlist check that lets an empty basename through.
        Oracle: a ValueError naming the refusal.
        """
        with pytest.raises(ValueError, match='refused'):
            remove_if_empty('/')

    def test_remove_if_empty_noop_on_non_empty_dir(self, tmp_path):
        """Verify remove_if_empty keeps an allowed directory that holds a file.

        Mutation: removing the tree instead of only an empty directory.
        Oracle: the directory and its file still exist.
        """
        target = tmp_path / 'hooks'
        target.mkdir()
        (target / 'keep.json').write_text('{}')
        remove_if_empty(str(target))
        assert target.exists()
        assert (target / 'keep.json').exists()

    def test_read_json_missing_file(self, tmp_path):
        """Verify read_json_file returns {} for a missing file.

        Mutation: raising FileNotFoundError, which breaks a first install.
        Oracle: the empty dict.
        """
        result = read_json_file(str(tmp_path / 'nope.json'))
        assert result == {}

    def test_read_json_with_comments(self, tmp_path):
        """Verify read_json_file parses a file that holds a // comment.

        Mutation: a reader that skips strip_json5 and fails on the comment.
        Oracle: the hand-written dict.
        """
        p = tmp_path / 'test.json'
        p.write_text('{\n  "key": "val" // comment\n}')
        result = read_json_file(str(p))
        assert result == {'key': 'val'}

    def test_write_json_atomic(self, tmp_path):
        """Verify write_json_file leaves the target and no .tmp file behind.

        Mutation: writing in place, or leaving the temporary file after the
            rename.
        Oracle: the target content read back, and the absent .tmp path.
        """
        p = str(tmp_path / 'out.json')
        write_json_file(p, {'hello': 'world'})
        assert pathlib.Path(p).exists()
        assert not pathlib.Path(p + '.tmp').exists()
        data = json.loads(pathlib.Path(p).open().read())
        assert data == {'hello': 'world'}


class TestHookManagement:
    """`add_claude_hooks_selective` and `remove_claude_hooks`.
    """

    def test_remove_claude_hooks(self):
        """Verify remove_claude_hooks drops memman entries and keeps others.

        Mutation: clearing the whole event list, or matching no memman
            command.
        Oracle: one surviving entry, free of the string 'memman'.
        """
        data = {
            'hooks': {
                'SessionStart': [
                    {'hooks': [{'type': 'command', 'command': '/path/to/hooks/memman/prime.sh'}]},
                    {'hooks': [{'type': 'command', 'command': '/other/tool.sh'}]},
                ],
            },
        }
        remove_claude_hooks(data)
        assert len(data['hooks']['SessionStart']) == 1
        assert 'memman' not in str(data['hooks']['SessionStart'][0])

    def test_add_claude_hooks_selective_remind_only(self):
        """Verify a remind-only call registers two hook events.

        Mutation: registering a Stop hook, or dropping either default event.
        Oracle: the hand-listed event names present or absent in the result.
        """
        data = {}
        add_claude_hooks_selective(data, '/hooks/dir', remind=True)
        hooks = data['hooks']
        assert 'SessionStart' in hooks
        assert 'UserPromptSubmit' in hooks
        assert 'Stop' not in hooks

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
        """Verify no PreToolUse entry appears unless task_recall is set.

        Mutation: defaulting task_recall to True.
        Oracle: 'PreToolUse' absent from the hook keys.
        """
        data = {}
        add_claude_hooks_selective(data, '/hooks/dir')
        hooks = data['hooks']
        assert 'PreToolUse' not in hooks

    def test_remove_claude_hooks_cleans_pretooluse(self):
        """Verify remove_claude_hooks drops a memman PreToolUse entry only.

        Mutation: leaving PreToolUse untouched, or dropping the foreign
            entry along with the memman one.
        Oracle: one surviving entry, the Bash matcher.
        """
        data = {
            'hooks': {
                'PreToolUse': [
                    {
                        'hooks': [{'type': 'command',
                                   'command': '/hooks/memman/task_recall.sh'}],
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
        """Verify a PreToolUse list without memman entries is unchanged.

        Mutation: removing PreToolUse entries by event rather than by
            command.
        Oracle: the one Bash entry still present.
        """
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

    def test_remove_claude_hooks_keeps_foreign_hook_in_shared_entry(self):
        """Verify a foreign hook sharing an entry with memman survives.

        Mutation: filtering whole entries, which deletes prompt-nudge.py
            along with user_prompt.sh.
        Oracle: the one-entry settings from the bug report, with only
            prompt-nudge.py left in the entry.
        """
        data = {'hooks': {'UserPromptSubmit': [{'hooks': [
            {'type': 'command', 'command': '~/.claude/hooks/prompt-nudge.py'},
            {'type': 'command',
             'command': '~/.claude/hooks/memman/user_prompt.sh'},
            ]}]}}
        remove_claude_hooks(data)
        assert data == {'hooks': {'UserPromptSubmit': [{'hooks': [
            {'type': 'command', 'command': '~/.claude/hooks/prompt-nudge.py'},
            ]}]}}

    def test_add_claude_hooks_rerun_keeps_entry_order(self):
        """Verify a rerun of setup leaves settings unchanged.

        Mutation: removing memman entries and appending them again,
            which moves them behind the foreign entries on each run.
        Oracle: a deep copy of the settings a first run produced, with
            a foreign PreToolUse entry placed after the memman entry.
        """
        data = {}
        add_claude_hooks_selective(data, '/hooks/memman', remind=True, task_recall=True)
        data['hooks']['PreToolUse'].insert(1, {
            'matcher': 'Bash',
            'hooks': [{'type': 'command', 'command': '/other/enforce.py'}],
            })
        before = copy.deepcopy(data)
        add_claude_hooks_selective(data, '/hooks/memman', remind=True, task_recall=True)
        assert data == before

    def test_add_claude_hooks_keeps_foreign_hook_in_shared_entry(self):
        """Verify setup keeps a foreign hook listed beside a memman hook.

        Mutation: filtering whole entries before re-adding, which leaves
            only user_prompt.sh as the bug report observed.
        Oracle: the bug report's entry, unchanged by the run.
        """
        entry = {'hooks': [
            {'type': 'command', 'command': '~/.claude/hooks/prompt-nudge.py'},
            {'type': 'command', 'command': '/hooks/memman/user_prompt.sh'},
            ]}
        data = {'hooks': {'UserPromptSubmit': [copy.deepcopy(entry)]}}
        add_claude_hooks_selective(data, '/hooks/memman', remind=True)
        assert data['hooks']['UserPromptSubmit'] == [entry]

    def test_add_claude_hooks_replaces_stale_matcher(self):
        """Verify a rerun drops a memman hook registered under an old matcher.

        Mutation: keeping every memman hook already present and only
            appending missing ones, which leaves the old Task entry live
            beside the new Agent|Task entry.
        Oracle: the installer's own matcher, alone in PreToolUse.
        """
        data = {'hooks': {'PreToolUse': [{
            'matcher': 'Task',
            'hooks': [{'type': 'command',
                       'command': '/hooks/memman/task_recall.sh'}],
            }]}}
        add_claude_hooks_selective(data, '/hooks/memman', task_recall=True)
        assert [e['matcher'] for e in data['hooks']['PreToolUse']] \
            == ['Agent|Task']

    def test_remove_claude_hooks_drops_retired_stop_hook(self):
        """Verify removal reaches an event memman no longer registers.

        Mutation: walking only the events current installs write, so a
            Stop hook from an older install survives install and
            uninstall while doctor says install repairs it.
        Oracle: a Stop entry naming stop.sh under the memman hooks dir.
        """
        data = {'hooks': {'Stop': [{'hooks': [
            {'type': 'command', 'command': '~/.claude/hooks/memman/stop.sh'},
            ]}]}}
        remove_claude_hooks(data)
        assert data == {}

    def test_remove_claude_hooks_keeps_hook_under_memman_checkout(self):
        """Verify a user hook whose path names memman is left alone.

        Mutation: judging ownership by 'memman' anywhere in the command,
            which deletes a user's hook under a memman source checkout.
        Oracle: a hook under ~/code/memman/scripts, outside the memman
            hooks directory.
        """
        data = {'hooks': {'PreToolUse': [{'matcher': 'Bash', 'hooks': [
            {'type': 'command', 'command': '~/code/memman/scripts/lint.sh'},
            ]}]}}
        before = copy.deepcopy(data)
        remove_claude_hooks(data)
        assert data == before

    def test_remove_claude_hooks_leaves_non_string_command(self):
        """Verify a hand-edited non-string command is left, not a crash.

        Mutation: judging ownership on str(command), which admits a list
            and raises TypeError hashing it into the keep lookup.
        Oracle: the settings unchanged, since a list is no memman command.
        """
        data = {'hooks': {'Stop': [{'hooks': [
            {'type': 'command', 'command': ['/hooks/memman/x.sh']},
            ]}]}}
        before = copy.deepcopy(data)
        remove_claude_hooks(data)
        assert data == before

    def test_add_claude_hooks_keeps_path_beside_home_absolute(
            self, monkeypatch):
        """Verify a hooks dir that only shares home's prefix stays absolute.

        Mutation: a bare startswith(home), which turns /home/ubuntux into
            the ~x form, a different user's home in the shell.
        Oracle: home /home/ubuntu against hooks dir /home/ubuntux/...,
            and /home/ubuntu/... for the ~ form.
        """
        monkeypatch.setattr(pathlib.Path, 'home',
                            lambda: pathlib.Path('/home/ubuntu'))
        beside = {}
        add_claude_hooks_selective(beside, '/home/ubuntux/.claude/hooks/memman')
        under = {}
        add_claude_hooks_selective(under, '/home/ubuntu/.claude/hooks/memman')
        assert beside['hooks']['SessionStart'][0]['hooks'][0]['command'] \
            == '/home/ubuntux/.claude/hooks/memman/prime.sh'
        assert under['hooks']['SessionStart'][0]['hooks'][0]['command'] \
            == '~/.claude/hooks/memman/prime.sh'

    def test_add_claude_hooks_appends_to_existing_pretooluse(self):
        """Verify task_recall appends to an existing PreToolUse list.

        Mutation: replacing the list, which drops the foreign Bash entry.
        Oracle: the matcher set {'Bash', 'Agent|Task'}.
        """
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


class TestPermissions:
    """`add_memman_permission`, `remove_memman_permission`.
    """

    def test_add_memman_permission(self):
        """Verify add_memman_permission adds every entry once, idempotently.

        Mutation: appending on every call, which duplicates entries.
        Oracle: each curated entry counted exactly once after two calls.
        """
        data = {}
        add_memman_permission(data, list_claude_permissions())
        allow = data['permissions']['allow']
        for entry in list_claude_permissions():
            assert allow.count(entry) == 1
        add_memman_permission(data, list_claude_permissions())
        for entry in list_claude_permissions():
            assert data['permissions']['allow'].count(entry) == 1

    def test_add_memman_permission_existing_allow(self):
        """Verify curated entries are appended after existing allow entries.

        Mutation: replacing the existing allow list.
        Oracle: the first entry unchanged, the rest equal to the curated list.
        """
        data = {'permissions': {'allow': ['Bash(git:*)']}}
        add_memman_permission(data, list_claude_permissions())
        allow = data['permissions']['allow']
        assert allow[0] == 'Bash(git:*)'
        assert allow[1:] == list(list_claude_permissions())

    def test_remove_memman_permission(self):
        """Verify remove_memman_permission drops memman entries only.

        Mutation: clearing allow, or leaving curated entries in place.
        Oracle: allow equal to the one foreign entry.
        """
        data = {
            'permissions': {
                'allow': ['Bash(git:*)', *list_claude_permissions()],
                },
            }
        remove_memman_permission(data)
        assert data['permissions']['allow'] == ['Bash(git:*)']

    def test_remove_memman_permission_missing(self):
        """Verify removal is a no-op when no memman entry exists.

        Mutation: removing an unrelated entry, or raising on absence.
        Oracle: allow unchanged.
        """
        data = {'permissions': {'allow': ['Bash(git:*)']}}
        remove_memman_permission(data)
        assert data['permissions']['allow'] == ['Bash(git:*)']

    def test_remove_memman_permission_sweeps_user_added(self):
        """Verify removal also drops hand-added Bash(memman ...) entries.

        Mutation: removing only the curated list, not every memman entry.
        Oracle: allow equal to the one foreign entry.
        """
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
        """Verify removal sweeps deny and ask as well as allow.

        Mutation: sweeping allow only.
        Oracle: permissions equal to the one foreign deny entry.
        """
        data = {
            'permissions': {
                'allow': ['Bash(memman recall:*)'],
                'deny': ['Bash(memman uninstall:*)', 'Bash(rm:*)'],
                'ask': ['Bash(memman scheduler stop)'],
                },
            }
        remove_memman_permission(data)
        assert data['permissions'] == {'deny': ['Bash(rm:*)']}

    def test_remove_memman_permission_keeps_rules_naming_memman_paths(self):
        """Verify removal keeps a rule that only names a memman path.

        Mutation: judging ownership by 'memman' anywhere in the rule,
            which deletes a user's rules for a memman source checkout.
        Oracle: rules for a checkout at ~/code/memman, none of which
            runs the memman CLI, beside one that does.
        """
        foreign = [
            'Read(~/code/memman/**)',
            'Bash(cd ~/code/memman && make test)',
            'Bash(memmanager:*)',
            ]
        data = {'permissions': {'allow': [*foreign, 'Bash(memman:*)']}}
        remove_memman_permission(data)
        assert data['permissions']['allow'] == foreign

    def test_remove_memman_permission_drops_empty_permissions(self):
        """Verify an emptied permissions dict is removed.

        Mutation: leaving an empty 'permissions' key in settings.
        Oracle: 'permissions' absent from the data.
        """
        data = {'permissions': {'allow': list(list_claude_permissions())}}
        remove_memman_permission(data)
        assert 'permissions' not in data

    def test_install_uninstall_roundtrip(self):
        """Verify add then remove restores the starting settings.

        Mutation: a removal that leaves residue, or an add that changes
            foreign entries.
        Oracle: equality with a deep copy taken before the add.
        """
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
            data, '/hooks/dir', remind=True,
            task_recall=True)
        allow = data.get('permissions', {}).get('allow', [])
        for entry in list_claude_permissions():
            assert entry not in allow


class TestPrimeHook:
    """Prime hook script execution.
    """

    def test_prime_hook_emits_guide_content(self, tmp_path):
        """Verify prime.sh relays the output of `memman prime`.

        Mutation: printing a fixed text instead of calling memman.
        Oracle: a shim memman on PATH that prints a marker line.
        """
        script = str(files('memman.setup.assets')
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
        """Verify prime.sh warns and exits 0 when memman is not on PATH.

        Mutation: a failing exit status, or silence, when the binary is
            missing.
        Oracle: exit status 0 and 'not on PATH' in the output, run with a PATH
            that holds no memman.
        """
        script = str(files('memman.setup.assets')
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
            preview and drops the rest without erroring; or the
            `replace` line dropped from guide.md, which leaves the
            agent no correction verb.
        Oracle: HOOK_STDOUT_LIMIT, the byte count above which the host
            replaces the payload with a preview, checked against the
            emitted bytes; plus the recall, remember, and replace
            commands of the contract.
        """
        result = CliRunner().invoke(
            cli, ['prime'], input='{}',
            env={'MEMMAN_DATA_DIR': str(tmp_path)})

        assert result.exit_code == 0
        assert len(result.output.encode()) < HOOK_STDOUT_LIMIT
        assert 'memman recall' in result.output
        assert 'memman remember' in result.output
        assert 'memman replace' in result.output


class TestUserPromptHook:
    """`user_prompt.sh` recall reminder and session-id hint.
    """

    def test_prints_the_bare_recall_reminder_regardless_of_session_id(
            self, tmp_path):
        """Verify the hook prints one fixed reminder, reading no session id.

        Mutation: reintroducing a `--session` hint keyed off the
            payload's session id, so a payload with one and a payload
            without one print different lines.
        Oracle: the exact shipped reminder line, unchanged whether the
            payload carries a session id or not.
        """
        with_id = _run_hook(
            _prompt_script(),
            '{"session_id": "sess-6"}',
            tmp_path)
        without_id = _run_hook(
            _prompt_script(),
            '{}',
            tmp_path)

        assert with_id.returncode == 0
        assert without_id.returncode == 0
        expected = '[memman] Recall: memman recall "<focused query>"\n'
        assert with_id.stdout == expected
        assert without_id.stdout == expected

    def test_carries_no_write_instruction(self, tmp_path):
        """Verify the reminder asks for a recall and never for a write.

        Mutation: restoring the trailing "After responding, evaluate:
            remember needed?" clause, which UserPromptSubmit delivers
            before the turn whose end it asks about.
        Oracle: the two phrases that made up the deleted clause, checked
            against a payload that carries a session id.
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
    """No shipped hook re-invokes the model after a turn has ended.
    """

    def test_no_shipped_hook_emits_a_block_decision(self, tmp_path):
        """Verify every shipped hook emits plain text, never a block decision.

        Mutation: a hook re-emitting {"decision": "block"}, which restarts
            the model after the turn already ended.
        Oracle: hand-written expectation that the shipped hook set emits
            text only - stdout parsed as JSON carries no decision key.
        """
        assets = files('memman.setup.assets')
        payload = json.dumps({
            'stop_hook_active': False,
            'session_id': 'sess-block-probe',
            })
        trees = (
            ('claude', assets.joinpath('claude')),
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


class TestPreToolUseHooksReachTheModel:
    """PreToolUse reminders ride the one channel Claude Code reads.
    """

    def test_pretooluse_hooks_emit_additional_context(self, tmp_path):
        """Verify every PreToolUse hook speaks additionalContext.

        Mutation: a PreToolUse hook echoing plain text, which Claude
            Code discards for every event but SessionStart and
            UserPromptSubmit, so the reminder never reaches the model.
        Oracle: the registered event set - every script the installer
            files under PreToolUse, each required to emit JSON naming
            the event and carrying a memman line.
        """
        registered: dict = {}
        add_claude_hooks_selective(
            registered, '/hooks/dir', remind=True,
            task_recall=True)
        scripts = [
            pathlib.Path(hook['command']).name
            for entry in registered['hooks']['PreToolUse']
            for hook in entry['hooks']
            ]
        assert scripts
        for name in scripts:
            script = str(
                files('memman.setup.assets')
                .joinpath(f'claude/{name}'))
            result = _run_hook(
                script, '{"session_id": "sess-ctx"}', tmp_path)
            assert result.returncode == 0, name
            emitted = json.loads(result.stdout)['hookSpecificOutput']
            assert emitted['hookEventName'] == 'PreToolUse', name
            assert '[memman]' in emitted['additionalContext'], name


class TestDocsMatchShippedHooks:
    """Prose and diagram counts track the shipped hook set.
    """

    def test_doc_hook_counts_match_the_asset_tree(self):
        """Verify every doc site naming a hook count names the real one.

        Mutation: a hook added or deleted on some doc sites only - the
            deletion that corrected three README counts and left the
            fourth reading six, and regenerated one diagram of two.
        Oracle: the shipped asset tree counted directly - the .sh files
            under assets/claude.
        """
        assets = (pathlib.Path(__file__).resolve().parents[1]
                  / 'src' / 'memman' / 'setup' / 'assets')
        shipped = len(list(assets.glob('claude/*.sh')))
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
        assert counted == {shipped}

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
                   / '02-system-architecture.drawio').read_text()
        label = re.search(r'id="a_hooks" value="([^"]*)"', diagram)
        assert label is not None
        text = re.sub(r'<[^>]+>', '/', html.unescape(label.group(1)))
        roles = [
            token.strip() for token in re.split(r'[/,]', text)
            if token.strip() and token.strip().lower() != 'hooks'
            ]
        assert len(roles) == shipped


class TestSetupCli:
    """`memman prime` CLI command.
    """

    def test_prime_command_emits_status_and_guide(self, tmp_path, monkeypatch):
        """Verify `memman prime` prints the status line and the shipped guide.

        Mutation: dropping the guide text, or the status line.
        Oracle: the guide.md asset read directly.
        """
        monkeypatch.setattr(pathlib.Path, 'home', lambda: tmp_path)
        _create_seeded_store('default', os.environ[config.DATA_DIR])
        runner = CliRunner()
        result = runner.invoke(cli, ['prime'], input='{}')
        assert result.exit_code == 0
        assert '[memman] Memory active' in result.output
        shipped = (files('memman.setup.assets')
                   .joinpath('claude/guide.md').read_text())
        assert shipped.strip() in result.output

    def test_prime_prints_the_recorded_model_notice(
            self, tmp_path, monkeypatch):
        """`memman prime` shows the notice the model check recorded.

        Mutation: prime not reading the model state, so a retiring or
            unroutable model goes unreported at session start.
        Oracle: a state naming the seeded model with a fixed notice.
        """
        monkeypatch.setattr(pathlib.Path, 'home', lambda: tmp_path)
        notice = 'LLM model under test retires on 2026-10-09'
        state = {
            'model': config.INSTALL_DEFAULTS[config.LLM_MODEL],
            'checked_at': 0,
            'notice': notice,
            }
        state_path = pathlib.Path(os.environ[config.DATA_DIR]) / 'model.state'
        state_path.write_text(json.dumps(state))
        result = CliRunner().invoke(cli, ['prime'], input='{}')
        assert result.exit_code == 0
        assert f'[memman] {notice}' in result.output

    def test_prime_honors_memman_store_env(self, tmp_path, monkeypatch):
        """`memman prime` counts the MEMMAN_STORE store, not the active one.

        Mutation: prime reading only the active-store file, so a session
            pinned to another store reports the wrong store's count.
        Oracle: one row in the active `default` store and two in `work`.
        """
        monkeypatch.setattr(pathlib.Path, 'home', lambda: tmp_path)
        data_dir = os.environ[config.DATA_DIR]
        for name, row_cnt in (('default', 1), ('work', 2)):
            path = store_dir(data_dir, name)
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            db = open_db(path)
            for row_idx in range(row_cnt):
                insert_insight(db, make_insight(id=f'{name}-{row_idx}'))
            db.close()
        write_active(data_dir, 'default')

        monkeypatch.setenv('MEMMAN_STORE', 'work')
        result = CliRunner().invoke(cli, ['prime'], input='{}')
        assert result.exit_code == 0
        assert '[memman] Memory active (2 insights).' in result.output


class TestSymlinks:
    """`claude_write_skill` / `claude_write_hook` symlink behavior.
    """

    def test_claude_write_skill_creates_symlink(self, tmp_path):
        """Verify claude_write_skill links to the shipped SKILL.md.

        Mutation: copying the file, or linking to another path.
        Oracle: the resolved link equals the resolved asset path.
        """
        config = tmp_path / 'claude'
        link_path = claude_write_skill(str(config))
        link = pathlib.Path(link_path)
        assert link.is_symlink()
        target = pathlib.Path(str(files('memman.setup.assets')
                                  .joinpath('claude/SKILL.md'))).resolve()
        assert link.resolve() == target

    def test_claude_write_hook_creates_symlink(self, tmp_path):
        """Verify claude_write_hook links to the shipped hook script.

        Mutation: copying the script, or linking to another path.
        Oracle: the resolved link equals the resolved prime.sh asset path.
        """
        config = tmp_path / 'claude'
        link_path = claude_write_hook(str(config), 'prime.sh')
        link = pathlib.Path(link_path)
        assert link.is_symlink()
        target = pathlib.Path(str(files('memman.setup.assets')
                                  .joinpath('claude/prime.sh'))).resolve()
        assert link.resolve() == target

    def test_symlink_replaces_stale_symlink(self, tmp_path):
        """Verify a reinstall replaces a dangling symlink with a live one.

        Mutation: skipping the write when a link path already exists.
        Oracle: the link exists after the call, and did not before.
        """
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
        """Verify a reinstall replaces a regular file with a symlink.

        Mutation: leaving the stale file in place.
        Oracle: the path is a symlink after the call.
        """
        config = tmp_path / 'claude'
        link = config / 'skills' / 'memman' / 'SKILL.md'
        link.parent.mkdir(parents=True)
        link.write_text('stale pre-symlink content')
        assert not link.is_symlink()
        claude_write_skill(str(config))
        assert link.is_symlink()

    def test_uninstall_removes_symlink_not_target(self, tmp_path):
        """Verify claude_uninstall removes the link, keeps the asset.

        Mutation: deleting through the link, which destroys the packaged file.
        Oracle: the asset's bytes read before and after.
        """
        config = tmp_path / '.claude'
        claude_write_skill(str(config))
        target = pathlib.Path(str(files('memman.setup.assets')
                                  .joinpath('claude/SKILL.md'))).resolve()
        target_bytes = target.read_bytes()
        claude_uninstall(str(config))
        assert not (config / 'skills' / 'memman' / 'SKILL.md').exists()
        assert target.exists()
        assert target.read_bytes() == target_bytes


class TestInstallConsent:
    """`_install_claude_code` TTY consent flow for permissions.
    """

    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        """Set up a clean Claude Code config dir and stub heavy deps.
        """
        config_dir = tmp_path / '.claude'
        config_dir.mkdir()
        monkeypatch.setenv('HOME', str(tmp_path))
        monkeypatch.setattr(claude_setup, '_init_default_store',
                            lambda data_dir: None)
        return {'config_dir': str(config_dir)}

    def _allow(self, config_dir: str) -> list:
        path = os.path.join(config_dir, 'settings.json')
        data = read_json_file(path)
        return data.get('permissions', {}).get('allow', [])

    def test_tty_consent_accept_writes_permissions(
            self, env, tmp_path, monkeypatch):
        """Verify accepting the TTY prompt writes every curated permission.

        Mutation: ignoring the confirm answer, or writing a partial list.
        Oracle: each entry of list_claude_permissions in allow.
        """
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
        """Verify declining the TTY prompt skips permissions and keeps hooks.

        Mutation: writing permissions despite a decline, or skipping hooks.
        Oracle: no curated entry in allow, and a non-empty hooks section.
        """
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
        """Verify no_wizard writes permissions without asking.

        Mutation: prompting despite no_wizard on a TTY.
        Oracle: a spy on click.confirm records no call, and every entry is in
            allow.
        """
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
        """Verify a non-TTY install writes permissions without asking.

        Mutation: prompting with no terminal to answer.
        Oracle: a spy on click.confirm records no call, and every entry is in
            allow.
        """
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


def test_shipped_package_names_no_claw_host():
    """Verify no shipped file contains an openclaw or nanoclaw reference.

    Mutation: leaving an openclaw or nanoclaw branch, asset, or docstring
        behind in the shipped package.
    Oracle: the text of every shipped file.
    """
    pattern = re.compile(r'openclaw|nanoclaw', re.I)
    root = pathlib.Path(str(files('memman')))
    offenders = [
        str(item.relative_to(root))
        for item in sorted(root.rglob('*'))
        if item.is_file()
        and item.suffix in {'.py', '.md', '.sh', '.js', '.json'}
        and pattern.search(item.read_text(errors='replace'))
        ]
    assert offenders == [], (
        'Shipped files still contain openclaw/nanoclaw references:\n'
        + '\n'.join(offenders))


def test_uninstall_raises_when_claude_code_cleanup_fails(
        monkeypatch, tmp_path):
    """Verify a failed Claude Code cleanup fails the uninstall first.

    Mutation: discarding `_uninstall_env`'s error result, so uninstall
        exits 0, removes the scheduler, and prints success.
    Oracle: a stubbed cleanup returning one error, and a spy proving
        the scheduler removal never ran.
    """
    monkeypatch.setattr(
        claude_setup, 'claude_uninstall',
        lambda config_dir: [RuntimeError('settings rewrite failed')])
    scheduler_calls = []
    monkeypatch.setattr(
        claude_setup, 'uninstall_scheduler',
        lambda data_dir: scheduler_calls.append(data_dir) or {})
    env = {
        'display': 'Claude Code', 'detected': True,
        'version': '', 'config_dir': str(tmp_path / '.claude'),
        }
    with pytest.raises(click.ClickException):
        claude_setup._run_uninstall_flow(
            env, claude_code=False, data_dir=str(tmp_path))
    assert scheduler_calls == []


def test_install_flow_forces_claude_code_when_undetected(
        monkeypatch, tmp_path):
    """Verify `claude_code=True` installs into Claude Code though undetected.

    Mutation: `_run_install_flow` ignoring `claude_code` and following
        detection, so `--claude-code` on a host without the `claude`
        binary installs the scheduler only.
    Oracle: a spy on `_install_claude_code` proving the install branch
        ran on the forced config dir.
    """
    installs = []
    monkeypatch.setattr(
        claude_setup, '_install_claude_code',
        lambda env, data_dir, no_wizard: installs.append(env['config_dir']))
    monkeypatch.setattr(
        claude_setup, 'install_scheduler',
        lambda data_dir, knobs: {'platform': 'systemd', 'actions': []})
    monkeypatch.setattr(
        claude_setup.openrouter_models, 'refresh_model_state',
        lambda data_dir, force: None)
    env = {
        'display': 'Claude Code', 'detected': False,
        'version': '', 'config_dir': str(tmp_path / '.claude'),
        }
    claude_setup._run_install_flow(
        env, claude_code=True, data_dir=str(tmp_path), knobs={})
    assert installs == [str(tmp_path / '.claude')]


def test_uninstall_flow_forces_claude_code_when_undetected(
        monkeypatch, tmp_path):
    """Verify `claude_code=True` cleans up Claude Code though undetected.

    Mutation: `_run_uninstall_flow` ignoring `claude_code` and following
        detection, so `--claude-code` leaves the hooks and skill behind.
    Oracle: a spy on `claude_uninstall` proving the cleanup ran on the
        forced config dir.
    """
    cleaned = []
    monkeypatch.setattr(
        claude_setup, 'claude_uninstall',
        lambda config_dir: cleaned.append(config_dir) or [])
    monkeypatch.setattr(
        claude_setup, 'uninstall_scheduler', lambda data_dir: {})
    env = {
        'display': 'Claude Code', 'detected': False,
        'version': '', 'config_dir': str(tmp_path / '.claude'),
        }
    claude_setup._run_uninstall_flow(
        env, claude_code=True, data_dir=str(tmp_path))
    assert cleaned == [str(tmp_path / '.claude')]
