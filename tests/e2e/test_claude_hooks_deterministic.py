"""Deterministic Claude Code hook contract tests.

Each shipped hook script under `src/memman/setup/assets/claude/` is
exercised with realistic Claude Code input JSON, against an isolated
HOME, in a subprocess. The contract checked is exactly what Claude
Code's hook subsystem cares about: exit code, stdout shape (every
hook but task_recall.sh prints a plain string, and task_recall.sh
prints JSON; none returns a decision payload).

No live LLM. No container. No Anthropic API key. Runs on every PR.
"""

import json
import shlex
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import memman
import pytest
from memman.store.sqlite import open_sqlite_backend

pytestmark = pytest.mark.e2e_cli

SESSION_ID = '00000000-0000-0000-0000-000000000abc'


def _hook(name: str) -> str:
    """Resolve a shipped Claude hook script path via importlib.resources.
    """
    return str(files('memman.setup.assets').joinpath(f'claude/{name}'))


def _hook_env(home: Path) -> dict[str, str]:
    """Hook env whose `memman` runs this interpreter on the package under test.

    An installed `memman` may run other source, such as the main checkout
    while the tests run in a worktree, so its prime payload would not
    match the guide.md these tests read.
    """
    bin_dir = home / 'bin'
    bin_dir.mkdir(exist_ok=True)
    shim = bin_dir / 'memman'
    source_root = Path(memman.__file__).parents[1]
    shim.write_text(
        '#!/bin/sh\n'
        f'PYTHONPATH={shlex.quote(str(source_root))}'
        f' exec {shlex.quote(sys.executable)}'
        ' -c "from memman.cli import cli; cli()" "$@"\n')
    shim.chmod(0o755)
    return {'HOME': str(home), 'PATH': f'{bin_dir}:/usr/bin:/bin'}


def _run_hook(script: str, input_json: dict, home: Path
              ) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['bash', script],
        input=json.dumps(input_json), capture_output=True, text=True,
        env=_hook_env(home))


# --- prime.sh - SessionStart ---

def test_prime_emits_memman_prefix(memman_home: tuple[Path, Path]):
    """Verify prime.sh exits 0 and prints the status line and the guide.

    Mutation: dropping the `memman prime` call, so only the
        "memman not on PATH" warning (which also carries `[memman]`)
        reaches stdout.
    Oracle: the literal `[memman] Memory active` status line and the
        shipped guide.md text, which only a real prime payload carries.
    """
    home, _ = memman_home
    open_sqlite_backend('default', str(home / '.memman'), create=True).close()
    out = _run_hook(_hook('prime.sh'),
                    {'session_id': SESSION_ID}, home)
    assert out.returncode == 0, out.stderr
    assert '[memman] Memory active' in out.stdout, out.stdout
    guide = (files('memman.setup.assets')
             .joinpath('claude/guide.md').read_text())
    assert guide in out.stdout, out.stdout


def test_prime_handles_empty_stdin(memman_home: tuple[Path, Path]):
    """Verify prime.sh runs `memman prime` and exits 0 on empty piped stdin.

    Mutation: prime.sh or `memman prime` reading stdin again and
        failing on an empty payload, or the trailing `exit 0` dropped.
    Oracle: exit code 0 and the `[memman] Memory active` status line,
        which only a real `memman prime` run prints.
    """
    home, _ = memman_home
    open_sqlite_backend('default', str(home / '.memman'), create=True).close()
    out = subprocess.run(
        ['bash', _hook('prime.sh')],
        input='', capture_output=True, text=True,
        env=_hook_env(home))
    assert out.returncode == 0, out.stderr
    assert '[memman] Memory active' in out.stdout, out.stdout


# --- user_prompt.sh - UserPromptSubmit ---

def test_user_prompt_emits_recall_reminder(memman_home: tuple[Path, Path]):
    """Verify user_prompt.sh prints the recall reminder with the command.

    Mutation: the reminder text losing the `memman recall` command
        or the `[memman] Recall` label.
    Oracle: the literal reminder substrings.
    """
    home, _ = memman_home
    out = _run_hook(_hook('user_prompt.sh'),
                    {'session_id': SESSION_ID}, home)
    assert out.returncode == 0, out.stderr
    assert '[memman] Recall' in out.stdout, out.stdout
    assert 'memman recall' in out.stdout, out.stdout


# --- task_recall.sh - PreToolUse(Agent|Task) ---

def test_task_recall_emits_reminder(memman_home: tuple[Path, Path]):
    """Verify task_recall.sh prints a `[memman]` recall reminder.

    Mutation: emptying the reminder text or changing it to a
        non-recall message.
    Oracle: the `[memman]` prefix and the word `recall`.
    """
    home, _ = memman_home
    out = _run_hook(_hook('task_recall.sh'), {}, home)
    assert out.returncode == 0
    assert '[memman]' in out.stdout
    assert 'recall' in out.stdout.lower()
