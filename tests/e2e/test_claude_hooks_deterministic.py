"""Deterministic Claude Code hook contract tests.

Each shipped hook script under `src/memman/setup/assets/claude/` is
exercised with realistic Claude Code input JSON, against an isolated
HOME, in a subprocess. The contract checked is exactly what Claude
Code's hook subsystem cares about: exit code, stdout shape (every
hook emits a plain prefix string; none returns a decision payload),
and any side-effect under `~/.memman/`.

No live LLM. No container. No Anthropic API key. Runs on every PR.
"""

import json
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e_cli

SESSION_ID = '00000000-0000-0000-0000-000000000abc'


def _hook(name: str) -> str:
    """Resolve a shipped Claude hook script path via importlib.resources.
    """
    return str(files('memman.setup.assets').joinpath(f'claude/{name}'))


def _run_hook(script: str, input_json: dict, home: Path
              ) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['bash', script],
        input=json.dumps(input_json), capture_output=True, text=True,
        env={'HOME': str(home), 'PATH': '/usr/bin:/bin:'
             + str(Path(shutil.which('memman')).parent)})


# --- prime.sh - SessionStart ---

def test_prime_emits_memman_prefix(memman_home: tuple[Path, Path]):
    """Verify prime.sh exits 0 and prints the status line and the guide.

    Mutation: dropping the `memman prime` pipe, so only the
        "memman not on PATH" warning (which also carries `[memman]`)
        reaches stdout.
    Oracle: the literal `[memman] Memory active` status line and the
        shipped guide.md text, which only a real prime payload carries.
    """
    home, _ = memman_home
    out = _run_hook(_hook('prime.sh'),
                    {'session_id': SESSION_ID}, home)
    assert out.returncode == 0, out.stderr
    assert '[memman] Memory active' in out.stdout, out.stdout
    guide = (files('memman.setup.assets')
             .joinpath('claude/guide.md').read_text())
    assert guide in out.stdout, out.stdout


def test_prime_handles_empty_stdin(memman_home: tuple[Path, Path]):
    """Verify prime.sh runs `memman prime` and exits 0 on empty piped stdin.

    Mutation: dropping the `[ -t 0 ]` guard or the trailing
        `exit 0`, so an empty piped payload makes the hook fail; or
        an empty payload sending prime down the "not on PATH" branch.
    Oracle: exit code 0 and the `[memman] Memory active` status line,
        which only a real `memman prime` run prints.
    """
    home, _ = memman_home
    out = subprocess.run(
        ['bash', _hook('prime.sh')],
        input='', capture_output=True, text=True,
        env={'HOME': str(home), 'PATH': '/usr/bin:/bin:'
             + str(Path(shutil.which('memman')).parent)})
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


# --- compact.sh - PreCompact ---

def test_compact_writes_flag_file(memman_home: tuple[Path, Path]):
    """Verify compact.sh writes a per-session flag with trigger and ts.

    Mutation: writing the flag under the wrong path, dropping the
        parsed trigger (always `auto`), or omitting `ts`.
    Oracle: the flag path built from the session id, and the
        `manual` trigger sent in the input.
    """
    home, _ = memman_home
    out = _run_hook(_hook('compact.sh'),
                    {'session_id': SESSION_ID,
                     'trigger': 'manual'}, home)
    assert out.returncode == 0, out.stderr

    flag = home / '.memman' / 'compact' / f'{SESSION_ID}.json'
    assert flag.exists(), f'compact flag not written at {flag}'
    payload = json.loads(flag.read_text())
    assert payload['trigger'] == 'manual'
    assert 'ts' in payload


def test_compact_no_session_id_no_op(memman_home: tuple[Path, Path]):
    """Verify compact.sh writes nothing when the input has no session id.

    Mutation: dropping the `[ -z "$SESSION_ID" ] && exit 0` guard,
        which would write a `.json` flag with an empty name.
    Oracle: the compact directory absent or empty.
    """
    home, _ = memman_home
    out = _run_hook(_hook('compact.sh'), {'trigger': 'auto'}, home)
    assert out.returncode == 0
    flag_dir = home / '.memman' / 'compact'
    assert not flag_dir.exists() or len(list(flag_dir.iterdir())) == 0


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


# --- exit_plan.sh - PreToolUse(ExitPlanMode) ---

def test_exit_plan_emits_remember_reminder(memman_home: tuple[Path, Path]):
    """Verify exit_plan.sh prints a `[memman]` remember reminder.

    Mutation: emptying the reminder text or changing it to a
        non-remember message.
    Oracle: the `[memman]` prefix and the word `remember`.
    """
    home, _ = memman_home
    out = _run_hook(_hook('exit_plan.sh'), {}, home)
    assert out.returncode == 0
    assert '[memman]' in out.stdout
    assert 'remember' in out.stdout.lower()
