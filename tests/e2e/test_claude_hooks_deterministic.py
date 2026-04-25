"""Deterministic Claude Code hook contract tests.

Each shipped hook script under `src/memman/setup/assets/claude/` is
exercised with realistic Claude Code input JSON, against an isolated
HOME, in a subprocess. The contract checked is exactly what Claude
Code's hook subsystem cares about: exit code, stdout shape (plain
prefix string for SessionStart-style hooks, `{"decision":"block", ...}`
JSON for Stop-style hooks), and any side-effect under `~/.memman/`.

No live LLM. No container. No Anthropic API key. Runs on every PR.
"""

import json
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
             + str(Path.home() / '.local' / 'bin')})


# ---------------------------------------------------------------------
# prime.sh — SessionStart
# ---------------------------------------------------------------------

def test_prime_emits_memman_prefix(memman_home: tuple[Path, Path]):
    home, _ = memman_home
    out = _run_hook(_hook('prime.sh'),
                    {'session_id': SESSION_ID}, home)
    assert out.returncode == 0, out.stderr
    assert '[memman]' in out.stdout, out.stdout


def test_prime_handles_empty_stdin(memman_home: tuple[Path, Path]):
    """prime.sh has a `[ -t 0 ]` guard for interactive runs.

    A piped empty input still triggers the non-tty path. The hook
    should not crash even if the JSON is empty.
    """
    home, _ = memman_home
    out = subprocess.run(
        ['bash', _hook('prime.sh')],
        input='', capture_output=True, text=True,
        env={'HOME': str(home), 'PATH': '/usr/bin:/bin:'
             + str(Path.home() / '.local' / 'bin')})
    assert out.returncode == 0, out.stderr


# ---------------------------------------------------------------------
# user_prompt.sh — UserPromptSubmit
# ---------------------------------------------------------------------

def test_user_prompt_emits_recall_reminder(memman_home: tuple[Path, Path]):
    home, _ = memman_home
    out = _run_hook(_hook('user_prompt.sh'),
                    {'session_id': SESSION_ID}, home)
    assert out.returncode == 0, out.stderr
    assert '[memman] Recall' in out.stdout, out.stdout
    assert 'memman recall' in out.stdout, out.stdout


def test_user_prompt_clears_stop_fired_flag(memman_home: tuple[Path, Path]):
    """user_prompt.sh rmdir's $HOME/.memman/stop_fired/<sid> so the next
    Stop hook fires again.
    """
    home, _ = memman_home
    flag_dir = home / '.memman' / 'stop_fired'
    flag_dir.mkdir(parents=True, exist_ok=True)
    (flag_dir / SESSION_ID).mkdir()
    assert (flag_dir / SESSION_ID).exists()

    out = _run_hook(_hook('user_prompt.sh'),
                    {'session_id': SESSION_ID}, home)
    assert out.returncode == 0, out.stderr
    assert not (flag_dir / SESSION_ID).exists(), (
        'stop_fired/<sid> should be removed after user_prompt.sh')


# ---------------------------------------------------------------------
# stop.sh — Stop
# ---------------------------------------------------------------------

def test_stop_blocks_on_first_call(memman_home: tuple[Path, Path]):
    home, _ = memman_home
    out = _run_hook(_hook('stop.sh'),
                    {'stop_hook_active': False,
                     'session_id': SESSION_ID}, home)
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload['decision'] == 'block'
    assert '[memman] Memory check' in payload['reason']


def test_stop_gated_on_second_call_same_session(
        memman_home: tuple[Path, Path]):
    home, _ = memman_home
    inp = {'stop_hook_active': False, 'session_id': SESSION_ID}
    first = _run_hook(_hook('stop.sh'), inp, home)
    assert first.returncode == 0
    assert first.stdout.strip(), 'first call should emit a payload'

    second = _run_hook(_hook('stop.sh'), inp, home)
    assert second.returncode == 0
    assert second.stdout.strip() == '', (
        f'second call same session should be silent, got {second.stdout!r}')


def test_stop_active_flag_returns_immediately(
        memman_home: tuple[Path, Path]):
    home, _ = memman_home
    out = _run_hook(_hook('stop.sh'),
                    {'stop_hook_active': True,
                     'session_id': SESSION_ID}, home)
    assert out.returncode == 0
    assert out.stdout.strip() == '', (
        f'stop_hook_active=true should be silent, got {out.stdout!r}')


def test_stop_no_session_id_falls_back_to_block(
        memman_home: tuple[Path, Path]):
    home, _ = memman_home
    out = _run_hook(_hook('stop.sh'),
                    {'stop_hook_active': False}, home)
    assert out.returncode == 0
    payload = json.loads(out.stdout)
    assert payload['decision'] == 'block', (
        'no session_id should fall back to always-block (safe default)')


# ---------------------------------------------------------------------
# compact.sh — PreCompact
# ---------------------------------------------------------------------

def test_compact_writes_flag_file(memman_home: tuple[Path, Path]):
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
    home, _ = memman_home
    out = _run_hook(_hook('compact.sh'), {'trigger': 'auto'}, home)
    assert out.returncode == 0
    flag_dir = home / '.memman' / 'compact'
    assert not flag_dir.exists() or len(list(flag_dir.iterdir())) == 0


# ---------------------------------------------------------------------
# task_recall.sh — PreToolUse(Task)
# ---------------------------------------------------------------------

def test_task_recall_emits_reminder(memman_home: tuple[Path, Path]):
    home, _ = memman_home
    out = _run_hook(_hook('task_recall.sh'), {}, home)
    assert out.returncode == 0
    assert '[memman]' in out.stdout
    assert 'recall' in out.stdout.lower()


# ---------------------------------------------------------------------
# exit_plan.sh — PreToolUse(ExitPlanMode)
# ---------------------------------------------------------------------

def test_exit_plan_emits_remember_reminder(memman_home: tuple[Path, Path]):
    home, _ = memman_home
    out = _run_hook(_hook('exit_plan.sh'), {}, home)
    assert out.returncode == 0
    assert '[memman]' in out.stdout
    assert 'remember' in out.stdout.lower()


# ---------------------------------------------------------------------
# Nanoclaw shipped hooks — same files the SKILL.md references
# ---------------------------------------------------------------------

def _nc_hook(name: str) -> str:
    return str(files('memman.setup.assets').joinpath(
        f'nanoclaw/hooks/{name}'))


def test_nanoclaw_prime_emits_status(memman_home: tuple[Path, Path]):
    home, _ = memman_home
    out = subprocess.run(
        ['bash', _nc_hook('prime.sh')],
        capture_output=True, text=True,
        env={'HOME': str(home), 'PATH': '/usr/bin:/bin:'
             + str(Path.home() / '.local' / 'bin')})
    assert out.returncode == 0
    assert '[memman]' in out.stdout
    assert 'Memory active' in out.stdout


def test_nanoclaw_user_prompt_emits_evaluate(
        memman_home: tuple[Path, Path]):
    home, _ = memman_home
    out = subprocess.run(
        ['bash', _nc_hook('user_prompt.sh')],
        capture_output=True, text=True,
        env={'HOME': str(home), 'PATH': '/usr/bin:/bin:'
             + str(Path.home() / '.local' / 'bin')})
    assert out.returncode == 0
    assert '[memman] Evaluate' in out.stdout


def test_nanoclaw_stop_blocks_when_inactive(
        memman_home: tuple[Path, Path]):
    home, _ = memman_home
    out = subprocess.run(
        ['bash', _nc_hook('stop.sh')],
        input='{"stop_hook_active": false}',
        capture_output=True, text=True,
        env={'HOME': str(home), 'PATH': '/usr/bin:/bin:'
             + str(Path.home() / '.local' / 'bin')})
    assert out.returncode == 0
    payload = json.loads(out.stdout)
    assert payload['decision'] == 'block'
    assert '[memman]' in payload['reason']


def test_nanoclaw_stop_silent_when_active(
        memman_home: tuple[Path, Path]):
    home, _ = memman_home
    out = subprocess.run(
        ['bash', _nc_hook('stop.sh')],
        input='{"stop_hook_active": true}',
        capture_output=True, text=True,
        env={'HOME': str(home), 'PATH': '/usr/bin:/bin:'
             + str(Path.home() / '.local' / 'bin')})
    assert out.returncode == 0
    assert out.stdout.strip() == ''
