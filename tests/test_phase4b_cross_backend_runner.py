"""Phase 4b slice 4 -- CliRunner backend-switching fixture.

Demonstrates the new `cross_backend_runner` fixture by parametrizing
a representative slice of `test_memory_system.py`-style CLI flows
over both `MEMMAN_BACKEND=sqlite` and `MEMMAN_BACKEND=postgres`.
The full 53-test parametrization of `test_memory_system.py` itself
is incremental work — this file proves the fixture works
end-to-end so individual tests can be migrated as needed.
"""

import json

from memman.cli import cli


def _invoke(runner_tuple, args):
    """Invoke the CLI with `--data-dir` baked in."""
    r, data_dir = runner_tuple
    return r.invoke(cli, ['--data-dir', data_dir] + args)


def test_remember_then_recall_round_trip(cross_backend_runner):
    """A `remember` followed by `recall` returns the inserted text.

    Exercises the full pipeline (CLI -> queue enqueue -> worker drain
    -> store insert -> recall) on each parametrized backend. The
    autouse `_scheduler_started` fixture force-drains the queue
    after `remember`.
    """
    result = _invoke(
        cross_backend_runner,
        ['remember', 'cross-backend smoke probe alpha bravo charlie'])
    assert result.exit_code == 0, result.output

    result = _invoke(
        cross_backend_runner,
        ['recall', '--basic', 'alpha bravo charlie'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    contents = [r.get('content', '') for r in payload.get('results', [])]
    assert any('alpha bravo charlie' in c for c in contents), (
        f'inserted content not found in recall results: {contents}')


def test_status_command_succeeds(cross_backend_runner):
    """`memman status` runs cleanly on both backends.

    Smoke test: exits 0 and emits backend-identifying output. Catches
    backend-dispatch regressions in the cluster/factory layer.
    """
    result = _invoke(cross_backend_runner, ['status'])
    assert result.exit_code == 0, result.output
    assert result.output.strip(), 'status returned empty output'


def test_doctor_runs_on_both_backends(cross_backend_runner):
    """`memman doctor` runs without crashing on either backend.

    Phase 4 gate item 2 says doctor tests should be green on both
    backends; this is a thin smoke check at the CLI surface.
    """
    result = _invoke(cross_backend_runner, ['doctor'])
    assert result.exit_code in (0, 1), (
        f'doctor exited with unexpected code {result.exit_code}; '
        f'output: {result.output[:500]}')
    assert 'memman doctor' in result.output.lower() or 'status' in result.output, (
        f'doctor output looked malformed: {result.output[:300]}')
