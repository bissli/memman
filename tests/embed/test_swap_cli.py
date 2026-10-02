"""CLI integration for `memman embed swap`.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner
from memman import config
from memman.cli import cli
from memman.embed import registry
from memman.embed.fingerprint import Fingerprint, write_fingerprint
from memman.embed.vector import serialize_vector
from memman.setup import scheduler as sched_mod
from memman.store.db import open_db
from memman.store.sqlite import SqliteBackend


class _FakeTargetProvider:
    """Embedder factory used by the CLI swap test.

    Seeded into the registry cache so the CLI's
    `registry.get_for(model)` returns this instance.
    """

    def __init__(self) -> None:
        self.model = 'fake-target-d384'
        self.dim = 0

    def available(self) -> bool:
        return True

    def prepare(self) -> None:
        if self.dim == 0:
            self.dim = 384

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.01 * (i + 1) for i in range(self.dim or 384)]
                for _ in texts]

    def unavailable_message(self) -> str:
        return ''


def _seed(backend: SqliteBackend, n: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with backend.transaction():
        for i in range(n):
            backend._db._exec(
                'insert into insights'
                ' (id, content, embedding, embedding_model,'
                '  created_at, updated_at)'
                ' values (?, ?, ?, ?, ?, ?)',
                (f'id-{i:03d}', f'content {i}',
                 serialize_vector([0.01 * i] * 512),
                 'voyage-3-lite', now, now))


@pytest.fixture
def _scheduler_stopped(monkeypatch):
    """Force scheduler state to STOPPED so the swap CLI accepts the run.
    """
    monkeypatch.setattr(
        sched_mod, 'read_state', lambda: sched_mod.STATE_STOPPED)


@pytest.fixture
def cli_env(
        tmp_path, monkeypatch, _isolate_env, _scheduler_stopped):
    """Provision a single-store data dir and register the fake target.
    """
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    store_dir_path = data_dir / 'data' / 'main'
    store_dir_path.mkdir(parents=True)
    db = open_db(str(store_dir_path))
    backend = SqliteBackend(db)
    write_fingerprint(
        backend,
        Fingerprint(model='voyage-3-lite', dim=512))
    _seed(backend, 4)
    db.close()
    target = _FakeTargetProvider()
    target.prepare()
    monkeypatch.setitem(
        registry._GET_FOR_CACHE, 'fake-target-d384', target)
    (data_dir / 'env').write_text(
        f'{config.DEFAULT_BACKEND}=sqlite\n'
        f'{config.BACKEND_FOR("main")}=sqlite\n')
    return str(data_dir)


def test_swap_command_completes(cli_env):
    """`embed swap --to MODEL` switches the fingerprint.

    Mutation: the command not writing the target fingerprint, leaving
        the store on voyage.
    Oracle: the JSON body's `state` and the target's hand-set model and
        384 dimension.
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['--data-dir', cli_env, '--store', 'main',
         'embed', 'swap',
         '--to', 'fake-target-d384'])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body['state'] == 'done'
    assert body['fingerprint'] == {
        'model': 'fake-target-d384',
        'dim': 384,
        }


def test_swap_abort_clears_inflight(cli_env):
    """--abort drops embedding_pending and clears swap meta.

    Mutation: `--abort` reporting success without calling `abort_swap`,
        leaving `embed_swap_state` set.
    Oracle: the reopened store's `embed_swap_state` meta is empty.
    """
    db = open_db(str(Path(cli_env) / 'data' / 'main'))
    backend = SqliteBackend(db)
    backend.swap_prepare(384)
    backend.meta.set('embed_swap_state', 'backfilling')
    backend.meta.set('embed_swap_target_model', 'fake-target-d384')
    backend.meta.set('embed_swap_target_dim', '384')
    db.close()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['--data-dir', cli_env, '--store', 'main',
         'embed', 'swap', '--abort'])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body['state'] == 'aborted'

    db = open_db(str(Path(cli_env) / 'data' / 'main'))
    backend = SqliteBackend(db)
    try:
        assert (backend.meta.get('embed_swap_state') or '') == ''
    finally:
        db.close()


def test_swap_rejects_resume_and_abort(cli_env):
    """--resume and --abort are mutually exclusive.

    Mutation: dropping the `abort and resume` guard, so the command
        aborts and resumes together.
    Oracle: nonzero exit with `mutually exclusive` in the output.
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['--data-dir', cli_env, '--store', 'main',
         'embed', 'swap', '--resume', '--abort'])
    assert result.exit_code != 0
    assert 'mutually exclusive' in result.output


def test_swap_resume_without_inflight_errors(cli_env):
    """--resume errors when no swap is in progress.

    Mutation: `--resume` starting a fresh swap or exiting 0 when
        `read_progress` reports no state.
    Oracle: nonzero exit with `no in-flight swap` in the output.
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['--data-dir', cli_env, '--store', 'main',
         'embed', 'swap', '--resume'])
    assert result.exit_code != 0
    assert 'no in-flight swap' in result.output
