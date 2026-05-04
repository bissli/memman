"""Phase 4b slice 5 -- install wizard probe assertions.

Phase 4 gate item 5: install probe asserts `SELECT 1`, pgvector
extension presence, and emits a PgBouncer recommendation on
non-localhost DSNs.
"""

import psycopg
import pytest

pytestmark = pytest.mark.postgres


def test_probe_dsn_succeeds_on_pgvector_database(pg_dsn):
    """SELECT 1 + pgvector check both pass on the test container."""
    from memman.setup.wizard import _probe_dsn
    _probe_dsn(pg_dsn)


def test_probe_dsn_raises_when_pgvector_missing(pg_dsn):
    """Drop pgvector and verify the probe complains; restore after."""
    from memman.setup.wizard import _probe_dsn

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute('DROP EXTENSION IF EXISTS vector CASCADE')
    try:
        with pytest.raises(RuntimeError, match='pgvector'):
            _probe_dsn(pg_dsn)
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute('CREATE EXTENSION IF NOT EXISTS vector')


def test_is_remote_dsn_recognizes_localhost():
    """Localhost DSNs return False (no PgBouncer hint emitted)."""
    from memman.setup.wizard import _is_remote_dsn
    assert _is_remote_dsn('host=localhost port=5432 dbname=x') is False
    assert _is_remote_dsn('host=127.0.0.1 port=5432') is False
    assert _is_remote_dsn(
        'postgresql://user:pass@localhost:5432/db') is False
    assert _is_remote_dsn(
        'postgresql://user:pass@127.0.0.1/db') is False


def test_is_remote_dsn_recognizes_non_localhost():
    """Remote hosts return True (hint will fire)."""
    from memman.setup.wizard import _is_remote_dsn
    assert _is_remote_dsn(
        'postgresql://user:pass@db.example.com:5432/foo') is True
    assert _is_remote_dsn('host=db.internal port=5432') is True


def test_probe_dsn_emits_pgbouncer_hint_on_remote_dsn(
        pg_dsn, capsys, monkeypatch):
    """Probe of a remote-shaped DSN emits the PgBouncer hint.

    We can't easily mint a real remote DSN inside the testcontainer;
    instead we monkeypatch `_is_remote_dsn` to return True for the
    test container's DSN and check that the hint appears in stdout.
    """
    from memman.setup import wizard as wiz_mod
    monkeypatch.setattr(wiz_mod, '_is_remote_dsn', lambda _dsn: True)
    wiz_mod._probe_dsn(pg_dsn)
    captured = capsys.readouterr()
    assert 'PgBouncer' in captured.out
