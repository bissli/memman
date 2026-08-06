"""Tests for scripts/rebuild_schema.py — guards, rollback, orphans.

The retained rebuild machinery from the 0.18.0 migration; the one-off
data repairs are gone, so these tests cover the structural pieces:
the orphan filter, the interpreter/schema guards, the rollback, and
the conditional gather. The script is loaded via importlib
(scripts/ is not a package); script-level tests hand-build a
0.17.3-schema store, since `open_db` now only creates the current
shape.
"""

import importlib.util
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from memman.embed.fingerprint import seed_default_fingerprint
from memman.migrate import PAYLOAD_VERSION, MigrateEdge, MigrateInsight
from memman.migrate import MigrationPayload
from memman.store.model import format_timestamp

_SPEC = importlib.util.spec_from_file_location(
    'rebuild_schema',
    Path(__file__).parent.parent / 'scripts' / 'rebuild_schema.py')
mig = importlib.util.module_from_spec(_SPEC)
# dataclass resolution reads sys.modules[cls.__module__], so the
# module must be registered before exec
sys.modules['rebuild_schema'] = mig
_SPEC.loader.exec_module(mig)

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)

OLD_SCHEMA = """
create table insights (
    id          text primary key,
    content     text not null,
    category    text default 'general',
    importance  integer default 3,
    entities    text default '[]',
    source      text default 'user',
    access_count integer default 0,
    keywords    text,
    summary     text,
    semantic_facts text,
    last_accessed_at text,
    embedding   blob,
    embedding_pending blob,
    effective_importance real default 0.5,
    linked_at   text,
    enriched_at text,
    created_at  text not null,
    updated_at  text not null,
    deleted_at  text,
    prompt_version text,
    model_id    text,
    embedding_model text
);
create table edges (
    source_id   text not null,
    target_id   text not null,
    edge_type   text not null,
    weight      real default 1.0,
    metadata    text default '{}',
    created_at  text not null,
    primary key (source_id, target_id, edge_type)
);
create table oplog (
    id          integer primary key autoincrement,
    operation   text not null,
    insight_id  text,
    detail      text default '',
    created_at  text not null,
    before      text,
    after       text
);
create table meta (key text primary key, value text not null);
"""


def _mi(id, created_at, *, source='user', deleted_at=None):
    return MigrateInsight(
        id=id, content=f'content {id}', category='fact',
        importance=3, entities=[], source=source, access_count=0,
        keywords=None, summary=None, semantic_facts=None,
        last_accessed_at=None, embedding=None,
        effective_importance=0.5, linked_at=None, enriched_at=None,
        created_at=created_at, updated_at=created_at,
        deleted_at=deleted_at, prompt_version=None, model_id=None,
        embedding_model=None, session_id=None, queue_uuid=None)


def _semantic(src, tgt):
    return MigrateEdge(
        source_id=src, target_id=tgt, edge_type='semantic',
        weight=0.7, metadata={}, created_at=NOW)


def _payload(insights, edges):
    return MigrationPayload(
        payload_version=PAYLOAD_VERSION,
        fingerprint=seed_default_fingerprint(),
        embedding_dim=512, embedding_dtype='float64',
        insights=insights, edges=edges, oplog=[],
        embedding_pending=[], swap_state=None, meta={})


def _make_old_store(data_dir, store):
    """Hand-build a pre-0.18.0 store: 3 insights, one live semantic
    pair, one orphan edge, one oplog row, and the meta fingerprint.
    """
    sdir = Path(data_dir) / 'data' / store
    sdir.mkdir(parents=True)
    conn = sqlite3.connect(str(sdir / 'memman.db'))
    try:
        conn.executescript(OLD_SCHEMA)
        rows = [
            ('a', NOW),
            ('b', NOW - timedelta(hours=2)),
            ('c', NOW - timedelta(hours=10)),
            ]
        for rid, ts in rows:
            conn.execute(
                'insert into insights'
                ' (id, content, source, created_at, updated_at)'
                ' values (?, ?, ?, ?, ?)',
                (rid, f'content {rid}', 'user',
                 format_timestamp(ts), format_timestamp(ts)))
        edges = [('a', 'b'), ('b', 'a'), ('b', 'ghost')]
        for src, tgt in edges:
            conn.execute(
                'insert into edges'
                ' (source_id, target_id, edge_type, weight,'
                '  metadata, created_at)'
                " values (?, ?, 'semantic', 0.7, '{}', ?)",
                (src, tgt, format_timestamp(NOW)))
        conn.execute(
            'insert into oplog (operation, insight_id, created_at)'
            " values ('remember', 'a', ?)",
            (format_timestamp(NOW),))
        conn.execute(
            "insert into meta (key, value) values"
            " ('embed_fingerprint', ?)",
            (seed_default_fingerprint().to_json(),))
        conn.commit()
    finally:
        conn.close()
    return sdir


def test_touched_ids_excludes_dead_endpoints():
    """`touched_ids` carries only live, non-soft-deleted endpoints.

    `refresh_effective_importance` selects `where deleted_at is null`
    and raises `ValueError` on a miss — after `apply` committed and
    the original directory was archived away.

    Mutation: leaving `touched_ids` unintersected with live ids.
    Oracle: orphan edges touching one live id, one soft-deleted id
        and one missing id yield exactly the live id.
    """
    p = _payload(
        [_mi('a', NOW), _mi('d', NOW, deleted_at=NOW)],
        [_semantic('a', 'ghost'), _semantic('d', 'ghost')])
    result = mig.repair_payload(p)
    assert result.orphan_edges_dropped == 2
    assert result.touched_ids == {'a'}


def test_repair_drops_orphan_edges(tmp_path):
    """An edge to a missing insight never reaches the FK-checked insert.

    Mutation: dropping the orphan filter — `apply` opens with
        `pragma foreign_keys=on`, so the orphan hits the insert as a
        FOREIGN KEY failure after the store was archived away.
    Oracle: the rebuild completes, the ghost edge is gone, and the
        live pair survives untouched.
    """
    _make_old_store(tmp_path, 's1')
    mig.migrate_store(
        mig.SqliteMigrator(str(tmp_path)), 's1', str(tmp_path))
    db_path = tmp_path / 'data' / 's1' / 'memman.db'
    with sqlite3.connect(str(db_path)) as conn:
        ghost = conn.execute(
            "select count(*) from edges"
            " where source_id = 'ghost' or target_id = 'ghost'"
            ).fetchone()[0]
        live = conn.execute(
            'select count(*) from edges').fetchone()[0]
    assert ghost == 0
    assert live == 2


def test_script_refuses_already_migrated_store(tmp_path):
    """A second run skips a migrated store instead of re-archiving it.

    Mutation: dropping the column-probe guard — the re-run would
        re-archive a healthy store and re-run `repair_payload`
        against data it already repaired.
    Oracle: `store_gate` answers 'skip' after a successful rebuild.
    """
    _make_old_store(tmp_path, 's2')
    mig.migrate_store(
        mig.SqliteMigrator(str(tmp_path)), 's2', str(tmp_path))
    assert mig.store_gate(str(tmp_path), 's2', False) == 'skip'


def test_post_archive_failure_restores_the_store(tmp_path, monkeypatch):
    """A failure after archive moves the original directory back.

    The failure mode this guards is a valid, empty, doctor-clean
    store with the real rows stranded in `archive/`.

    Mutation: dropping the rollback, or copying instead of moving.
    Oracle: after a forced `apply` failure, `data/<store>` holds the
        original 3 rows and `archive/<store>` holds nothing.
    """
    _make_old_store(tmp_path, 's3')
    monkeypatch.setattr(
        mig.SqliteMigrator, 'apply',
        lambda self, store, payload: (_ for _ in ()).throw(
            RuntimeError('forced apply failure')))
    with pytest.raises(RuntimeError, match='forced apply failure'):
        mig.migrate_store(
            mig.SqliteMigrator(str(tmp_path)), 's3', str(tmp_path))
    db_path = tmp_path / 'data' / 's3' / 'memman.db'
    assert db_path.exists()
    with sqlite3.connect(str(db_path)) as conn:
        assert conn.execute(
            'select count(*) from insights').fetchone()[0] == 3
    archived = list((tmp_path / 'archive' / 's3').glob('*')) if (
        tmp_path / 'archive' / 's3').exists() else []
    assert archived == []
    assert not (tmp_path / mig.BREADCRUMB_NAME).exists()


def test_new_schema_empty_store_is_not_treated_as_migrated(
        tmp_path, monkeypatch):
    """Zero rows on the new schema is the killed-mid-apply signature.

    `apply` commits the new schema via `open_db` BEFORE its
    transaction opens, so a SIGKILL leaves a new-schema db with zero
    rows while the real rows sit in `archive/`.

    Mutation: gating guard 2 on the column probe alone.
    Oracle: `store_gate` answers 'migrate' for an empty new-schema
        store, and a present breadcrumb makes `main` abort (exit 1)
        without touching anything.
    """
    from memman.store.db import open_db
    sdir = tmp_path / 'data' / 's4'
    open_db(str(sdir)).close()
    assert mig.store_gate(str(tmp_path), 's4', False) == 'migrate'

    (tmp_path / mig.BREADCRUMB_NAME).write_text('s4')
    # --log must stay inside tmp_path: the argparse default is the
    # shared host path an operator reads after a REAL rebuild, and an
    # unpointed test run appends pytest noise into it
    monkeypatch.setattr(
        'sys.argv',
        ['rebuild_schema.py', 's4', '--data-dir', str(tmp_path),
         '--log', str(tmp_path / 'rebuild.log')])
    assert mig.main() == 1
    assert (tmp_path / mig.BREADCRUMB_NAME).exists()


def test_script_rejects_wrong_interpreter(monkeypatch):
    """A memman whose payload version differs must abort.

    A mismatched interpreter would import another release's
    `SqliteMigrator` and silently rebuild the wrong schema.

    Mutation: dropping the import guard.
    Oracle: `assert_correct_interpreter` raises `SystemExit` when the
        loaded payload version differs.
    """
    monkeypatch.setattr(mig, 'PAYLOAD_VERSION', 99)
    with pytest.raises(SystemExit, match='PAYLOAD_VERSION'):
        mig.assert_correct_interpreter()


def test_gather_tolerates_premigration_store(tmp_path):
    """Gather from a store without the new columns must not raise.

    Mutation: adding `session_id`/`queue_uuid` to the select
        unconditionally — the rebuild would fail on the first store.
    Oracle: gather succeeds against a hand-built 0.17.3 store and
        fills both fields with None.
    """
    from memman.store.sqlite import SqliteMigrator
    _make_old_store(tmp_path, 's5')
    payload = SqliteMigrator(str(tmp_path)).gather('s5')
    assert len(payload.insights) == 3
    assert all(i.session_id is None for i in payload.insights)
    assert all(i.queue_uuid is None for i in payload.insights)


def test_migration_payload_round_trips_new_fields(tmp_path):
    """session_id/queue_uuid survive gather -> apply -> gather.

    Mutation: dropping either field from `MigrateInsight` or from
        one side of the gather/apply pair.
    Oracle: values written through the Backend insert come back
        identical after a full rebuild cycle.
    """
    from memman.store.db import open_db, set_meta, store_dir
    from memman.store.node import insert_insight
    from memman.store.sqlite import SqliteMigrator
    from tests.conftest import make_insight
    sdir = store_dir(str(tmp_path), 's6')
    db = open_db(sdir)
    try:
        insert_insight(db, make_insight(
            id='rt-1', session_id='sess-9', queue_uuid='uuid-9'))
        set_meta(db, 'embed_fingerprint',
                 seed_default_fingerprint().to_json())
    finally:
        db.close()
    m = SqliteMigrator(str(tmp_path))
    payload = m.gather('s6')
    assert payload.insights[0].session_id == 'sess-9'
    assert payload.insights[0].queue_uuid == 'uuid-9'
    m.apply('s7', payload)
    again = m.gather('s7')
    assert again.insights[0].session_id == 'sess-9'
    assert again.insights[0].queue_uuid == 'uuid-9'


def test_apply_rejects_unknown_payload_version(tmp_path):
    """`apply` refuses a payload whose version is not this build's.

    Mutation: leaving `PAYLOAD_VERSION` declarative (written by
        gather, read by nothing) — a stale payload would silently
        rebuild the wrong shape.
    Oracle: version 99 raises `MigrateError` before anything is
        written.
    """
    from memman.migrate import MigrateError
    from memman.store.sqlite import SqliteMigrator
    _make_old_store(tmp_path, 's8')
    m = SqliteMigrator(str(tmp_path))
    payload = m.gather('s8')
    payload.payload_version = 99
    with pytest.raises(MigrateError, match='payload version 99'):
        m.apply('s8_target', payload)
    assert not (tmp_path / 'data' / 's8_target' / 'memman.db').exists()


def test_open_db_names_the_migration_script(tmp_path):
    """A pre-migration store fails at open naming the rebuild script.

    This open-time error is the primary diagnostic — a pre-migration
    store never opens, so nothing that needs a live Backend can
    report on it.

    Mutation: dropping the `_migrate` error translation (a raw
        `no such column` OperationalError surfaces instead).
    Oracle: `open_db` raises RuntimeError naming both the store dir
        and `MIGRATION_SCRIPT`.
    """
    from memman.store.db import MIGRATION_SCRIPT, open_db
    _make_old_store(tmp_path, 's9')
    with pytest.raises(RuntimeError, match='rebuild with') as excinfo:
        open_db(str(tmp_path / 'data' / 's9'))
    assert MIGRATION_SCRIPT in str(excinfo.value)
    assert 's9' in str(excinfo.value)
