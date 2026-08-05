"""Tests for scripts/migrate_0180.py — repairs, guards, rollback.

The script is loaded via importlib (scripts/ is not a package). Pure
`repair_payload` tests build payloads directly; script-level tests
hand-build a 0.17.3-schema store, since `open_db` now only creates
the 0.18.0 shape.
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
    'migrate_0180',
    Path(__file__).parent.parent / 'scripts' / 'migrate_0180.py')
mig = importlib.util.module_from_spec(_SPEC)
# dataclass resolution reads sys.modules[cls.__module__], so the
# module must be registered before exec
sys.modules['migrate_0180'] = mig
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


def _backbone(src, tgt, direction='precedes'):
    return MigrateEdge(
        source_id=src, target_id=tgt, edge_type='temporal',
        weight=1.0,
        metadata={'sub_type': 'backbone', 'direction': direction},
        created_at=NOW)


def _payload(insights, edges):
    return MigrationPayload(
        payload_version=PAYLOAD_VERSION,
        fingerprint=seed_default_fingerprint(),
        embedding_dim=512, embedding_dtype='float64',
        insights=insights, edges=edges, oplog=[],
        embedding_pending=[], swap_state=None, meta={})


def _make_old_store(data_dir, store):
    """Hand-build a pre-0.18.0 store: 3 insights, backbone pair
    within the window (a<->b, 2 h), one beyond it (a<->c, 10 h), one
    orphan edge, one oplog row, and the meta fingerprint.
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
                " values (?, ?, ?, ?, ?)",
                (rid, f'content {rid}', f'queue:{rid}',
                 format_timestamp(ts), format_timestamp(ts)))
        edges = [
            ('a', 'b', '{"sub_type": "backbone", "direction": "precedes"}'),
            ('b', 'a', '{"sub_type": "backbone", "direction": "succeeds"}'),
            ('a', 'c', '{"sub_type": "backbone", "direction": "precedes"}'),
            ('c', 'a', '{"sub_type": "backbone", "direction": "succeeds"}'),
            ('b', 'ghost', '{"sub_type": "backbone", "direction": "precedes"}'),
            ]
        for src, tgt, meta in edges:
            conn.execute(
                'insert into edges'
                ' (source_id, target_id, edge_type, weight,'
                '  metadata, created_at)'
                " values (?, ?, 'temporal', 1.0, ?, ?)",
                (src, tgt, meta, format_timestamp(NOW)))
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


def test_repair_backfills_queue_source_to_user():
    """Synthetic `queue:N` sources become `'user'`; real ones survive.

    Mutation: dropping repair 1.
    Oracle: hand-built payload — one `queue:12` row rewritten, one
        `agent` row untouched, counter reports exactly 1.
    """
    p = _payload(
        [_mi('a', NOW, source='queue:12'),
         _mi('b', NOW, source='agent')], [])
    result = mig.repair_payload(p)
    assert [i.source for i in p.insights] == ['user', 'agent']
    assert result.sources_rewritten == 1


def test_backbone_within_window_converts_to_proximity():
    """A backbone pair 2 h apart is re-typed, not deleted.

    Mutation: deleting instead of converting (the superseded design).
    Oracle: both directions survive with weight `1/(1+2)`,
        `sub_type == 'proximity'`, and NO `direction` key — proximity
        edges are written without one, so a bare sub_type flip would
        leave a shape the write path never produces.
    """
    p = _payload(
        [_mi('a', NOW), _mi('b', NOW - timedelta(hours=2))],
        [_backbone('a', 'b'), _backbone('b', 'a', 'succeeds')])
    result = mig.repair_payload(p)
    assert result.edges_converted == 2
    assert result.edges_dropped == 0
    assert len(p.edges) == 2
    for e in p.edges:
        assert e.metadata['sub_type'] == 'proximity'
        assert 'direction' not in e.metadata
        assert e.weight == pytest.approx(1.0 / 3.0)
        assert e.metadata['hours_diff'] == '2.00'


def test_backbone_beyond_window_is_dropped():
    """A backbone pair 10 h apart is false adjacency and is dropped.

    Mutation: converting unconditionally (no window test).
    Oracle: both directions gone, `edges_dropped == 2`, zero
        conversions.
    """
    p = _payload(
        [_mi('a', NOW), _mi('c', NOW - timedelta(hours=10))],
        [_backbone('a', 'c'), _backbone('c', 'a', 'succeeds')])
    result = mig.repair_payload(p)
    assert result.edges_dropped == 2
    assert result.edges_converted == 0
    assert p.edges == []


def test_backbone_with_missing_endpoint_does_not_raise():
    """A backbone edge whose endpoint has no insights row passes through.

    `default` holds 110 such edges; without the guard the timestamp
    lookup raises on the one store the runbook declares mandatory.

    Mutation: dropping the missing-endpoint guard (KeyError), or
        counting the edge as converted/dropped instead of leaving it
        for the orphan filter.
    Oracle: no exception, and the edge lands in
        `orphan_edges_dropped`.
    """
    p = _payload(
        [_mi('a', NOW)],
        [_backbone('a', 'ghost')])
    result = mig.repair_payload(p)
    assert result.orphan_edges_dropped == 1
    assert result.edges_converted == 0
    assert result.edges_dropped == 0
    assert p.edges == []


def test_touched_ids_excludes_dead_endpoints():
    """`touched_ids` carries only live, non-soft-deleted endpoints.

    `refresh_effective_importance` selects `where deleted_at is null`
    and raises `ValueError` on a miss — after `apply` committed and
    the original directory was archived away.

    Mutation: leaving `touched_ids` unintersected with live ids, or
        collecting converted edges' endpoints too.
    Oracle: a dropped >4 h pair with one soft-deleted endpoint plus
        an orphan edge yields exactly the one live id.
    """
    p = _payload(
        [_mi('a', NOW),
         _mi('c', NOW - timedelta(hours=10), deleted_at=NOW),
         _mi('w1', NOW), _mi('w2', NOW - timedelta(hours=1))],
        [_backbone('a', 'c'), _backbone('c', 'a', 'succeeds'),
         _backbone('a', 'ghost'),
         _backbone('w1', 'w2'), _backbone('w2', 'w1', 'succeeds')])
    result = mig.repair_payload(p)
    assert result.touched_ids == {'a'}


def test_repair_drops_orphan_edges(tmp_path):
    """An edge to a missing insight never reaches the FK-checked insert.

    Mutation: dropping repair 3 — `apply` opens with
        `pragma foreign_keys=on`, so the orphan hits the insert as a
        FOREIGN KEY failure after the store was archived away.
    Oracle: the rebuild completes and the new store holds zero edges
        touching the ghost id.
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
        prox = conn.execute(
            "select weight, json_extract(metadata, '$.direction')"
            " from edges where json_extract(metadata,"
            " '$.sub_type') = 'proximity'").fetchall()
        beyond = conn.execute(
            "select count(*) from edges"
            " where source_id = 'c' or target_id = 'c'").fetchone()[0]
    assert ghost == 0
    assert len(prox) == 2
    assert all(w == pytest.approx(1.0 / 3.0) for w, _d in prox)
    assert all(d is None for _w, d in prox)
    assert beyond == 0


def test_script_refuses_already_migrated_store(tmp_path):
    """A second run skips a migrated store instead of re-archiving it.

    Mutation: dropping the column-probe guard — the re-run would
        re-archive and repair 2 would re-type legitimate session
        backbone edges into proximity edges with nothing signalling
        the damage.
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
    monkeypatch.setattr(
        'sys.argv',
        ['migrate_0180.py', 's4', '--data-dir', str(tmp_path)])
    assert mig.main() == 1
    assert (tmp_path / mig.BREADCRUMB_NAME).exists()


def test_script_rejects_wrong_interpreter(monkeypatch):
    """A memman whose PAYLOAD_VERSION is not 2 must abort.

    The pipx interpreter would import 0.17.3's `SqliteMigrator` and
    silently rebuild the old schema.

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
