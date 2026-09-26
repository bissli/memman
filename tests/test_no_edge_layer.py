"""A store holds insights, their meta and an oplog, and no edges.

Recall ranks the fused keyword, vector and recency anchors directly.
No table, verb, command, check or config key reads or writes an edge,
no row carries a session id, and a drain opens only the stores its
rows name.
"""

import json
import time
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner
from memman.cli import cli
from memman.doctor import run_all_checks
from memman.embed.fingerprint import bound_embedder
from memman.migrate import MigrateInsight, MigrationPayload
from memman.pipeline.remember import run_remember
from memman.queue import open_queue_db
from memman.search import recall as recall_mod
from memman.search.recall import ANCHOR_TOP_K, RERANK_SHORTLIST
from memman.search.recall import intent_aware_recall
from memman.store.factory import open_backend
from tests.conftest import _vec, invoke, make_insight, parse_remember
from tests.conftest import set_created_at


def _remember(runner, text):
    """Store `text` verbatim and return its id."""
    res = invoke(runner, ['remember', text])
    assert res.exit_code == 0, res.output
    return parse_remember(res, runner)['id']


def _table_names(backend):
    """Every table name in the store, on either backend."""
    if hasattr(backend, '_db'):
        rows = backend._db._query(
            "select name from sqlite_master where type = 'table'").fetchall()
        return {r[0] for r in rows}
    with backend._conn.cursor() as cur:
        cur.execute(
            'select table_name from information_schema.tables'
            ' where table_schema = %s', (backend._schema,))
        return {r[0] for r in cur.fetchall()}


def _insight_columns(backend):
    """Every column name of the insights table, on either backend."""
    if hasattr(backend, '_db'):
        rows = backend._db._query('pragma table_info(insights)').fetchall()
        return {r[1] for r in rows}
    with backend._conn.cursor() as cur:
        cur.execute(
            'select column_name from information_schema.columns'
            " where table_schema = %s and table_name = 'insights'",
            (backend._schema,))
        return {r[0] for r in cur.fetchall()}


def test_a_fresh_store_has_no_edges_table(backend):
    """Verify the baseline schema creates no edges table.

    Mutation: leaving the edges block in `_BASELINE_SCHEMA` or
        `PG_BASELINE_SCHEMA`, which recreates the table on every open
        and undoes the hand drop.
    Oracle: the catalog read directly, beside `insights`, which proves
        the read sees the store's tables.
    """
    tables = _table_names(backend)

    assert 'insights' in tables
    assert 'edges' not in tables


def test_a_fresh_store_has_no_session_column(backend):
    """Verify the insights table carries no session_id column.

    Mutation: leaving `session_id` in either baseline schema, so a
        fresh store keeps a column nothing reads.
    Oracle: the catalog's column list, beside `content`.
    """
    columns = _insight_columns(backend)

    assert 'content' in columns
    assert 'session_id' not in columns


def test_a_fresh_queue_has_no_session_column(tmp_path):
    """Verify the queue table carries no session_id column.

    Mutation: leaving `session_id` in the queue DDL, so every enqueue
        still writes a column no drain reads.
    Oracle: `pragma table_info(queue)` on a fresh queue.db.
    """
    conn = open_queue_db(str(tmp_path))
    try:
        columns = {r[1] for r in conn.execute('pragma table_info(queue)')}
    finally:
        conn.close()

    assert 'content' in columns
    assert 'session_id' not in columns


@pytest.mark.postgres
def test_postgres_storage_summary_sizes_every_table(pg_dsn):
    """Verify `storage_summary` sizes insights, oplog and meta, and no edges.

    Mutation: leaving 'edges' in the summary loop, whose regclass cast
        raises on a store without the table and drops the oplog and
        meta sizes with it.
    Oracle: the exact set of size keys on a fresh schema.
    """
    from memman.store.postgres import drop_postgres_store
    from memman.store.postgres import open_postgres_backend
    name = 'no_edge_summary'
    drop_postgres_store(name, pg_dsn)
    backend = open_postgres_backend(name, pg_dsn)
    try:
        summary = backend.storage_summary()
    finally:
        backend.close()
        drop_postgres_store(name, pg_dsn)

    assert {k for k in summary if k.endswith('_bytes')} == {
        'insights_bytes', 'oplog_bytes', 'meta_bytes'}


def test_the_backend_exposes_no_edge_or_session_verbs(backend):
    """Verify no verb that served edges or sessions survives.

    Mutation: keeping any verb whose only caller was the edge layer,
        the temporal backbone or the drain's whole-store vector cache.
    Oracle: the attribute list on a live backend, which covers the
        Protocol defaults as well as each binding.
    """
    left = [verb for verb in (
        'get_recent_in_window', 'get_latest_by_session', 'clear_linked_at',
        'iter_embeddings_as_vecs', 'get_all_embeddings', 'count_orphans',
        ) if hasattr(backend.nodes, verb)]
    left += [verb for verb in ('edges', 'write_lock')
             if hasattr(backend, verb)]

    assert left == []


def test_opening_a_store_writes_no_constants_hash(tmp_path):
    """Verify a checked open leaves the meta table without constants_hash.

    Mutation: keeping the constants reindex on open, which stamps the
        hash on a fresh store's first open.
    Oracle: the meta key read inside the same open.
    """
    from memman.session import active_store
    with active_store(data_dir=str(tmp_path), store='default') as backend:
        assert backend.meta.get('constants_hash') is None


def test_a_drain_reads_no_whole_store_vectors(mm_runner, monkeypatch):
    """Verify a drain stores a row without reading every stored vector.

    Mutation: keeping `_StoreContext.embed_cache`, which reads every
        vector in the store at each drain's store open.
    Oracle: the stored row, with both whole-store readers patched to
        raise.
    """
    from memman.store.sqlite import SqliteNodeStore

    def boom(self):
        raise AssertionError('whole-store vector read')

    monkeypatch.setattr(
        SqliteNodeStore, 'iter_embeddings_as_vecs', boom, raising=False)
    monkeypatch.setattr(
        SqliteNodeStore, 'get_all_embeddings', boom, raising=False)
    _, data_dir = mm_runner

    res = invoke(mm_runner, ['remember', 'the drain stores this row'])

    assert res.exit_code == 0, res.output
    with open_backend('default', data_dir, read_only=True) as backend:
        assert [i.content for i in backend.nodes.get_all_active()] == [
            'the drain stores this row']


def test_run_remember_reports_no_edges(tmp_backend):
    """Verify the write result names no edge counts and takes no store name.

    Mutation: keeping `edges_created` on the result, or the
        `store_name` keyword that only chose a semantic threshold.
    Oracle: the result dict's own keys.
    """
    ec = bound_embedder(tmp_backend)
    parent = make_insight(id='no-edge-1', content='Redis backs the cache')

    res = run_remember(tmp_backend, parent, 'Redis backs the cache', ec=ec)

    assert 'edges_created' not in res['facts'][0]


def test_the_newest_row_scores_above_an_older_one_on_recency(backend):
    """Verify the anchor score carries recency into the blended score.

    Mutation: deleting the anchor term or zeroing its weight, which
        scores both rows 0 on a query with no keyword or vector match.
    Oracle: two rows matching nothing, a day apart: the time channel
        ranks the newer first, so min-max gives it anchor 1.0 and the
        older 0.0.
    """
    now = datetime.now(timezone.utc)
    backend.nodes.insert(make_insight(id='older', content='alpha body'))
    backend.nodes.insert(make_insight(id='newer', content='beta body'))
    set_created_at(backend, 'older', now - timedelta(days=2))
    set_created_at(backend, 'newer', now - timedelta(days=1))

    rows = {r['insight'].id: r for r in intent_aware_recall(
        backend, 'zzz', None, 5)['results']}

    assert rows['newer']['signals']['anchor'] == pytest.approx(1.0)
    assert rows['newer']['score'] > rows['older']['score']


class _SpyingSession:
    """Recall session proxy that records the vector channel's k."""

    def __init__(self, inner, calls):
        self._inner = inner
        self._calls = calls

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def vector_anchors(self, query_vec, **kwargs):
        self._calls['vector_k'] = kwargs['k']
        return self._inner.vector_anchors(query_vec, **kwargs)


def test_the_vector_channel_alone_widens_to_the_rerank_shortlist(
        backend, monkeypatch):
    """Verify vector anchors take RERANK_SHORTLIST and keyword ANCHOR_TOP_K.

    Mutation: passing `anchor_k` to `vector_anchors`, which shrinks
        the rerank pool below its shortlist, or widening the keyword
        channel with it.
    Oracle: spies on the two channel calls.
    """
    backend.nodes.insert(make_insight(id='vec-row', content='vector row'))
    backend.nodes.update_embedding('vec-row', _vec(1.0), 'test-model')
    calls = {}
    real_session = backend.recall_session

    @contextmanager
    def spying_session():
        with real_session() as session:
            yield _SpyingSession(session, calls)

    monkeypatch.setattr(backend, 'recall_session', spying_session)
    real_keyword = recall_mod.keyword_search

    def spying_keyword(pool, query, limit, counts):
        calls['keyword_k'] = limit
        return real_keyword(pool, query, limit, counts)

    monkeypatch.setattr(recall_mod, 'keyword_search', spying_keyword)

    intent_aware_recall(backend, 'vector row', _vec(1.0), 5)

    assert calls == {'vector_k': RERANK_SHORTLIST, 'keyword_k': ANCHOR_TOP_K}


def test_the_time_channel_stays_at_anchor_top_k(backend):
    """Verify recency seeds at most ANCHOR_TOP_K rows.

    Mutation: widening the time channel along with the vector one,
        which floods the pool with recent rows whatever the query.
    Oracle: ANCHOR_TOP_K + 5 rows matching nothing, recalled
        unbounded: only the time channel reaches them.
    """
    for i in range(ANCHOR_TOP_K + 5):
        backend.nodes.insert(make_insight(id=f't-{i}', content=f'row {i}'))

    resp = intent_aware_recall(backend, 'zzz', None, 0)

    assert len(resp['results']) == ANCHOR_TOP_K


def test_supersede_reports_no_edge_count(mm_runner):
    """Verify `supersede` output names only the two rows.

    Mutation: keeping `edges_moved` in the JSON.
    Oracle: the exact output key set.
    """
    old = _remember(mm_runner, 'the broker is kombu')
    new = _remember(mm_runner, 'the broker is redis now')

    res = invoke(mm_runner, ['supersede', old, new])

    assert res.exit_code == 0, res.output
    assert set(json.loads(res.output)) == {'predecessor', 'successor'}


def test_unsupersede_reports_no_edge_count(mm_runner):
    """Verify `unsupersede` output names only the row and its old successor.

    Mutation: keeping `edges_created` in the JSON.
    Oracle: the exact output key set.
    """
    old = _remember(mm_runner, 'the broker is kombu')
    new = _remember(mm_runner, 'the broker is redis now')
    assert invoke(mm_runner, ['supersede', old, new]).exit_code == 0
    assert invoke(mm_runner, ['forget', new]).exit_code == 0

    res = invoke(mm_runner, ['unsupersede', old])

    assert res.exit_code == 0, res.output
    assert set(json.loads(res.output)) == {'id', 'was_superseded_by'}


def test_status_reports_no_edge_count(mm_runner):
    """Verify `status` carries no edge count.

    Mutation: keeping `edge_count` on `NodeStats` and in the output.
    Oracle: the output keys, beside the one-row total.
    """
    _remember(mm_runner, 'a status row')

    out = json.loads(invoke(mm_runner, ['status']).output)

    assert out['total_insights'] == 1
    assert 'edge_count' not in out


def test_the_prime_status_line_counts_insights_only(mm_runner):
    """Verify the SessionStart status line counts rows and names no edges.

    Mutation: the status line still reading an edge count, which on a
        store without the table falls back to the bare line with no
        count at all.
    Oracle: the first output line on a one-row store.
    """
    _remember(mm_runner, 'a prime row')

    result = CliRunner().invoke(cli, ['prime'], input='{}')

    first = result.output.splitlines()[0]
    assert '(1 insight' in first
    assert 'edge' not in first


def test_the_prime_guide_names_no_session_flag():
    """Verify the injected guide asks for no session id.

    Mutation: keeping `--session $SESSION_ID` in guide.md, or the
        substitution that fills it from the hook input.
    Oracle: the prime output for a hook input carrying a session id.
    """
    payload = json.dumps({'session_id': 'sess-z'})

    result = CliRunner().invoke(cli, ['prime'], input=payload)

    assert '--session' not in result.output
    assert 'sess-z' not in result.output


@pytest.mark.parametrize('command', [
    ['remember', 'a row', '--session', 's-1'],
    ['recall', 'a row', '--session', 's-1'],
    ])
def test_no_command_takes_a_session_flag(mm_runner, command):
    """Verify remember and recall reject `--session`.

    Mutation: keeping the option on either command, with its
        MEMMAN_SESSION_ID and CLAUDE_CODE_SESSION_ID fallbacks.
    Oracle: click's own unknown-option refusal.
    """
    res = invoke(mm_runner, command)

    assert res.exit_code != 0
    assert 'No such option' in res.output


def test_replace_takes_no_session_flag(mm_runner):
    """Verify replace rejects `--session`.

    Mutation: keeping the option on `replace`.
    Oracle: click's own unknown-option refusal.
    """
    old = _remember(mm_runner, 'the broker is kombu')

    res = invoke(mm_runner, ['replace', old, 'redis now', '--session', 's'])

    assert res.exit_code != 0
    assert 'No such option' in res.output


@pytest.mark.parametrize('command', ['link', 'related'])
def test_the_graph_group_has_no_link_or_related(mm_runner, command):
    """Verify `graph link` and `graph related` do not exist.

    Mutation: keeping either command.
    Oracle: click's own unknown-command refusal.
    """
    res = invoke(mm_runner, ['graph', command, 'a', 'b'])

    assert res.exit_code != 0
    assert 'No such command' in res.output


@pytest.mark.parametrize(('key', 'value'), [
    ('MEMMAN_SURFACE_default', 'code'),
    ('MEMMAN_AUTO_SEMANTIC_THRESHOLD_default', '0.5'),
    ])
def test_config_set_rejects_the_threshold_keys(mm_runner, key, value):
    """Verify the two semantic-threshold keys are unrecognized.

    Mutation: keeping their `PER_STORE_KEY_SPECS` rows.
    Oracle: the unrecognized-key refusal text.
    """
    res = invoke(mm_runner, ['config', 'set', key, value])

    assert res.exit_code != 0
    assert 'not a recognized config key' in res.output


def test_doctor_runs_no_edge_checks(backend):
    """Verify doctor runs no edge or threshold check and takes no store name.

    Mutation: keeping any of the four checks, or the `store_name`
        keyword that only chose the threshold surface.
    Oracle: the check names on a one-row store.
    """
    backend.nodes.insert(make_insight(id='doc-row', content='doctor row'))

    names = {c['name'] for c in run_all_checks(backend)['checks']}

    assert names.isdisjoint({
        'orphan_insights', 'dangling_edges', 'edge_degree',
        'embed_threshold'})


def test_supersession_integrity_reports_three_populations(backend):
    """Verify the integrity report has no edge population.

    Mutation: keeping `superseded_with_edges`, whose query joins a
        table the store no longer holds.
    Oracle: the exact key set.
    """
    counts = backend.nodes.supersession_integrity()

    assert set(counts) == {'dangling', 'self_pointer', 'unterminated'}


def test_the_migration_payload_carries_no_edges_or_sessions():
    """Verify the payload carries no edges and no session id.

    Mutation: keeping the `edges` payload field or the `session_id`
        insight field, which the target store has no table or column
        to load. tests/test_no_group3_fields.py pins the version the
        current shape carries.
    Oracle: the dataclass fields.
    """
    payload_fields = {f.name for f in fields(MigrationPayload)}
    insight_fields = {f.name for f in fields(MigrateInsight)}

    assert ('edges' not in payload_fields)
    assert ('session_id' not in insight_fields)


def test_the_wizard_prints_no_surface_note(monkeypatch, tmp_path, capsys):
    """Verify the interactive wizard names no per-store surface key.

    Mutation: keeping the surface note, which points at a key `config
        set` rejects and a doctor check that does not exist.
    Oracle: the wizard's captured output.
    """
    from memman.setup import wizard
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)

    wizard.run_wizard(str(tmp_path / 'memman'), backend='sqlite')

    assert 'MEMMAN_SURFACE' not in capsys.readouterr().out


def test_a_maintenance_pass_opens_no_untouched_store(mm_runner, monkeypatch):
    """Verify maintenance opens only the stores the drain touched.

    Mutation: keeping the all-stores pass, which opens every store on
        disk at every drain.
    Oracle: a spy on `open_backend` across a pass with no touched
        store, over two stores on disk.
    """
    from memman.maintenance import run_maintenance
    from memman.store import factory
    _, data_dir = mm_runner
    _remember(mm_runner, 'first store row')
    assert invoke(mm_runner, [
        '--store', 'quiet', 'remember', 'quiet store row']).exit_code == 0
    opened = []
    real_open = factory.open_backend

    def spying_open(store, *args, **kwargs):
        opened.append(store)
        return real_open(store, *args, **kwargs)

    monkeypatch.setattr(factory, 'open_backend', spying_open)
    conn = open_queue_db(data_dir)
    try:
        run_maintenance(
            queue_conn=conn, touched_stores=set(), store_contexts={},
            deadline_monotonic=time.monotonic() + 60)
    finally:
        conn.close()

    assert opened == []


@pytest.mark.scheduler_stopped
def test_a_rebuild_whose_enrichment_fails_still_terminates(
        tmp_path, monkeypatch):
    """Verify a failing enrichment still ends a rebuild with nothing pending.

    Mutation: selecting pending rows on `enriched_at is null` instead
        of `linked_at is null`, so a row whose enrichment raises stays
        selectable and the rebuild loop never exits.
    Oracle: the rebuild's own `remaining`, and the row's two stamps:
        `linked_at` set by the attempt, `enriched_at` left null.
    """
    from memman.graph import enrichment
    from memman.store.db import open_db
    from memman.store.node import insert_insight
    monkeypatch.delenv('MEMMAN_STORE', raising=False)
    store_path = tmp_path / 'data' / 'default'
    db = open_db(str(store_path))
    insert_insight(db, make_insight(id='fail-1', content='a row to rebuild'))
    db.close()

    def failing_enrich(insight, client):
        raise RuntimeError('enrichment down')

    monkeypatch.setattr(enrichment, 'enrich_with_llm', failing_enrich)

    result = CliRunner().invoke(cli, [
        '--data-dir', str(tmp_path), 'graph', 'rebuild'])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['remaining'] == 0
    db = open_db(str(store_path))
    linked, enriched = db._conn.execute(
        "select linked_at, enriched_at from insights where id = 'fail-1'"
        ).fetchone()
    db.close()
    assert linked is not None
    assert enriched is None
