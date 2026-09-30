"""A stored insight carries no category, entities, keywords, importance
or source.

Recall filters by neither category nor source, enrichment returns a
summary alone, and no ranking breaks a tie on importance. No command
takes `--cat`.
"""

import inspect
import json
from dataclasses import fields

import pytest
from memman.doctor import check_enrichment_coverage
from memman.migrate import MigrateInsight
from memman.pipeline.enrich import ENRICHMENT_SYSTEM_PROMPT, enrich_with_llm
from memman.queue import QueueRow, open_queue_db
from memman.search.keyword import keyword_search
from memman.search.recall import run_recall
from memman.store.backend import NodeStore, RecallSession
from memman.store.db import open_db
from memman.store.model import Insight
from memman.store.postgres import PostgresNodeStore, PostgresRecallSession
from memman.store.sqlite import SqliteNodeStore, SqliteRecallSession
from tests.conftest import invoke, make_insight, parse_remember

DROPPED_COLUMNS = {'category', 'importance', 'entities', 'source', 'keywords'}
DROPPED_HINTS = {'hint_imp', 'hint_source', 'hint_entities'}


def _insight_columns(backend):
    """Every column name of the insights table, on either backend.
    """
    if hasattr(backend, '_db'):
        rows = backend._db._query('pragma table_info(insights)').fetchall()
        return {r[1] for r in rows}
    with backend._conn.cursor() as cur:
        cur.execute(
            'select column_name from information_schema.columns'
            " where table_schema = %s and table_name = 'insights'",
            (backend._schema,))
        return {r[0] for r in cur.fetchall()}


def _index_definitions(backend):
    """Map each insights index name to its definition, on either backend.
    """
    if hasattr(backend, '_db'):
        rows = backend._db._query(
            "select name, sql from sqlite_master where type = 'index'"
            " and tbl_name = 'insights' and sql is not null").fetchall()
        return {r[0]: r[1] for r in rows}
    with backend._conn.cursor() as cur:
        cur.execute(
            'select indexname, indexdef from pg_indexes'
            " where schemaname = %s and tablename = 'insights'",
            (backend._schema,))
        return {r[0]: r[1] for r in cur.fetchall()}


def test_a_fresh_store_has_no_group3_columns(backend):
    """Verify the insights table drops every column in `DROPPED_COLUMNS`.

    Mutation: leaving any of the five columns in either baseline
        schema, so a fresh store keeps a column nothing reads.
    Oracle: the catalog's column list, beside `content`, which proves
        the read sees the table.
    """
    columns = _insight_columns(backend)

    assert 'content' in columns
    assert columns.isdisjoint(DROPPED_COLUMNS)


def test_no_index_names_a_group3_column(backend):
    """Verify no insights index names category, importance or source.

    Mutation: leaving `idx_insights_category`,
        `idx_insights_importance` or `idx_insights_source` in a
        baseline, which makes the release
        refuse every store the hand DDL has run on; or keeping
        `importance` in the `--basic` listing index.
    Oracle: the catalog's index definitions, beside the listing index,
        which proves the read sees the store's indexes.
    """
    definitions = _index_definitions(backend)
    listing = next(d for n, d in definitions.items()
                   if n.startswith('idx_insights_current_listing'))

    assert 'created_at' in listing
    assert not any(column in d
                   for d in definitions.values()
                   for column in ('category', 'importance', 'source'))


def test_the_sqlite_keyword_index_holds_content_alone(tmp_path):
    """Verify `insights_fts` indexes content and no trigger names entities.

    Mutation: keeping the `entities` column in `_FTS_STATEMENTS`, so
        the keyword channel still matches a name that only the caller
        typed.
    Oracle: `pragma table_info(insights_fts)` and the trigger text in
        `sqlite_master` on a fresh store.
    """
    db = open_db(str(tmp_path / 'fts'))
    try:
        columns = [r[1] for r in db._conn.execute(
            'pragma table_info(insights_fts)')]
        triggers = [r[0] for r in db._conn.execute(
            "select sql from sqlite_master where type = 'trigger'"
            " and name like 'insights_fts_%'")]
    finally:
        db.close()

    assert columns == ['content']
    assert len(triggers) == 3
    assert not any('entities' in t for t in triggers)


def test_a_fresh_queue_has_no_dropped_hints(tmp_path):
    """Verify the queue carries `replaced_id` and no category or hints.

    Mutation: leaving `category`, `hint_imp`, `hint_source` or
        `hint_entities` in the queue DDL, or keeping `hint_cat` or
        `hint_replaced_id`.
    Oracle: `pragma table_info(queue)` on a fresh queue.db.
    """
    conn = open_queue_db(str(tmp_path))
    try:
        columns = {r[1] for r in conn.execute('pragma table_info(queue)')}
    finally:
        conn.close()

    assert 'replaced_id' in columns
    assert columns.isdisjoint(
        DROPPED_HINTS | {'category', 'hint_cat', 'hint_replaced_id'})


@pytest.mark.parametrize('args', [
    ['remember', '--cat', 'decision', 'a row'],
    ['remember', '--imp', '3', 'a row'],
    ['remember', '--source', 'agent', 'a row'],
    ['remember', '--entity', 'Kombu', 'a row'],
    ['replace', 'deadbeef', '--cat', 'decision', 'a row'],
    ['replace', 'deadbeef', '--imp', '3', 'a row'],
    ['replace', 'deadbeef', '--source', 'agent', 'a row'],
    ['replace', 'deadbeef', '--entity', 'Kombu', 'a row'],
    ['recall', '--cat', 'fact', 'a query'],
    ['recall', '--basic', '--cat', 'fact', 'a query'],
    ['recall', '--source', 'agent', 'a query'],
    ])
def test_no_command_takes_a_group3_flag(mm_runner, args):
    """Verify the write verbs and recall reject every deleted flag.

    Mutation: keeping any of `--cat`, `--imp`, `--source` or
        `--entity` on `remember` or `replace`, or `--cat` or
        `--source` on `recall`.
    Oracle: Click's usage error, exit status 2.
    """
    res = invoke(mm_runner, args)

    assert res.exit_code == 2
    assert 'No such option' in res.output


def test_the_enrichment_prompt_asks_for_a_summary_alone():
    """Verify the enrichment call asks for and returns a summary alone.

    Mutation: keeping the keyword field in `ENRICHMENT_SYSTEM_PROMPT`,
        or passing a returned `keywords` list through, which bills
        output tokens nothing stores.
    Oracle: the prompt text, and the result keys on a reply that
        carries both fields.
    """

    class _Client:
        def complete(self, system, user, **kwargs):
            return json.dumps({
                'keywords': ['kombu'],
                'summary': 'A broker choice.',
                })

    ins = make_insight(
        id='enrich-1',
        content='The team picked RabbitMQ as the broker for the worker pool.')

    result = enrich_with_llm(ins, _Client())

    assert 'keyword' not in ENRICHMENT_SYSTEM_PROMPT.lower()
    assert result == {'summary': 'A broker choice.'}


def test_enrichment_coverage_grades_embedding_and_summary(backend):
    """Verify doctor's coverage check reads no keywords count.

    Mutation: keeping `missing_keywords` in the grade, which reads a
        column the store no longer has.
    Oracle: the detail keys on a one-row store, beside
        `missing_summary`.
    """
    backend.nodes.insert(make_insight(id='cov-1', content='coverage row'))

    detail = check_enrichment_coverage(backend)['detail']

    assert 'missing_summary' in detail
    assert 'missing_keywords' not in detail


def test_status_reports_no_top_entities(mm_runner):
    """Verify `status` counts no categories and lists no entities.

    Mutation: keeping `by_category` or `top_entities` on `NodeStats`
        and in the output.
    Oracle: the output keys, beside `total_insights`.
    """
    invoke(mm_runner, ['remember', 'a status row'])

    out = json.loads(invoke(mm_runner, ['status']).output)

    assert 'total_insights' in out
    assert 'by_category' not in out
    assert 'top_entities' not in out


def test_insights_show_prints_no_group3_field(mm_runner):
    """Verify `insights show` prints none of the dropped columns.

    Mutation: keeping `category`, `importance`, `entities` or
        `source` in `insight_to_full_dict`.
    Oracle: the printed keys, beside `content`.
    """
    res = invoke(mm_runner, ['remember', 'a shown row'])
    row_id = parse_remember(res, mm_runner)['id']

    shown = json.loads(invoke(mm_runner, ['insights', 'show', row_id]).output)

    assert 'content' in shown
    assert set(shown).isdisjoint(DROPPED_COLUMNS)


def test_insights_review_reports_no_importance(mm_runner):
    """Verify `insights review` flags a row and prints no importance.

    Mutation: keeping the `importance` key in `review_results`, which
        reads a field the insight no longer has.
    Oracle: the flagged row's keys, on content that carries an AWS
        instance id.
    """
    invoke(mm_runner, [
        'remember',
        ('An outage traced to instance i-0c220c2402a5245bc running out of'
         ' memory')])

    data = json.loads(invoke(mm_runner, ['insights', 'review']).output)

    assert data['total_flagged'] == 1
    assert set(data['review_results'][0]) == {
        'id', 'content', 'quality_warnings'}


def test_keyword_search_ties_ignore_importance():
    """Verify a score tie in the keyword channel does not read importance.

    Mutation: keeping importance in the heap entry or the eviction
        clause, which puts the planted high-importance row first.
    Oracle: the order of two equal-score rows, compared with the
        planted attribute swapped between them.
    """
    orders = []
    for high in ('tie-a', 'tie-b'):
        rows = [Insight(id='tie-a', content='alpha'),
                Insight(id='tie-b', content='alpha')]
        for row in rows:
            row.importance = 5 if row.id == high else 1
        hits = keyword_search(
            rows, 'alpha', limit=10, counts={'tie-a': 1, 'tie-b': 1})
        orders.append([ins.id for ins, _ in hits])

    assert orders[0] == orders[1]


@pytest.mark.parametrize('target', [
    run_recall,
    NodeStore.query,
    SqliteNodeStore.query,
    PostgresNodeStore.query,
    RecallSession.vector_anchors,
    SqliteRecallSession.vector_anchors,
    PostgresRecallSession.vector_anchors,
    ])
def test_recall_takes_no_category_or_source_filter(target):
    """Verify no recall verb keeps a category or source parameter.

    Mutation: keeping `category=` or `source=` on a verb whose only
        caller was the deleted recall flag, a dead parameter.
    Oracle: the signature's parameter names.
    """
    params = set(inspect.signature(target).parameters)

    assert params.isdisjoint({'category', 'source'})


def test_no_record_type_carries_a_group3_field():
    """Verify the insight, migration and queue rows drop every column.

    Mutation: leaving a field on `Insight`, `MigrateInsight` or
        `QueueRow`.
    Oracle: the dataclass fields.
    """
    insight_fields = {f.name for f in fields(Insight)}
    migrate_fields = {f.name for f in fields(MigrateInsight)}
    queue_fields = {f.name for f in fields(QueueRow)}

    assert 'content' in insight_fields
    assert insight_fields.isdisjoint(DROPPED_COLUMNS)
    assert migrate_fields.isdisjoint(DROPPED_COLUMNS)
    assert queue_fields.isdisjoint(DROPPED_HINTS | {'category'})


def test_the_node_store_offers_no_entity_verb():
    """Verify no backend keeps a verb that writes the entity list.

    Mutation: keeping `update_entities` on the Protocol or either
        backend after the column is gone.
    Oracle: attribute lookup on the Protocol and both classes.
    """
    for cls in (NodeStore, SqliteNodeStore, PostgresNodeStore):
        assert not hasattr(cls, 'update_entities')
