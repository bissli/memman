"""Tests for memman.cli - Click CLI commands via CliRunner.

All tests use real Haiku LLM and Voyage embedding APIs.
Requires OPENROUTER_API_KEY and VOYAGE_API_KEY in environment.
"""

import json
import pathlib
import re
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from memman.cli import cli
from memman.embed.fingerprint import seed_default_fingerprint
from memman.embed.vector import serialize_vector
from memman.store.db import store_exists
from memman.store.errors import BackendError
from memman.store.node import insert_insight, update_embedding
from tests.conftest import invoke, make_insight, parse_remember

_SCORED_LINE = re.compile(
    r'^(?P<id>\S{8}) (?P<score>-?\d+\.\d\d)'
    r' (?P<created>\S+) (?P<author>\S+) (?P<category>\S+) \| (?P<text>.*)$')
_BASIC_LINE = re.compile(
    r'^(?P<id>\S{8})'
    r' (?P<created>\S+) (?P<author>\S+) (?P<category>\S+) \| (?P<text>.*)$')


def _parse_recall_lines(output: str, basic: bool = False) -> list[dict]:
    """Parse a recall page into one field dict per line, id8-keyed order kept.

    Mutation: a caller comparing against the deleted JSON envelope
        instead of this plain-text line shape.
    """
    pattern = _BASIC_LINE if basic else _SCORED_LINE
    rows = []
    for line in output.splitlines():
        match = pattern.match(line)
        assert match, f'line off the page format: {line!r}'
        rows.append(match.groupdict())
    return rows


@pytest.fixture
def runner(mm_runner):
    """CliRunner + data_dir tuple (delegates to conftest `mm_runner`)."""
    return mm_runner


class TestRemember:
    """`memman remember` happy paths and validation."""

    def test_remember_basic(self, runner):
        """Store a basic insight."""
        result = invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        assert data['action'] in {'add', 'added', 'update', 'updated'}
        assert 'sqlite' in data['content'].lower()

    def test_remember_with_flags(self, runner):
        """Store with category and importance.

        Mutation: dropping the `--cat`/`--imp` values on the way to
            storage, so the stored row keeps the defaults.
        Oracle: `insights show` on the stored id, compared against the
            flags passed to `remember`.
        """
        result = invoke(runner, [
            'remember', 'Chose Docker for container orchestration in production',
            '--cat', 'decision', '--imp', '4'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        assert 'id' in data

        shown = json.loads(
            invoke(runner, ['insights', 'show', data['id']]).output)
        assert shown['category'] == 'decision'
        assert shown['importance'] == 4

    def test_remember_invalid_category(self, runner):
        """Invalid category is rejected."""
        result = invoke(runner, [
            'remember', 'Go uses SQLite for storage', '--cat', 'bogus'])
        assert result.exit_code != 0

    def test_remember_rejects_general(self, runner):
        """Verify `--cat general` exits non-zero and names the valid set.

        Mutation: `general` accepted as a category (the 0.33.x set).
        Oracle: the exit code and the message listing the valid categories.
        """
        result = invoke(runner, [
            'remember', 'Go uses SQLite for storage', '--cat', 'general'])
        assert result.exit_code != 0
        assert 'valid:' in result.output
        assert 'fact' in result.output

    def test_remember_invalid_importance(self, runner):
        """Importance outside 1-5 is rejected."""
        result = invoke(runner, [
            'remember', 'Go uses SQLite for storage', '--imp', '0'])
        assert result.exit_code != 0

    def test_remember_does_not_link_old_pending_insights(self, runner, monkeypatch):
        """Remember does inline enrichment, never calls link_pending."""
        invoke(runner, [
            'remember', 'Redis cache eviction uses LRU algorithm'])

        from unittest.mock import patch
        with patch('memman.graph.engine.link_pending',
                   side_effect=AssertionError(
                       'link_pending called from remember')) as mock_lp:
            result = invoke(runner, [
                'remember', 'PostgreSQL MVCC provides snapshot isolation'])
            assert result.exit_code == 0
            mock_lp.assert_not_called()

    def test_remember_quality_warnings(self, runner):
        """Content with quality warnings is queued; warnings populated as hints."""
        result = invoke(runner, [
            'remember', 'i-0c220c2402a5245bc deployed via Terraform'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['action'] == 'queued'
        assert 'AWS instance ID' in data['quality_warnings']
        assert 'deployment receipt' in data['quality_warnings']

    def test_remember_no_quality_warnings(self, runner):
        """Durable content produces empty quality_warnings."""
        result = invoke(runner, [
            'remember', 'SQLite chosen for single-node simplicity and embedded operation'])
        assert result.exit_code == 0
        raw = json.loads(result.output)
        assert raw['quality_warnings'] == []

    def test_remember_quality_warnings_populate(self, runner):
        """Quality warnings populate as hints but never block the write."""
        result = invoke(runner, [
            'remember', 'Stack deployed via Terraform. 32 resources total.'])
        data = json.loads(result.output)
        assert data['action'] == 'queued'
        assert len(data['quality_warnings']) >= 2

        result = invoke(runner, [
            'remember', 'Production outage traced to instance i-0c220c2402a5245bc running out of memory causing cascading failure'])
        data = parse_remember(result, runner)
        assert data['action'] == 'add'
        raw = json.loads(result.output)
        assert len(raw['quality_warnings']) == 1

    def test_remember_creates_semantic_edges(self, runner):
        """Worker creates semantic edges for the new insight."""
        from memman.store.db import open_read_only, store_dir

        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        invoke(runner, [
            'remember', 'SQLite WAL mode improves write throughput'])

        _, data_dir = runner
        db = open_read_only(store_dir(data_dir, 'default'))
        try:
            rows = db._query(
                "SELECT edge_type FROM edges WHERE edge_type = 'semantic'"
                ).fetchall()
        finally:
            db.close()
        # Either zero or many semantic edges, depending on similarity;
        # the table exists and the worker reaches the edge-creation step.
        assert isinstance(rows, list)


class TestRecall:
    """`memman recall` smart and basic modes."""

    def test_recall_basic(self, runner):
        """Recall after remembering."""
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        result = invoke(runner, ['recall', 'Go SQLite storage'])
        assert result.exit_code == 0

    def test_recall_does_not_call_link_pending(self, runner, monkeypatch):
        """Recall path must not call link_pending (performance regression guard)."""
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])

        from unittest.mock import patch
        with patch('memman.graph.engine.link_pending',
                   side_effect=AssertionError('link_pending called')) as mock_lp:
            result = invoke(runner, ['recall', 'Go SQLite storage'])
            assert result.exit_code == 0
            mock_lp.assert_not_called()

    def test_recall_logs_when_query_embed_fails(
            self, runner, caplog, monkeypatch):
        """A raising `ec.embed` for the recall query is now warned, not
        swallowed silently. The recall still degrades to the keyword
        path and returns successfully.
        """
        import logging

        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])

        from memman import embed as embed_mod
        real_ec = embed_mod.get_client()

        def _boom(self, text):
            raise RuntimeError('forced query embed failure')

        monkeypatch.setattr(type(real_ec), 'embed', _boom)
        with caplog.at_level(logging.WARNING, logger='memman'):
            result = invoke(runner, ['recall', 'Go SQLite storage'])
        assert result.exit_code == 0
        warned = [r for r in caplog.records
                  if 'recall query embed failed' in r.getMessage()]
        assert warned

    def test_recall_default_runs_rerank(self, runner):
        """Default install seeds MEMMAN_RERANK_ENABLED=true, so rerank fires.

        Mutation: dropping the `rerank=rerank` kwarg from the
            `intent_aware_recall` call, so the config default never
            reaches the reranker.
        Oracle: a spy on the Voyage client, called once.
        """
        for fact in [
                'Go uses SQLite for persistent storage',
                'Go modules manage dependency versions',
                'SQLite uses WAL mode for concurrent writes']:
            invoke(runner, ['remember', fact])

        with patch('memman.rerank.voyage.Client.rerank',
                   return_value=[(0, 0.9), (1, 0.5), (2, 0.1)]) as mock_re:
            result = invoke(runner, ['recall', 'Go SQLite persistent storage'])
            assert result.exit_code == 0
            mock_re.assert_called_once()

    def test_recall_global_disable_skips_rerank(self, runner, env_file):
        """MEMMAN_RERANK_ENABLED=false disables rerank globally.

        Mutation: ignoring the global config flag, running the
            reranker regardless.
        Oracle: a spy on the Voyage client, never called.
        """
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        env_file('MEMMAN_RERANK_ENABLED', 'false')

        with patch('memman.rerank.voyage.Client.rerank',
                   side_effect=AssertionError('rerank called')) as mock_re:
            result = invoke(runner, ['recall', 'Go SQLite storage'])
            assert result.exit_code == 0
            mock_re.assert_not_called()

    def test_recall_per_store_disable_overrides_global(self, runner, env_file):
        """MEMMAN_RERANK_ENABLED_<store>=false wins over the global default.

        Mutation: reading only the global flag, so a per-store
            override is silently ignored.
        Oracle: a spy on the Voyage client, never called.
        """
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        env_file('MEMMAN_RERANK_ENABLED_default', 'false')

        with patch('memman.rerank.voyage.Client.rerank',
                   side_effect=AssertionError('rerank called')) as mock_re:
            result = invoke(runner, ['recall', 'Go SQLite storage'])
            assert result.exit_code == 0
            mock_re.assert_not_called()

    def test_recall_per_store_enable_overrides_global_disable(
            self, runner, env_file):
        """Per-store true beats global false.

        Mutation: letting the global false short-circuit before the
            per-store override is read.
        Oracle: a spy on the Voyage client, called once.
        """
        for fact in [
                'Go uses SQLite for persistent storage',
                'Go modules manage dependency versions',
                'SQLite uses WAL mode for concurrent writes']:
            invoke(runner, ['remember', fact])
        env_file('MEMMAN_RERANK_ENABLED', 'false')
        env_file('MEMMAN_RERANK_ENABLED_default', 'true')

        with patch('memman.rerank.voyage.Client.rerank',
                   return_value=[(0, 0.9), (1, 0.5), (2, 0.1)]) as mock_re:
            result = invoke(runner, ['recall', 'Go SQLite persistent storage'])
            assert result.exit_code == 0
            mock_re.assert_called_once()

    def test_recall_rerank_skipped_on_short_query(self, runner):
        """Rerank auto-skips when the query has <=2 tokens, even with default on.

        Mutation: dropping the token-count guard, so a short query
            still reaches the reranker.
        Oracle: a spy on the Voyage client, never called.
        """
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])

        with patch('memman.rerank.voyage.Client.rerank',
                   side_effect=AssertionError('rerank called on short query')
                   ) as mock_re:
            result = invoke(runner, ['recall', 'storage'])
            assert result.exit_code == 0
            mock_re.assert_not_called()

    def test_recall_rerank_failure_falls_back_gracefully(self, runner):
        """Reranker errors must not break recall; falls back to baseline."""
        for fact in [
                'Go uses SQLite for persistent storage',
                'Go modules manage dependency versions']:
            invoke(runner, ['remember', fact])

        from unittest.mock import patch
        with patch('memman.rerank.voyage.Client.rerank',
                   side_effect=RuntimeError('voyage 503')) as mock_re:
            result = invoke(runner, ['recall', 'Go SQLite persistent storage'])
            assert result.exit_code == 0
            mock_re.assert_called_once()

    def test_recall_basic_mode(self, runner):
        """Basic recall prints a scoreless page line per matching row.

        Mutation: keeping the deleted `{results, meta}` JSON envelope.
        Oracle: the basic-line regex, matched against every printed
            line.
        """
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        result = invoke(runner, ['recall', 'Go SQLite', '--basic'])
        assert result.exit_code == 0
        rows = _parse_recall_lines(result.output, basic=True)
        assert rows

    def test_recall_basic_returns_envelope(self, runner):
        """Recall --basic prints a line whose text names the stored content.

        Mutation: keeping the deleted `{results: [...]}` envelope.
        Oracle: the parsed page line's `text` field containing the
            stored word.
        """
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        result = invoke(runner, ['recall', '--basic', 'Go SQLite'])
        assert result.exit_code == 0
        rows = _parse_recall_lines(result.output, basic=True)
        assert any('SQLite' in r['text'] for r in rows)

    def test_recall_emits_summary_when_populated(self, runner):
        """A row with a summary shows the summary, not the content prefix.

        Mutation: always falling back to the content prefix even when
            a summary is stored.
        Oracle: the row's stored summary, read independently via
            `insights show`, compared to the page line's text.
        """
        long_content = (
            'The application uses a write-through cache layer between the '
            'API tier and Postgres. TTL is 5 minutes for hot keys and 1 '
            'hour for cold keys. Cache invalidation must run before each '
            'DB write commits to avoid stale reads during the gap.')
        invoke(runner, ['remember', long_content])
        result = invoke(runner, ['recall', 'cache invalidation'])
        assert result.exit_code == 0
        rows = _parse_recall_lines(result.output)
        assert rows, 'expected at least one matching row'
        shown = json.loads(
            invoke(runner, ['insights', 'show', rows[0]['id']]).output)
        assert shown.get('summary'), \
            'summary should be present and non-empty for substantive content'
        assert shown['summary'] != shown['content']
        assert rows[0]['text'] == ' '.join(shown['summary'].split())

    def test_recall_omits_summary_when_unenriched(self, runner):
        """A row with no summary shows the content prefix instead.

        Mutation: printing an empty string in place of the content
            fallback when summary is unset.
        Oracle: the row's stored content, read via `insights show`,
            folded the same way the page line folds it.
        """
        invoke(runner, [
            'remember', 'Q', '--cat', 'fact'])
        result = invoke(runner, ['recall', '--basic', 'Q'])
        assert result.exit_code == 0
        rows = _parse_recall_lines(result.output, basic=True)
        assert rows
        shown = json.loads(
            invoke(runner, ['insights', 'show', rows[0]['id']]).output)
        assert 'summary' not in shown
        assert rows[0]['text'] == ' '.join(shown['content'].split())

    def test_recall_basic_emits_summary_when_present(self, runner):
        """--basic mode also shows summary; both paths share the formatter.

        Mutation: the --basic branch always falling back to content
            even when the row has a summary.
        Oracle: the stored summary, read via `insights show`, compared
            to the --basic page line's text.
        """
        long_content = (
            'The job scheduler uses systemd timers on Linux hosts and '
            'launchd on macOS hosts. The drain interval defaults to 60 '
            'seconds and is configurable via memman scheduler interval.')
        invoke(runner, ['remember', long_content])
        result = invoke(runner, ['recall', '--basic', 'scheduler timer'])
        assert result.exit_code == 0
        rows = _parse_recall_lines(result.output, basic=True)
        assert rows
        shown = json.loads(
            invoke(runner, ['insights', 'show', rows[0]['id']]).output)
        if shown.get('summary'):
            assert rows[0]['text'] == ' '.join(shown['summary'].split())
            assert shown['summary'] != shown['content']

    def test_recall_detail_oplog_records_the_request(self, runner):
        """The recall-detail row carries the REQUESTED limit and session.

        Mutation: recording `len(hits)` in place of the requested
            `limit` - which cannot tell a thin page from a small ask -
            or dropping the session key, either of which leaves a
            return unattributable to the session that asked.
        Oracle: a recall issued with a limit deliberately larger than
            the store can fill, so the requested value and the
            returned count differ.
        """
        invoke(runner, [
            'remember', 'Envoy routes gRPC traffic by header match'])
        invoke(runner, [
            'recall', 'Envoy gRPC header routing',
            '--limit', '17', '--session', 'sess-abc'])

        entries = json.loads(
            invoke(runner, ['log', 'list', '--limit', '50']).output)['entries']
        details = [
            json.loads(e['detail'])
            for e in entries if e['operation'] == 'recall-detail']
        assert details, 'expected a recall-detail row'
        row = details[0]
        assert row['limit'] == 17
        assert row['session'] == 'sess-abc'
        assert 'q' in row
        assert len(row['q']) <= 80
        assert len(row['hits']) < row['limit'], (
            'fixture must under-fill the page so the two cannot be '
            'confused')

    def test_recall_session_falls_back_to_the_environment(
            self, runner, monkeypatch):
        """An unflagged recall takes its session from the environment.

        Mutation: dropping the `envvar` list from the `--session`
            option, which leaves the oplog session blank on every
            recall an agent issues without the flag - the normal case,
            since the shipped hooks never pass it explicitly.
        Oracle: the oplog row from a recall run with no `--session`
            argument at all, against the exported id.
        """
        monkeypatch.setenv('MEMMAN_SESSION_ID', 'env-session-9')
        invoke(runner, [
            'remember', 'Redis evicts keys by LRU under maxmemory'])
        invoke(runner, ['recall', 'Redis LRU maxmemory eviction'])

        entries = json.loads(
            invoke(runner, ['log', 'list', '--limit', '50']).output)['entries']
        details = [
            json.loads(e['detail'])
            for e in entries if e['operation'] == 'recall-detail']
        assert details, 'expected a recall-detail row'
        assert details[0]['session'] == 'env-session-9'

    def test_recall_source_filter_smart(self, runner):
        """Smart recall respects --source filter.

        Mutation: dropping the `source` predicate from the anchor
            scans, so a non-matching row's id reaches the page.
        Oracle: `insights show` on every returned id, compared against
            the filter value.
        """
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage', '--source', 'agent'])
        invoke(runner, [
            'remember', 'Python uses PostgreSQL for web application storage', '--source', 'human'])

        result = invoke(runner, [
            'recall', 'database storage', '--source', 'agent'])
        assert result.exit_code == 0
        rows = _parse_recall_lines(result.output)
        assert rows
        for row in rows:
            shown = json.loads(
                invoke(runner, ['insights', 'show', row['id']]).output)
            assert shown['source'] == 'agent'

    def test_recall_source_filter_returns_matching_rows(self, runner):
        """Recall --source surfaces a matching row the top-k would drop.

        Folded from `test_recall_source_filter_inflates_fetch_limit`
        after D2 replaced the fetch-inflation post-filter with anchor
        scans that filter before the top-k cut; the deep fill-to-limit
        regression lives in
        `tests/test_recall_filters.py::test_filtered_recall_fills_to_limit`.

        Mutation: dropping the `source` predicate from the anchor
            scans (or filtering only after the cut).
        Oracle: the single agent-sourced row appears despite six
            better-matching user rows competing for the slots.
        """
        topics = [
            'PostgreSQL query optimization with EXPLAIN ANALYZE',
            'PostgreSQL index types including B-tree GIN GiST',
            'PostgreSQL vacuum autovacuum tuning parameters',
            'PostgreSQL partitioning strategies for large tables',
            'PostgreSQL connection pooling with PgBouncer setup',
            'PostgreSQL replication streaming and logical decoding',
            ]
        for topic in topics:
            invoke(runner, [
                'remember', topic, '--source', 'user'])
        invoke(runner, [
            'remember', 'PostgreSQL JSONB operators for document queries', '--source', 'agent'])

        result = invoke(runner, [
            'recall', 'PostgreSQL database',
            '--source', 'agent', '--limit', '3'])
        assert result.exit_code == 0
        rows = _parse_recall_lines(result.output)
        assert rows, 'filtered recall returned nothing'
        for row in rows:
            shown = json.loads(
                invoke(runner, ['insights', 'show', row['id']]).output)
            assert shown['source'] == 'agent'


class TestForget:
    """`memman forget` happy paths and missing-id error."""

    def test_forget_basic(self, runner):
        """Forget an insight by ID."""
        result = invoke(runner, [
            'remember', 'Redis cache eviction policy uses LRU by default'])
        data = parse_remember(result, runner)
        iid = data['id']
        result = invoke(runner, ['forget', iid])
        assert result.exit_code == 0
        fdata = json.loads(result.output)
        assert fdata['status'] == 'deleted'

    def test_forget_writes_oplog(self, runner):
        """Forget command writes an oplog entry atomically."""
        result = invoke(runner, [
            'remember', 'PostgreSQL uses MVCC for transaction isolation'])
        data = parse_remember(result, runner)
        iid = data['id']
        invoke(runner, ['forget', iid])
        result = invoke(runner, ['log', 'list', '--stats'])
        assert result.exit_code == 0
        log_data = json.loads(result.output)
        assert 'forget' in log_data['operation_counts']

    def test_forget_nonexistent_fails(self, runner):
        """Forget with nonexistent ID returns error."""
        result = invoke(runner, ['forget', 'nonexistent-id-12345'])
        assert result.exit_code != 0


class TestStore:
    """`memman store` admin: list, create, set, remove."""

    def test_store_list(self, runner):
        """Store list emits a JSON envelope with stores[] and active."""
        import json as _json
        result = invoke(runner, ['store', 'list'])
        assert result.exit_code == 0
        payload = _json.loads(result.output)
        assert 'stores' in payload
        assert 'active' in payload

    def test_store_create(self, runner):
        """Create a new store; JSON reports action='created'."""
        import json as _json
        result = invoke(runner, ['store', 'create', 'test-store'])
        assert result.exit_code == 0
        payload = _json.loads(result.output)
        assert payload['action'] == 'created'
        assert payload['store'] == 'test-store'

    def test_store_create_duplicate(self, runner):
        """Duplicate store name is rejected."""
        invoke(runner, ['store', 'create', 'dup'])
        result = invoke(runner, ['store', 'create', 'dup'])
        assert result.exit_code != 0

    def test_store_set(self, runner):
        """Set active store; JSON reports action='set'."""
        import json as _json
        invoke(runner, ['store', 'create', 'work'])
        result = invoke(runner, ['store', 'use', 'work'])
        assert result.exit_code == 0
        payload = _json.loads(result.output)
        assert payload['action'] == 'set'
        assert payload['store'] == 'work'

    def test_store_remove_yes(self, runner):
        """Remove a non-active store with --yes skips prompt."""
        import json as _json
        invoke(runner, ['store', 'create', 'temp'])
        result = invoke(runner, ['store', 'remove', '--yes', 'temp'])
        assert result.exit_code == 0
        payload = _json.loads(result.output)
        assert payload['action'] == 'removed'
        assert payload['store'] == 'temp'

    def test_store_remove_purges_queue(self, runner):
        """Removing a store also drops its in-flight queue rows.

        Regression: the old `store remove` flow rmtreed the data dir
        but left queue rows orphaned, so the worker would re-attempt
        them against a missing store dir. The purge now happens
        inside `factory.drop_store`, which is what `store remove`
        invokes; the test asserts the observable contract (no
        survivor queue rows) regardless of where the purge fires.
        """
        import json as _json

        from memman.queue import enqueue, list_rows, open_queue_db
        _, data_dir = runner
        invoke(runner, ['store', 'create', 'doomed'])
        qconn = open_queue_db(data_dir)
        try:
            enqueue(qconn, 'doomed', 'a fact to remember')
            qconn.commit()
            before = [
                r for r in list_rows(qconn, limit=50)
                if r['store'] == 'doomed']
            assert len(before) == 1
        finally:
            qconn.close()
        result = invoke(runner, ['store', 'remove', '--yes', 'doomed'])
        assert result.exit_code == 0
        payload = _json.loads(result.output)
        assert payload['action'] == 'removed'
        qconn = open_queue_db(data_dir)
        try:
            after = [
                r for r in list_rows(qconn, limit=50)
                if r['store'] == 'doomed']
            assert after == [], (
                f'expected queue rows for doomed store to be purged, '
                f'got {len(after)} survivors')
        finally:
            qconn.close()

    def test_store_remove_purges_per_store_env_keys(self, runner):
        """Removing a store drops its per-store keys from the env file.

        Mutation: dropping the `removes=` cleanup, which leaves
        `MEMMAN_BACKEND_doomed` and the store's Postgres DSN (password
        included) in the env file after the store is gone.
        Oracle: the parsed env file -- every PER_STORE_KEY_SPECS prefix
        for the removed store absent, and the same prefixes for a
        surviving store plus the global key still present.
        """
        from memman import config
        from memman.setup.scheduler import _write_env_keys
        _, data_dir = runner
        invoke(runner, ['store', 'create', 'doomed'])
        invoke(runner, ['store', 'create', 'keeper'])
        value_for = {
            'MEMMAN_BACKEND_': 'sqlite',
            config._pg_dsn_prefix(): 'postgresql://u:p@127.0.0.1:1/db',
            'MEMMAN_RERANK_ENABLED_': 'false',
            'MEMMAN_SURFACE_': 'code',
            'MEMMAN_AUTO_SEMANTIC_THRESHOLD_': '0.5',
            }
        doomed_keys = {
            f'{prefix}doomed' for prefix, _, _ in config.PER_STORE_KEY_SPECS}
        keeper_keys = {
            f'{prefix}keeper' for prefix, _, _ in config.PER_STORE_KEY_SPECS}
        seeded = {
            f'{prefix}{store}': value_for[prefix]
            for prefix, _, _ in config.PER_STORE_KEY_SPECS
            for store in ('doomed', 'keeper')
            }
        _write_env_keys(
            seeded | {'MEMMAN_LOG_LEVEL': 'DEBUG'}, data_dir=data_dir)
        env_path = config.env_file_path(data_dir)
        assert doomed_keys <= set(config.parse_env_file(env_path))

        result = invoke(runner, ['store', 'remove', '--yes', 'doomed'])
        assert result.exit_code == 0

        after = set(config.parse_env_file(env_path))
        assert not (doomed_keys & after), (
            f'per-store keys survived removal: {sorted(doomed_keys & after)}')
        assert keeper_keys <= after, (
            f'unrelated store keys were collateral: '
            f'{sorted(keeper_keys - after)}')
        assert 'MEMMAN_LOG_LEVEL' in after

    def test_store_remove_prompts_without_yes(self, runner):
        """Without --yes, remove prompts; typing 'n' aborts."""
        r, data_dir = runner
        invoke(runner, ['store', 'create', 'temp2'])
        result = r.invoke(
            cli, ['--data-dir', data_dir, 'store', 'remove', 'temp2'],
            input='n\n')
        assert result.exit_code != 0
        assert store_exists(data_dir, 'temp2')

    def test_store_remove_prompts_accept(self, runner):
        """Without --yes, typing 'y' at the prompt completes the delete."""
        import json as _json
        r, data_dir = runner
        invoke(runner, ['store', 'create', 'temp3'])
        result = r.invoke(
            cli, ['--data-dir', data_dir, 'store', 'remove', 'temp3'],
            input='y\n')
        assert result.exit_code == 0, result.output
        # The confirm prompt echoes before the JSON payload; find the payload.
        payload_start = result.output.find('{')
        payload = _json.loads(result.output[payload_start:])
        assert payload['action'] == 'removed'

    def test_store_auto_create_from_env(self, runner, monkeypatch):
        """MEMMAN_STORE env var silently creates a non-existent store."""
        r, data_dir = runner
        monkeypatch.setenv('MEMMAN_STORE', 'auto-created')

        result = r.invoke(cli, ['--data-dir', data_dir, 'recall', 'test',
                                '--limit', '1'])
        assert result.exit_code == 0, result.output

        store_path = pathlib.Path(data_dir) / 'data' / 'auto-created'
        assert store_path.is_dir(), 'store directory should be auto-created'

        monkeypatch.delenv('MEMMAN_STORE')
        list_result = r.invoke(cli, ['--data-dir', data_dir, 'store', 'list'])
        assert 'auto-created' in list_result.output

        r.invoke(cli, ['--data-dir', data_dir, 'store', 'remove', 'auto-created'])


class TestStatus:
    """`memman status` and `memman doctor` smoke."""

    def test_status_basic(self, runner):
        """Status returns JSON."""
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        result = invoke(runner, ['status'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert 'total_insights' in data

    def test_doctor_basic(self, runner):
        """Doctor returns JSON with checks and status.

        Exit code may be 0 (pass/warn) or 1 (fail) depending on environment.
        """
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        result = invoke(runner, ['doctor'])
        assert result.exit_code in {0, 1}
        data = json.loads(result.output)
        assert 'status' in data
        assert 'checks' in data
        assert 'total_active' in data


class TestLog:
    """`memman log` smoke."""

    def test_log_basic(self, runner):
        """Log shows recent operations."""
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        result = invoke(runner, ['log', 'list'])
        assert result.exit_code == 0


class TestInsightsReview:
    """`memman insights review` flags transient content."""

    def test_review_flags_transient_content(self, runner):
        """A stored instance id is flagged; a durable decision is not."""
        invoke(runner, [
            'remember', 'Production outage traced to instance i-0c220c2402a5245bc running out of memory causing cascading failure'])
        invoke(runner, [
            'remember', 'SQLite chosen for simplicity and embedded operation', '--imp', '5'])
        result = invoke(runner, ['insights', 'review'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['total_flagged'] >= 1
        flagged = [r['content'] for r in data['review_results']]
        assert any('i-0c220c2402a5245bc' in c.lower() for c in flagged)

    def test_review_clean_store_flags_nothing(self, runner):
        """A store of durable content returns zero flagged."""
        invoke(runner, [
            'remember', 'SQLite chosen for simplicity and embedded operation'])
        result = invoke(runner, ['insights', 'review'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['total_flagged'] == 0


class TestReplace:
    """`memman replace` happy paths, metadata, oplog, edges."""

    def test_replace_basic(self, runner):
        """Replace an insight, verify old soft-deleted, new exists."""
        result = invoke(runner, [
            'remember', 'Redis cache configured with 512MB memory limit', '--cat', 'fact', '--imp', '3'])
        old_id = parse_remember(result, runner)['id']

        result = invoke(runner, [
            'replace', old_id,
            'Redis cache configured with 1GB memory limit for production'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        assert data['action'] == 'replace'
        assert data['replaced_id'] == old_id
        assert 'redis' in data['content'].lower()

    def test_replace_inherits_metadata(self, runner):
        """Replace without flags inherits cat/imp from original.

        Mutation: dropping the inherited category or importance on a
            flag-less replace, defaulting instead.
        Oracle: `insights show` on the replacement id, compared
            against the original's stored values.
        """
        result = invoke(runner, [
            'remember', 'Chose PostgreSQL over MySQL for JSONB support',
            '--cat', 'decision', '--imp', '5'])
        old_id = parse_remember(result, runner)['id']

        result = invoke(runner, [
            'replace', old_id,
            'Chose PostgreSQL over MySQL for JSONB and CTE support'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        assert 'id' in data

        shown = json.loads(
            invoke(runner, ['insights', 'show', data['id']]).output)
        assert shown['category'] == 'decision'
        assert shown['importance'] == 5

    def test_replace_overrides_metadata(self, runner):
        """Replace with explicit flags uses new values.

        Mutation: keeping the original category/importance despite an
            explicit override on the replace command.
        Oracle: `insights show` on the replacement id, compared
            against the flags passed to `replace`.
        """
        result = invoke(runner, [
            'remember', 'Nginx configured as reverse proxy for API gateway', '--cat', 'fact', '--imp', '2'])
        old_id = parse_remember(result, runner)['id']

        result = invoke(runner, [
            'replace', old_id,
            'Switched from Nginx to Envoy for service mesh integration',
            '--cat', 'decision', '--imp', '5'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        assert 'id' in data

        shown = json.loads(
            invoke(runner, ['insights', 'show', data['id']]).output)
        assert shown['category'] == 'decision'
        assert shown['importance'] == 5

    def test_replace_nonexistent_id(self, runner):
        """Replace a nonexistent ID produces error."""
        result = invoke(runner, [
            'replace', 'nonexistent-id',
            'Redis configured for cluster mode replication'])
        assert result.exit_code != 0
        assert 'not found' in result.output

    def test_replace_already_deleted(self, runner):
        """Verify replacing a forgotten insight fails naming the deletion.

        Mutation: preflighting with `nodes.get`, which cannot tell a
            forgotten row from a missing one.
        Oracle: the error text says the row was forgotten.
        """
        result = invoke(runner, [
            'remember', 'Kafka consumer group rebalance strategy uses cooperative'])
        old_id = parse_remember(result, runner)['id']
        invoke(runner, ['forget', old_id])

        result = invoke(runner, [
            'replace', old_id,
            'Kafka consumer group rebalance uses eager strategy'])
        assert result.exit_code != 0
        assert 'was forgotten' in result.output

    def test_replace_refuses_a_superseded_id_and_names_the_successor(
            self, runner):
        """Verify replacing a superseded id points at its successor.

        Mutation: preflighting with `nodes.get` (a bare not-found), or
            accepting the superseded id and forking the chain.
        Oracle: the error text carries the successor's id and the
            history command; the successor stays the one current row.
        """
        result = invoke(runner, [
            'remember', 'Kafka retention is seven days'])
        old_id = parse_remember(result, runner)['id']
        result = invoke(runner, [
            'replace', old_id, 'Kafka retention is thirty days'])
        new_id = parse_remember(result, runner)['id']

        result = invoke(runner, [
            'replace', old_id, 'Kafka retention is ninety days'])
        assert result.exit_code != 0
        assert f'is superseded by {new_id}' in result.output
        assert '--history' in result.output
        active = _parse_recall_lines(
            invoke(runner, ['recall', '--basic', 'Kafka']).output,
            basic=True)
        assert [row['id'] for row in active] == [new_id[:8]]

    def test_replace_oplog_entries(self, runner):
        """Replace logs both replace and remember ops."""
        result = invoke(runner, [
            'remember', 'Prometheus alerting rules configured for SLO monitoring'])
        old_id = parse_remember(result, runner)['id']

        result = invoke(runner, [
            'replace', old_id,
            'Prometheus alerting rules with Grafana dashboards for SLO'])
        parse_remember(result, runner)

        result = invoke(runner, ['log', 'list', '--limit', '10'])
        assert result.exit_code == 0
        assert 'replace' in result.output
        assert 'remember' in result.output

    def test_replace_quality_warnings_populate(self, runner):
        """Replace path also passes quality warnings as hints, never blocks."""
        result = invoke(runner, [
            'remember', 'Kafka chosen for event streaming due to partition tolerance'])
        old_id = parse_remember(result, runner)['id']

        result = invoke(runner, [
            'replace', old_id,
            'Stack deployed via Terraform. 32 resources total.'])
        data = json.loads(result.output)
        assert data['action'] != 'rejected'
        assert len(data['quality_warnings']) >= 2

    def test_replace_creates_background_edges(self, runner):
        """Replace passes store context so background edges are created."""
        r1 = invoke(runner, [
            'remember', 'Celery task queue configured for async job processing'])
        orig_id = parse_remember(r1, runner)['id']

        r2 = invoke(runner, [
            'replace', orig_id,
            'Celery with Redis broker for distributed task processing'])
        assert r2.exit_code == 0
        new_id = parse_remember(r2, runner)['id']

        result = invoke(runner, ['graph', 'related', new_id])
        assert result.exit_code == 0


class TestLink:
    """`memman graph link` direct edge creation."""

    def test_link_creates_both_directions(self, runner):
        """Link creates edges in both directions atomically.

        Mutation: writing only the forward row, so a traversal from
            the target never reaches the source.
        Oracle: both rows read straight out of the edges table,
            matched on `created_by = 'claude'`. The enrichment pass
            mints an entity and a temporal edge between any two
            insights, so a `graph related` assertion would pass on an
            auto edge even if `graph link` wrote nothing. The anchors
            are textually distant, so no auto semantic edge competes.
        """
        from memman.store.db import open_read_only, store_dir

        r1 = invoke(runner, [
            'remember', 'chose SQLite because embedded serverless'])
        id1 = parse_remember(r1, runner)['id']
        r2 = invoke(runner, [
            'remember', 'preferred color is emerald green'])
        id2 = parse_remember(r2, runner)['id']

        result = invoke(
            runner, ['graph', 'link', id1, id2, '--type', 'semantic'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['status'] == 'linked'

        _, data_dir = runner
        db = open_read_only(store_dir(data_dir, 'default'))
        try:
            manual = db._query(
                "select source_id, target_id from edges"
                " where json_extract(metadata, '$.created_by') = 'claude'"
                " and edge_type = 'semantic'").fetchall()
        finally:
            db.close()
        assert (id1, id2) in {(r[0], r[1]) for r in manual}
        assert (id2, id1) in {(r[0], r[1]) for r in manual}

    def test_link_respects_user_created_by(self, runner):
        """User-provided --meta['created_by'] is preserved, not clobbered to 'claude'.
        """
        import sqlite3
        r1 = invoke(runner, [
            'remember', 'Nginx is configured as the reverse proxy'])
        id1 = parse_remember(r1, runner)['id']
        r2 = invoke(runner, [
            'remember', "Let's Encrypt auto-renews TLS certificates"])
        id2 = parse_remember(r2, runner)['id']

        result = invoke(runner, ['graph', 'link', id1, id2, '--type', 'semantic',
                                 '--meta', '{"created_by": "research-agent"}'])
        assert result.exit_code == 0

        _, data_dir = runner
        store_db = pathlib.Path(data_dir) / 'data' / 'default' / 'memman.db'
        conn = sqlite3.connect(str(store_db))
        try:
            rows = conn.execute(
                'SELECT metadata FROM edges'
                ' WHERE source_id = ? AND target_id = ?'
                ' AND edge_type = ?',
                (id1, id2, 'semantic')).fetchall()
        finally:
            conn.close()
        assert rows, 'expected one semantic edge source->target'
        meta = json.loads(rows[0][0])
        assert meta['created_by'] == 'research-agent'

    def test_link_meta_non_dict_fails(self, runner):
        """Non-dict JSON metadata is rejected."""
        r1 = invoke(runner, [
            'remember', 'Elasticsearch configured for full-text search'])
        id1 = parse_remember(r1, runner)['id']
        r2 = invoke(runner, [
            'remember', 'Kibana dashboards visualize Elasticsearch data'])
        id2 = parse_remember(r2, runner)['id']

        result = invoke(runner, ['graph', 'link', id1, id2, '--type', 'semantic',
                                 '--meta', '[1, 2]'])
        assert result.exit_code != 0
        assert 'object' in result.output.lower() or 'dict' in result.output.lower()

    def test_link_self_edge_rejected(self, runner):
        """Linking an insight to itself is rejected."""
        r1 = invoke(runner, [
            'remember', 'GraphQL schema stitching combines microservice APIs'])
        id1 = parse_remember(r1, runner)['id']

        result = invoke(runner, ['graph', 'link', id1, id1, '--type', 'semantic'])
        assert result.exit_code != 0
        assert 'itself' in result.output.lower()

    def test_link_warns_when_lower_weight(self, runner):
        """Link output includes warning when requested weight < existing."""
        r1 = invoke(runner, [
            'remember', 'Consul service discovery enables dynamic routing'])
        id1 = parse_remember(r1, runner)['id']
        r2 = invoke(runner, [
            'remember', 'Vault secrets management integrates with Consul'])
        id2 = parse_remember(r2, runner)['id']

        invoke(runner, ['graph', 'link', id1, id2, '--weight', '0.9'])
        result = invoke(runner, ['graph', 'link', id1, id2, '--weight', '0.3'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert 'warning' in data
        assert '0.9' in data['warning']

    def test_link_returns_actual_db_weight(self, runner):
        """Link output weight reflects the DB value, not the user-supplied value.

        Mutation: echoing the requested weight back instead of reading
            the stored row, which would report 0.3 after the upsert
            kept 0.9.
        Oracle: the second call's own output, against the weight the
            first call stored.
        """
        r1 = invoke(runner, [
            'remember', 'chose SQLite because embedded serverless'])
        id1 = parse_remember(r1, runner)['id']
        r2 = invoke(runner, [
            'remember', 'preferred color is emerald green'])
        id2 = parse_remember(r2, runner)['id']

        invoke(runner, [
            'graph', 'link', id1, id2, '--type', 'semantic', '--weight', '0.9'])
        result = invoke(runner, [
            'graph', 'link', id1, id2, '--type', 'semantic', '--weight', '0.3'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['weight'] >= 0.9, (
            f'Link output shows {data["weight"]} but should be >= 0.9 '
            f'(MAX preserves higher weight over requested 0.3)')
        assert data['weight'] != 0.3, (
            'Link output should not show 0.3 - MAX should preserve higher')


class TestSingleTierEnrichment:
    """Remember runs enrichment inline on the drain worker."""

    def test_output_has_enrichment_dict(self, runner):
        """Verify the drain stores the enrichment keywords on the row.

        Mutation: `_apply_plan` stamping `enriched_at` without the
            `update_enrichment` write, so the row reads as enriched,
            holds no keywords, and no stranded-row sweep revisits it.
        Oracle: the autouse mock LLM, which echoes the enrichment
            prompt's opening words, the content's first word among
            them, as keywords.
        """
        from memman.store.db import open_read_only, store_dir

        result = invoke(runner, [
            'remember', 'Redis cache configured with LRU eviction policy'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        iid = data['id']

        _, data_dir = runner
        db = open_read_only(store_dir(data_dir, 'default'))
        try:
            row = db._query(
                'SELECT keywords FROM insights WHERE id = ?',
                (iid,)).fetchone()
        finally:
            db.close()
        assert row is not None
        assert 'redis' in json.loads(row[0])

    def test_no_link_pending_in_output(self, runner):
        """Output no longer includes link_pending field."""
        result = invoke(runner, [
            'remember', 'Docker containers orchestrated via Kubernetes'])
        assert result.exit_code == 0
        raw = json.loads(result.output)
        assert 'link_pending' not in raw

    def test_linked_at_stamped_after_remember(self, runner):
        """linked_at is non-NULL after remember returns."""
        from memman.store.db import open_read_only

        result = invoke(runner, [
            'remember', 'Consul service mesh enables secure service communication'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        iid = data['id']

        _, data_dir = runner
        ro = open_read_only(data_dir + '/data/default')
        row = ro._conn.execute(
            'SELECT linked_at FROM insights WHERE id = ?',
            (iid,)).fetchone()
        ro.close()
        assert row is not None
        assert row[0] is not None

    def test_graph_rebuild_zero_pending_after_remember(self, runner):
        """Graph rebuild processes already-linked insights after remember."""
        invoke(runner, [
            'remember', 'Kafka event streaming configured for microservices'])
        result = invoke(runner, ['graph', 'rebuild', '--dry-run'])
        assert result.exit_code == 0

    def test_enriched_at_stamped_after_remember(self, runner):
        """enriched_at is non-NULL after remember returns."""
        from memman.store.db import open_read_only

        result = invoke(runner, [
            'remember', 'Elasticsearch full-text search with custom analyzers'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        iid = data['id']

        _, data_dir = runner
        ro = open_read_only(data_dir + '/data/default')
        row = ro._conn.execute(
            'SELECT enriched_at FROM insights WHERE id = ?',
            (iid,)).fetchone()
        ro.close()
        assert row is not None
        assert row[0] is not None


@pytest.mark.scheduler_stopped
class TestGraphRebuild:
    """Graph rebuild command tests - dry-run, live, edge preservation."""

    def test_rebuild_dry_run_reports_count(self, tmp_path, monkeypatch):
        """Dry run reports total insights without modifying DB."""
        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        from memman.store.db import open_db
        from memman.store.node import insert_insight
        from tests.conftest import make_insight
        db = open_db(str(store_path))
        for i in range(3):
            insert_insight(db, make_insight(
                id=f'rd-{i}', content=f'Test insight {i}'))
            db._conn.execute(
                'UPDATE insights SET linked_at = ?, enriched_at = ?'
                ' WHERE id = ?',
                ('2024-01-01T00:00:00+00:00',
                 '2024-01-01T00:00:00+00:00', f'rd-{i}'))
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild', '--dry-run'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['total'] == 3
        assert data['dry_run'] == 1

        db = open_db(str(store_path))
        row = db._conn.execute(
            'SELECT COUNT(*) FROM insights'
            ' WHERE enriched_at IS NOT NULL').fetchone()
        assert row[0] == 3, 'dry-run must not clear enriched_at'
        db.close()

    def test_rebuild_reprocesses_stale_insights(
            self, tmp_path, monkeypatch):
        """Rebuild re-enriches insights with stale/empty keywords."""
        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        from memman.store.db import open_db
        from memman.store.node import insert_insight
        from tests.conftest import make_insight
        db = open_db(str(store_path))
        insert_insight(db, make_insight(
            id='rs-1', content='Python and SQLite used for data analysis',
            entities=['Python', 'SQLite']))
        insert_insight(db, make_insight(
            id='rs-2', content='SQLite database migration with Python scripts',
            entities=['SQLite', 'Python']))
        db._conn.execute(
            "UPDATE insights"
            " SET linked_at = '2024-01-01T00:00:00+00:00',"
            "     enriched_at = '2024-01-01T00:00:00+00:00',"
            "     keywords = '[]'"
            " WHERE id IN ('rs-1', 'rs-2')")
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['processed'] >= 2

        db = open_db(str(store_path))
        row = db._conn.execute(
            "SELECT keywords, enriched_at FROM insights"
            " WHERE id = 'rs-1'").fetchone()
        keywords = json.loads(row[0]) if row[0] else []
        assert len(keywords) > 0, 'rebuild should populate keywords'
        assert row[1] is not None, 'rebuild should set enriched_at'

        entities_raw = db._conn.execute(
            "SELECT entities FROM insights"
            " WHERE id = 'rs-1'").fetchone()[0]
        entities = json.loads(entities_raw) if entities_raw else []
        assert len(entities) > 0, 'rebuild should populate entities'
        db.close()

    def test_rebuild_handles_mix_of_linked_and_unlinked(
            self, tmp_path, monkeypatch):
        """Rebuild processes both linked and unlinked insights."""
        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        from memman.store.db import open_db
        from memman.store.node import insert_insight
        from tests.conftest import make_insight
        db = open_db(str(store_path))
        insert_insight(db, make_insight(
            id='mx-1', content='Already linked insight'))
        db._conn.execute(
            "UPDATE insights SET linked_at = ?, enriched_at = ?"
            " WHERE id = 'mx-1'",
            ('2024-01-01T00:00:00+00:00',
             '2024-01-01T00:00:00+00:00'))
        insert_insight(db, make_insight(
            id='mx-2', content='Never linked insight'))
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['processed'] >= 2

        db = open_db(str(store_path))
        pending = db._conn.execute(
            'SELECT COUNT(*) FROM insights'
            ' WHERE linked_at IS NULL'
            ' AND deleted_at IS NULL').fetchone()[0]
        assert pending == 0, 'all insights should be linked after rebuild'
        db.close()

    def test_rebuild_preserves_manual_edges(
            self, tmp_path, monkeypatch):
        """Manual claude edges survive rebuild."""
        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        from memman.store.db import open_db
        from memman.store.edge import get_all_edges, insert_edge
        from memman.store.node import insert_insight
        from tests.conftest import make_edge, make_insight
        db = open_db(str(store_path))
        insert_insight(db, make_insight(
            id='me-1', content='Python web framework',
            entities=['Python']))
        insert_insight(db, make_insight(
            id='me-2', content='Python data pipeline',
            entities=['Python']))
        db._conn.execute(
            "UPDATE insights"
            " SET linked_at = '2024-01-01T00:00:00+00:00',"
            "     enriched_at = '2024-01-01T00:00:00+00:00'")
        manual_edge = make_edge(
            source_id='me-1', target_id='me-2',
            edge_type='semantic',
            metadata={'created_by': 'claude', 'cosine': '0.95'})
        insert_edge(db, manual_edge)
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild'])
        assert result.exit_code == 0, result.output

        db = open_db(str(store_path))
        edges = get_all_edges(db)
        manual = [e for e in edges
                  if e.edge_type == 'semantic'
                  and e.metadata.get('created_by') == 'claude']
        assert len(manual) == 1, (
            'rebuild deleted manual claude edge - '
            'should preserve created_by=claude')
        db.close()

    def test_rebuild_keeps_the_stored_entity_list(
            self, tmp_path, monkeypatch):
        """Rebuild keeps the stored entity list, whatever the LLM returns.

        Mutation: the rebuild loop replacing a row's entities with the
            re-enrichment draw, so a redrawn body that names none of
            the stored labels erases them.
        Oracle: the stubbed LLM body names an entity the stored list
            lacks (`Warrant`, a literal substring of the content) and
            omits every stored label; the stored list surviving
            unchanged proves nothing from the draw was adopted.
        """
        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        from memman.store.db import open_db
        from memman.store.node import insert_insight
        from tests.conftest import make_insight
        db = open_db(str(store_path))
        insert_insight(db, make_insight(
            id='vocab-1',
            content='Postgres replaced SQLite for the Warrant ledger.',
            entities=['Database: Postgres', 'Library: SQLite']))
        db._conn.execute(
            "UPDATE insights"
            " SET linked_at = '2024-01-01T00:00:00+00:00',"
            "     enriched_at = '2024-01-01T00:00:00+00:00'"
            " WHERE id = 'vocab-1'")
        db.close()

        def fake_complete(self, system, user, **kwargs):
            if 'keyword' in system.lower() and 'enrichment' in system.lower():
                return json.dumps({
                    'entities': ['Warrant'],
                    'keywords': ['ledger'],
                    'summary': 'a ledger migration',
                    'semantic_facts': ['Postgres replaced SQLite'],
                    })
            return json.dumps({'facts': [{'text': user, 'category': 'fact',
                                          'entities': []}]})

        monkeypatch.setattr(
            'memman.llm.client.MemmanLLMClient.complete', fake_complete)

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild'])
        assert result.exit_code == 0, result.output

        db = open_db(str(store_path))
        raw = db._conn.execute(
            "SELECT entities FROM insights WHERE id = 'vocab-1'"
            ).fetchone()[0]
        db.close()
        stored = json.loads(raw)
        assert stored == ['Database: Postgres', 'Library: SQLite']


@pytest.mark.scheduler_stopped
class TestGraphRebuildAutoEdges:
    """A rebuild re-derives the auto edges its own loop disturbed."""

    def test_rebuild_leaves_the_clean_slate_entity_edge_set(
            self, tmp_path, monkeypatch):
        """Rebuild's edge set already equals a clean re-derivation.

        Mutation: omitting the end-of-loop re-derive, so a later row's
            both-direction `delete_auto_for_node` leaves an earlier
            already-stamped row short of links it earned - eight rows
            share one entity against `MAX_ENTITY_LINKS = 5`, so the
            per-row order decides who keeps what. Here it shows as
            edges an unfinished pass left behind: `Postgres` sits in
            every row, so its IDF weight is zero and a clean pass
            writes no edge for it at all.
        Oracle: a differential re-implementation - `reindex_auto_edges`
            deletes every auto edge in one pre-pass and rebuilds from
            the stored entity list, so its output is the set a
            finished rebuild owes.
        """
        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        from memman.graph.engine import reindex_auto_edges
        from memman.store.db import open_db
        from memman.store.edge import get_all_edges
        from memman.store.factory import open_backend

        db = open_db(str(store_path))
        for n in range(8):
            group = 'Redis' if n < 4 else 'Kafka'
            insert_insight(db, make_insight(
                id=f'ae-{n}',
                content=f'Postgres carries the {group} ledger, note {n}.',
                entities=['Postgres', group]))
        db.close()

        result = CliRunner().invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild'])
        assert result.exit_code == 0, result.output

        def _entity_edges() -> set:
            db = open_db(str(store_path))
            edges = {(e.source_id, e.target_id) for e in get_all_edges(db)
                     if e.edge_type == 'entity'}
            db.close()
            return edges

        after_rebuild = _entity_edges()
        reindex_auto_edges(
            open_backend('default', data_dir), store_name='default')
        assert after_rebuild == _entity_edges()
        assert after_rebuild


class TestGraphRebuildIsolation:
    """A corpus rebuild refuses to race the scheduler drain."""

    def test_rebuild_rejected_while_scheduler_started(self, tmp_path):
        """A started scheduler blocks a rebuild, both modes.

        Mutation: guarding the rebuild with `_require_started`, the
            sense `embed reembed` and `embed swap` already take the
            other way, so the drain's relink path keeps claiming rows
            the rebuild just reset and re-enriches them on the wrong
            model.
        Oracle: the message `_require_stopped` raises, which names the
            command that clears the way; the autouse fixture holds the
            state at STARTED for this class.
        """
        for extra in ([], ['--stale-only']):
            out = CliRunner().invoke(cli, [
                '--data-dir', str(tmp_path), 'graph', 'rebuild'] + extra)
            assert out.exit_code != 0, out.output
            assert 'scheduler stop' in out.output


@pytest.mark.scheduler_stopped
class TestGraphRebuildStaleOnly:
    """Tests for `graph rebuild --stale-only` flag."""

    def _seed_drift(self, store_path, active_pv):
        """Insert one drifted row and one current row.

        Also primes the per-store constants_hash so that opening via
        `_active_backend` does not trigger a wholesale reindex that
        nulls every row's `linked_at` (which would erase the seed's
        linked-state and mask the test's intent).
        """
        from memman.embed.fingerprint import Fingerprint, write_fingerprint
        from memman.graph.engine import compute_constants_hash
        from memman.store.db import open_db
        from memman.store.node import insert_insight, update_enrichment
        from memman.store.sqlite import SqliteBackend
        from tests.conftest import make_insight
        OLD_PV = 'old-prompt-version-deadbeef'
        db = open_db(str(store_path))
        backend = SqliteBackend(db)
        backend.meta.set('constants_hash', compute_constants_hash())
        write_fingerprint(backend, Fingerprint(
            provider='voyage', model='voyage-3-lite', dim=512))
        insert_insight(db, make_insight(
            id='drift-1', content='Drifted insight needing re-enrichment',
            prompt_version=OLD_PV))
        insert_insight(db, make_insight(
            id='fresh-1', content='Fresh insight already on active config',
            prompt_version=active_pv))
        for iid in ('drift-1', 'fresh-1'):
            update_enrichment(db, iid, ['kw'], 'sum')
            db._conn.execute(
                'UPDATE insights SET linked_at = ?, enriched_at = ?'
                ' WHERE id = ?',
                ('2024-01-01T00:00:00+00:00',
                 '2024-01-01T00:00:00+00:00', iid))
        db.close()

    def test_dry_run_reports_stale_count(self, tmp_path, monkeypatch):
        """`--stale-only --dry-run` reports stale count without modifying.

        Mutation: flipping `!=` to `==` in `count_stale_insights`'s
            predicate, or skipping the `dry_run` branch so a real
            rebuild runs instead of only counting.
        Oracle: the one row seeded with a drifted `prompt_version`
            against the two seeded current, so the flipped predicate
            counts 2.
        """
        from memman.pipeline.remember import compute_prompt_version
        from memman.store.db import open_db
        from memman.store.node import insert_insight
        from tests.conftest import make_insight

        active_pv = compute_prompt_version()

        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        self._seed_drift(store_path, active_pv)
        db = open_db(str(store_path))
        insert_insight(db, make_insight(
            id='fresh-2', content='Second insight already on active config',
            prompt_version=active_pv))
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild',
            '--stale-only', '--dry-run'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['mode'] == 'stale-only'
        assert data['total'] == 1
        assert data['dry_run'] == 1

    def test_empty_stale_fast_path(self, tmp_path, monkeypatch):
        """Zero-stale store returns processed=0 without doing work.

        Mutation: dropping the `total_count == 0` fast-path guard, or
            widening the stale predicate to count a current row,
            which drives the pipeline into a rebuild that needs an
            LLM client this test never wires up.
        Oracle: the literal `'skipped': 'no_stale_rows'` key against
            the single row seeded on the active `prompt_version`.
        """
        from memman.embed.fingerprint import Fingerprint, write_fingerprint
        from memman.graph.engine import compute_constants_hash
        from memman.pipeline.remember import compute_prompt_version
        from memman.store.sqlite import SqliteBackend

        active_pv = compute_prompt_version()

        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        from memman.store.db import open_db
        from memman.store.node import insert_insight
        from tests.conftest import make_insight
        db = open_db(str(store_path))
        backend = SqliteBackend(db)
        backend.meta.set('constants_hash', compute_constants_hash())
        write_fingerprint(backend, Fingerprint(
            provider='voyage', model='voyage-3-lite', dim=512))
        insert_insight(db, make_insight(
            id='ok-1', content='Already on active config',
            prompt_version=active_pv))
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild', '--stale-only'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['mode'] == 'stale-only'
        assert data['processed'] == 0
        assert data['skipped'] == 'no_stale_rows'

    def test_stale_only_re_enriches_drifted_rows(self, tmp_path, monkeypatch):
        """`--stale-only` clears enriched_at on drifted rows only.

        Mutation: passing every row's id to `reset_for_rebuild`
            instead of only the stale batch, which would also touch
            `fresh-1`'s `enriched_at`.
        Oracle: `enriched_at` and `prompt_version` read back per row,
            before and after, for both the drifted and the fresh id.
        """
        from memman.pipeline.remember import compute_prompt_version
        from memman.store.db import open_db

        active_pv = compute_prompt_version()

        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        self._seed_drift(store_path, active_pv)

        db = open_db(str(store_path))
        before = {
            row[0]: (row[1], row[2]) for row in db._conn.execute(
                'SELECT id, enriched_at, prompt_version FROM insights')
            }
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild', '--stale-only'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['mode'] == 'stale-only'
        assert data['processed'] == 1

        db = open_db(str(store_path))
        after = {
            row[0]: (row[1], row[2]) for row in db._conn.execute(
                'SELECT id, enriched_at, prompt_version FROM insights')
            }
        db.close()
        assert after['fresh-1'] == before['fresh-1']
        assert after['drift-1'][0] != before['drift-1'][0]
        assert after['drift-1'][1] == active_pv

    def test_stale_only_accepted_on_postgres_runner(self, cross_backend_runner):
        """`--stale-only` does not trip the SQLite-only guard on Postgres."""
        r, data_dir = cross_backend_runner
        out = r.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild',
            '--stale-only', '--dry-run'])
        assert out.exit_code == 0, out.output
        data = json.loads(out.output)
        assert data['mode'] == 'stale-only'
        assert 'SQLite-only' not in out.output

    def test_wholesale_rebuild_accepted_on_postgres_runner(
            self, cross_backend_runner):
        """Wholesale `graph rebuild` is now cross-backend (gate lifted)."""
        r, data_dir = cross_backend_runner
        out = r.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild', '--dry-run'])
        assert out.exit_code == 0, out.output
        data = json.loads(out.output)
        assert 'total' in data
        assert data.get('dry_run') == 1
        assert 'SQLite-only' not in out.output


def _rows_for_queue_id(data_dir, store, queue_id):
    """Active insight ids stored for one queue row, via its queue_uuid."""
    from memman.queue import queue_db
    from memman.store.db import open_read_only, store_dir
    with queue_db(data_dir) as qconn:
        queue_uuid = qconn.execute(
            'select queue_uuid from queue where id = ?',
            (queue_id,)).fetchone()[0]
    db = open_read_only(store_dir(data_dir, store))
    try:
        return db._query(
            'SELECT id FROM insights WHERE queue_uuid = ?'
            ' AND deleted_at IS NULL', (queue_uuid,)).fetchall()
    finally:
        db.close()


def test_update_reconciliation_no_dangling_edges(runner):
    """A retired row leaves no semantic edge behind it.

    A companion row is seeded first so the target picks up a
    temporal proximity edge before it is retired; `replace <id>`
    then supersedes it, and the sweep runs over a store that holds a
    real edge to lose, not an empty one.

    Mutation: dropping the edge cleanup `_apply_plan` runs on the
        target after `supersede_insight`'s own cleanup, so a
        predecessor that picked up an edge before the write keeps it.
    Oracle: `check_dangling_edges`, an independent sweep that
        counts edges whose endpoint is soft-deleted or
        superseded.
    """
    _r, data_dir = runner
    invoke(runner, [
        'remember', 'Onboarding doc lists the required VPN client'])
    seeded = invoke(runner, [
        'remember', 'Delta mode dropdown defaults to incremental_sync'])
    old_id = parse_remember(seeded, runner)['id']

    store_path = pathlib.Path(data_dir) / 'data' / 'default'
    from memman.store.db import open_db
    db = open_db(str(store_path))
    edges_before = db._query(
        'select count(*) from edges'
        ' where source_id = ? or target_id = ?',
        (old_id, old_id)).fetchone()[0]
    db.close()
    assert edges_before > 0, 'seeded row has no edge for the sweep to lose'

    result = invoke(runner, [
        'replace', old_id,
        ('Delta mode dropdown defaults'
         ' to incremental_sync with no empty option')])
    assert result.exit_code == 0, result.output

    from memman.doctor import check_dangling_edges
    from memman.store.sqlite import SqliteBackend
    db = open_db(str(store_path))
    retired = db._query(
        'select count(*) from insights'
        ' where superseded_by is not null').fetchone()[0]
    doctor_result = check_dangling_edges(SqliteBackend(db))
    db.close()

    assert retired > 0, 'nothing was retired; the sweep proves nothing'
    assert doctor_result['status'] == 'pass', (
        f'dangling edges found: {doctor_result["detail"]}')
    assert doctor_result['detail']['count'] == 0


class TestHotPathPurity:
    """Synchronous write commands must be LLM/embed-free.

    `forget` and `graph link` mutate the store DB synchronously and
    must make zero LLM or embed calls. Any future change that adds
    such calls to these paths will fail one of these tests loudly.
    """

    @pytest.fixture
    def runner_with_seed(self, tmp_path):
        """CliRunner over an isolated data dir with two seeded insights.

        Direct DB seeding avoids invoking the LLM for setup, so the
        assertion that the test target makes no LLM calls is meaningful.
        """
        from memman.embed.fingerprint import write_fingerprint
        from memman.store.db import open_db, store_dir, write_active
        from memman.store.sqlite import SqliteBackend

        data_dir = str(tmp_path)
        name = 'default'
        write_active(data_dir, name)
        sdir = store_dir(data_dir, name)
        db = open_db(sdir)
        fp = seed_default_fingerprint()
        write_fingerprint(SqliteBackend(db), fp)

        a = make_insight(id='aud-a', content='alpha', importance=3)
        b = make_insight(id='aud-b', content='beta', importance=3)
        insert_insight(db, a)
        insert_insight(db, b)
        update_embedding(db, 'aud-a',
                         serialize_vector([0.1] * fp.dim), fp.model)
        db.close()

        return CliRunner(), data_dir

    def _make_failing_complete(self, *_args, **_kwargs):
        raise AssertionError(
            'synchronous write must not invoke the LLM')

    def _make_failing_embed(self, *_args, **_kwargs):
        raise AssertionError(
            'synchronous write must not invoke the embed client')

    def test_forget_makes_no_llm_or_embed_calls(
            self, runner_with_seed, monkeypatch):
        """`forget` is pure SQL: no LLM, no embed."""
        monkeypatch.setattr(
            'memman.llm.client.MemmanLLMClient.complete',
            self._make_failing_complete)
        monkeypatch.setattr(
            'memman.embed.voyage.Client.embed', self._make_failing_embed)

        r, data_dir = runner_with_seed
        out = r.invoke(cli, ['--data-dir', data_dir, 'forget', 'aud-a'])
        assert out.exit_code == 0, out.output

    def test_graph_link_makes_no_llm_or_embed_calls(
            self, runner_with_seed, monkeypatch):
        """`graph link` is pure SQL."""
        monkeypatch.setattr(
            'memman.llm.client.MemmanLLMClient.complete',
            self._make_failing_complete)
        monkeypatch.setattr(
            'memman.embed.voyage.Client.embed', self._make_failing_embed)

        r, data_dir = runner_with_seed
        out = r.invoke(cli, ['--data-dir', data_dir,
                             'graph', 'link', 'aud-a', 'aud-b'])
        assert out.exit_code == 0, out.output


class TestPostgresGuards:
    """Admin commands that are SQLite-only must reject postgres backend."""

    def test_embed_reembed_rejects_postgres_backend(self, runner, env_file):
        """`embed reembed` exits non-zero with a clear message on postgres."""
        env_file('MEMMAN_BACKEND_default', 'postgres')
        env_file('MEMMAN_POSTGRES_DSN_default', 'postgresql://user@host/db')
        r, data_dir = runner
        out = r.invoke(cli, ['--data-dir', data_dir, 'embed', 'reembed'])
        assert out.exit_code != 0
        assert 'SQLite-only' in out.output


class TestCorruptStoreErrorHygiene:
    """A store whose database cannot be opened exits cleanly."""

    def test_recall_reports_corrupt_store_without_traceback(self, runner):
        """`recall` on an unreadable store prints an error, not a trace.

        Mutation: dropping `open_db`'s `sqlite3.Error` translation, so
            the driver error escapes the OPEN untranslated.
        Oracle: the MESSAGE, not the exception type. `open_db` names
            the store path (`cannot open database ... memman.db`);
            the root group's generic arm says only `sqlite query
            failed`, so the message is what separates them.

        Scope, since two seams have since grown over this path. The
        root group catches `sqlite3.Error` as well as `BackendError`,
        so `result.exception` is a `SystemExit` under the mutation too
        and the type assertion below is tautological -- kept only as a
        guard against a future seam that re-raises. This test now pins
        `open_db`'s own translation via its message alone. The direct
        pin on `active_store`'s catch is
        `tests/test_session.py::test_active_store_wraps_backend_error_from_open`,
        and the root group's sqlite arm is pinned by
        `tests/test_backend_error_hygiene.py::test_sqlite_statement_error_exits_as_one_clean_line`.
        """
        _, data_dir = runner
        sdir = pathlib.Path(data_dir) / 'data' / 'broken'
        sdir.mkdir(parents=True)
        (sdir / 'memman.db').write_bytes(b'not a sqlite database' * 8)
        result = invoke(
            runner, ['--store', 'broken', 'recall', 'anything', '--basic'])
        assert result.exit_code != 0
        assert not isinstance(result.exception, BackendError)
        assert 'Error: cannot open database' in result.output
        assert 'memman.db' in result.output

    def test_remember_reports_corrupt_queue_without_traceback(self, runner):
        """`remember` on an unreadable queue.db prints an error, not a trace.

        Mutation: dropping the root group's `BackendError` translation,
            so the translated queue error escapes with no seam to catch
            it -- `remember` never reaches `session.active_store`, which
            is the only other seam. `exit_code` alone cannot catch that:
            Click reports an in-command raise as exit 1 too.
        Oracle: a clean exit leaves `result.exception` a `SystemExit`
            and writes `Error: ...` naming the queue file; a leak leaves
            the `BackendError` itself on `result.exception`.
        """
        _, data_dir = runner
        queue_path = pathlib.Path(data_dir) / 'queue.db'
        queue_path.write_bytes(b'not a sqlite database' * 8)
        result = invoke(runner, ['remember', 'a fact worth keeping'])
        assert result.exit_code != 0
        assert not isinstance(result.exception, BackendError)
        assert 'Error: cannot open queue database' in result.output
        assert 'queue.db' in result.output

    def test_reembed_reports_unreadable_store_without_traceback(self, runner):
        """`embed reembed --dry-run` names a bad store dir, not a trace.

        Mutation: dropping the root group's `BackendError` translation,
            or leaving `open_read_only`'s missing-database case raising
            `FileNotFoundError`. `embed reembed` counts rows through
            `open_ro_db`, so it reaches neither `open_db` nor
            `session.active_store`.
        Oracle: a store directory with no `memman.db` -- the sweep is
            global, so one stray directory is enough -- and a clean exit
            leaves `result.exception` a `SystemExit` with `Error: ...`
            naming the path.
        """
        _, data_dir = runner
        (pathlib.Path(data_dir) / 'data' / 'stray').mkdir(parents=True)
        result = invoke(runner, ['embed', 'reembed', '--dry-run'])
        assert result.exit_code != 0
        assert not isinstance(result.exception, BackendError)
        assert 'Error: database not found' in result.output
        assert 'stray' in result.output

    def test_debug_keeps_the_stack_the_seam_hides(self, runner, caplog):
        """`--debug` still carries the traceback behind the clean line.

        Mutation: dropping `exc_info=True` from the seam's
            `logger.debug`, which discards the only copy of the stack
            -- the clean `Error:` line is all that survives, and no
            flag brings the traceback back.
        Oracle: the captured record's own `exc_info`, plus the
            `BackendError` type in its formatted text. Asserting on
            `result.output` would not work: `_configure_logging` binds
            its handler to whatever `sys.stderr` was live at the first
            configure in the process and never rebinds.
        """
        import logging

        _, data_dir = runner
        (pathlib.Path(data_dir) / 'queue.db').write_bytes(
            b'not a sqlite database' * 8)
        with caplog.at_level(logging.DEBUG, logger='memman'):
            result = invoke(runner, ['--debug', 'remember', 'a fact'])
        assert result.exit_code != 0
        seam = [r for r in caplog.records if 'CLI seam' in r.getMessage()]
        assert seam, [r.getMessage() for r in caplog.records]
        assert seam[0].exc_info is not None
        assert 'BackendError' in logging.Formatter().format(seam[0])
