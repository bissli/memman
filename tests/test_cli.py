"""Tests for memman.cli - Click CLI commands via CliRunner.

The LLM and embedding clients are the autouse mocks from conftest.
"""

import json
import logging
import pathlib
import re
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from memman import config
from memman import embed as embed_mod
from memman.cli import cli
from memman.embed.fingerprint import Fingerprint, seed_default_fingerprint
from memman.embed.fingerprint import write_fingerprint
from memman.embed.vector import serialize_vector
from memman.pipeline.remember import compute_prompt_version
from memman.queue import enqueue, list_rows, open_queue_db
from memman.setup.scheduler import _write_env_keys
from memman.store.db import open_db, open_read_only, store_dir, store_exists
from memman.store.db import write_active
from memman.store.errors import BackendError
from memman.store.node import insert_insight, update_embedding
from memman.store.node import update_enrichment
from memman.store.sqlite import SqliteBackend
from tests.conftest import invoke, make_insight, parse_remember

_SCORED_LINE = re.compile(
    r'^(?P<id>\S{8}) (?P<score>-?\d+\.\d\d)'
    r' (?P<created>\S+) (?P<author>\S+) \| (?P<text>.*)$')
_BASIC_LINE = re.compile(
    r'^(?P<id>\S{8})'
    r' (?P<created>\S+) (?P<author>\S+) \| (?P<text>.*)$')


def _parse_recall_lines(output: str, basic: bool = False) -> list[dict]:
    """Parse a recall page into one field dict per line.

    Parameters
    ----------
    output : str
        Plain-text recall output, one row per line.
    basic : bool
        Match the scoreless `--basic` line format.

    Returns
    -------
    list[dict]
        Named groups of each line, in page order.
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
    """CliRunner + data_dir tuple (delegates to conftest `mm_runner`).
    """
    return mm_runner


class TestRemember:
    """`memman remember` happy paths and validation.
    """

    def test_remember_basic(self, runner):
        """Verify `remember` queues a row and reports its content.

        Mutation: returning the payload without the stored content, or an
            action outside the add/update vocabulary.
        Oracle: the drained row read back by `parse_remember`.
        """
        result = invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        assert data['action'] in {'add', 'added', 'update', 'updated'}
        assert 'sqlite' in data['content'].lower()

    def test_remember_does_not_enrich_old_pending_insights(
            self, runner, monkeypatch):
        """Remember does inline enrichment, never calls enrich_pending.

        Mutation: `remember` calling `enrich_pending` to sweep the
            backlog on every write, billing a batch scan on the
            write path.
        Oracle: a `side_effect` that raises if `enrich_pending` runs,
            plus `mock_lp.assert_not_called()`.
        """
        invoke(runner, [
            'remember', 'Redis cache eviction uses LRU algorithm'])

        with patch('memman.pipeline.enrich.enrich_pending',
                   side_effect=AssertionError(
                       'enrich_pending called from remember')) as mock_lp:
            result = invoke(runner, [
                'remember', 'PostgreSQL MVCC provides snapshot isolation'])
            assert result.exit_code == 0
            mock_lp.assert_not_called()

    def test_remember_quality_warnings(self, runner):
        """Verify transient content is queued with named quality warnings.

        Mutation: dropping a warning pattern (instance id, deployment
            receipt), or rejecting the write instead of queueing it.
        Oracle: content holding an instance id and a deploy phrase.
        """
        result = invoke(runner, [
            'remember', 'i-0c220c2402a5245bc deployed via Terraform'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['action'] == 'queued'
        assert 'AWS instance ID' in data['quality_warnings']
        assert 'deployment receipt' in data['quality_warnings']

    def test_remember_no_quality_warnings(self, runner):
        """Verify durable content carries an empty warning list.

        Mutation: a warning pattern that matches ordinary decision text.
        Oracle: a decision sentence with no transient marker.
        """
        result = invoke(runner, [
            'remember', 'SQLite chosen for single-node simplicity and embedded operation'])
        assert result.exit_code == 0
        raw = json.loads(result.output)
        assert raw['quality_warnings'] == []

    def test_remember_quality_warnings_populate(self, runner):
        """Verify warnings never block a write.

        Mutation: rejecting content that draws warnings, or dropping the
            warning list from the payload.
        Oracle: a two-warning sentence queued, and a long incident sentence
            stored with exactly one warning.
        """
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


class TestRecall:
    """`memman recall` smart and basic modes.
    """

    def test_recall_basic(self, runner):
        """Verify `recall` exits 0 after a `remember`.

        Mutation: recall raising on a store that holds one fresh row.
        Oracle: the exit code alone.
        """
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        result = invoke(runner, ['recall', 'Go SQLite storage'])
        assert result.exit_code == 0

    def test_recall_does_not_call_enrich_pending(self, runner, monkeypatch):
        """Recall never calls enrich_pending on its read path.

        Mutation: `recall` calling `enrich_pending` to opportunistically
            sweep the backlog on a read, billing a batch scan on every
            query.
        Oracle: a `side_effect` that raises if `enrich_pending` runs,
            plus `mock_lp.assert_not_called()`.
        """
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])

        with patch('memman.pipeline.enrich.enrich_pending',
                   side_effect=AssertionError('enrich_pending called')) as mock_lp:
            result = invoke(runner, ['recall', 'Go SQLite storage'])
            assert result.exit_code == 0
            mock_lp.assert_not_called()

    def test_recall_logs_when_query_embed_fails(
            self, runner, caplog, monkeypatch):
        """Verify a failing query embed is logged and recall still succeeds.

        Mutation: swallowing the embed error without a warning, or letting
            it abort the command instead of degrading to keyword recall.
        Oracle: a patched embed that raises, and the captured WARNING record.
        """

        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])

        real_ec = embed_mod.get_client()

        def _boom(self, text: str) -> None:
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
            `run_recall` call, so the config default never
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
        """Verify a reranker error does not break recall.

        Mutation: letting the rerank exception escape the recall command.
        Oracle: a patched rerank raising RuntimeError, called exactly once.
        """
        for fact in [
                'Go uses SQLite for persistent storage',
                'Go modules manage dependency versions']:
            invoke(runner, ['remember', fact])

        with patch('memman.rerank.voyage.Client.rerank',
                   side_effect=RuntimeError('voyage 503')) as mock_re:
            result = invoke(runner, ['recall', 'Go SQLite persistent storage'])
            assert result.exit_code == 0
            mock_re.assert_called_once()

    def test_recall_basic_mode(self, runner):
        """Basic recall prints a scoreless page line per matching row.

        Mutation: printing a `{results, meta}` JSON envelope.
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

        Mutation: printing a `{results: [...]}` JSON envelope.
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
            'remember', 'Q'])
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
        """The recall-detail row carries the REQUESTED limit.

        Mutation: recording `len(hits)` in place of the requested
            `limit`, which cannot tell a thin page from a small ask.
        Oracle: a recall issued with a limit deliberately larger than
            the store can fill, so the requested value and the
            returned count differ.
        """
        invoke(runner, [
            'remember', 'Envoy routes gRPC traffic by header match'])
        invoke(runner, [
            'recall', 'Envoy gRPC header routing', '--limit', '17'])

        entries = json.loads(
            invoke(runner, ['log', 'list', '--limit', '50']).output)['entries']
        details = [
            json.loads(e['detail'])
            for e in entries if e['operation'] == 'recall-detail']
        assert details, 'expected a recall-detail row'
        row = details[0]
        assert row['limit'] == 17
        assert 'q' in row
        assert len(row['q']) <= 80
        assert len(row['hits']) < row['limit'], (
            'fixture must under-fill the page so the two cannot be '
            'confused')


class TestForget:
    """`memman forget` happy paths and missing-id error.
    """

    def test_forget_basic(self, runner):
        """Verify `forget` deletes an insight by id.

        Mutation: reporting a status other than deleted, or failing on a
            live id.
        Oracle: the id returned by `remember`.
        """
        result = invoke(runner, [
            'remember', 'Redis cache eviction policy uses LRU by default'])
        data = parse_remember(result, runner)
        iid = data['id']
        result = invoke(runner, ['forget', iid])
        assert result.exit_code == 0
        fdata = json.loads(result.output)
        assert fdata['status'] == 'deleted'

    def test_forget_writes_oplog(self, runner):
        """Verify `forget` leaves a `forget` oplog entry.

        Mutation: deleting the row without writing the oplog record.
        Oracle: `log list --stats` operation counts.
        """
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
        """Verify `forget` on an unknown id exits non-zero.

        Mutation: reporting success for an id that does not exist.
        Oracle: exit code for a made-up id.
        """
        result = invoke(runner, ['forget', 'nonexistent-id-12345'])
        assert result.exit_code != 0


class TestStore:
    """`memman store` admin: list, create, set, remove.
    """

    def test_store_list(self, runner):
        """Verify `store list` emits stores and active keys.

        Mutation: dropping either key from the envelope.
        Oracle: the parsed JSON keys.
        """
        result = invoke(runner, ['store', 'list'])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert 'stores' in payload
        assert 'active' in payload

    def test_store_create(self, runner):
        """Verify `store create` reports action=created and the name.

        Mutation: reporting the wrong action or an empty store name.
        Oracle: hand-set `created` and `test-store`.
        """
        result = invoke(runner, ['store', 'create', 'test-store'])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload['action'] == 'created'
        assert payload['store'] == 'test-store'

    def test_store_create_duplicate(self, runner):
        """Verify creating an existing store fails.

        Mutation: dropping the exists check so the second create succeeds.
        Oracle: exit code of the second `store create dup`.
        """
        invoke(runner, ['store', 'create', 'dup'])
        result = invoke(runner, ['store', 'create', 'dup'])
        assert result.exit_code != 0

    def test_store_set(self, runner):
        """Verify `store use` reports action=set and the name.

        Mutation: switching the store without reporting it, or reporting
            the previous store.
        Oracle: hand-set `set` and `work`.
        """
        invoke(runner, ['store', 'create', 'work'])
        result = invoke(runner, ['store', 'use', 'work'])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload['action'] == 'set'
        assert payload['store'] == 'work'

    def test_store_remove_yes(self, runner):
        """Verify `store remove --yes` skips the prompt and removes.

        Mutation: prompting despite --yes, or removing without reporting.
        Oracle: exit 0 with no input and action=removed.
        """
        invoke(runner, ['store', 'create', 'temp'])
        result = invoke(runner, ['store', 'remove', '--yes', 'temp'])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload['action'] == 'removed'
        assert payload['store'] == 'temp'

    def test_store_remove_purges_queue(self, runner):
        """Verify removing a store drops its queue rows.

        Mutation: removing the data dir but leaving queue rows, which the
            worker then retries against a missing store dir.
        Oracle: queue rows for the store counted before and after removal.
        """

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
        payload = json.loads(result.output)
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
        """Verify removing a store drops its per-store env keys only.

        Mutation: dropping the `removes=` cleanup, which leaves
            `MEMMAN_BACKEND_doomed` and the store's Postgres DSN (password
            included) in the env file after the store is gone.
        Oracle: the parsed env file. Every PER_STORE_KEY_SPECS prefix for
            the removed store is absent, and the same prefixes for a
            surviving store plus the global key remain.
        """
        _, data_dir = runner
        invoke(runner, ['store', 'create', 'doomed'])
        invoke(runner, ['store', 'create', 'keeper'])
        value_for = {
            'MEMMAN_BACKEND_': 'sqlite',
            'MEMMAN_POSTGRES_DSN_': 'postgresql://u:p@127.0.0.1:1/db',
            'MEMMAN_RERANK_ENABLED_': 'false',
            }
        doomed_keys = {
            f'{prefix}doomed' for prefix, _ in config.PER_STORE_KEY_SPECS}
        keeper_keys = {
            f'{prefix}keeper' for prefix, _ in config.PER_STORE_KEY_SPECS}
        seeded = {
            f'{prefix}{store}': value_for[prefix]
            for prefix, _ in config.PER_STORE_KEY_SPECS
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
        """Verify answering `n` to the prompt aborts the removal.

        Mutation: removing the store despite a negative answer.
        Oracle: non-zero exit and `store_exists` still true.
        """
        r, data_dir = runner
        invoke(runner, ['store', 'create', 'temp2'])
        result = r.invoke(
            cli, ['--data-dir', data_dir, 'store', 'remove', 'temp2'],
            input='n\n')
        assert result.exit_code != 0
        assert store_exists(data_dir, 'temp2')

    def test_store_remove_prompts_accept(self, runner):
        """Verify answering `y` to the prompt completes the removal.

        Mutation: aborting on a positive answer, or omitting the JSON
            payload after the prompt echo.
        Oracle: exit 0 and action=removed in the payload.
        """
        r, data_dir = runner
        invoke(runner, ['store', 'create', 'temp3'])
        result = r.invoke(
            cli, ['--data-dir', data_dir, 'store', 'remove', 'temp3'],
            input='y\n')
        assert result.exit_code == 0, result.output
        # The confirm prompt echoes before the JSON payload.
        payload_start = result.output.find('{')
        payload = json.loads(result.output[payload_start:])
        assert payload['action'] == 'removed'

    def test_store_auto_create_from_env(self, runner, monkeypatch):
        """Verify MEMMAN_STORE creates a missing store on first use.

        Mutation: failing on an unknown store name from the env var, or
            creating no directory.
        Oracle: the store directory on disk and the name in `store list`.
        """
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
    """`memman status` and `memman doctor` smoke.
    """

    def test_status_basic(self, runner):
        """Verify `status` emits JSON with total_insights.

        Mutation: dropping the total from the status payload.
        Oracle: the parsed JSON key.
        """
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        result = invoke(runner, ['status'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert 'total_insights' in data

    def test_doctor_basic(self, runner):
        """Verify `doctor` emits JSON with status, checks, and total_active.

        Mutation: dropping a top-level key from the doctor report.
        Oracle: the parsed JSON keys; exit 0 or 1 per environment.
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
    """`memman log` smoke.
    """

    def test_log_basic(self, runner):
        """Verify `log list` exits 0 after a write.

        Mutation: raising on a store that holds oplog rows.
        Oracle: the exit code alone.
        """
        invoke(runner, [
            'remember', 'Go uses SQLite for persistent storage'])
        result = invoke(runner, ['log', 'list'])
        assert result.exit_code == 0


class TestInsightsReview:
    """`memman insights review` flags transient content.
    """

    def test_review_flags_transient_content(self, runner):
        """A stored instance id is flagged; a durable decision is not.

        Mutation: a seeding `remember` call silently failing to store,
            which would leave `total_flagged` at 0 for the wrong
            reason.
        Oracle: both seed calls checked for a zero exit code before
            the review assertion runs.
        """
        r1 = invoke(runner, [
            'remember', 'Production outage traced to instance i-0c220c2402a5245bc running out of memory causing cascading failure'])
        assert r1.exit_code == 0, r1.output
        r2 = invoke(runner, [
            'remember', 'SQLite chosen for simplicity and embedded operation'])
        assert r2.exit_code == 0, r2.output
        result = invoke(runner, ['insights', 'review'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['total_flagged'] >= 1
        flagged = [r['content'] for r in data['review_results']]
        assert any('i-0c220c2402a5245bc' in c.lower() for c in flagged)

    def test_review_clean_store_flags_nothing(self, runner):
        """Verify durable content draws no review flag.

        Mutation: a review pattern that matches ordinary decision text.
        Oracle: total_flagged of 0 for one decision sentence.
        """
        invoke(runner, [
            'remember', 'SQLite chosen for simplicity and embedded operation'])
        result = invoke(runner, ['insights', 'review'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['total_flagged'] == 0


class TestReplace:
    """`memman replace` happy paths, metadata, oplog.
    """

    def test_replace_basic(self, runner):
        """Verify `replace` reports the action and the replaced id.

        Mutation: reporting action=add, or a wrong replaced_id.
        Oracle: the id returned by the first `remember`.
        """
        result = invoke(runner, [
            'remember', 'Redis cache configured with 512MB memory limit'])
        old_id = parse_remember(result, runner)['id']

        result = invoke(runner, [
            'replace', old_id,
            'Redis cache configured with 1GB memory limit for production'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        assert data['action'] == 'replace'
        assert data['replaced_id'] == old_id
        assert 'redis' in data['content'].lower()

    def test_replace_nonexistent_id(self, runner):
        """Verify replacing an unknown id fails with a not-found message.

        Mutation: creating a new row for an id that does not exist.
        Oracle: non-zero exit and the `not found` text.
        """
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

    def test_replace_refuses_a_replaced_id_and_names_the_successor(
            self, runner):
        """Verify replacing a replaced id points at its successor.

        Mutation: preflighting with `nodes.get` (a bare not-found), or
            accepting the replaced id and forking the chain.
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
        assert f'was replaced by {new_id}' in result.output
        assert '--history' in result.output
        active = _parse_recall_lines(
            invoke(runner, ['recall', '--basic', 'Kafka']).output,
            basic=True)
        assert [row['id'] for row in active] == [new_id[:8]]

    def test_replace_oplog_entries(self, runner):
        """Verify a replace logs both a replace and a remember op.

        Mutation: writing only one of the two oplog entries.
        Oracle: both operation names in `log list` output.
        """
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
        """Verify the replace path passes warnings as hints without blocking.

        Mutation: rejecting replacement text that draws warnings, or
            skipping the warning scan on the replace path.
        Oracle: a two-warning sentence, accepted with warnings listed.
        """
        result = invoke(runner, [
            'remember', 'Kafka chosen for event streaming due to partition tolerance'])
        old_id = parse_remember(result, runner)['id']

        result = invoke(runner, [
            'replace', old_id,
            'Stack deployed via Terraform. 32 resources total.'])
        data = json.loads(result.output)
        assert data['action'] != 'rejected'
        assert len(data['quality_warnings']) >= 2


class TestSingleTierEnrichment:
    """Remember runs enrichment inline on the drain worker.
    """

    def test_output_has_enrichment_dict(self, runner):
        """Verify the drain stores the enrichment summary on the row.

        Mutation: `_apply_plan` stamping `enriched_at` without the
            `update_enrichment` write, so the row reads as enriched
            and holds no summary, and no stranded-row sweep revisits
            it.
        Oracle: the autouse mock LLM, which echoes the content's first
            100 characters as the summary.
        """

        content = ('Redis cache configured with LRU eviction policy, '
                   'replicated across three availability zones for '
                   'automatic failover and durability')
        result = invoke(runner, ['remember', content])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        iid = data['id']

        _, data_dir = runner
        db = open_read_only(store_dir(data_dir, 'default'))
        try:
            row = db._query(
                'select summary from insights where id = ?',
                (iid,)).fetchone()
        finally:
            db.close()
        assert row is not None
        assert 'redis' in row[0].lower()

    def test_enrich_attempted_at_stamped_after_remember(self, runner):
        """enrich_attempted_at is non-NULL after remember returns.

        Mutation: the drain worker's `_apply_plan` skipping
            `stamp_enrich_attempted`, leaving a written row eligible
            for a second, redundant enrich pass.
        Oracle: the row's `enrich_attempted_at` column read back
            through a read-only handle, not-None.
        """

        result = invoke(runner, [
            'remember', 'Consul service mesh enables secure service communication'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        iid = data['id']

        _, data_dir = runner
        ro = open_read_only(data_dir + '/data/default')
        row = ro._conn.execute(
            'select enrich_attempted_at from insights where id = ?',
            (iid,)).fetchone()
        ro.close()
        assert row is not None
        assert row[0] is not None

    def test_enrich_zero_pending_after_remember(self, runner):
        """No row is left pending enrichment after remember returns.

        Mutation: `remember` skipping its inline enrichment pass, so
            a written row waits for the next `memman enrich` or drain
            instead of stamping `enrich_attempted_at` immediately.
        Oracle: a direct read-only count of rows with NULL
            `enrich_attempted_at`, independent of the `enrich
            --dry-run` JSON (which reports total active rows, not
            the pending count).
        """

        invoke(runner, [
            'remember', 'Kafka event streaming configured for microservices'])
        result = invoke(runner, ['enrich', '--dry-run'])
        assert result.exit_code == 0

        _, data_dir = runner
        ro = open_read_only(data_dir + '/data/default')
        row = ro._conn.execute(
            'select count(*) from insights'
            ' where enrich_attempted_at is null and deleted_at is null').fetchone()
        ro.close()
        assert row[0] == 0

    def test_enriched_at_stamped_after_remember(self, runner):
        """Verify enriched_at is set once `remember` returns.

        Mutation: the drain worker never stamping `enriched_at`, so a
            written row stays eligible for re-enrichment.
        Oracle: the column read back through a read-only handle.
        """

        result = invoke(runner, [
            'remember', 'Elasticsearch full-text search with custom analyzers'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        iid = data['id']

        _, data_dir = runner
        ro = open_read_only(data_dir + '/data/default')
        row = ro._conn.execute(
            'select enriched_at from insights where id = ?',
            (iid,)).fetchone()
        ro.close()
        assert row is not None
        assert row[0] is not None


def test_data_dir_flag_moves_implicit_env_resolution(tmp_path):
    """`--data-dir` must move every no-argument `config.env_file_path()` read.

    Mutation: the root `cli` callback storing `--data-dir` only in
        `ctx.obj` without exporting `MEMMAN_DATA_DIR`, so a call site
        that resolves the env file with no `data_dir` argument (e.g.
        `session.active_store` -> `embed.get_client()`) keeps reading
        the directory named by the `MEMMAN_DATA_DIR` env var instead
        of the one the flag names.
    Oracle: two data dirs whose `MEMMAN_EMBED_PROVIDER` rows diverge;
        `status` (which opens the store through `active_store` and
        eagerly calls `get_client()`) must fail naming the flag's
        directory's unregistered provider.
    """

    other_dir = tmp_path / 'other'
    other_dir.mkdir()
    rows = dict(config.INSTALL_DEFAULTS)
    rows[config.EMBED_PROVIDER] = 'bogus-provider'
    rows['MEMMAN_OPENROUTER_API_KEY'] = 'mock-key-for-testing'
    rows['MEMMAN_LLM_API_KEY'] = 'mock-llm-api-key-for-testing'
    (other_dir / config.ENV_FILENAME).write_text(
        '\n'.join(f'{k}={v}' for k, v in rows.items()) + '\n')

    result = CliRunner().invoke(cli, [
        '--data-dir', str(other_dir), 'status'])
    assert result.exit_code != 0, result.output
    assert 'bogus-provider' in result.output


@pytest.mark.scheduler_stopped
def test_enrich_is_top_level_and_graph_rebuild_is_gone(tmp_path):
    """`memman enrich` answers at the top level and no `graph` group remains.

    Mutation: keeping the `graph` group or leaving `enrich` unwired at
        the top level, so the old path still answers or the new one
        does not.
    Oracle: click's own unknown-command exit code (2) for `graph
        rebuild`, against `enrich --dry-run`'s exit code (0).
    """
    data_dir = str(tmp_path / 'memman')
    old = CliRunner().invoke(cli, [
        '--data-dir', data_dir, 'graph', 'rebuild', '--dry-run'])
    assert old.exit_code == 2, old.output

    new = CliRunner().invoke(cli, [
        '--data-dir', data_dir, 'enrich', '--dry-run'])
    assert new.exit_code == 0, new.output


@pytest.mark.scheduler_stopped
class TestEnrich:
    """`enrich` command tests - dry-run, live.
    """

    def test_rebuild_dry_run_reports_count(self, tmp_path, monkeypatch):
        """Dry run reports total insights without modifying DB.

        Mutation: `dry_run` falling through to the reset/enrich loop
            before its early return, clearing `enriched_at` on every
            row it reports on.
        Oracle: `data['total']` against the seeded row count, and a
            post-run `enriched_at IS NOT NULL` count unchanged at 3.
        """
        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path / 'memman')
        store_path = tmp_path / 'memman' / 'data' / 'default'
        db = open_db(str(store_path))
        for i in range(3):
            insert_insight(db, make_insight(
                id=f'rd-{i}', content=f'Test insight {i}'))
            db._conn.execute(
                'update insights set enrich_attempted_at = ?, enriched_at = ?'
                ' where id = ?',
                ('2024-01-01T00:00:00+00:00',
                 '2024-01-01T00:00:00+00:00', f'rd-{i}'))
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'enrich', '--dry-run'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['total'] == 3
        assert data['dry_run'] == 1

        db = open_db(str(store_path))
        row = db._conn.execute(
            'select count(*) from insights'
            ' where enriched_at is not null').fetchone()
        assert row[0] == 3, 'dry-run must not clear enriched_at'
        db.close()

    def test_rebuild_reprocesses_stale_insights(
            self, tmp_path, monkeypatch):
        """Rebuild re-enriches insights with a stale/empty summary.

        Mutation: the rebuild loop skipping a row whose `enriched_at`
            is already set, so a blanked summary is never
            re-populated.
        Oracle: two rows seeded with `summary = ''` and a stale
            `enriched_at`; after rebuild both carry a non-empty
            summary and a fresh `enriched_at`.
        """
        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path / 'memman')
        store_path = tmp_path / 'memman' / 'data' / 'default'
        db = open_db(str(store_path))
        insert_insight(db, make_insight(
            id='rs-1', content=(
                'Python and SQLite used for data analysis across the '
                'reporting pipeline with scheduled batch jobs and '
                'nightly validation checks')))
        insert_insight(db, make_insight(
            id='rs-2', content=(
                'SQLite database migration with Python scripts run '
                'nightly across every regional replica before the '
                'reporting jobs start')))
        db._conn.execute(
            "update insights"
            " set enrich_attempted_at = '2024-01-01T00:00:00+00:00',"
            "     enriched_at = '2024-01-01T00:00:00+00:00',"
            "     summary = ''"
            " where id in ('rs-1', 'rs-2')")
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'enrich'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['processed'] >= 2

        db = open_db(str(store_path))
        row = db._conn.execute(
            "select summary, enriched_at from insights"
            " where id = 'rs-1'").fetchone()
        assert row[0], 'rebuild should populate summary'
        assert row[1] is not None, 'rebuild should set enriched_at'
        db.close()

    def test_rebuild_handles_mix_of_attempted_and_unattempted(
            self, tmp_path, monkeypatch):
        """Rebuild processes both attempted and unattempted insights.

        Mutation: the batch loop stopping after one `enrich_pending`
            call per slice instead of looping to `count == 0`,
            leaving a batch half-processed when a row's LLM call
            takes more than one internal retry to land.
        Oracle: a direct post-run count of rows with NULL
            `enrich_attempted_at`, independent of the `processed`
            figure in the command's own JSON.
        """
        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path / 'memman')
        store_path = tmp_path / 'memman' / 'data' / 'default'
        db = open_db(str(store_path))
        insert_insight(db, make_insight(
            id='mx-1', content='Already linked insight'))
        db._conn.execute(
            "update insights set enrich_attempted_at = ?, enriched_at = ?"
            " where id = 'mx-1'",
            ('2024-01-01T00:00:00+00:00',
             '2024-01-01T00:00:00+00:00'))
        insert_insight(db, make_insight(
            id='mx-2', content='Never linked insight'))
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'enrich'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['processed'] >= 2

        db = open_db(str(store_path))
        pending = db._conn.execute(
            'select count(*) from insights'
            ' where enrich_attempted_at is null'
            ' and deleted_at is null').fetchone()[0]
        assert pending == 0, (
            'all insights should be enrich-attempted after rebuild')
        db.close()


class TestEnrichIsolation:
    """A corpus rebuild refuses to race the scheduler drain.
    """

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
                '--data-dir', str(tmp_path), 'enrich'] + extra)
            assert out.exit_code != 0, out.output
            assert 'scheduler stop' in out.output


@pytest.mark.scheduler_stopped
class TestEnrichStaleOnly:
    """Tests for `enrich --stale-only` flag.
    """

    def _seed_drift(self, store_path: pathlib.Path, active_pv: str) -> None:
        """Insert one drifted row and one current row.

        Parameters
        ----------
        store_path : Path
            Store directory to seed.
        active_pv : str
            Prompt version stamped on the current row.
        """
        OLD_PV = 'old-prompt-version-deadbeef'
        db = open_db(str(store_path))
        backend = SqliteBackend(db)
        write_fingerprint(backend, Fingerprint(
            provider='voyage', model='voyage-3-lite', dim=512))
        insert_insight(db, make_insight(
            id='drift-1', content='Drifted insight needing re-enrichment',
            prompt_version=OLD_PV))
        insert_insight(db, make_insight(
            id='fresh-1', content='Fresh insight already on active config',
            prompt_version=active_pv))
        for iid in ('drift-1', 'fresh-1'):
            update_enrichment(db, iid, 'sum')
            db._conn.execute(
                'update insights set enrich_attempted_at = ?, enriched_at = ?'
                ' where id = ?',
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

        active_pv = compute_prompt_version()

        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path / 'memman')
        store_path = tmp_path / 'memman' / 'data' / 'default'
        self._seed_drift(store_path, active_pv)
        db = open_db(str(store_path))
        insert_insight(db, make_insight(
            id='fresh-2', content='Second insight already on active config',
            prompt_version=active_pv))
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'enrich',
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

        active_pv = compute_prompt_version()

        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path / 'memman')
        store_path = tmp_path / 'memman' / 'data' / 'default'
        db = open_db(str(store_path))
        backend = SqliteBackend(db)
        write_fingerprint(backend, Fingerprint(
            provider='voyage', model='voyage-3-lite', dim=512))
        insert_insight(db, make_insight(
            id='ok-1', content='Already on active config',
            prompt_version=active_pv))
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'enrich', '--stale-only'])
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

        active_pv = compute_prompt_version()

        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path / 'memman')
        store_path = tmp_path / 'memman' / 'data' / 'default'
        self._seed_drift(store_path, active_pv)

        db = open_db(str(store_path))
        before = {
            row[0]: (row[1], row[2]) for row in db._conn.execute(
                'select id, enriched_at, prompt_version from insights')
            }
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'enrich', '--stale-only'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['mode'] == 'stale-only'
        assert data['processed'] == 1

        db = open_db(str(store_path))
        after = {
            row[0]: (row[1], row[2]) for row in db._conn.execute(
                'select id, enriched_at, prompt_version from insights')
            }
        db.close()
        assert after['fresh-1'] == before['fresh-1']
        assert after['drift-1'][0] != before['drift-1'][0]
        assert after['drift-1'][1] == active_pv

    def test_stale_only_re_enriches_stranded_rows(self, tmp_path, monkeypatch):
        """`--stale-only` re-enriches a stranded row and skips an enriched one.

        Mutation: the stale predicate skipping every null
            `prompt_version`, which leaves the stranded row behind; or
            taking every null key, which re-bills `legacy-1`.
        Oracle: `enriched_at` and `prompt_version` read back per row
            against a stranded row and an enriched row, both seeded
            with a null `prompt_version`.
        """

        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path / 'memman')
        store_path = tmp_path / 'memman' / 'data' / 'default'
        db = open_db(str(store_path))
        write_fingerprint(SqliteBackend(db), Fingerprint(
            provider='voyage', model='voyage-3-lite', dim=512))
        insert_insight(db, make_insight(
            id='strand-1', content='Stranded insight whose enrichment failed'))
        insert_insight(db, make_insight(
            id='legacy-1', content='Enriched insight from before provenance'))
        update_enrichment(db, 'legacy-1', 'sum')
        db._conn.execute(
            'update insights set enrich_attempted_at = ? where id = ?',
            ('2024-01-01T00:00:00+00:00', 'strand-1'))
        db._conn.execute(
            'update insights set enrich_attempted_at = ?, enriched_at = ?'
            ' where id = ?',
            ('2024-01-01T00:00:00+00:00',
             '2024-01-01T00:00:00+00:00', 'legacy-1'))
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'enrich', '--stale-only'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['processed'] == 1

        db = open_db(str(store_path))
        after = {
            row[0]: (row[1], row[2]) for row in db._conn.execute(
                'select id, enriched_at, prompt_version from insights')
            }
        db.close()
        assert after['legacy-1'] == ('2024-01-01T00:00:00+00:00', None)
        assert after['strand-1'][0] is not None
        assert after['strand-1'][1] == compute_prompt_version()

    def test_stale_only_accepted_on_postgres_runner(self, cross_backend_runner):
        """`--stale-only` does not trip the SQLite-only guard on Postgres.

        Mutation: reintroducing a backend-kind check ahead of the
            `--stale-only` branch that raises on any non-sqlite
            backend.
        Oracle: exit code 0 and no `'SQLite-only'` text in the
            output against the Postgres-backed runner.
        """
        r, data_dir = cross_backend_runner
        out = r.invoke(cli, [
            '--data-dir', data_dir, 'enrich',
            '--stale-only', '--dry-run'])
        assert out.exit_code == 0, out.output
        data = json.loads(out.output)
        assert data['mode'] == 'stale-only'
        assert 'SQLite-only' not in out.output

    def test_wholesale_rebuild_accepted_on_postgres_runner(
            self, cross_backend_runner):
        """Wholesale `enrich` runs on any backend, Postgres included.

        Mutation: reintroducing a backend-kind check ahead of the
            wholesale (non-stale) enrich path that raises on any
            non-sqlite backend.
        Oracle: exit code 0, a `total` and `dry_run` key in the JSON,
            and no `'SQLite-only'` text in the output against the
            Postgres-backed runner.
        """
        r, data_dir = cross_backend_runner
        out = r.invoke(cli, [
            '--data-dir', data_dir, 'enrich', '--dry-run'])
        assert out.exit_code == 0, out.output
        data = json.loads(out.output)
        assert 'total' in data
        assert data.get('dry_run') == 1
        assert 'SQLite-only' not in out.output


class TestHotPathPurity:
    """Synchronous write commands must be LLM/embed-free.

    `forget` mutates the store DB synchronously and must make zero
    LLM or embed calls. Any future change that adds such calls to
    this path will fail this test loudly.
    """

    @pytest.fixture
    def runner_with_seed(self, tmp_path):
        """CliRunner and data dir holding two directly seeded insights.

        Seeding the store directly keeps the LLM out of setup, so the
        assertion that the target makes no LLM call is meaningful.
        """

        data_dir = str(tmp_path / 'memman')
        name = 'default'
        write_active(data_dir, name)
        sdir = store_dir(data_dir, name)
        db = open_db(sdir)
        fp = seed_default_fingerprint()
        write_fingerprint(SqliteBackend(db), fp)

        a = make_insight(id='aud-a', content='alpha')
        b = make_insight(id='aud-b', content='beta')
        insert_insight(db, a)
        insert_insight(db, b)
        update_embedding(db, 'aud-a',
                         serialize_vector([0.1] * fp.dim), fp.model)
        db.close()

        return CliRunner(), data_dir

    def _make_failing_complete(self, *_args, **_kwargs) -> None:
        """Stand-in for the LLM call that fails the test when reached.
        """
        raise AssertionError(
            'synchronous write must not invoke the LLM')

    def _make_failing_embed(self, *_args, **_kwargs) -> None:
        """Stand-in for the embed call that fails the test when reached.
        """
        raise AssertionError(
            'synchronous write must not invoke the embed client')

    def test_forget_makes_no_llm_or_embed_calls(
            self, runner_with_seed, monkeypatch):
        """Verify `forget` makes no LLM or embed call.

        Mutation: `forget` calling the LLM or embed client, billing a
            request on a pure delete.
        Oracle: patched client methods that raise AssertionError.
        """
        monkeypatch.setattr(
            'memman.llm.client.MemmanLLMClient.complete',
            self._make_failing_complete)
        monkeypatch.setattr(
            'memman.embed.voyage.Client.embed', self._make_failing_embed)

        r, data_dir = runner_with_seed
        out = r.invoke(cli, ['--data-dir', data_dir, 'forget', 'aud-a'])
        assert out.exit_code == 0, out.output


class TestPostgresGuards:
    """Admin commands that are SQLite-only must reject postgres backend.
    """

    def test_embed_reembed_rejects_postgres_backend(self, runner, env_file):
        """Verify `embed reembed` refuses a postgres store.

        Mutation: dropping the SQLite-only guard, so a postgres store is
            swept with the sqlite code path.
        Oracle: non-zero exit and the `SQLite-only` message.
        """
        env_file('MEMMAN_BACKEND_default', 'postgres')
        env_file('MEMMAN_POSTGRES_DSN_default', 'postgresql://user@host/db')
        r, data_dir = runner
        out = r.invoke(cli, ['--data-dir', data_dir, 'embed', 'reembed'])
        assert out.exit_code != 0
        assert 'SQLite-only' in out.output


class TestCorruptStoreErrorHygiene:
    """A store whose database cannot be opened exits cleanly.
    """

    def test_recall_reports_corrupt_store_without_traceback(self, runner):
        """`recall` on an unreadable store prints an error, not a trace.

        Mutation: dropping `open_db`'s `sqlite3.Error` translation, so
            the driver error escapes the OPEN untranslated.
        Oracle: the MESSAGE, not the exception type. `open_db` names
            the store path (`cannot open database ... memman.db`);
            the root group's generic arm says only `sqlite query
            failed`, so the message is what separates them.

        Scope: the root group catches `sqlite3.Error` as well as
        `BackendError`, so `result.exception` is a `SystemExit` under
        the mutation too and the type assertion below adds nothing
        beyond a guard against a seam that re-raises. This test pins
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
            `open_read_only`, so it reaches neither `open_db` nor
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
