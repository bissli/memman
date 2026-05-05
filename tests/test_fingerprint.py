"""Tests for the pluggable embed provider + fingerprint flow.

Covers: provider registry resolution, Fingerprint serialization,
assert_consistent semantics (match / unseeded / mismatch),
install-time seeding, the embed reembed sweep (initialize, swap,
resumability, scheduler-stopped gate), and CLI block-on-mismatch
behavior for recall/remember/worker.
"""

import json

import pytest
from click.testing import CliRunner
from memman.cli import cli
from memman.embed.fingerprint import Fingerprint, active_fingerprint
from memman.embed.fingerprint import assert_consistent, stored_fingerprint
from memman.embed.fingerprint import write_fingerprint
from memman.embed.vector import serialize_vector
from memman.exceptions import ConfigError, EmbedFingerprintError
from memman.store.db import get_meta, open_db
from memman.store.node import insert_insight, update_embedding
from memman.store.sqlite import SqliteBackend
from tests.conftest import make_insight


def _seed_voyage(db) -> None:
    """Helper: write the canonical voyage fingerprint to meta."""
    write_fingerprint(SqliteBackend(db), Fingerprint(
            provider='voyage', model='voyage-3-lite', dim=512))


def _seed_row_with_embedding(db, *, id: str, content: str = 'x',
                             model: str = 'voyage-3-lite',
                             dim: int = 512) -> None:
    """Seed a row with a synthetic embedding of given model+dim."""
    insight = make_insight(
        id=id, content=content, embedding_model=model)
    insert_insight(db, insight)
    fake_vec = [0.1] * dim
    update_embedding(db, id, serialize_vector(fake_vec), model)


def _invoke(args: list) -> 'click.testing.Result':
    """Run the CLI with a CliRunner, returning the result."""
    return CliRunner().invoke(cli, args)


def _parse_recall_json(output: str) -> dict:
    """Extract the JSON object from recall output (strips WARNING logs)."""
    brace = output.index('{')
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(output[brace:])
    return obj


class TestFingerprintRegistry:
    """Provider registry resolution and Fingerprint serialization."""

    def test_unknown_provider_raises_config_error(self, env_file):
        """Unknown MEMMAN_EMBED_PROVIDER raises ConfigError."""
        env_file('MEMMAN_EMBED_PROVIDER', 'bogus')
        from memman.embed import get_client
        with pytest.raises(ConfigError) as excinfo:
            get_client()
        assert 'bogus' in str(excinfo.value)
        assert 'voyage' in str(excinfo.value)

    def test_default_provider_is_voyage(self, monkeypatch):
        """Unset MEMMAN_EMBED_PROVIDER picks voyage and matches its triple."""
        monkeypatch.delenv('MEMMAN_EMBED_PROVIDER', raising=False)
        fp = active_fingerprint()
        assert fp.provider == 'voyage'
        assert fp.model == 'voyage-3-lite'
        assert fp.dim == 512

    def test_fingerprint_round_trip_json(self):
        """Fingerprint to_json/from_json is stable and lossless."""
        fp = Fingerprint(provider='openai', model='text-3-small', dim=1536)
        blob = fp.to_json()
        parsed = json.loads(blob)
        assert parsed == {
            'provider': 'openai', 'model': 'text-3-small', 'dim': 1536}
        assert Fingerprint.from_json(blob) == fp

    def test_fingerprint_from_json_malformed(self):
        """Corrupt JSON raises EmbedFingerprintError, not stdlib errors."""
        with pytest.raises(EmbedFingerprintError):
            Fingerprint.from_json('not-json-at-all')

    def test_fingerprint_from_json_missing_keys(self):
        """Missing required key raises EmbedFingerprintError."""
        with pytest.raises(EmbedFingerprintError):
            Fingerprint.from_json('{"provider": "voyage"}')


class TestFingerprintConsistency:
    """assert_consistent semantics and install-time seeding."""

    @pytest.mark.no_autoseed_fingerprint
    def test_passes_on_match(self, tmp_db):
        """Seeded matching fingerprint -> no error."""
        _seed_voyage(tmp_db)
        assert_consistent(SqliteBackend(tmp_db))

    @pytest.mark.no_autoseed_fingerprint
    def test_raises_on_unseeded(self, tmp_db):
        """No meta.embed_fingerprint -> EmbedFingerprintError with hint."""
        with pytest.raises(EmbedFingerprintError) as excinfo:
            assert_consistent(SqliteBackend(tmp_db))
        msg = str(excinfo.value)
        assert 'embed reembed' in msg
        assert 'initialize' in msg

    @pytest.mark.no_autoseed_fingerprint
    def test_raises_on_mismatch(self, tmp_db):
        """Seeded fingerprint != active -> EmbedFingerprintError with hint."""
        write_fingerprint(SqliteBackend(tmp_db), Fingerprint(provider='openai', model='m', dim=1024))
        with pytest.raises(EmbedFingerprintError) as excinfo:
            assert_consistent(SqliteBackend(tmp_db))
        msg = str(excinfo.value)
        assert 'mismatch' in msg.lower()
        assert 'embed reembed' in msg

    def test_init_default_store_seeds_fingerprint(self, tmp_path):
        """_init_default_store writes meta.embed_fingerprint at create time."""
        from memman.setup.claude import _init_default_store
        from memman.store.db import store_dir

        _init_default_store(str(tmp_path))
        db = open_db(store_dir(str(tmp_path), 'default'))
        try:
            stored = stored_fingerprint(SqliteBackend(db))
        finally:
            db.close()
        assert stored is not None
        assert stored.provider == 'voyage'
        assert stored.dim == 512


@pytest.fixture
def _scheduler_stopped(monkeypatch):
    """Force read_state to STATE_STOPPED for embed reembed tests."""
    from memman.setup import scheduler as sched_mod
    monkeypatch.setattr(
        sched_mod, 'read_state', lambda: sched_mod.STATE_STOPPED)


class TestReembed:
    """embed reembed CLI: initialize, swap, resumability, worker blocking."""

    def test_initializes_unseeded_db(self, tmp_path, _scheduler_stopped):
        """Running embed reembed on an unseeded DB writes the fingerprint
        without re-embedding rows that already match the active client.
        """
        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(db, id='r1', content='hello')
            _seed_row_with_embedding(db, id='r2', content='world')
        finally:
            db.close()

        result = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'reembed'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out['total_scanned'] == 2
        assert out['total_reembedded'] == 0
        assert out['fingerprint']['provider'] == 'voyage'
        assert len(out['stores']) == 1
        assert out['stores'][0]['store'] == 'default'

        db = open_db(sdir)
        try:
            stored = stored_fingerprint(SqliteBackend(db))
            assert stored is not None
            assert stored.provider == 'voyage'
            assert get_meta(db, 'embed_reembed_state') == 'idle'
            assert (
                (get_meta(db, 'embed_reembed_cursor') or '') == '')
        finally:
            db.close()

    def test_dry_run_writes_nothing(self, tmp_path):
        """--dry-run reports counts without DB writes."""
        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(db, id='r1')
        finally:
            db.close()

        result = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'reembed',
            '--dry-run'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out['dry_run'] == 1

        db = open_db(sdir)
        try:
            assert stored_fingerprint(SqliteBackend(db)) is None
        finally:
            db.close()

    def test_rejects_when_scheduler_started(self, tmp_path):
        """Refuses to run when scheduler is started (autouse fixture)."""
        result = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'reembed'])
        assert result.exit_code != 0
        assert 'scheduler stop' in result.output.lower()

    def test_passes_dry_run_when_started(self, tmp_path):
        """--dry-run is allowed even when scheduler is started."""
        result = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'reembed',
            '--dry-run'])
        assert result.exit_code == 0, result.output

    @pytest.mark.no_autoseed_fingerprint
    def test_recall_on_fresh_store_returns_empty(self, tmp_path):
        """Recall on a brand-new store auto-seeds the fingerprint and
        returns empty results, not EmbedFingerprintError.
        """
        result = _invoke([
            '--data-dir', str(tmp_path),
            'recall', 'anything', '--limit', '5'])
        assert result.exit_code == 0, (
            f'recall failed: exit={result.exit_code} '
            f'output={result.output}')
        payload = _parse_recall_json(result.output)
        assert payload['results'] == []

        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            assert stored_fingerprint(SqliteBackend(db)) is not None
        finally:
            db.close()

    @pytest.mark.no_autoseed_fingerprint
    def test_custom_store_recall_on_fresh_returns_empty(self, tmp_path):
        """Recall on a never-used --store custom name auto-seeds and returns empty.
        """
        result = _invoke([
            '--data-dir', str(tmp_path), '--store', 'custom',
            'recall', 'x', '--limit', '5'])
        assert result.exit_code == 0, (
            f'recall failed: exit={result.exit_code} '
            f'output={result.output}')
        payload = _parse_recall_json(result.output)
        assert payload['results'] == []

    @pytest.mark.no_autoseed_fingerprint
    def test_remember_on_fresh_store_seeds_and_drains(self, tmp_path):
        """Remember on a fresh store seeds the fingerprint AND the
        worker drain succeeds.
        """
        result = _invoke([
            '--data-dir', str(tmp_path), 'remember', 'a fresh memory'])
        assert result.exit_code == 0, result.output

        drain_result = _invoke([
            '--data-dir', str(tmp_path),
            'scheduler', 'drain', '--pending'])
        assert drain_result.exit_code == 0, drain_result.output

        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            assert stored_fingerprint(SqliteBackend(db)) is not None
        finally:
            db.close()

    @pytest.mark.no_autoseed_fingerprint
    def test_seed_if_fresh_short_circuits_on_present_insights(self, tmp_path):
        """seed_if_fresh declines to seed when insights are non-empty and
        fingerprint is missing -- corruption, not fresh state.
        """
        from memman.embed.fingerprint import seed_if_fresh
        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(db, id='r1', content='alpha')
            assert stored_fingerprint(SqliteBackend(db)) is None
            wrote = seed_if_fresh(SqliteBackend(db))
            assert wrote is False
            assert stored_fingerprint(SqliteBackend(db)) is None
        finally:
            db.close()

    @pytest.mark.no_autoseed_fingerprint
    def test_seed_if_fresh_raises_on_unavailable_client(
            self, tmp_path, monkeypatch):
        """When the embed client reports unavailable on a fresh store,
        recall surfaces the unavailable-client message, not the
        misleading 'embed reembed' hint.
        """
        monkeypatch.setattr(
            'memman.embed.voyage.Client.available', lambda self: False)
        result = _invoke([
            '--data-dir', str(tmp_path), 'recall', 'x'])
        assert result.exit_code != 0
        assert 'embed reembed' not in result.output

    @pytest.mark.no_autoseed_fingerprint
    def test_recall_blocks_on_corrupted_store(self, tmp_path):
        """Recall raises EmbedFingerprintError when fingerprint is None
        AND insights already exist (real corruption, not a fresh DB).
        """
        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(db, id='r1', content='alpha')
        finally:
            db.close()

        result = _invoke([
            '--data-dir', str(tmp_path), 'recall', 'anything'])
        assert result.exit_code != 0
        assert 'embed reembed' in result.output

    def test_converges_after_provider_swap(
            self, tmp_path, _scheduler_stopped, monkeypatch, env_file):
        """Swap to a stub provider with a different dim, run reembed,
        assert all rows are re-embedded and the fingerprint advances.
        """
        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(db, id='r1', content='alpha')
            _seed_row_with_embedding(db, id='r2', content='beta')
            _seed_voyage(db)
        finally:
            db.close()

        class _StubClient:
            name = 'stub'
            model = 'stub-1024'
            dim = 1024

            def available(self):
                return True

            def embed(self, text):
                return [0.5] * self.dim

            def unavailable_message(self):
                return 'stub down'

        from memman import embed as embed_mod
        monkeypatch.setitem(
            embed_mod.PROVIDERS, 'stub', _StubClient)
        env_file('MEMMAN_EMBED_PROVIDER', 'stub')

        result = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'reembed'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out['total_scanned'] == 2
        assert out['total_reembedded'] == 2
        assert out['fingerprint']['provider'] == 'stub'
        assert out['fingerprint']['dim'] == 1024

        db = open_db(sdir)
        try:
            stored = stored_fingerprint(SqliteBackend(db))
            assert stored.provider == 'stub'
            assert stored.dim == 1024
            rows = db._query(
                'SELECT id, LENGTH(embedding) FROM insights'
                ' WHERE deleted_at IS NULL ORDER BY id').fetchall()
            for _id, blob_len in rows:
                assert blob_len == 1024 * 8
        finally:
            db.close()

    @pytest.mark.no_autoseed_fingerprint
    @pytest.mark.no_auto_drain
    def test_drain_blocks_on_mismatch(self, tmp_path, monkeypatch):
        """Worker drain fails on a fingerprint-mismatched store.

        `remember` itself only enqueues, so it succeeds even against a
        stale store. The fingerprint check happens when the worker opens
        the store DB; the queue row lands in `failed` with a clear error.
        """
        from memman.queue import open_queue_db
        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            write_fingerprint(SqliteBackend(db), Fingerprint(
                    provider='openai', model='other', dim=1024))
        finally:
            db.close()

        result = _invoke([
            '--data-dir', str(tmp_path), 'remember', 'something'])
        assert result.exit_code == 0, result.output

        drain_result = _invoke([
            '--data-dir', str(tmp_path),
            'scheduler', 'drain', '--pending'])
        assert drain_result.exit_code == 0, drain_result.output

        qconn = open_queue_db(str(tmp_path))
        try:
            status, last_err = qconn.execute(
                'select status, last_error from queue order by id desc limit 1'
                ).fetchone()
        finally:
            qconn.close()
        assert status in {'pending', 'failed'}
        assert last_err is not None
        assert 'fingerprint' in last_err.lower()

    def test_embed_status_consistent(self, tmp_path):
        """Embed status reports consistent=True when active matches stored."""
        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            _seed_voyage(db)
        finally:
            db.close()

        result = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'status'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out['consistent'] is True
        assert out['active']['provider'] == 'voyage'
        assert out['stored']['provider'] == 'voyage'

    @pytest.mark.no_autoseed_fingerprint
    def test_embed_status_unseeded_reports_inconsistent(self, tmp_path):
        """Embed status reports consistent=False when DB has no fingerprint."""
        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        db.close()

        result = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'status'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out['consistent'] is False
        assert out['stored'] is None
        assert 'embed reembed' in out['hint']

    def test_doctor_reports_fingerprint_pass(self, tmp_path):
        """check_embed_fingerprint passes when active matches stored."""
        from memman.doctor import check_embed_fingerprint
        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            _seed_voyage(db)
            result = check_embed_fingerprint(SqliteBackend(db))
        finally:
            db.close()
        assert result['status'] == 'pass'
        assert result['detail']['active']['provider'] == 'voyage'
        assert result['detail']['stored']['provider'] == 'voyage'

    @pytest.mark.no_autoseed_fingerprint
    def test_doctor_reports_fingerprint_fail_unseeded(self, tmp_path):
        """check_embed_fingerprint fails on unseeded DB."""
        from memman.doctor import check_embed_fingerprint
        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            result = check_embed_fingerprint(SqliteBackend(db))
        finally:
            db.close()
        assert result['status'] == 'fail'
        assert 'embed reembed' in result['detail']['error']

    @pytest.mark.no_autoseed_fingerprint
    def test_doctor_reports_fingerprint_fail_mismatch(self, tmp_path):
        """check_embed_fingerprint fails when stored != active."""
        from memman.doctor import check_embed_fingerprint
        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            write_fingerprint(SqliteBackend(db), Fingerprint(provider='openai', model='m', dim=1024))
            result = check_embed_fingerprint(SqliteBackend(db))
        finally:
            db.close()
        assert result['status'] == 'fail'
        assert result['detail']['active']['provider'] == 'voyage'
        assert result['detail']['stored']['provider'] == 'openai'
        assert 'embed reembed' in result['detail']['error']

    @pytest.mark.no_autoseed_fingerprint
    def test_embed_status_reports_mismatch(self, tmp_path):
        """Embed status reports consistent=False with the right hint
        when stored fingerprint diverges from the active client.
        """
        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            write_fingerprint(SqliteBackend(db), Fingerprint(provider='openai', model='m', dim=1024))
        finally:
            db.close()

        result = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'status'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out['consistent'] is False
        assert out['stored']['provider'] == 'openai'
        assert out['active']['provider'] == 'voyage'
        assert 'scheduler stop' in out['hint']
        assert 'embed reembed' in out['hint']

    def test_idempotent_on_repeat(self, tmp_path, _scheduler_stopped):
        """Running reembed twice with the same active provider:
        second run must report total_reembedded=0.
        """
        from memman.store.db import store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(db, id='r1', content='hello',
                                     model='old-model')
        finally:
            db.close()

        first = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'reembed'])
        assert first.exit_code == 0, first.output
        first_out = json.loads(first.output)
        assert first_out['total_reembedded'] == 1

        second = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'reembed'])
        assert second.exit_code == 0, second.output
        second_out = json.loads(second.output)
        assert second_out['total_reembedded'] == 0

    def test_blocks_when_provider_unavailable(
            self, tmp_path, _scheduler_stopped, monkeypatch, env_file):
        """If the active client's available() returns False, reembed
        refuses to run with the unavailable_message().
        """
        class _UnavailableClient:
            name = 'fake'
            model = 'fake-model'
            dim = 1

            def available(self):
                return False

            def embed(self, text):
                return [0.0]

            def unavailable_message(self):
                return 'fake provider down: set FAKE_API_KEY'

        from memman import embed as embed_mod
        monkeypatch.setitem(
            embed_mod.PROVIDERS, 'fake', _UnavailableClient)
        env_file('MEMMAN_EMBED_PROVIDER', 'fake')

        result = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'reembed'])
        assert result.exit_code != 0
        assert 'fake provider down' in result.output

    def test_resumable_from_cursor(
            self, tmp_path, _scheduler_stopped, monkeypatch):
        """Pre-seed state=in_progress with a cursor past the first row;
        re-running reembed must skip the first row.
        """
        from memman.store.db import set_meta, store_dir
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(db, id='r1', content='alpha',
                                     model='old-model')
            _seed_row_with_embedding(db, id='r2', content='beta',
                                     model='old-model')
            set_meta(db, 'embed_reembed_state', 'in_progress')
            set_meta(db, 'embed_reembed_cursor', 'r1')
        finally:
            db.close()

        embed_calls = []

        class _StubClient:
            name = 'voyage'
            model = 'voyage-3-lite'
            dim = 512

            def available(self):
                return True

            def embed(self, text):
                embed_calls.append(text)
                return [0.0] * 512

            def unavailable_message(self):
                return 'down'

        from memman import embed as embed_mod
        monkeypatch.setitem(
            embed_mod.PROVIDERS, 'voyage', _StubClient)

        result = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'reembed'])
        assert result.exit_code == 0, result.output
        assert embed_calls == ['beta']

    def test_worker_blocks_on_mismatch(self, tmp_path, monkeypatch):
        """The worker asserts fingerprint consistency before processing rows.

        Enforced when `_StoreContext` opens the store DB at the start
        of a drain rather than per-row inside `_process_queue_row`.
        """
        from memman.cli import _StoreContext
        from memman.store.db import store_dir

        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            write_fingerprint(SqliteBackend(db), Fingerprint(provider='openai', model='m', dim=1024))
        finally:
            db.close()

        with pytest.raises(EmbedFingerprintError):
            _StoreContext(sdir)
