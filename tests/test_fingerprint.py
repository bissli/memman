"""Tests for the embed fingerprint flow.

Covers: the default embed model, Fingerprint serialization,
install-time seeding, the embed reembed sweep (initialize, swap,
resumability, scheduler-stopped gate), and per-store sovereignty
(worker + recall bind to each store's stored fingerprint regardless
of env-active model).
"""

import json

import pytest
from click.testing import CliRunner
from memman.cli import _StoreContext, cli
from memman.doctor import check_embed_fingerprint
from memman.embed import get_client
from memman.embed.fingerprint import Fingerprint, bound_embedder
from memman.embed.fingerprint import seed_default_fingerprint, seed_if_fresh
from memman.embed.fingerprint import stored_fingerprint, write_fingerprint
from memman.embed.vector import serialize_vector
from memman.exceptions import EmbedFingerprintError
from memman.setup import scheduler as sched_mod
from memman.setup.claude import _init_default_store
from memman.store.db import DB, get_meta, open_db, set_meta, store_dir
from memman.store.node import insert_insight, update_embedding
from memman.store.sqlite import SqliteBackend, open_sqlite_backend
from tests.conftest import EMBEDDING_DIM, make_insight


def _seed_voyage(db: DB) -> None:
    """Helper: write the canonical voyage fingerprint to meta.
    """
    write_fingerprint(SqliteBackend(db), Fingerprint(
            model='voyage-3-lite', dim=512))


def _seed_row_with_embedding(db: DB, *, id: str, content: str = 'x',
                             model: str = 'voyage-3-lite',
                             dim: int = 512) -> None:
    """Seed a row with a synthetic embedding of given model+dim.
    """
    insight = make_insight(
        id=id, content=content, embedding_model=model)
    insert_insight(db, insight)
    fake_vec = [0.1] * dim
    update_embedding(db, id, serialize_vector(fake_vec), model)


def _invoke(args: list) -> 'click.testing.Result':
    """Run the CLI with a CliRunner, returning the result.
    """
    return CliRunner().invoke(cli, args)


class TestFingerprintRegistry:
    """Provider registry resolution and Fingerprint serialization.
    """

    def test_default_model_is_voyage_lite(self, monkeypatch):
        """Verify an unset MEMMAN_EMBED_MODEL yields the shipped default model.

        Mutation: The shipped default embed model drifting from
            voyageai/voyage-4-lite.
        Oracle: Hand-written literal model name.
        """
        monkeypatch.delenv('MEMMAN_EMBED_MODEL', raising=False)
        fp = seed_default_fingerprint()
        assert fp.model == 'voyageai/voyage-4-lite'

    def test_default_fingerprint_carries_the_probed_dim(self):
        """Verify the seed fingerprint's dim comes from the model's reply.

        Mutation: seeding from a client that never probed, so dim is 0
            and a fresh Postgres schema falls back to a fixed
            `vector(N)` width the model does not return.
        Oracle: the mocked embed reply width, `EMBEDDING_DIM`.
        """
        assert seed_default_fingerprint().dim == EMBEDDING_DIM

    def test_fingerprint_round_trip_json(self):
        """Verify Fingerprint to_json/from_json is stable and lossless.

        Mutation: to_json dropping a field or renaming a key, or from_json
            coercing dim to a string.
        Oracle: Hand-written JSON dict and equality with the original.
        """
        fp = Fingerprint(model='text-3-small', dim=1536)
        blob = fp.to_json()
        parsed = json.loads(blob)
        assert parsed == {'model': 'text-3-small', 'dim': 1536}
        assert Fingerprint.from_json(blob) == fp

    def test_fingerprint_from_json_malformed(self):
        """Verify corrupt JSON raises EmbedFingerprintError.

        Mutation: from_json narrowing its except clause so JSONDecodeError
            escapes.
        Oracle: pytest.raises on a non-JSON string.
        """
        with pytest.raises(EmbedFingerprintError):
            Fingerprint.from_json('not-json-at-all')

    def test_fingerprint_from_json_missing_keys(self):
        """Verify a missing required key raises EmbedFingerprintError.

        Mutation: from_json dropping KeyError from its except clause, or
            defaulting the missing model and dim.
        Oracle: pytest.raises on JSON that holds only `model`.
        """
        with pytest.raises(EmbedFingerprintError):
            Fingerprint.from_json('{"model": "voyage-3-lite"}')


class TestFingerprintConsistency:
    """Install-time seeding and bound_embedder behavior.
    """

    @pytest.mark.no_autoseed_fingerprint
    def test_bound_embedder_raises_on_unseeded(self, tmp_db):
        """Verify bound_embedder on an unseeded store raises with a fix hint.

        Mutation: bound_embedder falling back to the env-active client for an
            unseeded store, or naming the SQLite-only `embed reembed`.
        Oracle: The `embed swap --to` command, which runs on both backends.
        """
        with pytest.raises(EmbedFingerprintError) as excinfo:
            bound_embedder(SqliteBackend(tmp_db))
        assert 'embed swap --to' in str(excinfo.value)

    def test_init_default_store_seeds_fingerprint(self, tmp_path):
        """Verify _init_default_store writes the fingerprint at creation.

        Mutation: _init_default_store creating the store without calling
            seed_if_fresh.
        Oracle: The fingerprint read back from a reopened store on disk.
        """
        _init_default_store(str(tmp_path))
        db = open_db(store_dir(str(tmp_path), 'default'))
        try:
            stored = stored_fingerprint(SqliteBackend(db))
        finally:
            db.close()
        assert stored is not None
        assert stored.model == 'voyageai/voyage-4-lite'
        assert stored.dim == 512


@pytest.fixture
def _scheduler_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force read_state to STATE_STOPPED for embed reembed tests.
    """
    monkeypatch.setattr(
        sched_mod, 'read_state', lambda: sched_mod.STATE_STOPPED)


class TestReembed:
    """embed reembed CLI: initialize, swap, resumability, worker blocking.
    """

    def test_initializes_unseeded_db(self, tmp_path, _scheduler_stopped):
        """Verify reembed seeds an unseeded store, skipping matching rows.

        Mutation: The row match test in _reembed_one_store comparing the wrong
            field, so matching rows are re-embedded; or the final write leaving
            state in_progress or a stale cursor.
        Oracle: Two seeded rows: total_scanned 2, total_reembedded 0, meta
            state idle, cursor empty.
        """
        data_dir = str(tmp_path / 'memman')
        sdir = store_dir(data_dir, 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(
                db, id='r1', content='hello',
                model='voyageai/voyage-4-lite')
            _seed_row_with_embedding(
                db, id='r2', content='world',
                model='voyageai/voyage-4-lite')
        finally:
            db.close()

        result = _invoke([
            '--data-dir', data_dir, 'embed', 'reembed'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out['total_scanned'] == 2
        assert out['total_reembedded'] == 0
        assert out['fingerprint']['model'] == 'voyageai/voyage-4-lite'
        assert len(out['stores']) == 1
        assert out['stores'][0]['store'] == 'default'

        db = open_db(sdir)
        try:
            stored = stored_fingerprint(SqliteBackend(db))
            assert stored is not None
            assert stored.model == 'voyageai/voyage-4-lite'
            assert get_meta(db, 'embed_reembed_state') == 'idle'
            assert (
                (get_meta(db, 'embed_reembed_cursor') or '') == '')
        finally:
            db.close()

    def test_dry_run_writes_nothing(self, tmp_path):
        """Verify --dry-run reports counts and writes nothing.

        Mutation: The dry-run branch still writing the fingerprint or the
            in_progress state.
        Oracle: stored_fingerprint is still None after the run.
        """
        data_dir = str(tmp_path / 'memman')
        sdir = store_dir(data_dir, 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(db, id='r1')
        finally:
            db.close()

        result = _invoke([
            '--data-dir', data_dir, 'embed', 'reembed',
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
        """Verify reembed refuses to run while the scheduler is started.

        Mutation: Removing the _require_stopped call from embed_reembed.
        Oracle: Non-zero exit and the `scheduler stop` instruction in the
            output.
        """
        result = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'reembed'])
        assert result.exit_code != 0
        assert 'scheduler stop' in result.output.lower()

    def test_passes_dry_run_when_started(self, tmp_path):
        """Verify --dry-run runs while the scheduler is started.

        Mutation: Calling _require_stopped before the dry_run check.
        Oracle: Exit code 0 under the autouse started-scheduler fixture.
        """
        result = _invoke([
            '--data-dir', str(tmp_path / 'memman'), 'embed', 'reembed',
            '--dry-run'])
        assert result.exit_code == 0, result.output

    @pytest.mark.no_autoseed_fingerprint
    def test_recall_on_fresh_store_returns_empty(self, tmp_path):
        """Verify recall on a brand-new store auto-seeds and prints nothing.

        Mutation: An unhandled EmbedFingerprintError reaching the CLI, or a
            page line printed for a store holding no row.
        Oracle: Exit 0 with empty stdout (the zero-anchor warning goes to
            stderr), and a fingerprint on disk.
        """
        data_dir = str(tmp_path / 'memman')
        open_sqlite_backend('default', data_dir, create=True).close()
        result = _invoke([
            '--data-dir', data_dir,
            'recall', 'anything', '--limit', '5'])
        assert result.exit_code == 0, (
            f'recall failed: exit={result.exit_code} '
            f'output={result.output}')
        assert result.stdout == ''

        sdir = store_dir(data_dir, 'default')
        db = open_db(sdir)
        try:
            assert stored_fingerprint(SqliteBackend(db)) is not None
        finally:
            db.close()

    @pytest.mark.no_autoseed_fingerprint
    def test_custom_store_recall_on_fresh_returns_empty(self, tmp_path):
        """Verify recall on a never-used --store auto-seeds and is empty.

        Mutation: An unhandled EmbedFingerprintError reaching the CLI on a
            never-used store name.
        Oracle: Exit 0 with empty output.
        """
        open_sqlite_backend(
            'custom', str(tmp_path / 'memman'), create=True).close()
        result = _invoke([
            '--data-dir', str(tmp_path / 'memman'), '--store', 'custom',
            'recall', 'x', '--limit', '5'])
        assert result.exit_code == 0, (
            f'recall failed: exit={result.exit_code} '
            f'output={result.output}')
        assert result.stdout == ''

    @pytest.mark.no_autoseed_fingerprint
    def test_remember_on_fresh_store_seeds_and_drains(self, tmp_path):
        """Verify remember on a fresh store seeds it and the drain works.

        Mutation: The write path skipping seed_if_fresh, so the drain fails on
            a missing fingerprint.
        Oracle: Exit 0 for remember and for drain, and a stored fingerprint.
        """
        data_dir = str(tmp_path / 'memman')
        open_sqlite_backend('default', data_dir, create=True).close()
        result = _invoke([
            '--data-dir', data_dir, 'remember', 'a fresh memory'])
        assert result.exit_code == 0, result.output

        drain_result = _invoke([
            '--data-dir', data_dir,
            'scheduler', 'drain'])
        assert drain_result.exit_code == 0, drain_result.output

        sdir = store_dir(data_dir, 'default')
        db = open_db(sdir)
        try:
            assert stored_fingerprint(SqliteBackend(db)) is not None
        finally:
            db.close()

    @pytest.mark.no_autoseed_fingerprint
    def test_seed_if_fresh_short_circuits_on_present_insights(self, tmp_path):
        """Verify seed_if_fresh declines to seed a store that holds rows.

        A missing fingerprint beside existing insights is corruption, not a
        fresh store.

        Mutation: Dropping the count_total guard, so seeding hides corruption.
        Oracle: Return value False and the fingerprint still None.
        """
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(db, id='r1', content='alpha')
            assert stored_fingerprint(SqliteBackend(db)) is None
            wrote = seed_if_fresh(SqliteBackend(db), get_client())
            assert wrote is False
            assert stored_fingerprint(SqliteBackend(db)) is None
        finally:
            db.close()

    @pytest.mark.no_autoseed_fingerprint
    def test_seed_if_fresh_raises_on_unavailable_client(
            self, tmp_path, monkeypatch):
        """Verify an unavailable client surfaces its own message on recall.

        Mutation: seed_if_fresh swallowing the unavailable-client error and
            falling through to the corrupted-store message.
        Oracle: The client's `is not reachable` text, and no `embed
            reembed` hint.
        """
        data_dir = str(tmp_path / 'memman')
        open_sqlite_backend('default', data_dir, create=True).close()
        monkeypatch.setattr(
            'memman.embed.client.Client.available', lambda self: False)
        result = _invoke([
            '--data-dir', data_dir, 'recall', 'x'])
        assert result.exit_code != 0
        assert 'is not reachable' in result.output
        assert 'embed reembed' not in result.output

    @pytest.mark.no_autoseed_fingerprint
    def test_recall_blocks_on_corrupted_store(self, tmp_path):
        """Verify recall fails on a populated store with no fingerprint.

        Mutation: _StoreContext seeding or ignoring a missing fingerprint when
            insights exist.
        Oracle: Non-zero exit and the `embed swap --to` hint in the output.
        """
        data_dir = str(tmp_path / 'memman')
        sdir = store_dir(data_dir, 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(db, id='r1', content='alpha')
        finally:
            db.close()

        result = _invoke([
            '--data-dir', data_dir, 'recall', 'anything'])
        assert result.exit_code != 0
        assert 'embed swap --to' in result.output

    def test_converges_after_model_swap(
            self, tmp_path, _scheduler_stopped, monkeypatch, env_file):
        """Verify reembed after a model swap converges every row.

        Mutation: The match test treating rows with a different model or dim as
            current, or the fingerprint not advancing.
        Oracle: Two rows re-embedded, stored model stub-1024, and each blob 1024
            * 8 bytes.
        """
        data_dir = str(tmp_path / 'memman')
        sdir = store_dir(data_dir, 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(db, id='r1', content='alpha')
            _seed_row_with_embedding(db, id='r2', content='beta')
            _seed_voyage(db)
        finally:
            db.close()

        class _StubClient:
            model = 'stub-1024'
            dim = 1024

            def available(self):
                return True

            def embed(self, text):
                return [0.5] * self.dim

            def unavailable_message(self):
                return 'stub down'

        monkeypatch.setattr('memman.cli.get_client', _StubClient)

        result = _invoke([
            '--data-dir', data_dir, 'embed', 'reembed'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out['total_scanned'] == 2
        assert out['total_reembedded'] == 2
        assert out['fingerprint']['model'] == 'stub-1024'
        assert out['fingerprint']['dim'] == 1024

        db = open_db(sdir)
        try:
            stored = stored_fingerprint(SqliteBackend(db))
            assert stored.model == 'stub-1024'
            assert stored.dim == 1024
            rows = db._query(
                'select id, length(embedding) from insights'
                ' where deleted_at is null order by id').fetchall()
            for _id, blob_len in rows:
                assert blob_len == 1024 * 8
        finally:
            db.close()

    @pytest.mark.no_autoseed_fingerprint
    @pytest.mark.no_auto_drain
    def test_drain_binds_per_store_fingerprint(self, tmp_path, monkeypatch):
        """Verify _StoreContext binds the stored fingerprint.

        The store fingerprint is the runtime authority for which client embeds.

        Mutation: _StoreContext binding get_client() instead of bound_embedder.
        Oracle: A stored `other` fingerprint against the voyage env
            default.
        """
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            write_fingerprint(SqliteBackend(db), Fingerprint(
                    model='other', dim=1024))
        finally:
            db.close()

        ctx = _StoreContext('default', str(tmp_path))
        try:
            assert ctx.ec.model == 'other'
        finally:
            ctx.close()

    def test_embed_status_reports_stored_fingerprint(self, tmp_path):
        """Verify embed status reports the stored fingerprint.

        Mutation: embed_status reading the env-active fingerprint, or omitting
            credentials_available.
        Oracle: Seeded voyage-3-lite/512 values and credentials_available True.
        """
        data_dir = str(tmp_path / 'memman')
        sdir = store_dir(data_dir, 'default')
        db = open_db(sdir)
        try:
            _seed_voyage(db)
        finally:
            db.close()

        result = _invoke([
            '--data-dir', data_dir, 'embed', 'status'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out['stored']['model'] == 'voyage-3-lite'
        assert out['stored']['dim'] == 512
        assert out['credentials_available'] is True

    @pytest.mark.no_autoseed_fingerprint
    def test_embed_status_unseeded_reports_no_fingerprint(self, tmp_path):
        """Verify embed status on an unseeded store reports stored=None.

        Mutation: embed_status inventing a stored value or dropping the hint.
        Oracle: stored is None and the hint names `embed swap --to`.
        """
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        db.close()

        result = _invoke([
            '--data-dir', str(tmp_path), 'embed', 'status'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out['stored'] is None
        assert 'memman --store default embed swap --to' in out['hint']

    def test_doctor_reports_fingerprint_pass(self, tmp_path):
        """Verify the doctor check passes with a fingerprint and creds.

        Mutation: The check reporting fail or dropping the stored detail.
        Oracle: Status pass, the seeded voyage model, and credentials True.
        """
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            _seed_voyage(db)
            result = check_embed_fingerprint(SqliteBackend(db))
        finally:
            db.close()
        assert result['status'] == 'pass'
        assert result['detail']['stored']['model'] == 'voyage-3-lite'
        assert result['detail']['credentials_available'] is True

    @pytest.mark.no_autoseed_fingerprint
    def test_doctor_reports_fingerprint_pass_when_empty_and_unseeded(
            self, tmp_path):
        """Verify an empty store with no fingerprint passes the check.

        A fresh `memman install` has no insights and no fingerprint. The first
        write seeds it.

        Mutation: The empty-store branch reporting fail, which flags a
            regression on every fresh install.
        Oracle: Status pass on a store that holds no row.
        """
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            result = check_embed_fingerprint(SqliteBackend(db))
        finally:
            db.close()
        assert result['status'] == 'pass'

    @pytest.mark.no_autoseed_fingerprint
    def test_doctor_reports_fingerprint_fail_when_populated_and_unseeded(
            self, tmp_path):
        """Verify a populated store with no fingerprint fails the check.

        Mutation: The check passing when count_active is above zero and the
            fingerprint is missing.
        Oracle: Status fail and the `embed swap --to` fix in detail.error.
        """
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(db, id='r1')
            result = check_embed_fingerprint(SqliteBackend(db))
        finally:
            db.close()
        assert result['status'] == 'fail'
        assert 'embed swap --to' in result['detail']['error']

    def test_doctor_fingerprint_fail_names_swap_when_model_unreachable(
            self, tmp_path, monkeypatch):
        """Verify an unreachable stored model points doctor at `embed swap`.

        Mutation: The failure text naming only the key and endpoint, so an
            operator whose endpoint dropped the model has no next step.
        Oracle: The client's `is not reachable` text plus `embed swap --to`.
        """
        monkeypatch.setattr(
            'memman.embed.client.Client.available', lambda self: False)
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            _seed_voyage(db)
            result = check_embed_fingerprint(SqliteBackend(db))
        finally:
            db.close()
        assert result['status'] == 'fail'
        assert 'is not reachable' in result['detail']['error']
        assert 'embed swap --to' in result['detail']['error']

    def test_embed_status_unreachable_hint_names_swap(
            self, tmp_path, monkeypatch):
        """Verify embed status names `embed swap` for an unreachable model.

        Mutation: The hint naming only the key and endpoint.
        Oracle: The `memman --store default embed swap --to` command.
        """
        monkeypatch.setattr(
            'memman.embed.client.Client.available', lambda self: False)
        data_dir = str(tmp_path / 'memman')
        sdir = store_dir(data_dir, 'default')
        db = open_db(sdir)
        try:
            _seed_voyage(db)
        finally:
            db.close()

        result = _invoke([
            '--data-dir', data_dir, 'embed', 'status'])
        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert 'memman --store default embed swap --to' in out['hint']

    def test_idempotent_on_repeat(self, tmp_path, _scheduler_stopped):
        """Verify a second reembed with the same model re-embeds nothing.

        Mutation: The first run not persisting the new model on each row, so
            the second run re-embeds again.
        Oracle: total_reembedded 1 on the first run, 0 on the second.
        """
        data_dir = str(tmp_path / 'memman')
        sdir = store_dir(data_dir, 'default')
        db = open_db(sdir)
        try:
            _seed_row_with_embedding(db, id='r1', content='hello',
                                     model='old-model')
        finally:
            db.close()

        first = _invoke([
            '--data-dir', data_dir, 'embed', 'reembed'])
        assert first.exit_code == 0, first.output
        first_out = json.loads(first.output)
        assert first_out['total_reembedded'] == 1

        second = _invoke([
            '--data-dir', data_dir, 'embed', 'reembed'])
        assert second.exit_code == 0, second.output
        second_out = json.loads(second.output)
        assert second_out['total_reembedded'] == 0

    def test_blocks_when_client_unavailable(
            self, tmp_path, _scheduler_stopped, monkeypatch, env_file):
        """Verify reembed refuses to run when the client is unavailable.

        Mutation: Removing the ec.available() check in embed_reembed.
        Oracle: Non-zero exit and the client unavailable_message text.
        """
        class _UnavailableClient:
            model = 'fake-model'
            dim = 1

            def available(self):
                return False

            def embed(self, text):
                return [0.0]

            def unavailable_message(self):
                return 'fake provider down: set FAKE_API_KEY'

        monkeypatch.setattr('memman.cli.get_client', _UnavailableClient)

        result = _invoke([
            '--data-dir', str(tmp_path / 'memman'), 'embed', 'reembed'])
        assert result.exit_code != 0
        assert 'fake provider down' in result.output

    def test_resumable_from_cursor(
            self, tmp_path, _scheduler_stopped, monkeypatch):
        """Verify reembed resumes past the stored cursor.

        State is pre-seeded to in_progress with the cursor at the first row.

        Mutation: Ignoring the stored cursor, or resetting it, so the first row
            is re-embedded.
        Oracle: The stub client saw only the text `beta`.
        """
        data_dir = str(tmp_path / 'memman')
        sdir = store_dir(data_dir, 'default')
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
            model = 'voyage-3-lite'
            dim = 512

            def available(self):
                return True

            def embed(self, text):
                embed_calls.append(text)
                return [0.0] * 512

            def unavailable_message(self):
                return 'down'

        monkeypatch.setattr('memman.cli.get_client', _StubClient)

        result = _invoke([
            '--data-dir', data_dir, 'embed', 'reembed'])
        assert result.exit_code == 0, result.output
        assert embed_calls == ['beta']

    def test_worker_binds_store_fingerprint(self, tmp_path, monkeypatch):
        """Verify _StoreContext binds the stored fingerprint.

        Mutation: _StoreContext binding get_client() instead of bound_embedder.
        Oracle: A stored `m` fingerprint against the voyage env default.
        """
        sdir = store_dir(str(tmp_path), 'default')
        db = open_db(sdir)
        try:
            write_fingerprint(SqliteBackend(db), Fingerprint(model='m', dim=1024))
        finally:
            db.close()

        ctx = _StoreContext('default', str(tmp_path))
        try:
            assert ctx.ec.model == 'm'
        finally:
            ctx.close()
