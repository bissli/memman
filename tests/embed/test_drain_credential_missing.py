"""Drain row-level visibility of missing embedder credentials.

When a store is fingerprinted to a provider whose creds are absent,
`_StoreContext` does not crash. Queue rows fail visibly with
`EmbedCredentialError` and the drain emits a structured trace event
`embedder_credential_missing` so the operator can detect the
condition without scraping queue failure counts.
"""

import pytest
from click.testing import CliRunner
from memman import embed as embed_mod
from memman.cli import _StoreContext, cli
from memman.embed.fingerprint import Fingerprint, write_fingerprint
from memman.exceptions import ConfigError, EmbedCredentialError
from memman.queue import open_queue_db
from memman.store.db import open_db, store_dir
from memman.store.sqlite import SqliteBackend


def _seed_fingerprint(sdir: str, fp: Fingerprint) -> None:
    """Write a fingerprint to the store DB.
    """
    db = open_db(sdir)
    try:
        write_fingerprint(SqliteBackend(db), fp)
    finally:
        db.close()


class _UncredentialedStub:
    """Embed client whose constructor raises ConfigError, as openrouter
    does when its key is absent.
    """

    name = 'unfunded-stub'

    def __init__(self):
        raise ConfigError(
            'unfunded-stub provider has no credentials in this process')


class TestCredentialMissingFailureMode:
    """Missing creds for a fingerprinted store produce a clean failure.
    """

    @pytest.fixture
    def _registered_unfunded(self, monkeypatch):
        """Register the `unfunded-stub` provider for the test's lifetime.
        """
        monkeypatch.setitem(
            embed_mod.PROVIDERS, 'unfunded-stub', _UncredentialedStub)

    @pytest.mark.no_autoseed_fingerprint
    @pytest.mark.no_auto_drain
    def test_storectx_opens_with_placeholder_when_creds_missing(
            self, tmp_path, _registered_unfunded):
        """A store fingerprinted to an uncredentialed provider still opens.

        `_StoreContext` succeeds with a placeholder client.

        Mutation: `_StoreContext` letting the provider's `ConfigError`
            propagate, so a store without creds cannot be opened.
        Oracle: the placeholder's `name`, `available()` False, and
            `EmbedCredentialError` from `embed()`.
        """
        sdir = store_dir(str(tmp_path), 'unfunded')
        _seed_fingerprint(sdir, Fingerprint(
            provider='unfunded-stub', model='stub-1024', dim=1024))

        ctx = _StoreContext('unfunded', str(tmp_path))
        try:
            assert ctx.ec.name == 'unfunded-stub'
            assert ctx.ec.available() is False
            with pytest.raises(EmbedCredentialError):
                ctx.ec.embed('hello')
        finally:
            ctx.close()

    @pytest.mark.no_autoseed_fingerprint
    @pytest.mark.no_auto_drain
    def test_drain_marks_row_failed_on_credential_error(
            self, tmp_path, _registered_unfunded):
        """A queued row for a store with no credential records the error.

        Mutation: the drain swallowing the credential error and marking
            the row `done`, or crashing instead of recording it.
        Oracle: the queue row's `status` and `last_error` text.
        """
        runner = CliRunner()
        data_dir = str(tmp_path / 'memman')
        sdir = store_dir(data_dir, 'unfunded')
        _seed_fingerprint(sdir, Fingerprint(
            provider='unfunded-stub', model='stub-1024', dim=1024))

        # The content must be substantive enough that LLM extraction
        # yields a fact. Otherwise the drain never tries to embed, hides
        # the credential error, and the row ends 'done'.
        result = runner.invoke(cli, [
            '--data-dir', data_dir, '--store', 'unfunded',
            'remember',
            ('Production Redis instance evicts keys via the allkeys-lfu'
             ' policy with a 16GB memory budget per shard.')])
        assert result.exit_code == 0, result.output

        drain_result = runner.invoke(cli, [
            '--data-dir', data_dir,
            'scheduler', 'drain'])
        assert drain_result.exit_code == 0, drain_result.output

        qconn = open_queue_db(data_dir)
        try:
            status, last_err = qconn.execute(
                'select status, last_error from queue'
                ' order by id desc limit 1').fetchone()
        finally:
            qconn.close()
        assert status in {'pending', 'failed'}
        assert last_err is not None
        assert ('EmbedCredentialError' in last_err
                or 'cannot run' in last_err.lower())
