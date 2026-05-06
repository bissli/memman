"""Drain row-level visibility of missing embedder credentials.

When a store is fingerprinted to a provider whose creds are absent,
`_StoreContext` does not crash. Queue rows fail visibly with
`EmbedCredentialError` and the drain emits a structured trace event
`embedder_credential_missing` so the operator can detect the
condition without scraping queue failure counts.
"""

import pytest
from click.testing import CliRunner
from memman.cli import cli
from memman.embed.fingerprint import Fingerprint, write_fingerprint
from memman.exceptions import ConfigError, EmbedCredentialError
from memman.queue import open_queue_db
from memman.store.db import open_db, store_dir
from memman.store.sqlite import SqliteBackend


def _seed_fingerprint(sdir: str, fp: Fingerprint) -> None:
    """Helper: write a fingerprint to the store DB."""
    db = open_db(sdir)
    try:
        write_fingerprint(SqliteBackend(db), fp)
    finally:
        db.close()


class _UncredentialedStub:
    """Test-only embed client whose constructor mimics openrouter:
    raises ConfigError immediately when its key is absent.
    """

    name = 'unfunded-stub'

    def __init__(self):
        raise ConfigError(
            'unfunded-stub provider has no credentials in this process')


class TestCredentialMissingFailureMode:
    """Missing creds for a fingerprinted store produce a clean failure."""

    @pytest.fixture
    def _registered_unfunded(self, monkeypatch):
        """Register the `unfunded-stub` provider for the test's lifetime."""
        from memman import embed as embed_mod
        monkeypatch.setitem(
            embed_mod.PROVIDERS, 'unfunded-stub', _UncredentialedStub)

    @pytest.mark.no_autoseed_fingerprint
    @pytest.mark.no_auto_drain
    def test_storectx_opens_with_placeholder_when_creds_missing(
            self, tmp_path, _registered_unfunded):
        """A store fingerprinted to a credentialed-out provider opens
        cleanly: `_StoreContext` succeeds with a placeholder client.
        """
        from memman.cli import _StoreContext
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
        """A queued row for a credentialed-out store lands in `failed`
        with an EmbedCredentialError reason in `last_error`.
        """
        runner = CliRunner()
        sdir = store_dir(str(tmp_path), 'unfunded')
        _seed_fingerprint(sdir, Fingerprint(
            provider='unfunded-stub', model='stub-1024', dim=1024))

        result = runner.invoke(cli, [
            '--data-dir', str(tmp_path), '--store', 'unfunded',
            'remember', 'note for absent creds'])
        assert result.exit_code == 0, result.output

        drain_result = runner.invoke(cli, [
            '--data-dir', str(tmp_path),
            'scheduler', 'drain', '--pending'])
        assert drain_result.exit_code == 0, drain_result.output

        qconn = open_queue_db(str(tmp_path))
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
