"""Tests for `memman.session.active_store` error wrapping.

`active_store` opens the per-store backend via
`factory.open_backend` and runs the embedding fingerprint pre-flight.
Both layers can raise `ConfigError`; the helper must wrap both as
`click.ClickException` so the CLI emits a clean exit message instead
of a bare traceback.
"""

import pytest
from click import ClickException
from memman.session import active_store


def test_active_store_wraps_backend_config_error_from_open(
        tmp_path, env_file, monkeypatch):
    """A misconfigured store (postgres backend, no DSN) raises
    `ClickException` instead of leaking the backend `ConfigError`.

    Pre-fix: `open_backend` was called outside the try block, so the
    raw backend `ConfigError` propagated as a traceback. Post-fix:
    `store.errors.ConfigError` subclasses `exceptions.ConfigError`
    and the `try/except ConfigError` wraps the open call.
    """
    env_file('MEMMAN_BACKEND_oops', 'postgres')
    data_dir = str(tmp_path / 'memman')
    with pytest.raises(ClickException) as exc:
        with active_store(
                data_dir=data_dir, store='oops', unchecked=True):
            pass
    assert 'oops' in str(exc.value).lower() or 'dsn' in str(exc.value).lower()


def test_active_store_wraps_runtime_config_error_from_assert(
        tmp_path, monkeypatch):
    """A `memman.exceptions.ConfigError` raised by the fingerprint
    assert path is wrapped as `ClickException`.

    Locks the existing post-open catch geometry so the F.1 refactor
    doesn't regress it.
    """
    from memman.exceptions import ConfigError as RuntimeConfigError

    def _raise(*args, **kwargs):
        raise RuntimeConfigError('forced')

    monkeypatch.setattr(
        'memman.embed.fingerprint.assert_consistent', _raise)
    data_dir = str(tmp_path / 'memman')
    with pytest.raises(ClickException) as exc:
        with active_store(data_dir=data_dir, store='default'):
            pass
    assert 'forced' in str(exc.value)
