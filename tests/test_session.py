"""Tests for `memman.session.active_store` per-store sovereignty.

`active_store` opens the per-store backend via
`factory.open_backend` and resolves the embedder from each store's
own stored fingerprint via `bound_embedder`. The CLI boundary wraps
backend `ConfigError` as `click.ClickException` so misconfigured
stores produce a clean exit message instead of a bare traceback.
"""

import pytest
from click import ClickException
from memman.embed import fingerprint as fp_mod
from memman.embed import registry as ec_registry
from memman.session import active_store
from memman.store.factory import open_backend


def test_active_store_wraps_backend_config_error_from_open(
        tmp_path, env_file, monkeypatch):
    """A misconfigured store (postgres backend, no DSN) raises
    `ClickException` instead of leaking the backend `ConfigError`.

    Mutation: dropping `ConfigError` from the caught tuple in
    `active_store`, so the missing-DSN error escapes as a traceback.
    Oracle: `pytest.raises(ClickException)` naming the store or the DSN.
    """
    env_file('MEMMAN_BACKEND_oops', 'postgres')
    data_dir = str(tmp_path / 'memman')
    with pytest.raises(ClickException) as exc, active_store(
            data_dir=data_dir, store='oops', unchecked=True):
        pass
    assert 'oops' in str(exc.value).lower() or 'dsn' in str(exc.value).lower()


def test_active_store_wraps_backend_error_from_open(tmp_path):
    """A corrupt store database raises `ClickException`, not `BackendError`.

    Mutation: dropping `BackendError` from the caught tuple in
        `active_store`, so a store-open failure escapes the CLI seam
        and prints a Python traceback.
    Oracle: `pytest.raises(ClickException)` discriminates -- an
        unwrapped `BackendError` is not a `ClickException` and escapes
        it -- plus the path named in the message.
    """
    data_dir = tmp_path / 'memman'
    store_dir = data_dir / 'data' / 'broken'
    store_dir.mkdir(parents=True)
    (store_dir / 'memman.db').write_bytes(b'not a sqlite database' * 8)
    with pytest.raises(ClickException) as excinfo, active_store(
            data_dir=str(data_dir), store='broken', unchecked=True):
        pass
    assert 'memman.db' in str(excinfo.value)


def test_active_store_yields_store_bound_ec_per_store(
        tmp_path, env_file, monkeypatch):
    """Two stores with different stored fingerprints in one process
    each yield their own bound embedder, regardless of env-active.

    The store's `meta.embed_fingerprint` picks the embedder, never the
    `MEMMAN_EMBED_MODEL` env var.

    Mutation: `bound_embedder` resolving the model from the env
    instead of the store's fingerprint, so both stores get one client.
    Oracle: two stores seeded with hand-picked fingerprints (8 and 16
    dims); each bound embedder reports its own model and dim.
    """

    data_dir = str(tmp_path / 'memman')

    def _seed(name: str, model: str, dim: int) -> None:
        backend = open_backend(name, data_dir)
        try:
            fp_mod.write_fingerprint(
                backend,
                fp_mod.Fingerprint(model=model, dim=dim))
        finally:
            backend.close()

    monkeypatch.setattr(
        ec_registry, 'get_for',
        lambda model, dim=0: _StubEC(
            model=model, dim=8 if model == 'stub-a-d8' else 16))

    _seed('store_a', 'stub-a-d8', 8)
    _seed('store_b', 'stub-b-d16', 16)

    with active_store(
            data_dir=data_dir, store='store_a',
            unchecked=True) as backend_a:
        ec_a = fp_mod.bound_embedder(backend_a)
    with active_store(
            data_dir=data_dir, store='store_b',
            unchecked=True) as backend_b:
        ec_b = fp_mod.bound_embedder(backend_b)

    assert ec_a.model == 'stub-a-d8'
    assert ec_a.dim == 8
    assert ec_b.model == 'stub-b-d16'
    assert ec_b.dim == 16


class _StubEC:
    """Minimal embed client stub for per-store binding tests.
    """

    def __init__(self, *, model: str, dim: int) -> None:
        self.model = model
        self.dim = dim

    def available(self) -> bool:
        return True

    def unavailable_message(self) -> str:
        return ''

    def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim
