"""Tests for `memman.session.active_store` per-store sovereignty.

`active_store` opens the per-store backend via
`factory.open_backend` and resolves the embedder from each store's
own stored fingerprint via `bound_embedder`. The CLI boundary wraps
backend `ConfigError` as `click.ClickException` so misconfigured
stores produce a clean exit message instead of a bare traceback.
"""

import pytest
from click import ClickException
from memman.session import active_store


def test_active_store_wraps_backend_config_error_from_open(
        tmp_path, env_file, monkeypatch):
    """A misconfigured store (postgres backend, no DSN) raises
    `ClickException` instead of leaking the backend `ConfigError`.
    """
    env_file('MEMMAN_BACKEND_oops', 'postgres')
    data_dir = str(tmp_path / 'memman')
    with pytest.raises(ClickException) as exc, active_store(
            data_dir=data_dir, store='oops', unchecked=True):
        pass
    assert 'oops' in str(exc.value).lower() or 'dsn' in str(exc.value).lower()


def test_active_store_yields_store_bound_ec_per_store(
        tmp_path, env_file, monkeypatch):
    """Two stores with different stored fingerprints in one process
    each yield their own bound embedder, regardless of env-active.

    Per-store sovereignty: the env var
    `MEMMAN_EMBED_PROVIDER` no longer drives recall/remember in an
    existing store; the store's `meta.embed_fingerprint` does.
    """
    from memman.embed import fingerprint as fp_mod
    from memman.embed import registry as ec_registry
    from memman.store.factory import open_backend

    data_dir = str(tmp_path / 'memman')

    def _seed(name: str, provider: str, model: str, dim: int) -> None:
        backend = open_backend(name, data_dir)
        try:
            fp_mod.write_fingerprint(
                backend,
                fp_mod.Fingerprint(
                    provider=provider, model=model, dim=dim))
        finally:
            backend.close()

    monkeypatch.setattr(
        ec_registry, 'get_for',
        lambda provider, model: _StubEC(
            provider=provider, model=model,
            dim=8 if provider == 'stub-a' else 16))

    _seed('store_a', 'stub-a', 'stub-a-d8', 8)
    _seed('store_b', 'stub-b', 'stub-b-d16', 16)

    with active_store(
            data_dir=data_dir, store='store_a',
            unchecked=True) as backend_a:
        ec_a = fp_mod.bound_embedder(backend_a)
    with active_store(
            data_dir=data_dir, store='store_b',
            unchecked=True) as backend_b:
        ec_b = fp_mod.bound_embedder(backend_b)

    assert ec_a.provider == 'stub-a'
    assert ec_a.dim == 8
    assert ec_b.provider == 'stub-b'
    assert ec_b.dim == 16


class _StubEC:
    """Minimal embed client stub for per-store binding tests."""

    def __init__(self, *, provider: str, model: str, dim: int) -> None:
        self.provider = provider
        self.name = provider
        self.model = model
        self.dim = dim

    def available(self) -> bool:
        return True

    def unavailable_message(self) -> str:
        return ''

    def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim
