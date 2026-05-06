"""Tests that opening a store binds the right per-store embedder.

The runtime authority for which embedder is used moves from the
env-resolved global provider to the store's stored fingerprint.
`_StoreContext` is the natural attachment point: it reads
`meta.embed_fingerprint`, calls `registry.get_for(provider, model)`,
and exposes the bound client as `ctx.ec`.
"""

from memman.embed.fingerprint import Fingerprint, write_fingerprint
from memman.store.db import open_db, store_dir
from memman.store.sqlite import SqliteBackend


class TestStoreContextBinding:
    """`_StoreContext` resolves `ec` from the store's stored fingerprint."""

    def test_voyage_store_binds_voyage_embedder(self, tmp_path):
        """A voyage-fingerprinted store binds a voyage client."""
        from memman.cli import _StoreContext
        sdir = store_dir(str(tmp_path), 'voy')
        db = open_db(sdir)
        try:
            write_fingerprint(SqliteBackend(db), Fingerprint(
                provider='voyage', model='voyage-3-lite', dim=512))
        finally:
            db.close()

        ctx = _StoreContext('voy', str(tmp_path))
        try:
            assert ctx.ec.name == 'voyage'
            assert ctx.ec.model == 'voyage-3-lite'
        finally:
            ctx.close()

    def test_two_stores_get_their_own_embedders(
            self, tmp_path, monkeypatch, env_file):
        """Two stores in one data dir with different (provider, model)
        fingerprints: one process opens both sequentially; each
        receives an embedder bound to its own fingerprint, even when
        the env-resolved active provider points elsewhere.
        """
        from memman.cli import _StoreContext

        class _FakeStubClient:
            name = 'stub'

            def __init__(self):
                self.model = 'stub-default'
                self.dim = 0

            def available(self):
                return True

            def embed(self, text):
                return [0.1] * self.dim if self.dim else [0.1] * 8

            def embed_batch(self, texts):
                return [self.embed(t) for t in texts]

            def unavailable_message(self):
                return 'never'

        from memman import embed as embed_mod
        monkeypatch.setitem(embed_mod.PROVIDERS, 'stub', _FakeStubClient)

        for name, fp in (
                ('a', Fingerprint('voyage', 'voyage-3-lite', 512)),
                ('b', Fingerprint('stub', 'stub-x', 8))):
            sdir = store_dir(str(tmp_path), name)
            db = open_db(sdir)
            try:
                write_fingerprint(SqliteBackend(db), fp)
            finally:
                db.close()

        env_file('MEMMAN_EMBED_PROVIDER', 'voyage')
        ctx_a = _StoreContext('a', str(tmp_path))
        try:
            assert ctx_a.ec.name == 'voyage'
            assert ctx_a.ec.model == 'voyage-3-lite'
        finally:
            ctx_a.close()

        ctx_b = _StoreContext('b', str(tmp_path))
        try:
            assert ctx_b.ec.name == 'stub'
            assert ctx_b.ec.model == 'stub-x'
        finally:
            ctx_b.close()
