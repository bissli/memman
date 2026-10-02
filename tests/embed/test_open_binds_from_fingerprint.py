"""Tests that opening a store binds the right per-store embedder.

A store's stored fingerprint decides which embedder the store uses. `_StoreContext` reads
`meta.embed_fingerprint`, calls `registry.get_for(model)`,
and exposes the bound client as `ctx.ec`.
"""

from memman.cli import _StoreContext
from memman.embed.fingerprint import Fingerprint, write_fingerprint
from memman.store.db import open_db, store_dir
from memman.store.sqlite import SqliteBackend


class TestStoreContextBinding:
    """`_StoreContext` resolves `ec` from the store's stored fingerprint.
    """

    def test_voyage_store_binds_voyage_embedder(self, tmp_path):
        """A voyage-fingerprinted store binds a voyage client.

        Mutation: `_StoreContext` binding the env-resolved default
            client instead of the stored fingerprint's, or ignoring the
            stored model.
        Oracle: the fingerprint's model written by the test.
        """
        sdir = store_dir(str(tmp_path), 'voy')
        db = open_db(sdir)
        try:
            write_fingerprint(SqliteBackend(db), Fingerprint(
                model='voyage-3-lite', dim=512))
        finally:
            db.close()

        ctx = _StoreContext('voy', str(tmp_path))
        try:
            assert ctx.ec.model == 'voyage-3-lite'
        finally:
            ctx.close()

    def test_two_stores_get_their_own_embedders(
            self, tmp_path, monkeypatch, env_file):
        """Two stores with different fingerprints each bind their own embedder.

        One process opens both in turn. The env-resolved embed model
        points elsewhere.

        Mutation: `_StoreContext` caching one client per process, or
            binding the env model, so store `b` gets the voyage
            client.
        Oracle: each store's own fingerprint model.
        """

        for name, fp in (
                ('a', Fingerprint('voyage-3-lite', 512)),
                ('b', Fingerprint('stub-x', 8))):
            sdir = store_dir(str(tmp_path), name)
            db = open_db(sdir)
            try:
                write_fingerprint(SqliteBackend(db), fp)
            finally:
                db.close()

        env_file('MEMMAN_EMBED_MODEL', 'voyage-3-lite')
        ctx_a = _StoreContext('a', str(tmp_path))
        try:
            assert ctx_a.ec.model == 'voyage-3-lite'
        finally:
            ctx_a.close()

        ctx_b = _StoreContext('b', str(tmp_path))
        try:
            assert ctx_b.ec.model == 'stub-x'
        finally:
            ctx_b.close()
