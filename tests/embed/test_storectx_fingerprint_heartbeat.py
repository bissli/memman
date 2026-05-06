"""Per-row fingerprint heartbeat in `_StoreContext`.

`_StoreContext` caches `ec` once on construction. If a swap completes
mid-drain the cached `ec` would silently produce vectors of the
wrong dim. The heartbeat re-reads `stored_fingerprint(backend)`
before each row's embed call and aborts the row with
`EmbedFingerprintError` on mismatch, releasing it for retry.
"""

import pytest
from memman.embed.fingerprint import Fingerprint, write_fingerprint
from memman.exceptions import EmbedFingerprintError
from memman.store.db import open_db, store_dir
from memman.store.sqlite import SqliteBackend


class TestHeartbeat:
    """`_StoreContext.assert_fingerprint_unchanged` raises on mid-drain swap."""

    def test_passes_when_unchanged(self, tmp_path):
        """No change between construction and call -> no error."""
        from memman.cli import _StoreContext
        sdir = store_dir(str(tmp_path), 'h1')
        db = open_db(sdir)
        try:
            write_fingerprint(SqliteBackend(db), Fingerprint(
                provider='voyage', model='voyage-3-lite', dim=512))
        finally:
            db.close()

        ctx = _StoreContext('h1', str(tmp_path))
        try:
            ctx.assert_fingerprint_unchanged()
        finally:
            ctx.close()

    @pytest.mark.no_autoseed_fingerprint
    def test_raises_when_fingerprint_flipped(self, tmp_path):
        """A fingerprint flip after construction triggers
        `EmbedFingerprintError` on the next heartbeat call.
        """
        from memman.cli import _StoreContext
        sdir = store_dir(str(tmp_path), 'h2')
        db = open_db(sdir)
        try:
            write_fingerprint(SqliteBackend(db), Fingerprint(
                provider='voyage', model='voyage-3-lite', dim=512))
        finally:
            db.close()

        ctx = _StoreContext('h2', str(tmp_path))
        try:
            db = open_db(sdir)
            try:
                write_fingerprint(SqliteBackend(db), Fingerprint(
                    provider='openai',
                    model='text-embedding-3-small',
                    dim=1536))
            finally:
                db.close()
            with pytest.raises(EmbedFingerprintError):
                ctx.assert_fingerprint_unchanged()
        finally:
            ctx.close()
