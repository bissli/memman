"""Unit tests for memman.backup.prune (keep-last-N by UTC stamp).
"""

from pathlib import Path

from memman.backup import prune


def _touch_bundle(target: Path, stamp: str, host: str = 'h1') -> Path:
    """Create a fake bundle + sidecar named with `stamp`.
    """
    bundle = target / f'memman-backup-{host}-{stamp}.tar.gz'
    bundle.write_text('x')
    (target / f'{bundle.name}.manifest.json').write_text('{}')
    return bundle


class TestPrune:
    """Retention deletes oldest bundles beyond `keep`.
    """

    def test_keeps_newest_n_by_timestamp(self, tmp_path):
        """Verify prune keeps the newest `keep` bundles and removes older ones.

        Mutation: sorting ascending, so the newest bundles are removed.
        Oracle: three hand-stamped bundles with the two newest expected left.
        """
        for stamp in ('20260101T000000Z', '20260102T000000Z',
                      '20260103T000000Z'):
            _touch_bundle(tmp_path, stamp)
        removed = prune(str(tmp_path), keep=2)
        remaining = sorted(
            p.name for p in tmp_path.glob('memman-backup-*.tar.gz'))
        assert remaining == [
            'memman-backup-h1-20260102T000000Z.tar.gz',
            'memman-backup-h1-20260103T000000Z.tar.gz']
        assert len(removed) == 1

    def test_removes_sidecar_with_bundle(self, tmp_path):
        """Verify a removed bundle takes its manifest sidecar with it.

        Mutation: deleting the bundle and leaving the `.manifest.json` behind.
        Oracle: filesystem existence after prune(keep=0).
        """
        bundle = _touch_bundle(tmp_path, '20260101T000000Z')
        prune(str(tmp_path), keep=0)
        assert not bundle.exists()
        assert not (tmp_path / f'{bundle.name}.manifest.json').exists()

    def test_keep_geq_count_is_noop(self, tmp_path):
        """Verify keep larger than the bundle count removes nothing.

        Mutation: an off-by-one slice that removes a bundle when keep exceeds
            the count.
        Oracle: an empty removed list and two bundles on disk.
        """
        _touch_bundle(tmp_path, '20260101T000000Z')
        _touch_bundle(tmp_path, '20260102T000000Z')
        removed = prune(str(tmp_path), keep=5)
        assert removed == []
        assert len(list(tmp_path.glob('memman-backup-*.tar.gz'))) == 2

    def test_malformed_name_skipped(self, tmp_path):
        """Verify a bundle without a parseable stamp is never pruned.

        Mutation: pruning or crashing on a file whose name has no timestamp.
        Oracle: filesystem existence after prune(keep=0).
        """
        good = _touch_bundle(tmp_path, '20260102T000000Z')
        bad = tmp_path / 'memman-backup-nostamp.tar.gz'
        bad.write_text('x')
        prune(str(tmp_path), keep=0)
        assert not good.exists()
        assert bad.exists()
