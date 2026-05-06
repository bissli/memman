"""Backend-neutral default implementations for selected NodeStore verbs.

`BaseNodeStore` mixes Python-side defaults that compose from
`get_all_active` and `get_all_embeddings`. Concrete backends override
each verb when a SQL pushdown is materially faster.
"""

from collections.abc import Iterator
from typing import Any

from memman.store.model import Id, Insight


class BaseNodeStore:
    """Mixin with Python-side defaults for selected NodeStore verbs."""

    def has_active_with_source(self, source: str) -> bool:
        """Default: scan `get_all_active` for any row with the source.
        """
        return any(ins.source == source for ins in self.get_all_active())

    def get_without_embedding(
            self, *, limit: int = 100) -> list[Insight]:
        """Default: cross `get_all_active` with embedded-id set.

        Returns rows that have no embedding, ordered by
        `importance desc, created_at desc`. `get_all_active`
        already returns `created_at desc`, so a stable sort by
        `-importance` preserves the secondary order.
        """
        embedded = {
            rid for rid, _content, _blob
            in self.get_all_embeddings()}  # type: ignore[attr-defined]
        rows = [
            ins for ins
            in self.get_all_active()  # type: ignore[attr-defined]
            if ins.id not in embedded]
        rows.sort(key=lambda i: -i.importance)
        return rows[:limit]

    def review_content_quality(
            self, *, limit: int) -> list[dict[str, Any]]:
        """Default: scan `get_all_active` for transient patterns.
        """
        from memman.search.quality import check_content_quality
        flagged: list[dict[str, Any]] = []
        for ins in self.get_all_active():  # type: ignore[attr-defined]
            warnings = check_content_quality(ins.content)
            if warnings:
                flagged.append(
                    {'insight': ins, 'quality_warnings': warnings})
        flagged.sort(
            key=lambda x: len(x['quality_warnings']),
            reverse=True)
        return flagged[:limit]

    def iter_embeddings_as_vecs(
            self) -> Iterator[tuple[Id, list[float]]]:
        """Default: deserialize the blobs from `get_all_embeddings`.
        """
        from memman.embed.vector import deserialize_vector
        for rid, _content, blob in (
                self.get_all_embeddings()):  # type: ignore[attr-defined]
            if blob is None:
                continue
            vec = deserialize_vector(blob)
            if vec is not None:
                yield rid, vec
