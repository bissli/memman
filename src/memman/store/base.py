"""Backend-neutral default implementations for select Protocol verbs.

`BaseNodeStore` provides Python-side defaults for the four
`NodeStore` verbs that compose cleanly from other Protocol verbs:

  - `has_active_with_source`     (filter `get_all_active`)
  - `get_without_embedding`      (cross `get_all_active` with
                                  `get_all_embeddings` ids)
  - `review_content_quality`     (loop over `get_all_active`)
  - `iter_embeddings_as_vecs`    (deserialize blobs from
                                  `get_all_embeddings`)

Concrete backends inherit:

    class SqliteNodeStore(BaseNodeStore, NodeStore):
        ...

Backends that have a faster SQL pushdown (Postgres for the first
three; SQLite for `iter_embeddings_as_vecs` because the blob
deserialize already runs in Python) keep their override. The
mixin's value is documenting the canonical fallback so a new
backend can ship with only the primitive verbs implemented.

The plan called out 5-8 candidates; the honest count is 4
because `Insight` does not carry `linked_at` (so
`get_pending_link_ids` / `count_pending_links` / `clear_linked_at`
have no Protocol-level state to compose from) and
`increment_access_count` skips `updated_at` (so it cannot stand
in for `boost_retention`).
"""

from collections.abc import Iterator
from typing import Any

from memman.store.model import Id, Insight


class BaseNodeStore:
    """Mixin with Python-side defaults for selected NodeStore verbs.

    Composes from `get_all_active` and `get_all_embeddings`. Concrete
    backends override individual verbs when a SQL pushdown is
    materially faster.
    """

    def has_active_with_source(self, source: str) -> bool:
        """Default: scan `get_all_active` for any row with the source.
        """
        for ins in self.get_all_active():  # type: ignore[attr-defined]
            if ins.source == source:
                return True
        return False

    def get_without_embedding(
            self, *, limit: int = 100) -> list[Insight]:
        """Default: cross `get_all_active` with embedded-id set.

        Returns rows that have no embedding, ordered by
        `importance desc, created_at desc`. `get_all_active`
        already returns `created_at desc`, so a stable sort by
        `-importance` preserves the secondary order.
        """
        if limit <= 0:
            limit = 100
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
            self, *, limit: int = 50) -> list[dict[str, Any]]:
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


class BaseEdgeStore:
    """Placeholder mixin for future EdgeStore defaults.

    No verbs default cleanly today: `by_node_and_type` /
    `by_source_and_type` would do a full per-node edge fetch in
    Python (no Protocol verb exposes the indexed scan), and
    `count_with_entity` would need a fixed-large-limit dance over
    `find_with_entity`. Both backends keep their indexed
    overrides. The class exists so a new backend can opt into
    `BaseEdgeStore` for symmetry once a defaultable verb appears.
    """
