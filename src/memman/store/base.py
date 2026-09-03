"""Backend-neutral default implementations for selected NodeStore verbs.

`BaseNodeStore` mixes Python-side defaults that compose from
`get_all_active` and `get_all_embeddings`. Concrete backends override
each verb when a SQL pushdown is materially faster.
"""

from collections.abc import Iterator
from typing import Any

from memman.store.model import Id


class BaseNodeStore:
    """Mixin with Python-side defaults for selected NodeStore verbs.

    No default exists for `has_active_with_queue_uuid` on purpose: a
    Python `==` scan would match legacy rows whose `queue_uuid` is
    None against a None argument, where SQL's `= ?` never matches
    NULL. Each backend implements it in SQL.
    """

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
