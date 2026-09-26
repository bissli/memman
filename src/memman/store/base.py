"""Backend-neutral default implementations for selected NodeStore verbs.

`BaseNodeStore` mixes Python-side defaults that compose from
`get_all_active`. Concrete backends override each verb when a SQL
pushdown is materially faster.
"""

from typing import Any


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
