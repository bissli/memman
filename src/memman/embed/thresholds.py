"""Per-(provider, model, surface) AUTO_SEMANTIC_THRESHOLD calibration.

Each entry in `_thresholds_generated._THRESHOLDS` is the cosine cutoff
that optimizes retrieval quality for one surface against labeled
corpora. `surface` is a closed-set tag in `{'code', 'claw'}` that
groups calibrated thresholds by the agent-register they target.

Two resolution APIs are exposed:

- `resolve(provider, model, surface)` returns the calibrated float or
  `None`. Use it when the caller wants to know whether a triple is
  calibrated and is prepared to handle the unknown case explicitly.

- `resolve_with_fallback(provider, model, surface)` returns a tuple
  `(threshold, source)` that always supplies a usable threshold. The
  fallback for an uncalibrated triple is `SURFACE_FALLBACK[surface]` --
  the median of all calibrated thresholds for that surface. This is a
  store-independent constant derived deterministically from the
  shipped table at import time; it has bounded retrieval-quality loss
  (mean nDCG@5 loss ~0.014 vs calibrated, max ~0.08 across the shipped
  triples) and is the right shape for an "unknown model" fallback
  precisely because it does not depend on the user's own embedding
  distribution (which would be store-dependent and surface-blind).

Operators with a quality-critical store running an uncalibrated model
can set `MEMMAN_AUTO_SEMANTIC_THRESHOLD_<store>` to a measured value;
that override is consulted upstream in `graph.engine` and takes
precedence over both the calibrated table and the median fallback.

`SURFACE_FALLBACK` is computed once at import. Long-lived processes
that observe `_thresholds_generated.py` being rewritten (e.g. after
`make retune`) must be restarted to pick up the new medians; the
calibrated-table lookup reloads with the module but the cached
fallback dict does not.
"""

import statistics

from memman.config import SURFACE_VALUES
from memman.embed._thresholds_generated import _THRESHOLDS

_LEGACY_FALLBACK_VALUE: float = 0.62


def _compute_surface_fallback() -> dict[str, float]:
    """Compute per-surface median from the calibrated table.

    Iterates `config.SURFACE_VALUES` -- single source of truth for the
    closed surface set. Falls back to the legacy 0.62 universal default
    for any surface with zero calibrated entries -- avoids a
    `statistics.median([])` crash at import time when the shipped table
    is empty (first emit, partial regen, or hand-editing).
    """
    out: dict[str, float] = {}
    for surf in sorted(SURFACE_VALUES):
        vals = [v for (p, m, s), v in _THRESHOLDS.items() if s == surf]
        out[surf] = (
            statistics.median(vals) if vals else _LEGACY_FALLBACK_VALUE)
    return out


SURFACE_FALLBACK: dict[str, float] = _compute_surface_fallback()


def resolve(provider: str, model: str,
            surface: str = 'code') -> float | None:
    """Return the calibrated AUTO_SEMANTIC_THRESHOLD, or None.

    None means the (provider, model, surface) triple has no calibrated
    entry. Use `resolve_with_fallback` to get a usable threshold for
    uncalibrated triples instead of `None`.

    `surface` defaults to `'code'` so stores without an explicit
    `MEMMAN_SURFACE_<store>` setting resolve via the code-surface row.
    """
    return _THRESHOLDS.get((provider, model, surface))


def resolve_with_fallback(
        provider: str, model: str,
        surface: str = 'code') -> tuple[float, str]:
    """Return `(threshold, source)`; always supplies a usable float.

    `source` is `'calibrated'` when the triple is in the shipped table
    and `'surface_median'` when the surface-wide median fallback is
    used. Raises `KeyError` for an unknown surface -- `surface` is a
    closed set and unknown values are caller bugs.
    """
    direct = _THRESHOLDS.get((provider, model, surface))
    if direct is not None:
        return direct, 'calibrated'
    return SURFACE_FALLBACK[surface], 'surface_median'


def is_calibrated(provider: str, model: str,
                  surface: str = 'code') -> bool:
    """Return True iff `(provider, model, surface)` has a calibrated entry.
    """
    return (provider, model, surface) in _THRESHOLDS
