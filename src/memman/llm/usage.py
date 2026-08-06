"""Per-stage LLM token accounting.

Accumulates the provider-reported `usage` block of every completion
attempt into a module-level ledger keyed by pipeline stage.
`MemmanLLMClient.complete` records one entry per HTTP attempt --
success, malformed, empty and error responses alike -- so the ledger
measures what the endpoint was asked to do, not just what the
pipeline kept: an empty body retried twice is three billed
completions, and a success-only recorder books zero for two of them.

Notes
-----
- `calls` counts HTTP-200 attempts (billed completions); non-2xx
  attempts land in `http_errors` instead, so a 429 storm retried
  three times cannot inflate the billed-call signal 3x. Token
  fields sum the provider-reported `usage` values from either kind
  of attempt (an error body reporting usage was still billed).
- An HTTP-200 response with no `usage` block increments
  `missing_usage` and adds nothing to the token fields -- "provider
  reported zero" and "provider reported nothing" must stay
  distinguishable.
- The ledger is never reset. Consumers take a `snapshot()` before a
  unit of work and diff with `delta()` after, so row-level and
  drain-level readings coexist without clobbering each other.
- The lock is required, not defensive: enrichment and causal run
  concurrently on a two-worker `ThreadPoolExecutor`, so their
  attempts interleave and attribution by event order is
  unrecoverable.
"""

import threading

STAGE_EXTRACTION = 'extraction'
STAGE_RECONCILIATION = 'reconciliation'
STAGE_QUERY_EXPANSION = 'query_expansion'
STAGE_ENRICHMENT = 'enrichment'
STAGE_CAUSAL = 'causal'
STAGE_PROBE = 'probe'
# Off-pipeline measurement tooling (experiments/ harnesses, eval
# judges) -- keeps their traffic out of the six pipeline buckets.
STAGE_HARNESS = 'harness'

VALID_STAGES = frozenset({
    STAGE_EXTRACTION, STAGE_RECONCILIATION, STAGE_QUERY_EXPANSION,
    STAGE_ENRICHMENT, STAGE_CAUSAL, STAGE_PROBE, STAGE_HARNESS,
    })

_COUNTER_KEYS = (
    'calls', 'prompt_tokens', 'completion_tokens', 'total_tokens',
    'missing_usage', 'http_errors')

_LOCK = threading.Lock()
_LEDGER: dict[str, dict[str, int]] = {}


def record(stage: str, usage: dict | None, *,
           http_error: bool = False) -> None:
    """Add one completion attempt to the stage's bucket.

    Parameters
    ----------
    stage : str
        One of `VALID_STAGES`; unknown stages raise so a typo cannot
        create a silent phantom bucket.
    usage : dict | None
        The response body's `usage` block, or None when the provider
        reported nothing (counted in `missing_usage` for billed
        attempts).
    http_error : bool, default False
        True for a non-2xx attempt: booked as `http_errors`, not
        `calls`, and never as `missing_usage` -- an unbilled
        rejection is neither a billed completion nor a billed
        completion whose usage went unreported.
    """
    if stage not in VALID_STAGES:
        raise ValueError(
            f'unknown LLM stage {stage!r};'
            f' valid stages: {sorted(VALID_STAGES)}')
    with _LOCK:
        bucket = _LEDGER.setdefault(
            stage, dict.fromkeys(_COUNTER_KEYS, 0))
        if http_error:
            bucket['http_errors'] += 1
        else:
            bucket['calls'] += 1
        if not isinstance(usage, dict):
            if not http_error:
                bucket['missing_usage'] += 1
            return
        prompt = int(usage.get('prompt_tokens') or 0)
        completion = int(usage.get('completion_tokens') or 0)
        total = int(usage.get('total_tokens') or (prompt + completion))
        bucket['prompt_tokens'] += prompt
        bucket['completion_tokens'] += completion
        bucket['total_tokens'] += total


def snapshot() -> dict[str, dict[str, int]]:
    """Return a copy of the ledger for later diffing with `delta`."""
    with _LOCK:
        return {s: dict(v) for s, v in _LEDGER.items()}


def delta(before: dict[str, dict[str, int]],
          after: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Per-stage difference between two snapshots.

    Parameters
    ----------
    before : dict[str, dict[str, int]]
        Earlier `snapshot()`.
    after : dict[str, dict[str, int]]
        Later `snapshot()`.

    Returns
    -------
    dict[str, dict[str, int]]
        Stages with any activity between the snapshots; idle stages
        are omitted.
    """
    out: dict[str, dict[str, int]] = {}
    for stage, vals in after.items():
        base = before.get(stage, {})
        diff = {k: v - base.get(k, 0) for k, v in vals.items()}
        if any(diff.values()):
            out[stage] = diff
    return out
