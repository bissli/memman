#!/usr/bin/env python3
r"""Re-enrich stale insights across stores via `memman graph rebuild`.

Iterates a list of stores, runs `memman graph rebuild` per store, and
streams a tqdm progress bar showing total-row progress plus the per-
store completion log. Per-store output is captured and appended to a
log file under `/tmp` so the rebuild can run in the background and
the operator can tail it later.

Cross-store parallelism (`--parallel N`) runs N store-rebuilds at
once via a subprocess thread pool; each rebuild process internally
fires 2 concurrent LLM calls, so the steady-state in-flight chat
completion count is ~2 * N. The default of 4 keeps that under
typical OpenRouter rate ceilings for sonnet-class models. Bump for
higher tiers; set 1 to revert to serial.

Different SQLite stores live in different DB files, so cross-store
concurrency does not race the per-store `reembed_lock` (a no-op on
SQLite anyway). Same-store parallelism is unsafe and not exposed.

Usage
-----
    python scripts/rebuild_stale.py [STORE ...] [--log PATH] \\
                                    [--memman PATH] [--parallel N] \\
                                    [--continue-on-error]

With no STORE arguments, runs against every store reported by
`memman store list`. Skips stores whose active-insight count is zero.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from tqdm import tqdm

DEFAULT_PARALLEL = 4


def _resolve_memman(explicit: str | None) -> str:
    """Return the absolute path to the memman binary to invoke."""
    if explicit:
        return explicit
    found = shutil.which('memman')
    if not found:
        raise SystemExit(
            'memman binary not on PATH; pass --memman /path/to/memman')
    return found


def _list_stores(memman: str) -> list[str]:
    """Return every store name reported by `memman store list`."""
    out = subprocess.run(
        [memman, 'store', 'list'],
        capture_output=True, text=True, check=True)
    return list(json.loads(out.stdout).get('stores') or [])


def _store_active_count(memman: str, store: str) -> int:
    """Return the active (non-deleted) insight count for `store`.

    Uses `memman --store <store> status` (cheap, no LLM). The status
    JSON exposes the active count as `total_insights` (deleted rows
    are tracked separately under `deleted_insights`).
    """
    out = subprocess.run(
        [memman, '--store', store, 'status'],
        capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return 0
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return 0
    raw = data.get('total_insights')
    if raw is None:
        raw = sum((data.get('by_category') or {}).values())
    return int(raw or 0)


_LOG_LOCK = Lock()


def _rebuild_store(memman: str, store: str, log_path: Path,
                   on_row) -> tuple[int, int, str]:
    """Run rebuild for one store, streaming progress via stderr.

    Each `done` event from the child (emitted because we pass
    `--progress-jsonl`) calls `on_row()` so the parent's outer bar
    can tick per row. Non-progress stderr lines are buffered and
    appended to `log_path` along with the final stdout once the
    child exits. Returns `(returncode, processed_count, raw_output)`.
    Log writes are serialized across worker threads via `_LOG_LOCK`.
    """
    proc = subprocess.Popen(
        [memman, '--store', store, 'graph', 'rebuild',
         '--progress-jsonl'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1)

    stderr_lines: list[str] = []
    assert proc.stderr is not None
    for line in proc.stderr:
        stripped = line.strip()
        if stripped.startswith('{'):
            try:
                evt = json.loads(stripped)
            except json.JSONDecodeError:
                evt = None
            if (evt is not None
                and evt.get('event') == 'progress'
                    and evt.get('stage') == 'done'):
                on_row()
                continue
        stderr_lines.append(line)

    assert proc.stdout is not None
    stdout = proc.stdout.read()
    proc.wait()

    started = datetime.now(timezone.utc).isoformat()
    stderr_text = ''.join(stderr_lines)
    with _LOG_LOCK, log_path.open('a') as fh:
        fh.write(f'\n\n=== {started} :: rebuild {store} ===\n')
        fh.write(f'returncode={proc.returncode}\n')
        fh.write(f'stdout={stdout}\n')
        if stderr_text:
            fh.write(f'stderr={stderr_text}\n')

    processed = 0
    if proc.returncode == 0 and stdout.strip():
        try:
            payload = json.loads(stdout)
            processed = int(payload.get('processed', 0))
        except json.JSONDecodeError:
            pass
    return proc.returncode, processed, (stdout + stderr_text).strip()


def main() -> int:
    """Drive the rebuild + progress bar."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        'stores', nargs='*',
        help='Store names to rebuild (default: every store)')
    ap.add_argument(
        '--log', default='/tmp/memman_rebuild.log',
        help='Append per-store rebuild output here (default: %(default)s)')
    ap.add_argument(
        '--memman', default=None,
        help='Path to the memman binary (default: shutil.which)')
    ap.add_argument(
        '--parallel', type=int, default=DEFAULT_PARALLEL,
        help=(
            'Run N store-rebuilds concurrently (default: %(default)s).'
            ' Each rebuild fires 2 concurrent LLM calls internally,'
            ' so steady-state in-flight chat completions = 2 * N.'
            ' Set 1 for serial.'))
    ap.add_argument(
        '--continue-on-error', action='store_true',
        help='Keep going past per-store rebuild failures')
    args = ap.parse_args()

    memman = _resolve_memman(args.memman)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    targets = args.stores or _list_stores(memman)
    sized = [(s, _store_active_count(memman, s)) for s in targets]
    sized = [(s, n) for s, n in sized if n > 0]

    if not sized:
        print('no stores with active insights to rebuild', file=sys.stderr)
        return 0

    parallel = max(1, min(args.parallel, len(sized)))
    sized.sort(key=lambda pair: pair[1], reverse=parallel > 1)
    total_rows = sum(n for _, n in sized)
    weights = dict(sized)

    print(
        f'rebuilding {len(sized)} stores, {total_rows} active rows total,'
        f' parallel={parallel}; log -> {log_path}',
        file=sys.stderr)

    bar = tqdm(
        total=total_rows, unit='row', desc='rebuild',
        dynamic_ncols=True, smoothing=0.05)
    failures: list[tuple[str, str]] = []
    overall_t0 = time.monotonic()

    in_flight: set[str] = set()
    in_flight_lock = Lock()

    def _set_postfix() -> None:
        with in_flight_lock:
            bar.set_postfix_str(','.join(sorted(in_flight)) or 'idle')

    aborted = False

    try:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {}
            for store, expected in sized:
                fut = pool.submit(
                    _wrapped_rebuild, memman, store, expected, log_path,
                    bar, in_flight, in_flight_lock, _set_postfix)
                futures[fut] = (store, expected)

            for fut in as_completed(futures):
                store, expected = futures[fut]
                t_start, rc, processed, output = fut.result()
                elapsed = time.monotonic() - t_start
                bar.write(
                    f'[{store}] rc={rc} processed={processed}'
                    f' elapsed={elapsed:.1f}s')
                if rc != 0:
                    failures.append((store, output[:200]))
                    if not args.continue_on_error and not aborted:
                        aborted = True
                        bar.write(
                            f'aborting after store {store!r}'
                            f' (rc={rc}); waiting for in-flight workers'
                            ' to drain')
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
    finally:
        bar.close()

    overall_elapsed = time.monotonic() - overall_t0
    print(
        f'\nrebuild complete in {overall_elapsed:.1f}s'
        f' across {len(sized) - len(failures)} stores'
        f' ({total_rows} rows requested)',
        file=sys.stderr)
    if failures:
        print(f'{len(failures)} store(s) failed:', file=sys.stderr)
        for f_store, f_out in failures:
            print(f'  {f_store}: {f_out}', file=sys.stderr)
        return 1
    return 0


def _wrapped_rebuild(memman, store, expected, log_path, bar,
                     in_flight, in_flight_lock, set_postfix):
    """Run `_rebuild_store`, ticking `bar` per row and tracking in_flight.

    Reconciles the bar at the end so each store contributes exactly
    `expected` units regardless of how many `done` events streamed
    (e.g., if the child processed fewer rows than the pre-flight
    `_store_active_count` snapshot suggested).
    """
    with in_flight_lock:
        in_flight.add(store)
    set_postfix()
    t_start = time.monotonic()
    seen = 0

    def _on_row() -> None:
        nonlocal seen
        seen += 1
        bar.update(1)

    try:
        rc, processed, output = _rebuild_store(
            memman, store, log_path, _on_row)
        if seen < expected:
            bar.update(expected - seen)
    finally:
        with in_flight_lock:
            in_flight.discard(store)
        set_postfix()
    return t_start, rc, processed, output


if __name__ == '__main__':
    raise SystemExit(main())
