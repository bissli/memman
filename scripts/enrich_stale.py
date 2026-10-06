r"""Run `memman enrich --stale-only` over many stores with one progress bar.

Usage
-----
    python scripts/enrich_stale.py [STORE ...] [--log PATH] \\
                                    [--memman PATH] [--parallel N] \\
                                    [--continue-on-error]

With no STORE, every store `memman store list` reports runs. A store
whose `memman status` shows no stale rows is skipped. Each store's
output is appended to the log.

Notes
-----
- Each store run makes two concurrent LLM calls, so `--parallel N`
  holds about 2N in flight. Lower N on rate-limit errors.
- Stores run in parallel, never one store twice at once: two runs on
  one store race its `reembed_lock`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from tqdm import tqdm

DEFAULT_PARALLEL = 4


def _resolve_memman(explicit: str | None) -> str:
    """Path of the memman binary to run.

    Parameters
    ----------
    explicit : str or None
        The `--memman` value, which wins when set.

    Returns
    -------
    str
        `explicit`, else the `memman` beside the running interpreter, so
        a dev virtualenv runs its own build, else the one on PATH.

    Raises
    ------
    SystemExit
        No memman binary is found.
    """
    if explicit:
        return explicit
    sibling = Path(sys.executable).with_name('memman')
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    found = shutil.which('memman')
    if not found:
        raise SystemExit(
            'memman binary not on PATH; pass --memman /path/to/memman')
    return found


def _list_stores(memman: str) -> list[str]:
    """Every store name `memman store list` reports.
    """
    out = subprocess.run(
        [memman, 'store', 'list'],
        capture_output=True, text=True, check=True)
    return list(json.loads(out.stdout).get('stores') or [])


def _store_stale_count(memman: str, store: str) -> int:
    """Stale row count of `store`, from `memman --store <store> status`.

    Returns
    -------
    int
        0 when the store does not open or reports no count, so the
        caller skips it.
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
    raw = data.get('stale_insights')
    if raw is None:
        return 0
    return int(raw or 0)


_LOG_LOCK = Lock()


def _rebuild_store(memman: str, store: str, log_path: Path,
                   on_row: Callable[[], None]) -> tuple[int, int, str]:
    """Run `memman enrich --stale-only` on one store.

    Parameters
    ----------
    memman : str
        memman binary.
    store : str
        Store name.
    log_path : Path
        File the run's stdout and stderr are appended to.
    on_row : Callable[[], None]
        Called once per enriched row.

    Returns
    -------
    tuple[int, int, str]
        Exit code, rows processed, and the run's combined output.
    """
    proc = subprocess.Popen(
        [memman, '--store', store, 'enrich',
         '--stale-only', '--progress-jsonl'],
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
        except json.JSONDecodeError as exc:
            with _LOG_LOCK, log_path.open('a') as fh:
                fh.write(
                    f'json-parse-failed for store {store}: {exc};'
                    f' first 200 chars: {stdout[:200]!r}\n')
    return proc.returncode, processed, (stdout + stderr_text).strip()


def main() -> int:
    """Enrich every selected store under one progress bar.

    Returns
    -------
    int
        0, or 1 when any store failed.
    """
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
    sized = [(s, _store_stale_count(memman, s)) for s in targets]
    sized = [(s, n) for s, n in sized if n > 0]

    if not sized:
        print('no stores with stale insights to rebuild', file=sys.stderr)
        return 0

    parallel = max(1, min(args.parallel, len(sized)))
    sized.sort(key=lambda pair: pair[1], reverse=parallel > 1)
    total_rows = sum(n for _, n in sized)
    weights = dict(sized)

    print(
        f'rebuilding {len(sized)} stores, {total_rows} stale rows total,'
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
                try:
                    t_start, rc, processed, output = fut.result()
                except CancelledError:
                    continue
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


def _wrapped_rebuild(
        memman: str, store: str, expected: int, log_path: Path,
        bar: tqdm, in_flight: set[str], in_flight_lock: Any,
        set_postfix: Callable[[], None],
        ) -> tuple[float, int, int, str]:
    """Run `_rebuild_store` on one store inside the shared progress bar.

    Parameters
    ----------
    memman : str
        memman binary.
    store : str
        Store name.
    expected : int
        The store's stale count, which the bar advances by in total
        even when the run enriches fewer rows.
    log_path : Path
        File the run's output is appended to.
    bar : tqdm
        Shared progress bar, ticked once per enriched row.
    in_flight : set[str]
        Names of the stores running now. `store` is in it for the run.
    in_flight_lock : Any
        Lock guarding `in_flight`.
    set_postfix : Callable[[], None]
        Redraws the bar's list of running stores.

    Returns
    -------
    tuple[float, int, int, str]
        Start time from `time.monotonic`, exit code, rows processed,
        and the run's combined output.
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
