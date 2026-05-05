"""Helpers for memman end-to-end tests.

Subprocess wrappers around the installed `memman` CLI and JSON-walk
assertion helpers (`assert_jq`, `assert_jq_gte`, `assert_contains`,
`extract_id`).

Every CLI invocation goes through `run_cli` so the HOME redirect, env
inheritance, and key gating live in exactly one place.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

_SENTINEL = object()


def run_cli(args: list[str], home: Path, data_dir: Path | None = None,
            extra_env: dict[str, str] | None = None,
            check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run `memman <args>` in a subprocess with HOME redirected.

    The HOME redirect picks up the scheduler markers the `memman_home`
    fixture writes; data_dir, when given, is prepended as `--data-dir`.
    Returns the CompletedProcess; raises on non-zero when check=True.
    """
    cmd = ['memman']
    if data_dir is not None:
        cmd += ['--data-dir', str(data_dir)]
    cmd += args
    env = os.environ.copy()
    env['HOME'] = str(home)
    for k in ('MEMMAN_STORE', 'MEMMAN_DATA_DIR'):
        env.pop(k, None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(cmd, capture_output=True, text=True,
                          env=env, check=check)


def json_out(result: subprocess.CompletedProcess[str]) -> dict:
    """Parse stdout as JSON, raising a clear error if it isn't.
    """
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f'expected JSON stdout, got: {result.stdout!r}\n'
            f'stderr: {result.stderr!r}') from exc


def _walk(data: Any, path: str) -> Any:
    """Walk a dotted/indexed path through nested dicts/lists.

    Mirrors jq filters used in the bash script. `facts.0.action` goes
    `data['facts'][0]['action']`. Raises KeyError/IndexError if the
    path is absent (matches jq's failure mode under `-r`).
    """
    cur = data
    for token in path.split('.'):
        if isinstance(cur, list):
            cur = cur[int(token)]
        else:
            cur = cur[token]
    return cur


def assert_jq(data: dict, path: str, expected: Any, label: str = '') -> None:
    """Assert a JSON-walked path equals expected.
    """
    actual = _walk(data, path)
    assert actual == expected, (
        f'{label}: {path} expected {expected!r}, got {actual!r}')


def assert_jq_gte(data: dict, path: str, expected: int,
                  label: str = '') -> None:
    """Assert a JSON-walked path is >= expected (numeric).
    """
    actual = _walk(data, path)
    assert int(actual) >= int(expected), (
        f'{label}: {path}={actual} expected >= {expected}')


def assert_jq_lte(data: dict, path: str, expected: int,
                  label: str = '') -> None:
    """Assert a JSON-walked path is <= expected (numeric).
    """
    actual = _walk(data, path)
    assert int(actual) <= int(expected), (
        f'{label}: {path}={actual} expected <= {expected}')


def assert_contains(text: str, needle: str, label: str = '') -> None:
    """Assert needle appears somewhere in text.
    """
    assert needle in text, f'{label}: {needle!r} not found in: {text!r}'


def assert_not_contains(text: str, needle: str, label: str = '') -> None:
    """Assert needle does not appear in text.
    """
    assert needle not in text, (
        f'{label}: {needle!r} should not be in: {text!r}')


def extract_id_from_facts(data: dict) -> str:
    """Extract `.facts[0].id` or fallback `.id`.

    Legacy shape — used by a few CLI commands that still return a
    `{facts: [...]}` envelope (e.g., `graph rebuild --dry-run`). Most
    write paths now go through the queue and return `{queue_id: N}`
    instead; use `extract_queue_id` for those.
    """
    facts = data.get('facts')
    if facts:
        return facts[0]['id']
    return data['id']


def extract_queue_id(data: dict) -> int:
    """Return `queue_id` from a queued `remember` response."""
    return int(data['queue_id'])


def find_insight_by_recall(home: Path, data_dir: Path,
                           keyword: str) -> str:
    """Recall the most recent insight matching `keyword`; return its id.

    Used after a drain pass to translate a write back to its insight
    without a queue-side back-reference (the queue table has no
    `insight_id` column). The keyword must be unique within the data
    dir so the top-1 result is the intended insight.

    `recall --basic` returns insight rows directly (not wrapped under
    `{insight: ...}`); this helper reads `results[0]['id']`.
    """
    out = run_cli(['recall', '--basic', keyword, '--limit', '1'],
                  home, data_dir)
    data = json_out(out)
    results = data.get('results', [])
    if not results:
        raise AssertionError(
            f'recall found no insight for keyword {keyword!r}; '
            f'drain may not have run')
    return str(results[0]['id'])


def find_result(results: list[dict], insight_id: str) -> dict | None:
    """Return the result row whose insight.id matches, or None.
    """
    for row in results:
        if row.get('insight', {}).get('id') == insight_id:
            return row
    return None
