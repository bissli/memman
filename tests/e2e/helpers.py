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
    env['MEMMAN_CACHE_DIR'] = str(home / '.memman' / 'cache')
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


def extract_id(data: dict) -> str:
    """Extract `.facts[0].id` or fallback `.id` like the bash helper.
    """
    facts = data.get('facts')
    if facts:
        return facts[0]['id']
    return data['id']


def find_result(results: list[dict], insight_id: str) -> dict | None:
    """Return the result row whose insight.id matches, or None.
    """
    for row in results:
        if row.get('insight', {}).get('id') == insight_id:
            return row
    return None
