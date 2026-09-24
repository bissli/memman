"""Fixtures for memman end-to-end tests.

The unit suite at `tests/conftest.py` autouses two patches
(`_scheduler_started`, `_mock_apis`) that are explicitly guarded
against the `tests/e2e/` subtree - see those fixtures' first-line
short-circuit. We re-emphasize the contract here:

- e2e tests must run the *real* `memman` CLI in a subprocess with no
  in-process monkeypatches affecting it.
- e2e tests gate on real `OPENROUTER_API_KEY` / `VOYAGE_API_KEY` via
  the `live_keys` fixture rather than on a `--live` flag.
- the `memman_home` fixture handles HOME redirection (env + Path.home)
  and the started-state file. Hosts without systemd/launchd opt into
  serve mode via `MEMMAN_SCHEDULER_KIND=serve` in the test environment.
"""

import os
import sqlite3
from pathlib import Path

import pytest
from memman import config

_PLACEHOLDER = 'placeholder-for-non-live-tests'

_SECRET_KEYS = (
    'MEMMAN_OPENROUTER_API_KEY',
    'MEMMAN_VOYAGE_API_KEY',
    'MEMMAN_LLM_API_KEY',
    )


def resolve_e2e_secret(name: str) -> str:
    """Resolve an API-key env var for e2e fixtures.

    Order matches the canonical runtime resolver: shell `os.environ`
    first (CI exports secrets), then `~/.memman/env` (local dev keeps
    them in the runtime config file), then a stable placeholder so
    non-live tests still emit a parseable env file.
    """
    user_env = Path.home() / '.memman' / config.ENV_FILENAME
    val = os.environ.get(name)
    if not val and user_env.exists():
        val = config.parse_env_file(user_env).get(name)
    return val or _PLACEHOLDER


def build_e2e_env_body() -> str:
    """Render a `KEY=VALUE`-per-line env-file body for e2e tests.

    Starts from `config.INSTALL_DEFAULTS` so model/provider/endpoint
    constants stay in sync with what `memman install` writes, then
    overlays e2e-specific secrets resolved from shell env or
    `~/.memman/env`.
    """
    rows: dict[str, str] = dict(config.INSTALL_DEFAULTS)
    rows[config.DEFAULT_BACKEND] = 'sqlite'
    for name in _SECRET_KEYS:
        rows[name] = resolve_e2e_secret(name)
    if rows['MEMMAN_LLM_API_KEY'] == _PLACEHOLDER:
        rows['MEMMAN_LLM_API_KEY'] = rows['MEMMAN_OPENROUTER_API_KEY']
    return ''.join(f'{k}={v}\n' for k, v in rows.items())


def seed_fingerprint(db_path: Path) -> None:
    """Write a voyage-3-lite/512 fingerprint into the store DB.

    Pre-seeding lets `seed_if_fresh` short-circuit on a fresh store,
    avoiding the live Voyage probe on every store-open. Used by e2e
    fixtures that don't need a real embed pipeline (CLI plumbing
    tests, list/use/remove behavior).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            'create table if not exists meta '
            '(key text primary key, value text not null)')
        conn.execute(
            "insert or replace into meta (key, value) values "
            "('embed_fingerprint', "
            '\'{"provider":"voyage","model":"voyage-3-lite","dim":512}\')')
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def memman_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                ) -> tuple[Path, Path]:
    """Per-test isolated HOME with scheduler state pre-written.

    Returns `(home_dir, data_dir)`. Tests pass `data_dir` to
    `run_cli` so each test gets its own SQLite store under
    `<tmp>/memman_data/`. `MEMMAN_SCHEDULER_KIND=serve` is set so
    scheduler-stop / -start commands route to the serve branch
    rather than raising. Two-step HOME patch (env + Path.home class
    attr) - the env var alone is not enough on macOS, where
    `Path.home()` may use `pwd.getpwuid` instead of `$HOME`.
    """
    home = tmp_path
    dot = home / '.memman'
    dot.mkdir(parents=True, exist_ok=True)
    (dot / 'scheduler.state').write_text('started\n')
    (dot / 'scheduler.state').chmod(0o600)

    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: home)
    monkeypatch.setenv('MEMMAN_SCHEDULER_KIND', 'serve')

    data_dir = tmp_path / 'memman_data'
    data_dir.mkdir(exist_ok=True)
    return home, data_dir


@pytest.fixture(scope='session')
def live_keys() -> None:
    """Gate live-key tests; fail-loud under MEMMAN_E2E_REQUIRE_LIVE=1.

    Tests that exercise enrichment, intent expansion, or any path that
    hits OpenRouter / Voyage take this fixture as a dependency. With
    `MEMMAN_E2E_REQUIRE_LIVE=1` set, missing keys raise instead of
    skipping silently.

    Resolution order: `os.environ` first (CI path: secrets exported
    via workflow), then the developer's canonical `~/.memman/env`.
    """
    require = os.environ.get('MEMMAN_E2E_REQUIRE_LIVE') == '1'
    for name in ('MEMMAN_OPENROUTER_API_KEY', 'MEMMAN_VOYAGE_API_KEY'):
        val = resolve_e2e_secret(name)
        if val in {_PLACEHOLDER, 'mock-key-for-testing'}:
            msg = f'{name} not set; live e2e test cannot run'
            if require:
                pytest.fail(msg)
            pytest.skip(msg)


from tests.conftest import _safe_store_name as _safe  # noqa: E402,F401


def _pg_vec(seed: int, dim: int = 512) -> list[float]:
    """Deterministic vector of length `dim` for Postgres e2e fixtures.

    The arithmetic-progression form `(seed + i) * 0.001` produces
    distinct vectors per seed while staying inside pgvector's value
    range for any practical dim. Three e2e files used to inline a
    byte-identical copy of this helper.
    """
    return [(seed + i) * 0.001 for i in range(dim)]
