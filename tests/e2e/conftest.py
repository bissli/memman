"""Fixtures for memman end-to-end tests.

The unit suite at `tests/conftest.py` autouses two patches
(`_scheduler_started`, `_mock_apis`) that are explicitly guarded
against the `tests/e2e/` subtree — see those fixtures' first-line
short-circuit. We re-emphasize the contract here:

- e2e tests must run the *real* `memman` CLI in a subprocess with no
  in-process monkeypatches affecting it.
- e2e tests gate on real `OPENROUTER_API_KEY` / `VOYAGE_API_KEY` via
  the `live_keys` fixture rather than on a `--live` flag.
- the `memman_home` fixture handles HOME redirection (env + Path.home),
  the started-state file, and MEMMAN_CACHE_DIR redirect. Hosts without
  systemd/launchd opt into serve mode via `MEMMAN_SCHEDULER_KIND=serve`
  in the test environment.
"""

import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def memman_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                ) -> tuple[Path, Path]:
    """Per-test isolated HOME with scheduler state pre-written.

    Returns `(home_dir, data_dir)`. Tests pass `data_dir` to
    `run_cli` so each test gets its own SQLite store under
    `<tmp>/memman_data/`. `MEMMAN_SCHEDULER_KIND=serve` is set so
    scheduler-stop / -start commands route to the serve branch
    rather than raising. Two-step HOME patch (env + Path.home class
    attr) — the env var alone is not enough on macOS, where
    `Path.home()` may use `pwd.getpwuid` instead of `$HOME`.
    """
    home = tmp_path
    dot = home / '.memman'
    dot.mkdir(parents=True, exist_ok=True)
    (dot / 'scheduler.state').write_text('started\n')
    (dot / 'scheduler.state').chmod(0o600)
    (dot / 'cache').mkdir(exist_ok=True)

    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: home)
    monkeypatch.setenv('MEMMAN_CACHE_DIR', str(dot / 'cache'))
    monkeypatch.setenv('MEMMAN_SCHEDULER_KIND', 'serve')

    data_dir = tmp_path / 'memman_data'
    data_dir.mkdir(exist_ok=True)
    return home, data_dir


@pytest.fixture(scope='session')
def live_keys() -> dict[str, str]:
    """Skip the test cleanly when real LLM/embedding keys are absent.

    Tests that exercise enrichment, causal inference, intent
    expansion, or any path that hits OpenRouter / Voyage take this
    fixture as a dependency. Returns the actual key values for
    callers that need to pass them through into containers.
    """
    keys = {}
    for name in ('OPENROUTER_API_KEY', 'VOYAGE_API_KEY'):
        val = os.environ.get(name)
        if not val or val == 'mock-key-for-testing':
            pytest.skip(f'{name} not set; skipping live e2e test')
        keys[name] = val
    return keys


@pytest.fixture(scope='session')
def docker_available() -> None:
    """Skip when the Docker daemon is unreachable.

    Used by container-level tests. Importing `docker` and pinging the
    daemon catches both 'no daemon' and 'permission denied' before
    testcontainers raises a less-descriptive error.
    """
    if shutil.which('docker') is None:
        pytest.skip('docker CLI not on PATH; skipping container test')
    try:
        import docker as _docker
        client = _docker.from_env()
        client.ping()
    except Exception as exc:
        pytest.skip(f'Docker daemon unreachable: {exc}')


@pytest.fixture(scope='session')
def repo_root() -> Path:
    """Absolute path to the memman repository root.
    """
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope='session')
def nanoclaw_image(docker_available: None, repo_root: Path) -> str:
    """Build (or reuse) the nanoclaw test image once per session.

    The image is tagged `memman-e2e-nanoclaw:dev` and kept across
    sessions so Docker's layer cache makes subsequent builds fast.
    Cleanup is deliberate: we do *not* remove the image on session
    exit. testcontainers' Ryuk reaper handles container cleanup.
    """
    from testcontainers.core.image import DockerImage

    tag = 'memman-e2e-nanoclaw:dev'
    DockerImage(
        path=str(repo_root),
        dockerfile_path='tests/e2e/Dockerfile.nanoclaw',
        tag=tag,
        clean_up=False,
    ).build()
    return tag


def _writable_mount_dir(parent: Path, group: str) -> Path:
    """Create a host-side mount dir the container's `node` user (uid
    1000) can write to. CI runners use uid 1001 by default; chmod 0o777
    avoids the EACCES that bind-mount uid mismatch otherwise causes.
    """
    d = parent / 'data' / group
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o777)
    state = d / 'scheduler.state'
    state.write_text('started\n')
    return d
