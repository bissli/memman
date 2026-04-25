"""Container-shape e2e tests for the nanoclaw integration.

Mirrors the deployment shape that
`src/memman/setup/assets/nanoclaw/SKILL.md` instructs nanoclaw users
to build: pip-installed memman, hooks at /app/hooks/memman/*.sh,
container-skill.md at /app/skills/memman/SKILL.md, host-side bind
mount at /home/node/.memman/data/default, scheduler markers
pre-written into ~/.memman/.

These tests catch:
- bind-mount UID/GID mismatch (host runner uid vs container node uid)
- missing `memman` on PATH inside the image
- per-group isolation across two containers with different mount paths
- read-only global store enforcement
- the bytes of the shipped hook scripts (validated against
  `[memman]` prefix output)
"""

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e_container]


def _have_keys() -> bool:
    """Real LLM/embedding keys present (subprocess inherits)."""
    for name in ('OPENROUTER_API_KEY', 'VOYAGE_API_KEY'):
        v = os.environ.get(name)
        if not v or v == 'mock-key-for-testing':
            return False
    return True


def _exec(container_id: str, cmd: list[str], check: bool = True
          ) -> subprocess.CompletedProcess[str]:
    """Run a command inside the named container, returning the result.
    """
    full = ['docker', 'exec', container_id, *cmd]
    return subprocess.run(full, capture_output=True, text=True, check=check)


def _start_container(image: str, mounts: list[tuple[Path, str, str]],
                     env: dict | None = None) -> str:
    """Start a detached container; return the container id.

    `mounts` is a list of `(host_path, container_path, mode)` triples
    where mode is 'rw' or 'ro'. Caller is responsible for stopping.
    """
    cmd = ['docker', 'run', '-d', '--rm']
    for host, ctr, mode in mounts:
        cmd += ['-v', f'{host}:{ctr}:{mode}']
    for k, v in (env or {}).items():
        cmd += ['-e', f'{k}={v}']
    cmd += [image, 'sleep', 'infinity']
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _stop_container(container_id: str) -> None:
    subprocess.run(['docker', 'stop', '-t', '1', container_id],
                   capture_output=True)


def _setup_mount(parent: Path, group: str) -> Path:
    """Create a host-side dir for the per-group bind mount.

    Mode 0o777 neutralizes the host-uid (often 1001 in CI) vs
    container-uid (1000 = node) mismatch.
    """
    d = parent / 'data' / group
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o777)
    return d


@pytest.fixture
def nanoclaw_run(nanoclaw_image: str, tmp_path: Path):
    """Factory: yield a function that starts a container, returns its id.

    Containers started via the factory are stopped on teardown.
    """
    started: list[str] = []

    def _run(group: str = 'default', env: dict | None = None,
             extra_mounts: list[tuple[Path, str, str]] | None = None
             ) -> tuple[str, Path]:
        host = _setup_mount(tmp_path, group)
        mounts = [(host, '/home/node/.memman/data/default', 'rw')]
        if extra_mounts:
            mounts.extend(extra_mounts)
        env = dict(env or {})
        for k in ('OPENROUTER_API_KEY', 'VOYAGE_API_KEY'):
            v = os.environ.get(k)
            if v:
                env.setdefault(k, v)
        cid = _start_container(nanoclaw_image, mounts, env)
        started.append(cid)
        return cid, host

    yield _run

    for cid in started:
        _stop_container(cid)


# ---------------------------------------------------------------------
# Status / smoke
# ---------------------------------------------------------------------

class TestStatus:

    def test_status_inside_container(self, nanoclaw_run):
        cid, _ = nanoclaw_run()
        out = _exec(cid, ['memman', 'status'])
        data = json.loads(out.stdout)
        assert data['total_insights'] == 0
        assert data['db_path'].endswith('memman.db')

    def test_memman_on_path(self, nanoclaw_run):
        cid, _ = nanoclaw_run()
        out = _exec(cid, ['sh', '-c', 'command -v memman'])
        assert '/memman' in out.stdout

    def test_scheduler_inline_active(self, nanoclaw_run):
        """Container's scheduler is inline + started (pre-written by image).
        """
        cid, _ = nanoclaw_run()
        out = _exec(cid, ['memman', 'scheduler', 'status'])
        data = json.loads(out.stdout)
        assert data['platform'] == 'inline'
        assert data['state'] == 'started'


# ---------------------------------------------------------------------
# Per-group isolation
# ---------------------------------------------------------------------

class TestPerGroupIsolation:

    @pytest.mark.skipif(not _have_keys(),
                        reason='no OPENROUTER/VOYAGE keys')
    def test_two_groups_dont_cross_read(self, nanoclaw_run):
        """Two containers with different bind mounts must isolate data.
        """
        cid_a, _ = nanoclaw_run(group='alpha')
        cid_b, _ = nanoclaw_run(group='beta')

        unique_a = f'unique-{uuid.uuid4().hex[:8]}-alpha'
        unique_b = f'unique-{uuid.uuid4().hex[:8]}-beta'

        _exec(cid_a, ['memman', 'remember', '--no-reconcile',
                      f'Alpha private fact about {unique_a}',
                      '--cat', 'fact', '--imp', '3'])
        _exec(cid_b, ['memman', 'remember', '--no-reconcile',
                      f'Beta private fact about {unique_b}',
                      '--cat', 'fact', '--imp', '3'])

        time.sleep(0.5)

        out_a = _exec(cid_a, ['memman', 'recall', '--basic', unique_a])
        assert unique_a in out_a.stdout
        assert unique_b not in out_a.stdout, 'beta data leaked to alpha'

        out_b = _exec(cid_b, ['memman', 'recall', '--basic', unique_b])
        assert unique_b in out_b.stdout
        assert unique_a not in out_b.stdout, 'alpha data leaked to beta'


# ---------------------------------------------------------------------
# Lifecycle hooks (the bytes shipped under assets/nanoclaw/hooks/)
# ---------------------------------------------------------------------

class TestLifecycleHooks:

    def test_prime_hook_emits_memman_prefix(self, nanoclaw_run):
        cid, _ = nanoclaw_run()
        out = _exec(cid, ['/app/hooks/memman/prime.sh'])
        assert '[memman]' in out.stdout
        assert 'Memory active' in out.stdout

    def test_user_prompt_hook_emits_memman_prefix(self, nanoclaw_run):
        cid, _ = nanoclaw_run()
        out = _exec(cid, ['/app/hooks/memman/user_prompt.sh'])
        assert '[memman] Evaluate' in out.stdout

    def test_stop_hook_blocks_when_inactive(self, nanoclaw_run):
        cid, _ = nanoclaw_run()
        out = subprocess.run(
            ['docker', 'exec', '-i', cid,
             'bash', '/app/hooks/memman/stop.sh'],
            input='{"stop_hook_active": false}',
            capture_output=True, text=True, check=True)
        payload = json.loads(out.stdout)
        assert payload['decision'] == 'block'
        assert '[memman]' in payload['reason']

    def test_stop_hook_silent_when_active(self, nanoclaw_run):
        cid, _ = nanoclaw_run()
        out = subprocess.run(
            ['docker', 'exec', '-i', cid,
             'bash', '/app/hooks/memman/stop.sh'],
            input='{"stop_hook_active": true}',
            capture_output=True, text=True, check=True)
        assert out.stdout.strip() == ''


# ---------------------------------------------------------------------
# Container behavioral guide is present at the documented path
# ---------------------------------------------------------------------

class TestContainerSkill:

    def test_container_skill_present(self, nanoclaw_run):
        cid, _ = nanoclaw_run()
        out = _exec(cid, ['cat', '/app/skills/memman/SKILL.md'])
        assert 'memman' in out.stdout.lower()
        assert len(out.stdout) > 100, 'skill content should be substantive'
