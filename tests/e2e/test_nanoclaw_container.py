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
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e_container]


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
    cmd += [image]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _stop_container(container_id: str) -> None:
    subprocess.run(['docker', 'stop', '-t', '1', container_id],
                   capture_output=True)


def _setup_mount(parent: Path, group: str) -> Path:
    """Create a host-side dir for the per-group bind mount.

    Mode 0o777 neutralizes the host-uid (often 1001 in CI) vs
    container-uid (1000 = node) mismatch. Pre-seeds the SQLite DB
    with a fingerprint meta row so memman's first store open does
    not trigger seed_if_fresh, which would otherwise probe the
    Voyage API (requires a live key) on every fresh container.
    """
    import sqlite3

    d = parent / 'data' / group
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o777)
    db_path = d / 'memman.db'
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
    db_path.chmod(0o666)
    return d


_BASE_ENV = (
    'MEMMAN_LLM_PROVIDER=openrouter\n'
    'MEMMAN_LLM_MODEL_FAST=anthropic/claude-haiku-4.5\n'
    'MEMMAN_LLM_MODEL_SLOW_CANONICAL=anthropic/claude-sonnet-4.6\n'
    'MEMMAN_LLM_MODEL_SLOW_METADATA=anthropic/claude-sonnet-4.6\n'
    'MEMMAN_EMBED_PROVIDER=voyage\n'
    'MEMMAN_RERANK_PROVIDER=voyage\n'
    'MEMMAN_OPENROUTER_ENDPOINT=https://openrouter.ai/api/v1\n'
    'MEMMAN_VOYAGE_RERANK_MODEL=rerank-2.5-lite\n'
    'MEMMAN_BACKEND=sqlite\n'
    'OPENROUTER_API_KEY=placeholder-for-non-live-tests\n'
    'VOYAGE_API_KEY=placeholder-for-non-live-tests\n')


def _write_env_file(container_id: str, body: str) -> None:
    """Atomic-overwrite `/home/node/.memman/env` inside the container."""
    subprocess.run(
        ['docker', 'exec', '-i', container_id,
         'sh', '-c', 'cat > /home/node/.memman/env'],
        input=body, text=True, capture_output=True, check=True)


def _seed_env_file(container_id: str,
                   keys: dict[str, str] | None = None) -> None:
    """Write the env file inside the container.

    Modern memman requires the env file to declare providers,
    endpoints, and API keys before any store open succeeds (status,
    remember, etc. all trigger seed_if_fresh -> get_client). The
    base env carries placeholder API keys so non-live tests open
    cleanly; live-key tests pass `keys` to overwrite with real
    OpenRouter + Voyage credentials before running the drain.
    """
    if keys:
        body = _BASE_ENV.replace(
            'OPENROUTER_API_KEY=placeholder-for-non-live-tests',
            f'OPENROUTER_API_KEY={keys["OPENROUTER_API_KEY"]}',
            ).replace(
            'VOYAGE_API_KEY=placeholder-for-non-live-tests',
            f'VOYAGE_API_KEY={keys["VOYAGE_API_KEY"]}',
            )
    else:
        body = _BASE_ENV
    _write_env_file(container_id, body)


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
        _seed_env_file(cid)
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
        assert data['storage_path'].endswith('memman.db')

    def test_memman_on_path(self, nanoclaw_run):
        cid, _ = nanoclaw_run()
        out = _exec(cid, ['sh', '-c', 'command -v memman'])
        assert '/memman' in out.stdout

    def test_scheduler_serve_pid1(self, nanoclaw_run):
        """PID 1 is `memman scheduler serve`; status reports serve mode.
        """
        cid, _ = nanoclaw_run()
        out = _exec(cid, ['cat', '/proc/1/cmdline'])
        cmdline = out.stdout.replace('\0', ' ')
        assert 'scheduler' in cmdline and 'serve' in cmdline, cmdline

        status_out = _exec(cid, ['memman', 'scheduler', 'status'])
        data = json.loads(status_out.stdout)
        assert data['platform'] == 'serve'
        assert data['state'] == 'started'


# ---------------------------------------------------------------------
# Per-group isolation
# ---------------------------------------------------------------------

class TestPerGroupIsolation:

    @pytest.mark.requires_live_keys
    def test_two_groups_dont_cross_read(self, nanoclaw_run, live_keys):
        """Two containers with different bind mounts must isolate data.
        """
        cid_a, _ = nanoclaw_run(group='alpha')
        cid_b, _ = nanoclaw_run(group='beta')
        _seed_env_file(cid_a, live_keys)
        _seed_env_file(cid_b, live_keys)

        unique_a = f'unique-{uuid.uuid4().hex[:8]}-alpha'
        unique_b = f'unique-{uuid.uuid4().hex[:8]}-beta'

        _exec(cid_a, ['memman', 'remember', '--no-reconcile',
                      f'Alpha private fact about {unique_a}',
                      '--cat', 'fact', '--imp', '3'])
        _exec(cid_b, ['memman', 'remember', '--no-reconcile',
                      f'Beta private fact about {unique_b}',
                      '--cat', 'fact', '--imp', '3'])

        _exec(cid_a, ['memman', 'scheduler', 'serve', '--once'])
        _exec(cid_b, ['memman', 'scheduler', 'serve', '--once'])

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
        content = out.stdout
        assert len(content) > 500, 'skill content should be substantive'
        assert '## Storing what you learn' in content
        assert '## Guardrails' in content


# ---------------------------------------------------------------------
# Reconciliation: write -> drain -> graph state
# ---------------------------------------------------------------------

class TestReconciliation:

    @pytest.mark.requires_live_keys
    def test_reconciliation_drains_and_creates_edges(
            self, nanoclaw_run, live_keys):
        """Two facts + one synchronous drain land insights and edges.

        Uses `scheduler serve --once` (not `scheduler trigger`, which
        raises in serve mode). Asserts the *global* edge_count from
        `memman status` — recall results carry no per-result edge_count.
        Temporal backbone edges between two facts on the same source
        are guaranteed by graph/temporal.py.
        """
        cid, _ = nanoclaw_run()
        _seed_env_file(cid, live_keys)

        unique = uuid.uuid4().hex[:8]
        fact_a = (f'The user prefers tabs over spaces in Python code '
                  f'on project alpha-{unique}, per their style guide.')
        fact_b = (f'The deployment for project beta-{unique} runs on '
                  f'Kubernetes 1.29 with autoscaling enabled.')

        _exec(cid, ['memman', 'remember', fact_a,
                    '--cat', 'fact', '--imp', '3'])
        _exec(cid, ['memman', 'remember', fact_b,
                    '--cat', 'fact', '--imp', '3'])

        drain = subprocess.run(
            ['docker', 'exec', cid, 'memman', 'scheduler', 'serve',
             '--once'],
            capture_output=True, text=True, timeout=90, check=True)
        assert drain.returncode == 0, drain.stderr

        status = json.loads(
            _exec(cid, ['memman', 'status']).stdout)
        assert status['edge_count'] >= 2, (
            f'expected >=2 edges after drain, got {status}')

        queue = json.loads(_exec(cid, [
            'memman', 'scheduler', 'queue', 'list']).stdout)
        assert queue['stats']['pending'] == 0, (
            f'queue not drained: {queue}')

        recall = json.loads(_exec(cid, [
            'memman', 'recall', f'alpha-{unique}', '--limit', '1']).stdout)
        assert recall['results'], (
            f'recall returned no rows for alpha-{unique}')


# ---------------------------------------------------------------------
# Error hygiene: corrupted on-disk state surfaces a clean error
# ---------------------------------------------------------------------

class TestCorruptDb:

    def test_recall_returns_clean_error_on_corrupt_db(self, nanoclaw_run):
        """Truncate the SQLite file; recall must not leak a traceback.
        """
        cid, _ = nanoclaw_run()
        _exec(cid, ['memman', 'remember', '--no-reconcile',
                    'sentinel fact for corruption test',
                    '--cat', 'fact', '--imp', '2'])

        db = '/home/node/.memman/data/default/memman.db'
        _exec(cid, ['truncate', '-s', '0', db])

        result = subprocess.run(
            ['docker', 'exec', cid, 'memman', 'recall', 'sentinel'],
            capture_output=True, text=True, check=False)
        assert result.returncode != 0, 'recall should fail on empty db'
        assert 'Traceback' not in result.stderr, (
            f'leaked Python traceback in stderr: {result.stderr}')


# ---------------------------------------------------------------------
# Session lifecycle: prime -> remember+remember -> drain -> recall
# ---------------------------------------------------------------------

class TestSessionLifecycle:

    @pytest.mark.requires_live_keys
    def test_session_accumulates_and_recalls(
            self, nanoclaw_run, live_keys):
        """One realistic session sequence accumulates state.

        prime.sh boots, two `--no-reconcile` remembers queue facts,
        `scheduler serve --once` drains synchronously (the embedding
        call inside run_remember is why this needs live keys), then
        status sees both insights and `recall --basic` finds the
        stored content. Hook isolation is covered by TestLifecycleHooks;
        the unique value here is end-to-end accumulation across writes.
        """
        cid, _ = nanoclaw_run()
        _seed_env_file(cid, live_keys)

        out = _exec(cid, ['/app/hooks/memman/prime.sh'])
        assert '[memman] Memory active' in out.stdout

        _exec(cid, ['memman', 'remember', '--no-reconcile',
                    'Lifecycle fact A',
                    '--cat', 'fact', '--imp', '3'])
        _exec(cid, ['memman', 'remember', '--no-reconcile',
                    'Lifecycle fact B',
                    '--cat', 'fact', '--imp', '3'])
        _exec(cid, ['memman', 'scheduler', 'serve', '--once'])

        status = json.loads(_exec(cid, ['memman', 'status']).stdout)
        assert status['total_insights'] == 2, status

        recall = _exec(cid, ['memman', 'recall', '--basic', 'Lifecycle'])
        assert 'Lifecycle fact A' in recall.stdout
        assert 'Lifecycle fact B' in recall.stdout


# ---------------------------------------------------------------------
# Read-only global mount: documented in assets/nanoclaw/SKILL.md
# ---------------------------------------------------------------------

class TestGlobalMountReadOnly:

    def test_global_mount_rejects_writes(self, nanoclaw_run, tmp_path):
        """The /home/node/.memman/data/global/ mount honors :ro.

        Positive cross-check via /proc/self/mountinfo (mirrors
        TestStatus.test_scheduler_serve_pid1's /proc/1/cmdline check)
        plus the EROFS error string distinguishes "ro mount enforced"
        from any other write-failure mode.
        """
        global_host = tmp_path / 'global'
        global_host.mkdir()
        (global_host / 'sentinel').write_text('host-side write OK\n')
        extra = [(global_host,
                  '/home/node/.memman/data/global', 'ro')]

        cid, _ = nanoclaw_run(extra_mounts=extra)

        out = _exec(cid, ['cat',
                          '/home/node/.memman/data/global/sentinel'])
        assert 'host-side write OK' in out.stdout

        mountinfo = _exec(cid, ['cat', '/proc/self/mountinfo']).stdout
        global_lines = [ln for ln in mountinfo.splitlines()
                        if '/home/node/.memman/data/global' in ln]
        assert global_lines, 'global mount not present in mountinfo'
        line = global_lines[0]
        assert ' ro,' in line or ' ro ' in line, (
            f'global mount not flagged ro: {line}')

        write = _exec(
            cid,
            ['sh', '-c',
             'echo bad > /home/node/.memman/data/global/leak'],
            check=False)
        assert write.returncode != 0
        err = (write.stderr + write.stdout).lower()
        assert 'read-only file system' in err, err
