"""Fresh shell and backend boundaries for the Codex integration."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from memman import config
from memman.store.errors import ConfigError
from memman.store.factory import drop_store, open_backend


@pytest.mark.parametrize('operation', [open_backend, drop_store])
@pytest.mark.parametrize('absolute', [False, True])
def test_backend_rejects_paths_before_opening_or_dropping(
        tmp_path, operation, absolute):
    """Verify open_backend and drop_store reject path-like store names.

    Mutation: The valid_store_name guard is removed from open_backend or
    drop_store, so a traversal or absolute name reaches the filesystem.
    Oracle: A marker file outside the store root with hand-written bytes; it
    must stay untouched and no data dir may appear.
    """
    data_dir = tmp_path / 'backend-data'
    outside = data_dir / 'outside'
    outside.mkdir(parents=True)
    marker = outside / 'memman.db'
    marker.write_bytes(b'untouched outside the store root')
    name = str(outside) if absolute else '../outside'
    with pytest.raises(ConfigError, match='invalid store name'):
        operation(name, str(data_dir))
    assert marker.read_bytes() == b'untouched outside the store root'
    assert not (data_dir / 'data').exists()


@pytest.mark.no_default_env
def test_subprocess_install_env_store_and_uninstall(tmp_path):
    """Verify real processes install, honor a store from env, and uninstall.

    Mutation: remember ignores MEMMAN_STORE for the active file, install skips
    the default store or scheduler state, or uninstall leaves the skill, state,
    or key.
    Oracle: Fresh child processes with a scrubbed env and no host agents,
    services, or provider traffic; queue.db rows, the active file, and env
    keys.
    """
    home = Path.home()
    data_dir = tmp_path / 'fresh data'
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    binary = bindir / 'memman'
    binary.write_text(f'#!{sys.executable}\nfrom memman.cli import cli\ncli()\n')
    binary.chmod(0o755)
    # Use a non-OpenRouter endpoint so installation has no catalog lookup.
    # Only the embedding availability probe is stubbed in the child; its
    # configuration, provider construction, and store fingerprint stay real.
    env = dict(os.environ)
    for name in list(env):
        if name.startswith('MEMMAN_') or name in {
                'VOYAGE_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY',
                'CODEX_HOME'}:
            env.pop(name)
    env.update({
        'HOME': str(home), 'PATH': str(bindir),
        config.DATA_DIR: str(data_dir),
        config.SCHEDULER_KIND: 'serve',
        config.LLM_ENDPOINT: 'http://127.0.0.1:1/v1',
        config.LLM_MODEL: 'test-model',
        config.VOYAGE_API_KEY: 'test-only-voyage-key',
    })

    def invoke(*args):
        script = (
            'from memman.embed.voyage import Client; '
            'Client.available = lambda self: True; '
            'from memman.cli import cli; cli()')
        result = subprocess.run(
            [sys.executable, '-c', script, *args],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    invoke('install', '--codex', '--no-wizard')
    skill = home / '.agents/skills/memman'
    assert (skill / 'SKILL.md').is_file()
    assert not (home / '.claude').exists()
    assert (data_dir / 'data/default/memman.db').is_file()
    assert (home / '.memman/scheduler.state').read_text().strip() == 'started'
    invoke('store', 'create', 'configured')
    invoke('store', 'create', 'from_env')
    invoke('store', 'use', 'configured')
    env[config.STORE] = 'from_env'
    saved = json.loads(invoke('remember', 'The canary deployment uses blue.'))
    assert saved['store'] == 'from_env'
    # Observe the persisted queue directly: no worker/network is required to
    # prove that the child shell's environment selected the write destination.
    with sqlite3.connect(data_dir / 'queue.db') as connection:
        assert connection.execute('select store from queue').fetchall() == [
            ('from_env',)]
    assert (data_dir / 'active').read_text().strip() == 'configured'
    invoke('uninstall', '--codex')
    assert not skill.is_symlink()
    assert not (home / '.memman/scheduler.state').exists()
    values = config.parse_env_file(data_dir / 'env')
    assert config.VOYAGE_API_KEY not in values
    assert (data_dir / 'queue.db').exists()
    assert (data_dir / 'data/from_env/memman.db').is_file()
