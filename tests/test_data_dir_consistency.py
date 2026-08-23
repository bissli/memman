"""One `data_dir` must select one env file for every setting.

`config.get` resolves against `MEMMAN_DATA_DIR`, while
`get_store_backend` / `get_store_pg_dsn` take an explicit `data_dir`.
Composing the two read per-store keys from the passed directory and
their defaults from the ambient one, so `memman --data-dir X` routed
a store by X's per-store keys and the home directory's defaults.
"""

import pytest
from memman import config
from memman.store.factory import resolve_store_backend
from memman.store.factory import resolve_store_pg_dsn


@pytest.fixture
def split_dirs(tmp_path, monkeypatch):
    """Point `MEMMAN_DATA_DIR` at a decoy holding opposite defaults."""
    real = tmp_path / 'real'
    decoy = tmp_path / 'decoy'
    real.mkdir()
    decoy.mkdir()
    (real / 'env').write_text(
        'MEMMAN_DEFAULT_BACKEND=postgres\n'
        'MEMMAN_DEFAULT_POSTGRES_DSN=postgresql://u@real:5432/d\n')
    (decoy / 'env').write_text(
        'MEMMAN_DEFAULT_BACKEND=sqlite\n'
        'MEMMAN_DEFAULT_POSTGRES_DSN=postgresql://u@decoy:5432/d\n')
    monkeypatch.setenv('MEMMAN_DATA_DIR', str(decoy))
    config.reset_file_cache()
    return str(real)


def test_default_backend_comes_from_the_passed_data_dir(split_dirs):
    """Backend resolution reads the default from `data_dir`.

    Mutation: composing `config.get(DEFAULT_BACKEND)` instead of the
    scoped read, which resolves against `MEMMAN_DATA_DIR` and returns
    the decoy's answer.
    Oracle: the two directories declare opposite backends, so the
    result names which file was read.
    """
    assert resolve_store_backend('shop', split_dirs) == 'postgres'


def test_default_dsn_comes_from_the_passed_data_dir(split_dirs):
    """DSN resolution reads the default from `data_dir` too.

    Mutation: leaving `resolve_store_pg_dsn` on `config.get`, which
    hands back the decoy DSN -- a store then opens against the wrong
    server while its backend came from the right file.
    Oracle: the two DSNs differ by host, so the host names the file.
    """
    dsn = resolve_store_pg_dsn('shop', split_dirs)

    assert dsn is not None
    assert '@real:' in dsn


def test_per_store_key_still_wins_over_the_scoped_default(split_dirs,
                                                          tmp_path):
    """A per-store key overrides the default from the same file.

    Mutation: reading the scoped default first, which would ignore
    explicit per-store routing.
    Oracle: the file declares `postgres` by default and `sqlite` for
    this one store.
    """
    env = tmp_path / 'real' / 'env'
    env.write_text(env.read_text() + 'MEMMAN_BACKEND_shop=sqlite\n')
    config.reset_file_cache()

    assert resolve_store_backend('shop', split_dirs) == 'sqlite'
