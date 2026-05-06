"""Per-store env-key helpers for the routing plan.

Slice 2.1: read-only helpers and validator extension. No call site
dispatches on these keys yet; this slice only adds the data-shape
surface and confirms the validator accepts per-store-suffixed keys.
"""

import pytest


def test_backend_for_builds_namespaced_key():
    """BACKEND_FOR(store) -> 'MEMMAN_BACKEND_<store>'.
    """
    from memman import config
    assert config.BACKEND_FOR('main') == 'MEMMAN_BACKEND_main'
    assert config.BACKEND_FOR('shared-2') == 'MEMMAN_BACKEND_shared-2'


def test_pg_dsn_for_builds_namespaced_key():
    """PG_DSN_FOR(store) -> 'MEMMAN_PG_DSN_<store>'.
    """
    from memman import config
    assert config.PG_DSN_FOR('main') == 'MEMMAN_PG_DSN_main'


def test_default_backend_constant():
    """`DEFAULT_BACKEND` and `DEFAULT_PG_DSN` use the documented names.
    """
    from memman import config
    assert config.DEFAULT_BACKEND == 'MEMMAN_DEFAULT_BACKEND'
    assert config.DEFAULT_PG_DSN == 'MEMMAN_DEFAULT_PG_DSN'


def test_get_store_backend_returns_value_when_set(env_file):
    """`get_store_backend` returns the per-store value or None.
    """
    from memman import config
    env_file('MEMMAN_BACKEND_main', 'postgres')
    assert config.get_store_backend('main') == 'postgres'
    assert config.get_store_backend('other') is None


def test_get_store_pg_dsn_returns_value_when_set(env_file):
    """`get_store_pg_dsn` returns the per-store DSN or None.
    """
    from memman import config
    env_file('MEMMAN_PG_DSN_main', 'postgresql://example/x')
    assert config.get_store_pg_dsn('main') == 'postgresql://example/x'
    assert config.get_store_pg_dsn('other') is None


def test_validator_accepts_per_store_pg_dsn_keys():
    """`MEMMAN_PG_DSN_<store>` does not trip the postgres validator.
    """
    from memman.store.config import validate_for
    validate_for('postgres', {
        'MEMMAN_PG_DSN_main': 'postgresql://x',
        'MEMMAN_PG_DSN_shared': 'postgresql://y',
        })


def test_validator_rejects_per_store_pg_key_with_invalid_suffix():
    """An invalid suffix (slashes, spaces) is rejected.
    """
    from memman.store.config import validate_for
    from memman.store.errors import ConfigError
    with pytest.raises(ConfigError):
        validate_for('postgres', {
            'MEMMAN_PG_DSN_/etc/passwd': 'oops',
            })


def test_validator_rejects_unknown_per_store_canonical_key():
    """A `MEMMAN_PG_<unknown>_<store>` key is still rejected.
    """
    from memman.store.config import validate_for
    from memman.store.errors import ConfigError
    with pytest.raises(ConfigError):
        validate_for('postgres', {
            'MEMMAN_PG_FAKE_KEY_main': 'value',
            })
