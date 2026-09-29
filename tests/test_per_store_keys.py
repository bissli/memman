"""Per-store env-key helpers for the routing plan.

Read-only helpers and a validator that accepts per-store-suffixed
keys.
"""

import pytest
from memman import config
from memman.store.config import validate_all
from memman.store.errors import ConfigError
from tests.conftest import invoke


def test_backend_for_builds_namespaced_key():
    """Verify BACKEND_FOR builds 'MEMMAN_BACKEND_<store>'.

    Mutation: dropping the store suffix or changing the prefix.
    Oracle: hand-written key strings, including a hyphenated store.
    """
    assert config.BACKEND_FOR('main') == 'MEMMAN_BACKEND_main'
    assert config.BACKEND_FOR('shared-2') == 'MEMMAN_BACKEND_shared-2'


def test_pg_dsn_for_builds_namespaced_key():
    """Verify POSTGRES_DSN_FOR builds 'MEMMAN_POSTGRES_DSN_<store>'.

    Mutation: changing the prefix, or dropping the store suffix.
    Oracle: hand-written keys for a plain and a hyphenated store name.
    """
    assert config.POSTGRES_DSN_FOR('main') == 'MEMMAN_POSTGRES_DSN_main'
    assert config.POSTGRES_DSN_FOR('shared-2') == (
        'MEMMAN_POSTGRES_DSN_shared-2')


def test_default_backend_constant():
    """Verify DEFAULT_BACKEND and DEFAULT_PG_DSN use the documented names.

    Mutation: renaming either key, which orphans values operators already
        wrote to their env files.
    Oracle: the literal documented key strings.
    """
    assert config.DEFAULT_BACKEND == 'MEMMAN_DEFAULT_BACKEND'
    assert config.DEFAULT_PG_DSN == 'MEMMAN_DEFAULT_POSTGRES_DSN'


def test_get_store_backend_returns_value_when_set(env_file):
    """Verify get_store_backend returns the per-store value or None.

    Mutation: returning another store's value, or a default string instead
        of None for an unset store.
    Oracle: one store set to 'postgres' and an unset store.
    """
    env_file('MEMMAN_BACKEND_main', 'postgres')
    assert config.get_store_backend('main') == 'postgres'
    assert config.get_store_backend('other') is None


def test_get_store_pg_dsn_returns_value_when_set(env_file):
    """Verify get_store_pg_dsn returns the per-store DSN or None.

    Mutation: returning another store's DSN, or a default instead of None
        for an unset store.
    Oracle: one store set to a hand-written DSN and an unset store.
    """
    env_file('MEMMAN_POSTGRES_DSN_main', 'postgresql://example/x')
    assert config.get_store_pg_dsn('main') == 'postgresql://example/x'
    assert config.get_store_pg_dsn('other') is None


def test_validator_accepts_per_store_pg_dsn_keys():
    """Verify per-store DSN keys pass the postgres validator.

    Mutation: rejecting a suffixed key as unknown.
    Oracle: validate_all returning without ConfigError.
    """
    validate_all({
        'MEMMAN_POSTGRES_DSN_main': 'postgresql://x',
        'MEMMAN_POSTGRES_DSN_shared': 'postgresql://y',
        })


def test_validator_rejects_per_store_pg_key_with_invalid_suffix():
    """Verify a suffix with slashes is rejected.

    Mutation: accepting any suffix, which lets a path-like store name
        through.
    Oracle: ConfigError for 'MEMMAN_POSTGRES_DSN_/etc/passwd'.
    """
    with pytest.raises(ConfigError):
        validate_all({
            'MEMMAN_POSTGRES_DSN_/etc/passwd': 'oops',
            })


def test_validator_rejects_unknown_per_store_canonical_key():
    """Verify an unknown per-store postgres key is still rejected.

    Mutation: a suffix pattern loose enough to accept any
        MEMMAN_POSTGRES_ key.
    Oracle: ConfigError for 'MEMMAN_POSTGRES_FAKE_KEY_main'.
    """
    with pytest.raises(ConfigError):
        validate_all({
            'MEMMAN_POSTGRES_FAKE_KEY_main': 'value',
            })


def test_config_set_per_store_backend(mm_runner):
    """Verify `config set MEMMAN_BACKEND_<store> postgres` is accepted.

    Mutation: the config-set validator rejecting the per-store routing key.
    Oracle: exit code 0 and the 'set MEMMAN_BACKEND_work' message.
    """
    result = invoke(mm_runner, [
        'config', 'set', 'MEMMAN_BACKEND_work', 'postgres'])
    assert result.exit_code == 0, result.output
    assert 'set MEMMAN_BACKEND_work' in result.output


def test_config_set_per_store_pg_dsn(mm_runner):
    """Verify `config set MEMMAN_POSTGRES_DSN_<store> <url>` is accepted.

    Mutation: the config-set validator rejecting the per-store DSN key.
    Oracle: exit code 0.
    """
    result = invoke(mm_runner, [
        'config', 'set', 'MEMMAN_POSTGRES_DSN_work',
        'postgresql://localhost/x'])
    assert result.exit_code == 0, result.output


def test_config_set_rejects_bare_memman_backend(mm_runner):
    """Verify the bare `MEMMAN_BACKEND` key is rejected with a hint.

    Mutation: accepting the bare key, or rejecting it without naming the
        default and per-store forms.
    Oracle: nonzero exit and both replacement key names in the output.
    """
    result = invoke(mm_runner, [
        'config', 'set', 'MEMMAN_BACKEND', 'postgres'])
    assert result.exit_code != 0
    assert 'MEMMAN_DEFAULT_BACKEND' in result.output
    assert 'MEMMAN_BACKEND_<store>' in result.output


def test_config_set_rejects_bare_memman_pg_dsn(mm_runner):
    """Verify the bare `MEMMAN_POSTGRES_DSN` key is rejected with a hint.

    Mutation: accepting the bare key, or rejecting it without naming the
        default and per-store forms.
    Oracle: nonzero exit and both replacement key names in the output.
    """
    result = invoke(mm_runner, [
        'config', 'set', 'MEMMAN_POSTGRES_DSN', 'postgresql://x'])
    assert result.exit_code != 0
    assert 'MEMMAN_DEFAULT_POSTGRES_DSN' in result.output
    assert 'MEMMAN_POSTGRES_DSN_<store>' in result.output


def test_config_set_rejects_unrecognized_key_with_hint(mm_runner):
    """Verify an unrecognized key is rejected with the shape-list hint.

    Mutation: accepting an unknown key, or rejecting it without listing
        the accepted per-store shapes.
    Oracle: nonzero exit and the hand-written hint fragments.
    """
    result = invoke(mm_runner, [
        'config', 'set', 'MEMMAN_NOT_A_KEY', 'value'])
    assert result.exit_code != 0
    assert 'not a recognized config key' in result.output
    assert 'MEMMAN_BACKEND_<store>' in result.output
    assert 'MEMMAN_POSTGRES_DSN_<store>' in result.output
