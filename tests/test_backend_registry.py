"""Backend registry surface tests.

Pins the load-bearing contract of the `BACKENDS` static registry:
- `known_backends()` reads from the static dict literal.
- `descriptor(name)` raises ConfigError on unknown backends.
- `--to` Click choices are dynamic; a synthetic third backend
  added at test time appears in the help output.
- `extras.detect_active_extras` reads through `extras_packages`.

These tests guard the abstraction: adding a third RDBMS backend
should require ONE new entry in the `BACKENDS` dict and a
`_build_<name>_descriptor()` factory, with no
edits to factory.py / cli.py / doctor.py / extras.py dispatch.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner
from memman import extras
from memman.cli import cli
from memman.store.errors import ConfigError
from memman.store.factory import BACKENDS, BackendDescriptor, all_descriptors
from memman.store.factory import descriptor, known_backends


def test_known_backends_reads_from_static_dict():
    """Verify known_backends() returns exactly the keys of BACKENDS.

    Mutation: known_backends() returning a hard-coded or filtered set that
        drifts from the registry dict.
    Oracle: frozenset of BACKENDS keys.
    """
    assert known_backends() == frozenset(BACKENDS.keys())


def test_known_backends_includes_sqlite_and_postgres():
    """Verify both shipped backends are registered.

    Mutation: dropping either backend's entry from BACKENDS.
    Oracle: the literal names 'sqlite' and 'postgres'.
    """
    names = known_backends()
    assert 'sqlite' in names
    assert 'postgres' in names


def test_descriptor_lookup_unknown_raises():
    """Verify descriptor() raises ConfigError naming the backends involved.

    Mutation: raising KeyError, or omitting the registered names from the
        error message.
    Oracle: the unknown name and known_backends() as substrings of the
        message.
    """
    with pytest.raises(ConfigError) as exc:
        descriptor('nonexistent_backend')
    msg = str(exc.value)
    assert 'nonexistent_backend' in msg
    for name in known_backends():
        assert name in msg


def test_all_descriptors_returns_BackendDescriptor_instances():
    """Verify every descriptor is a BackendDescriptor with callable hooks.

    Mutation: a registry entry built with a missing or non-callable
        open_backend, list_stores_keys, or drop_store_fn.
    Oracle: isinstance and callable checks per descriptor.
    """
    for d in all_descriptors():
        assert isinstance(d, BackendDescriptor)
        assert d.name in known_backends()
        assert callable(d.open_backend)
        assert callable(d.list_stores_keys)
        assert callable(d.drop_store_fn)


def test_postgres_descriptor_declares_extras_packages():
    """Verify the postgres descriptor lists psycopg in extras_packages.

    Mutation: leaving extras_packages empty, so extras detection never
        sees the postgres extra.
    Oracle: the literal package name 'psycopg'.
    """
    pg = descriptor('postgres')
    assert pg.extras_packages, (
        'postgres descriptor must declare extras_packages so'
        ' extras.detect_active_extras finds it')
    assert 'psycopg' in pg.extras_packages


def test_sqlite_descriptor_declares_no_extras():
    """Verify the sqlite descriptor declares no extras packages.

    Mutation: listing a third-party package as an sqlite extra.
    Oracle: the empty tuple.
    """
    sql = descriptor('sqlite')
    assert sql.extras_packages == ()


def test_postgres_probe_failure_log_masks_dsn_password(monkeypatch, caplog):
    """Verify a failed Postgres store probe logs the DSN without its password.

    Mutation: logging the raw DSN in the probe-failure warning.
    Oracle: a stub connection that raises, and the literal password.
    """
    def refuse(*args, **kwargs):
        raise OSError('connection refused')
    monkeypatch.setattr('memman.store.postgres._connection', refuse)
    env_values = {
        'MEMMAN_DEFAULT_POSTGRES_DSN': 'postgresql://memman:s3cret@db/memman',
        }
    with caplog.at_level('WARNING', logger='memman'):
        descriptor('postgres').list_stores_keys('/unused', env_values)
    assert 'probe failed' in caplog.text
    assert 's3cret' not in caplog.text


def test_extras_detect_active_extras_reads_from_registry():
    """Verify detect_active_extras returns only registered backend names.

    Mutation: detect_active_extras returning a name absent from BACKENDS,
        such as a hard-coded package name.
    Oracle: known_backends() membership for each returned name.
    """
    active = extras.detect_active_extras()
    assert isinstance(active, list)
    for name in active:
        assert name in known_backends(), (
            f'detect_active_extras returned {name!r} but it is not'
            f' a registered backend')


def test_cli_to_choice_dynamic_from_registry():
    """Verify migrate --help lists every registered backend.

    Mutation: hard-coding the --to choices, so a registered backend is
        missing from the help output.
    Oracle: known_backends() names as substrings of the help text.
    """

    runner = CliRunner()
    result = runner.invoke(cli, ['migrate', '--help'])
    assert result.exit_code == 0
    for name in known_backends():
        assert name in result.output, (
            f'--to choice {name!r} missing from migrate --help output')
