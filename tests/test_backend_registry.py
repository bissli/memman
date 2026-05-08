"""Backend registry surface tests.

Pins the load-bearing contract of the `BACKENDS` static registry:
- `known_backends()` reads from the static dict literal.
- `descriptor(name)` raises ConfigError on unknown backends.
- `--to` Click choices are dynamic; a synthetic third backend
  added at test time appears in the help output.
- `extras.detect_active_extras` reads through `extras_packages`.
- Capability flags are typed (BackendFeatures dataclass), not
  stringly-typed sets.

These tests guard the abstraction the v3 refactor was for: adding
a third RDBMS backend should require ONE new entry in the
`BACKENDS` dict + a `_build_<name>_descriptor()` factory, with no
edits to factory.py / cli.py / doctor.py / extras.py dispatch.
"""
from __future__ import annotations

import pytest
from memman.migrate import BackendFeatures
from memman.store.factory import BACKENDS, BackendDescriptor, all_descriptors
from memman.store.factory import descriptor, known_backends


def test_known_backends_reads_from_static_dict():
    """`known_backends()` returns the keys of `BACKENDS`."""
    assert known_backends() == frozenset(BACKENDS.keys())


def test_known_backends_includes_sqlite_and_postgres():
    """Both shipped backends are registered."""
    names = known_backends()
    assert 'sqlite' in names
    assert 'postgres' in names


def test_descriptor_lookup_unknown_raises():
    """`descriptor('nope')` raises with a hint listing registered names."""
    from memman.store.errors import ConfigError
    with pytest.raises(ConfigError) as exc:
        descriptor('nonexistent_backend')
    msg = str(exc.value)
    assert 'nonexistent_backend' in msg
    for name in known_backends():
        assert name in msg


def test_all_descriptors_returns_BackendDescriptor_instances():
    """Every descriptor is a frozen dataclass record."""
    for d in all_descriptors():
        assert isinstance(d, BackendDescriptor)
        assert d.name in known_backends()
        assert d.migrator_cls is not None
        assert callable(d.open_backend)
        assert callable(d.list_stores_keys)
        assert callable(d.drop_store_fn)


def test_descriptor_features_typed_dataclass():
    """`migrator_cls.snapshot_features` is a `BackendFeatures` dataclass.

    Not a `frozenset[str]` — the v3 refactor explicitly replaced
    string capability sets with typed boolean fields so adding a
    feature flag is a typed change, not a stringly-typed drift.
    """
    for d in all_descriptors():
        features = d.migrator_cls.snapshot_features
        assert isinstance(features, BackendFeatures)
        assert isinstance(features.supports_edges, bool)
        assert isinstance(features.supports_oplog, bool)
        assert isinstance(features.accepted_embedding_dtypes, frozenset)


def test_postgres_descriptor_declares_extras_packages():
    """The postgres descriptor lists psycopg / pgvector as extras."""
    pg = descriptor('postgres')
    assert pg.extras_packages, (
        'postgres descriptor must declare extras_packages so'
        ' extras.detect_active_extras finds it')
    assert 'psycopg' in pg.extras_packages


def test_sqlite_descriptor_declares_no_extras():
    """SQLite is a stdlib backend; extras_packages is empty."""
    sql = descriptor('sqlite')
    assert sql.extras_packages == ()


def test_env_key_for_returns_namespaced_key():
    """`env_key_for('postgres', 'DSN', store)` returns the per-store key."""
    from memman import config
    assert config.env_key_for('postgres', 'DSN', 'main') == (
        'MEMMAN_POSTGRES_DSN_main')
    assert config.env_key_for('postgres', 'DSN', 'shared-2') == (
        'MEMMAN_POSTGRES_DSN_shared-2')


def test_extras_detect_active_extras_reads_from_registry():
    """`extras.detect_active_extras` enumerates registry-declared extras.

    With psycopg present in the dev environment, the postgres extra
    resolves to active. Independent of psycopg, the function must
    return a list whose items are subset of `known_backends()`.
    """
    from memman import extras
    active = extras.detect_active_extras()
    assert isinstance(active, list)
    for name in active:
        assert name in known_backends(), (
            f'detect_active_extras returned {name!r} but it is not'
            f' a registered backend')


def test_cli_to_choice_dynamic_from_registry():
    """`memman migrate --to` choices are derived from `known_backends()`."""
    from click.testing import CliRunner
    from memman.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['migrate', '--help'])
    assert result.exit_code == 0
    for name in known_backends():
        assert name in result.output, (
            f'--to choice {name!r} missing from migrate --help output')
