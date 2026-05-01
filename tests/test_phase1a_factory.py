"""Phase 1a factory dispatch tests."""

import pytest
from memman import config
from memman.store.errors import ConfigError
from memman.store.factory import BACKENDS, open_cluster
from memman.store.sqlite import SqliteCluster


def test_sqlite_registered_in_backends():
    """SQLite is the always-on backend; postgres registers in Phase 2.
    """
    assert 'sqlite' in BACKENDS


def test_open_cluster_default_returns_sqlite_cluster():
    """With MEMMAN_BACKEND unset, factory returns SqliteCluster."""
    cluster = open_cluster()
    assert isinstance(cluster, SqliteCluster)


def test_open_cluster_unknown_backend_raises_configerror(monkeypatch):
    """Unknown MEMMAN_BACKEND value yields ConfigError with hint."""
    monkeypatch.setattr(config, 'get', lambda key: (
        'plutonium' if key == config.BACKEND else None))
    with pytest.raises(ConfigError, match='unknown'):
        open_cluster()


def test_open_cluster_explicit_sqlite_value(monkeypatch):
    """MEMMAN_BACKEND=sqlite returns SqliteCluster."""
    monkeypatch.setattr(config, 'get', lambda key: (
        'sqlite' if key == config.BACKEND else None))
    cluster = open_cluster()
    assert isinstance(cluster, SqliteCluster)


def test_open_cluster_case_insensitive(monkeypatch):
    """MEMMAN_BACKEND=SQLite is normalised to lowercase before lookup.
    """
    monkeypatch.setattr(config, 'get', lambda key: (
        'SQLite' if key == config.BACKEND else None))
    cluster = open_cluster()
    assert isinstance(cluster, SqliteCluster)
