"""Backend-namespaced env key validation.

Validates that `factory.open_cluster()` rejects typo'd keys that
fall in the active backend's namespace (`MEMMAN_PG_*` for postgres,
`MEMMAN_SQLITE_*` for sqlite) before any connection attempt, with
a `did you mean` hint when one is close. Cross-backend keys
(`OPENROUTER_API_KEY`, `MEMMAN_BACKEND`, `MEMMAN_EMBED_PROVIDER`)
are never scanned by either validator. Inactive-backend keys are
tolerated -- a sqlite-active install may carry `MEMMAN_PG_DSN`
from a prior postgres trial without erroring.
"""

import pytest

from memman.store.config import (
    PostgresBackendConfig,
    SqliteBackendConfig,
    )
from memman.store.errors import ConfigError


class TestPostgresValidation:
    """`PostgresBackendConfig._validate` for `MEMMAN_PG_*` keys."""

    def test_known_key_passes(self):
        """A correctly-spelled MEMMAN_PG_DSN passes silently.
        """
        env = {'MEMMAN_PG_DSN': 'postgresql://localhost/x'}
        PostgresBackendConfig._validate(env)

    def test_typo_raises_with_hint(self):
        """A typo'd MEMMAN_PG_DSL raises ConfigError with a did-you-mean.
        """
        env = {'MEMMAN_PG_DSL': 'postgresql://localhost/x'}
        with pytest.raises(ConfigError) as exc:
            PostgresBackendConfig._validate(env)
        msg = str(exc.value)
        assert 'MEMMAN_PG_DSL' in msg
        assert 'MEMMAN_PG_DSN' in msg
        assert 'did you mean' in msg.lower()

    def test_unknown_key_without_close_match(self):
        """An unknown MEMMAN_PG_* key with no close match still errors.
        """
        env = {'MEMMAN_PG_FOOBARBAZ': 'x'}
        with pytest.raises(ConfigError) as exc:
            PostgresBackendConfig._validate(env)
        assert 'MEMMAN_PG_FOOBARBAZ' in str(exc.value)

    def test_cross_backend_keys_ignored(self):
        """Cross-backend keys are not scanned by Postgres validator.
        """
        env = {
            'OPENROUTER_API_KEY': 'k',
            'MEMMAN_BACKEND': 'postgres',
            'MEMMAN_EMBED_PROVIDER': 'voyage',
            'MEMMAN_DATA_DIR': '/tmp/x',
            }
        PostgresBackendConfig._validate(env)


class TestSqliteValidation:
    """`SqliteBackendConfig._validate` for `MEMMAN_SQLITE_*` keys.

    SqliteBackendConfig owns no keys today. The class exists for
    symmetry as a future extension point; validation is a no-op
    unless an unknown `MEMMAN_SQLITE_*` key shows up.
    """

    def test_no_keys_owned_today(self):
        """Empty env passes silently.
        """
        SqliteBackendConfig._validate({})

    def test_postgres_keys_tolerated_when_sqlite_active(self):
        """MEMMAN_PG_* keys are not scanned by SQLite validator.
        """
        env = {'MEMMAN_PG_DSN': 'postgresql://localhost/x'}
        SqliteBackendConfig._validate(env)

    def test_unknown_sqlite_namespaced_key_errors(self):
        """Any MEMMAN_SQLITE_* key today is unknown.
        """
        env = {'MEMMAN_SQLITE_FOO': 'x'}
        with pytest.raises(ConfigError) as exc:
            SqliteBackendConfig._validate(env)
        assert 'MEMMAN_SQLITE_FOO' in str(exc.value)


class TestOpenClusterIntegration:
    """`factory.open_cluster` calls the active backend's _validate."""

    def test_open_cluster_rejects_postgres_typo(
            self, monkeypatch, tmp_path):
        """A MEMMAN_PG_DSL typo with backend=postgres errors before
        the connection attempt.
        """
        from memman import config
        from memman.store import factory

        data_dir = tmp_path / 'memman'
        data_dir.mkdir(exist_ok=True)
        env_path = data_dir / 'env'
        env_path.write_text(
            'MEMMAN_BACKEND=postgres\n'
            'MEMMAN_PG_DSL=postgresql://localhost/x\n')
        monkeypatch.setenv('MEMMAN_DATA_DIR', str(data_dir))
        config.reset_file_cache()

        with pytest.raises(ConfigError) as exc:
            factory.open_cluster()
        assert 'MEMMAN_PG_DSL' in str(exc.value)
        assert 'MEMMAN_PG_DSN' in str(exc.value)

    def test_open_cluster_tolerates_inactive_pg_keys(
            self, monkeypatch, tmp_path):
        """Sqlite-active install tolerates MEMMAN_PG_* leftovers.
        """
        from memman import config
        from memman.store import factory

        data_dir = tmp_path / 'memman'
        data_dir.mkdir(exist_ok=True)
        env_path = data_dir / 'env'
        env_path.write_text(
            'MEMMAN_BACKEND=sqlite\n'
            'MEMMAN_PG_DSN=postgresql://localhost/x\n')
        monkeypatch.setenv('MEMMAN_DATA_DIR', str(data_dir))
        config.reset_file_cache()

        cluster = factory.open_cluster()
        assert cluster is not None
