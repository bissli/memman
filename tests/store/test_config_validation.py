"""Backend-namespaced env key validation.

Validates that `factory.open_backend()` rejects typo'd keys that
fall in the active backend's namespace (`MEMMAN_PG_*` for postgres,
`MEMMAN_SQLITE_*` for sqlite) before any connection attempt, with
a `did you mean` hint pointing at the per-store form. Bare canonical
keys (e.g. `MEMMAN_PG_DSN`) are also rejected -- the per-store
routing model requires the `_<store>` suffix or the
`MEMMAN_DEFAULT_PG_DSN` fallback. Cross-backend keys
(`OPENROUTER_API_KEY`, `MEMMAN_DEFAULT_BACKEND`,
`MEMMAN_DEFAULT_PG_DSN`, `MEMMAN_EMBED_PROVIDER`) are never
scanned by either validator. Inactive-backend keys are tolerated
-- a sqlite-active install may carry `MEMMAN_PG_DSN_<store>` from
a prior postgres trial without erroring.
"""

import pytest
from memman.store.config import PostgresBackendConfig, SqliteBackendConfig
from memman.store.errors import ConfigError


class TestPostgresValidation:
    """`PostgresBackendConfig._validate` for `MEMMAN_PG_*` keys."""

    def test_per_store_key_passes(self):
        """A per-store MEMMAN_PG_DSN_<store> passes silently.
        """
        env = {'MEMMAN_PG_DSN_default': 'postgresql://localhost/x'}
        PostgresBackendConfig._validate(env)

    def test_bare_pg_dsn_rejected_with_hint(self):
        """The bare canonical MEMMAN_PG_DSN is no longer accepted.
        """
        env = {'MEMMAN_PG_DSN': 'postgresql://localhost/x'}
        with pytest.raises(ConfigError) as exc:
            PostgresBackendConfig._validate(env)
        msg = str(exc.value)
        assert 'MEMMAN_PG_DSN' in msg
        assert 'MEMMAN_PG_DSN_<store>' in msg

    def test_typo_raises_with_hint(self):
        """A typo'd MEMMAN_PG_DSL raises ConfigError with a did-you-mean
        pointing at the per-store form.
        """
        env = {'MEMMAN_PG_DSL': 'postgresql://localhost/x'}
        with pytest.raises(ConfigError) as exc:
            PostgresBackendConfig._validate(env)
        msg = str(exc.value)
        assert 'MEMMAN_PG_DSL' in msg
        assert 'MEMMAN_PG_DSN_<store>' in msg
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
            'MEMMAN_DEFAULT_BACKEND': 'postgres',
            'MEMMAN_DEFAULT_PG_DSN': 'postgresql://localhost/x',
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


class TestOpenBackendIntegration:
    """`factory.open_backend` calls the active backend's _validate."""

    def test_open_backend_rejects_postgres_typo(
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
            'MEMMAN_BACKEND_default=postgres\n'
            'MEMMAN_PG_DSN_default=postgresql://localhost/x\n'
            'MEMMAN_PG_DSL=postgresql://localhost/x\n')
        monkeypatch.setenv('MEMMAN_DATA_DIR', str(data_dir))
        config.reset_file_cache()

        with pytest.raises(ConfigError) as exc:
            factory.open_backend('default', str(data_dir))
        assert 'MEMMAN_PG_DSL' in str(exc.value)
        assert 'MEMMAN_PG_DSN' in str(exc.value)


class TestValidateAll:
    """`validate_all` runs every registered backend's namespace check."""

    def test_empty_suffix_rejected(self):
        """`MEMMAN_PG_DSN_` (no suffix) is an invalid store name.
        """
        from memman.store.config import validate_for
        env = {'MEMMAN_PG_DSN_': 'x'}
        with pytest.raises(ConfigError):
            validate_for('postgres', env)

    def test_invalid_store_name_suffix_rejected(self):
        """A suffix containing slashes / spaces is rejected.
        """
        from memman.store.config import validate_for
        env = {'MEMMAN_PG_DSN_bad/name': 'x'}
        with pytest.raises(ConfigError):
            validate_for('postgres', env)

    def test_validate_all_catches_inactive_namespace_typo(self):
        """`validate_all` runs both registered config classes, so a
        `MEMMAN_PG_DSN_typo` key is rejected even when the active
        backend is sqlite.

        Closes the validator gap where typos in an inactive backend's
        namespace passed silently because `validate_for(backend, ...)`
        only ran one class.
        """
        from memman.store.config import validate_all
        env = {
            'MEMMAN_DEFAULT_BACKEND': 'sqlite',
            'MEMMAN_PG_FAKE_KEY_main': 'x',
            }
        with pytest.raises(ConfigError):
            validate_all(env)

    def test_validate_all_catches_inactive_postgres_typo_when_active_is_sqlite(
            self):
        """End-to-end inactive-namespace coverage: a `MEMMAN_PG_*`
        typo is caught when sqlite is active.
        """
        from memman.store.config import validate_all
        env = {
            'MEMMAN_DEFAULT_BACKEND': 'sqlite',
            'MEMMAN_PG_DSL': 'postgresql://x',
            }
        with pytest.raises(ConfigError):
            validate_all(env)

    def test_did_you_mean_hints_point_at_per_store_form(self):
        """Property: every did-you-mean hint produced by either
        registered config class points at the per-store form
        (`<canonical>_<store>`), never at a bare canonical.
        """
        from memman.store.config import _REGISTRY
        for cls in _REGISTRY.values():
            for owned in cls.OWNED_KEYS:
                typo = owned + 'L'
                env = {typo: 'value'}
                try:
                    cls._validate(env)
                except ConfigError as exc:
                    msg = str(exc)
                    if 'did you mean' in msg.lower():
                        assert '<store>' in msg, (
                            f'hint for {typo!r} should point at the'
                            f' per-store form: {msg!r}')
