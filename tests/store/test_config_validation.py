"""Backend-namespaced env key validation.

Validates that `factory.open_backend()` rejects typo'd
`MEMMAN_POSTGRES_*` keys before any connection attempt, whatever the
active backend, with a `did you mean` hint pointing at the per-store
form. Bare canonical keys (e.g. `MEMMAN_POSTGRES_DSN`) are also
rejected -- the per-store routing model requires the `_<store>`
suffix or the `MEMMAN_DEFAULT_POSTGRES_DSN` fallback. Cross-backend
keys (`MEMMAN_API_KEY`, `MEMMAN_DEFAULT_BACKEND`,
`MEMMAN_DEFAULT_POSTGRES_DSN`, `MEMMAN_EMBED_MODEL`) are never
scanned. Inactive-backend keys are tolerated -- a sqlite-active
install may carry `MEMMAN_POSTGRES_DSN_<store>` from a prior postgres
trial without erroring.
"""

import pytest
from memman import config
from memman.store import factory
from memman.store.config import PostgresBackendConfig, validate_all
from memman.store.errors import ConfigError


class TestPostgresValidation:
    """`PostgresBackendConfig._validate` for `MEMMAN_POSTGRES_*` keys.
    """

    def test_per_store_key_passes(self):
        """A per-store MEMMAN_POSTGRES_DSN_<store> passes silently.

        Mutation: `_validate` rejecting the suffixed form it tells
            users to write.
        Oracle: no `ConfigError` on a well-formed per-store key.
        """
        env = {'MEMMAN_POSTGRES_DSN_default': 'postgresql://localhost/x'}
        PostgresBackendConfig._validate(env)

    def test_bare_pg_dsn_rejected_with_hint(self):
        """The bare canonical MEMMAN_POSTGRES_DSN is rejected.

        Mutation: `_validate` accepting an owned key without a store
            suffix, so a DSN routes to no store.
        Oracle: `ConfigError` naming the bare key and the
            `MEMMAN_POSTGRES_DSN_<store>` form.
        """
        env = {'MEMMAN_POSTGRES_DSN': 'postgresql://localhost/x'}
        with pytest.raises(ConfigError) as exc:
            PostgresBackendConfig._validate(env)
        msg = str(exc.value)
        assert 'MEMMAN_POSTGRES_DSN' in msg
        assert 'MEMMAN_POSTGRES_DSN_<store>' in msg

    def test_typo_raises_with_hint(self):
        """A typo'd MEMMAN_POSTGRES_DSL raises ConfigError with a hint.

        The hint points at the per-store form.

        Mutation: dropping the `difflib` suggestion, or suggesting the
            bare canonical key.
        Oracle: message holds the typo, the per-store form, and
            `did you mean`.
        """
        env = {'MEMMAN_POSTGRES_DSL': 'postgresql://localhost/x'}
        with pytest.raises(ConfigError) as exc:
            PostgresBackendConfig._validate(env)
        msg = str(exc.value)
        assert 'MEMMAN_POSTGRES_DSL' in msg
        assert 'MEMMAN_POSTGRES_DSN_<store>' in msg
        assert 'did you mean' in msg.lower()

    def test_unknown_key_without_close_match(self):
        """An unknown MEMMAN_POSTGRES_* key with no close match still errors.

        Mutation: `_validate` raising only when a close match exists,
            so a wholly unknown key passes.
        Oracle: `ConfigError` naming the unknown key.
        """
        env = {'MEMMAN_POSTGRES_FOOBARBAZ': 'x'}
        with pytest.raises(ConfigError) as exc:
            PostgresBackendConfig._validate(env)
        assert 'MEMMAN_POSTGRES_FOOBARBAZ' in str(exc.value)

    def test_cross_backend_keys_ignored(self):
        """The Postgres validator does not scan cross-backend keys.

        Mutation: the prefix scan widened to `MEMMAN_`, so shared keys
            such as `MEMMAN_DEFAULT_POSTGRES_DSN` raise.
        Oracle: no `ConfigError` on a hand-built env of shared keys.
        """
        env = {
            'MEMMAN_API_KEY': 'k',
            'MEMMAN_DEFAULT_BACKEND': 'postgres',
            'MEMMAN_DEFAULT_POSTGRES_DSN': 'postgresql://localhost/x',
            'MEMMAN_EMBED_MODEL': 'voyageai/voyage-4-lite',
            'MEMMAN_DATA_DIR': '/tmp/x',
            }
        PostgresBackendConfig._validate(env)


class TestOpenBackendIntegration:
    """`factory.open_backend` runs the Postgres namespace scan on every open.
    """

    def test_open_backend_rejects_postgres_typo(
            self, monkeypatch, tmp_path):
        """A MEMMAN_POSTGRES_DSL typo errors before any connection attempt.

        Mutation: `open_backend` connecting before validating, or not
            validating at all, so the typo surfaces as a connection
            failure.
        Oracle: `ConfigError` naming the typo and the correct key, with
            no reachable server behind the DSN.
        """
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(exist_ok=True)
        env_path = data_dir / 'env'
        env_path.write_text(
            'MEMMAN_BACKEND_default=postgres\n'
            'MEMMAN_POSTGRES_DSN_default=postgresql://localhost/x\n'
            'MEMMAN_POSTGRES_DSL=postgresql://localhost/x\n')
        monkeypatch.setenv('MEMMAN_DATA_DIR', str(data_dir))
        config.reset_file_cache()

        with pytest.raises(ConfigError) as exc:
            factory.open_backend('default', str(data_dir))
        assert 'MEMMAN_POSTGRES_DSL' in str(exc.value)
        assert 'MEMMAN_POSTGRES_DSN' in str(exc.value)


class TestValidateAll:
    """`validate_all` runs the Postgres namespace scan whatever the backend.
    """

    def test_empty_suffix_rejected(self):
        """`MEMMAN_POSTGRES_DSN_` (no suffix) is an invalid store name.

        Mutation: `validate_all` accepting an empty store suffix, so a
            truncated key routes a DSN to no store.
        Oracle: `ConfigError` on the bare-underscore key.
        """
        env = {'MEMMAN_POSTGRES_DSN_': 'x'}
        with pytest.raises(ConfigError):
            validate_all(env)

    def test_invalid_store_name_suffix_rejected(self):
        """A suffix containing slashes / spaces is rejected.

        Mutation: `validate_all` skipping the store-name check on the
            per-store suffix.
        Oracle: `ConfigError` on a suffix holding a slash.
        """
        env = {'MEMMAN_POSTGRES_DSN_bad/name': 'x'}
        with pytest.raises(ConfigError):
            validate_all(env)

    def test_validate_all_catches_inactive_namespace_typo(self):
        """`validate_all` rejects a typo'd Postgres key with sqlite active.

        Mutation: `validate_all` gating the namespace scan on
            `MEMMAN_DEFAULT_BACKEND == 'postgres'`, so a typo in an
            inactive namespace goes uncaught until a backend switch.
        Oracle: `ConfigError` raised with sqlite active.
        """
        env = {
            'MEMMAN_DEFAULT_BACKEND': 'sqlite',
            'MEMMAN_POSTGRES_FAKE_KEY_main': 'x',
            }
        with pytest.raises(ConfigError):
            validate_all(env)

    def test_did_you_mean_hints_point_at_per_store_form(self):
        """Every near-miss key raises with a hint at the per-store form.

        Mutation: the hint built from `suggestions[0]` alone, dropping
            the `+ '_<store>'` suffix, so it re-suggests the still-
            rejected bare canonical key; or a near-miss key that raises
            nothing, or no hint.
        Oracle: for each owned key plus 'L', `validate_all` with sqlite
            active raises `ConfigError` whose message is the hand-written
            "did you mean '<owned>_<store>'?" hint.
        """
        for owned in PostgresBackendConfig.OWNED_KEYS:
            typo = owned + 'L'
            env = {'MEMMAN_DEFAULT_BACKEND': 'sqlite', typo: 'value'}
            with pytest.raises(ConfigError) as excinfo:
                validate_all(env)
            assert f"did you mean '{owned}_<store>'?" in str(excinfo.value)
