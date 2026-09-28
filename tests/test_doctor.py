"""Tests for memman.doctor health-check module."""

import json
import os
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

try:
    import psycopg
except ImportError:
    psycopg = None

from click.testing import CliRunner
from memman.cli import cli
from memman.doctor import check_drain_heartbeat, check_env_completeness
from memman.doctor import check_env_permissions, check_scheduler_heartbeat
from memman.doctor import check_scheduler_state
from memman.store.node import insert_insight, update_embedding
from memman.store.node import update_enrichment
from tests.conftest import make_insight


def _fake_embedding(dim: int = 512) -> bytes:
    """Return a deterministic embedding blob of the given dimension."""
    return struct.pack(f'<{dim}d', *([0.1] * dim))


def _insert_healthy_insight(db, id: str, content: str = 'Healthy test insight with enough content') -> None:
    """Insert an insight with all enrichment fields populated."""
    from memman.pipeline.remember import compute_prompt_version
    ins = make_insight(
        id=id, content=content, prompt_version=compute_prompt_version())
    insert_insight(db, ins)
    update_enrichment(db, id, 'summary text')
    update_embedding(db, id, _fake_embedding(), 'voyage-3-lite')


class TestSqliteIntegrity:

    def test_pass_on_fresh_db(self, tmp_backend):
        """Fresh database passes integrity check."""
        from memman.doctor import check_integrity
        result = check_integrity(tmp_backend)
        assert result['name'] == 'integrity'
        assert result['status'] == 'pass'
        assert result['detail']['result'] == 'ok'


class TestEnrichmentCoverage:

    def test_full_pass(self, tmp_db, tmp_backend):
        """Verify rows with a summary and an embedding pass.

        Mutation: `enrichment_coverage` selecting a column the baseline
            no longer declares, such as `semantic_facts`, which raises
            rather than reports, or counting it as a missing field.
        Oracle: two rows carrying exactly the two graded fields.
        """
        from memman.doctor import check_enrichment_coverage
        _insert_healthy_insight(tmp_db, 'e-1')
        _insert_healthy_insight(tmp_db, 'e-2')
        result = check_enrichment_coverage(tmp_backend)
        assert result['status'] == 'pass'
        assert result['detail']['coverage_pct'] == 100.0

    def test_partial_warn(self, tmp_db, tmp_backend):
        """Some fields missing returns warn when coverage >= 90%."""
        from memman.doctor import check_enrichment_coverage
        for i in range(10):
            _insert_healthy_insight(tmp_db, f'e-{i}', f'Content for insight number {i}')
        ins = make_insight(id='e-bare', content='Bare insight without enrichment')
        insert_insight(tmp_db, ins)
        result = check_enrichment_coverage(tmp_backend)
        assert result['status'] == 'warn'
        assert result['detail']['missing_embedding'] == 1


class TestEmbeddingConsistency:

    def test_consistent_pass(self, tmp_db, tmp_backend):
        """All embeddings same size returns pass."""
        from memman.doctor import check_embedding_consistency
        _insert_healthy_insight(tmp_db, 'emb-1')
        _insert_healthy_insight(tmp_db, 'emb-2')
        result = check_embedding_consistency(tmp_backend)
        assert result['status'] == 'pass'

    def test_mixed_fail(self, tmp_db, tmp_backend):
        """Different embedding sizes returns fail."""
        from memman.doctor import check_embedding_consistency
        _insert_healthy_insight(tmp_db, 'emb-1')
        ins2 = make_insight(id='emb-2', content='Different dim embedding')
        insert_insight(tmp_db, ins2)
        update_embedding(tmp_db, 'emb-2', _fake_embedding(dim=256),
                         'voyage-3-lite')
        result = check_embedding_consistency(tmp_backend)
        assert result['status'] == 'fail'
        assert len(result['detail']['sizes']) > 1


class TestProvenanceDrift:

    def test_no_rows_pass(self, tmp_db, tmp_backend):
        """Empty store: no stale rows."""
        from memman.doctor import check_provenance_drift
        result = check_provenance_drift(tmp_backend)
        assert result['name'] == 'provenance_drift'
        assert result['status'] == 'pass'
        assert result['detail']['stale_rows'] == 0

    def test_reports_the_active_model(self, tmp_db, tmp_backend):
        """provenance_drift names the configured model as `active_model`.

        Mutation: the key renamed or dropped, or its value read from a
            variable other than MEMMAN_LLM_MODEL.
        Oracle: the model the autouse fixture seeds into the env file.
        """
        from memman import config
        from memman.doctor import check_provenance_drift
        result = check_provenance_drift(tmp_backend)
        assert result['detail']['active_model'] == \
            config.INSTALL_DEFAULTS[config.LLM_MODEL]

    def test_all_current_pass(self, tmp_db, tmp_backend):
        """All rows stamped with the active prompt_version: pass.

        Mutation: `_is_provenance_stale` reporting a row stale even
            when its `prompt_version` equals `active_pv`.
        Oracle: the row's `prompt_version` set to the freshly
            computed `active_pv` before the check runs.
        """
        from memman.doctor import check_provenance_drift
        from memman.pipeline.remember import compute_prompt_version

        active_pv = compute_prompt_version()

        _insert_healthy_insight(tmp_db, 'p-1')
        tmp_db._exec(
            'UPDATE insights SET prompt_version = ? WHERE id = ?',
            (active_pv, 'p-1'))

        result = check_provenance_drift(tmp_backend)
        assert result['status'] == 'pass'
        assert result['detail']['stale_rows'] == 0

    def test_null_prompt_version_not_stale(self, tmp_db, tmp_backend):
        """A row with no `prompt_version` never counts as stale.

        Mutation: `_is_provenance_stale` comparing `None != active_pv`
            as True, so a never-enriched row counts as drifted even
            though `count_stale_insights` excludes it with its
            `prompt_version is not null` clause.
        Oracle: a row inserted with `prompt_version=None`, checked
            against `check_provenance_drift`'s `stale_rows` output.
        """
        from memman.doctor import check_provenance_drift

        ins = make_insight(id='p-null', prompt_version=None)
        insert_insight(tmp_db, ins)

        result = check_provenance_drift(tmp_backend)
        assert result['status'] == 'pass'
        assert result['detail']['stale_rows'] == 0

    def test_drift_warns(self, tmp_db, tmp_backend):
        """A drifted prompt_version surfaces as warn with a remedy.

        Mutation: comparing the row's key against a constant, or
            dropping the warn so drift a rebuild CAN fix goes
            unreported.
        Oracle: two drifted rows against one carrying the active key,
            counted.
        """
        from memman.doctor import check_provenance_drift
        from memman.pipeline.remember import compute_prompt_version

        active_pv = compute_prompt_version()

        for i in range(2):
            _insert_healthy_insight(tmp_db, f'p-stale-{i}')
        _insert_healthy_insight(tmp_db, 'p-fresh')
        tmp_db._exec(
            'UPDATE insights SET prompt_version = ?'
            " WHERE id IN ('p-stale-0', 'p-stale-1')",
            ('deadbeefdeadbeef',))
        tmp_db._exec(
            'UPDATE insights SET prompt_version = ?'
            " WHERE id = 'p-fresh'",
            (active_pv,))

        result = check_provenance_drift(tmp_backend)
        assert result['status'] == 'warn'
        assert result['detail']['stale_rows'] == 2
        assert 'remediation' in result['detail']


class TestStaleHelpers:
    """Cross-backend tests for iter_stale_insight_ids and count_stale_insights.
    """

    def _seed_stale_matrix(self, backend, active_pv):
        """Seed the canonical predicate rows; return expected stale ids.

        Mapping, by prompt_version: B=current not stale, C=OLD STALE,
        E=OLD STALE.
        """
        OLD_PV = 'old-prompt-version-deadbeef'
        rows = [
            ('row-b', active_pv),
            ('row-c', OLD_PV),
            ('row-e', OLD_PV),
            ]
        for rid, pv in rows:
            backend.nodes.insert(make_insight(
                id=rid, content=f'content for {rid} long enough',
                prompt_version=pv))
        return ['row-c', 'row-e']

    def test_iter_returns_only_drifted_rows(self, backend):
        """iter_stale_insight_ids excludes the current row.

        Mutation: the `!= active_pv` term dropped, which reports the
            current row as stale too.
        Oracle: the hand-built three-row matrix from
            `_seed_stale_matrix`, whose only stale ids are the two
            seeded on `OLD_PV`.
        """
        from memman.pipeline.remember import compute_prompt_version

        active_pv = compute_prompt_version()
        expected = self._seed_stale_matrix(backend, active_pv)

        ids = backend.nodes.iter_stale_insight_ids(active_pv)
        assert sorted(ids) == sorted(expected)

    def test_count_matches_iter(self, backend):
        """count_stale_insights agrees with len(iter_stale_insight_ids).

        Mutation: `count_stale_insights`'s SQL predicate drifting from
            `iter_stale_insight_ids`'s, so the two disagree on the
            seeded matrix.
        Oracle: the hand-counted stale total of 2 from
            `_seed_stale_matrix`.
        """
        from memman.pipeline.remember import compute_prompt_version

        active_pv = compute_prompt_version()
        self._seed_stale_matrix(backend, active_pv)

        n = backend.nodes.count_stale_insights(active_pv)
        ids = backend.nodes.iter_stale_insight_ids(active_pv)
        assert n == len(ids) == 2

    def test_count_matches_doctor_stale_rows(self, backend):
        """count_stale_insights agrees with check_provenance_drift's stale_rows.

        Mutation: `check_provenance_drift`'s per-row
            `_is_provenance_stale` predicate diverging from
            `count_stale_insights`'s SQL predicate (e.g. treating a
            NULL `prompt_version` as stale), so the two disagree on
            the same seeded matrix.
        Oracle: the store helper's own count, cross-checked against
            the doctor check's `stale_rows` on the identical rows.
        """
        from memman.doctor import check_provenance_drift
        from memman.pipeline.remember import compute_prompt_version

        active_pv = compute_prompt_version()
        self._seed_stale_matrix(backend, active_pv)

        helper_count = backend.nodes.count_stale_insights(active_pv)
        doctor_result = check_provenance_drift(backend)
        assert helper_count == doctor_result['detail']['stale_rows']

    def test_empty_store(self, backend):
        """Empty store returns 0 / [] from both helpers."""
        from memman.pipeline.remember import compute_prompt_version

        active_pv = compute_prompt_version()
        assert backend.nodes.iter_stale_insight_ids(active_pv) == []
        assert backend.nodes.count_stale_insights(active_pv) == 0


class TestRunAllChecks:

    def test_structure(self, tmp_db, tmp_backend):
        """Verify output shape: status, checks list, total_active."""
        from memman.doctor import run_all_checks
        _insert_healthy_insight(tmp_db, 'all-1')
        result = run_all_checks(tmp_backend)
        assert 'status' in result
        assert 'checks' in result
        assert 'total_active' in result
        assert isinstance(result['checks'], list)
        assert result['status'] in {'pass', 'warn', 'fail'}

    def test_empty_db(self, tmp_db, tmp_backend):
        """Empty store returns status 'empty' with no checks."""
        from memman.doctor import run_all_checks
        result = run_all_checks(tmp_backend)
        assert result['status'] == 'empty'
        assert result['total_active'] == 0
        assert result['checks'] == []

    def test_healthy_db(self, tmp_db, tmp_backend):
        """Fully healthy DB returns status 'pass'."""
        from memman.doctor import run_all_checks
        ids = [f'h-{i}' for i in range(6)]
        for id in ids:
            _insert_healthy_insight(tmp_db, id, f'Healthy content for {id} insight')
        result = run_all_checks(tmp_backend)
        assert result['status'] == 'pass', [
            (c['name'], c['status'], c.get('detail'))
            for c in result['checks'] if c['status'] != 'pass']
        assert all(c['status'] == 'pass' for c in result['checks'])


class TestEnvCompleteness:
    """check_env_completeness against INSTALLABLE_KEYS."""

    @pytest.fixture
    def write_env(self, tmp_path, monkeypatch):
        """Write a custom env file under a fresh data dir."""
        from memman import config
        data_dir = tmp_path / 'memman'
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(config.DATA_DIR, str(data_dir))

        def _write(contents: str) -> None:
            (data_dir / config.ENV_FILENAME).write_text(contents)
            config.reset_file_cache()

        return _write

    def test_pass_when_all_present(self, write_env):
        """All INSTALLABLE_KEYS in the file -> status pass."""
        from memman import config
        lines = [f'{key}=value-for-{key}' for key in config.INSTALLABLE_KEYS]
        write_env('\n'.join(lines) + '\n')
        out = check_env_completeness()
        assert out['status'] == 'pass'

    def test_warns_when_non_secret_missing(self, write_env):
        """Missing non-secret key -> warn with key in detail.missing."""
        from memman import config
        lines = [
            f'{key}=v' for key in config.INSTALLABLE_KEYS
            if key != config.LLM_MODEL
            ]
        write_env('\n'.join(lines) + '\n')
        out = check_env_completeness()
        assert out['status'] == 'warn'
        assert config.LLM_MODEL in out['detail']['missing']
        assert 'memman install' in out['detail']['fix']

    def test_ignores_optional_secret(self, write_env):
        """Missing OPENAI_EMBED_API_KEY (optional secret) does not fail."""
        from memman import config
        lines = [
            f'{key}=v' for key in config.INSTALLABLE_KEYS
            if key != config.OPENAI_EMBED_API_KEY
            ]
        write_env('\n'.join(lines) + '\n')
        out = check_env_completeness()
        assert out['status'] == 'pass'
        assert config.OPENAI_EMBED_API_KEY not in out.get('detail', {}).get(
            'missing', [])

    @pytest.mark.parametrize(('provider', 'key_attr'), [
        ('voyage', 'VOYAGE_API_KEY'),
        ('openai', 'OPENAI_EMBED_API_KEY'),
        ('openrouter', 'OPENROUTER_API_KEY'),
        ])
    def test_passes_on_fresh_install(
            self, write_env, monkeypatch, provider, key_attr):
        """The env file a fresh install writes passes the check.

        Mutation: requiring every embed provider's key whatever the
            configured provider (a fresh voyage install warned
            `MEMMAN_OPENROUTER_API_KEY` missing).
        Oracle: the file `collect_install_knobs` builds with only the
            chosen provider's key and the LLM key exported, with Voyage
            reranking off unless the provider is voyage.
        """
        from memman import config
        for key in (*config.INSTALLABLE_KEYS,
                    *config.NATIVE_INSTALL_KEY_FALLBACKS.values()):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(config.EMBED_PROVIDER, provider)
        monkeypatch.setenv(getattr(config, key_attr), 'provider-key')
        monkeypatch.setenv(config.LLM_API_KEY, 'llm-key')
        if provider != 'voyage':
            monkeypatch.setenv(config.RERANK_ENABLED, 'false')
        write_env('')
        knobs = config.collect_install_knobs(os.environ[config.DATA_DIR])
        write_env(''.join(f'{k}={v}\n' for k, v in knobs.items()))
        out = check_env_completeness()
        assert out['status'] == 'pass', out['detail']

    def test_warns_when_voyage_rerank_lacks_its_key(self, write_env):
        """Reranking on with no Voyage key -> warn, with no provider read.

        Mutation: deriving the required keys from the embed provider
            alone, so an openai install reranking on Voyage never learns
            its reranker has no key; or a restored
            `MEMMAN_RERANK_PROVIDER` read, which lands in `missing`
            beside the Voyage key.
        Oracle: rerank/voyage.py requires `MEMMAN_VOYAGE_API_KEY`.
        """
        from memman import config
        values = dict.fromkeys(config.INSTALLABLE_KEYS, 'v')
        values.update({
            config.EMBED_PROVIDER: 'openai',
            config.RERANK_ENABLED: 'true',
            config.VOYAGE_API_KEY: '',
            })
        write_env(''.join(f'{k}={v}\n' for k, v in values.items()))
        out = check_env_completeness()
        assert out['status'] == 'warn'
        assert out['detail']['missing'] == [config.VOYAGE_API_KEY]

    def test_disabled_rerank_does_not_require_voyage_key(self, write_env):
        """Reranking off -> the Voyage key is not required on rerank's account.

        Mutation: requiring `MEMMAN_VOYAGE_API_KEY` unconditionally once
            the provider switch is gone, rather than gating on whether
            any rerank switch is truthy.
        Oracle: with `MEMMAN_RERANK_ENABLED=false` and a non-voyage embed
            provider that owns its own key, `missing` carries neither key.
        """
        from memman import config
        values = dict.fromkeys(config.INSTALLABLE_KEYS, 'v')
        values.update({
            config.EMBED_PROVIDER: 'openai',
            config.RERANK_ENABLED: 'false',
            config.VOYAGE_API_KEY: '',
            })
        write_env(''.join(f'{k}={v}\n' for k, v in values.items()))
        out = check_env_completeness()
        assert out['status'] == 'pass', out['detail']

    @pytest.mark.parametrize(('global_rerank', 'store_rerank'), [
        ('false', 'true'),
        (None, None),
        ('', None),
        ])
    def test_warns_when_rerank_is_on_only_where_recall_reads_it(
            self, write_env, global_rerank, store_rerank):
        """Voyage reranking on for recall, keyless -> the key is missing.

        Mutation: reading only the global `MEMMAN_RERANK_ENABLED` as
            written, so a store turned on by `MEMMAN_RERANK_ENABLED_<store>`
            or an unset or empty global (which recall reads as on) passes
            with no Voyage key, and recall silently keeps the unreranked
            order.
        Oracle: `recall` in cli.py reads the per-store key first, then the
            global with `default=True`.
        """
        from memman import config
        values = dict.fromkeys(config.INSTALLABLE_KEYS, 'v')
        values.update({
            config.EMBED_PROVIDER: 'openai',
            config.VOYAGE_API_KEY: '',
            })
        if global_rerank is None:
            del values[config.RERANK_ENABLED]
        else:
            values[config.RERANK_ENABLED] = global_rerank
        if store_rerank is not None:
            values[config.RERANK_ENABLED_FOR('default')] = store_rerank
        write_env(''.join(f'{k}={v}\n' for k, v in values.items()))
        out = check_env_completeness()
        assert out['status'] == 'warn'
        assert config.VOYAGE_API_KEY in out['detail']['missing']

    def test_ignores_optional_backup_keys(self, write_env):
        """Absent BACKUP_CRON/TARGET (opt-in feature) does not warn."""
        from memman import config
        optional_backup = {
            config.BACKUP_CRON, config.BACKUP_TARGET, config.BACKUP_KEEP}
        lines = [
            f'{key}=v' for key in config.INSTALLABLE_KEYS
            if key not in optional_backup
            ]
        write_env('\n'.join(lines) + '\n')
        out = check_env_completeness()
        assert out['status'] == 'pass'
        missing = out.get('detail', {}).get('missing', [])
        assert config.BACKUP_CRON not in missing
        assert config.BACKUP_TARGET not in missing


class TestCheckPerStoreKeys:
    """`check_per_store_keys` validates `MEMMAN_BACKEND_<store>` shape."""

    def test_pass_when_no_stores(self, tmp_path):
        """Empty data dir -> pass with empty stores list."""
        from memman.doctor import check_per_store_keys
        out = check_per_store_keys(str(tmp_path / 'memman'))
        assert out['name'] == 'per_store_keys'
        assert out['status'] == 'pass'
        assert out['detail']['stores'] == []

    def test_pass_when_per_store_key_resolves(self, tmp_path, env_file):
        """SQLite store with explicit per-store key -> pass."""
        from memman import config
        from memman.doctor import check_per_store_keys

        data_dir = str(tmp_path / 'memman')
        Path(data_dir, 'data', 'one').mkdir(parents=True, exist_ok=True)
        Path(data_dir, 'data', 'one', 'memman.db').write_bytes(b'')
        env_file(config.BACKEND_FOR('one'), 'sqlite')

        out = check_per_store_keys(data_dir)
        assert out['status'] == 'pass'
        names = [s['store'] for s in out['detail']['stores']]
        assert 'one' in names

    def test_pass_when_falling_back_to_default(self, tmp_path, env_file):
        """No per-store key, default sqlite -> pass with fallback flag."""
        from memman import config
        from memman.doctor import check_per_store_keys

        data_dir = str(tmp_path / 'memman')
        Path(data_dir, 'data', 'fallback').mkdir(parents=True, exist_ok=True)
        Path(data_dir, 'data', 'fallback', 'memman.db').write_bytes(b'')
        env_file(config.DEFAULT_BACKEND, 'sqlite')

        out = check_per_store_keys(data_dir)
        assert out['status'] == 'pass'
        match = next(s for s in out['detail']['stores']
                     if s['store'] == 'fallback')
        assert match['backend'] == 'sqlite'
        assert match['source'] == 'default'

    def test_fails_on_unknown_backend_value(self, tmp_path, env_file):
        """`MEMMAN_BACKEND_<store>=mongo` -> fail (unknown backend)."""
        from memman import config
        from memman.doctor import check_per_store_keys

        data_dir = str(tmp_path / 'memman')
        Path(data_dir, 'data', 'bad').mkdir(parents=True, exist_ok=True)
        Path(data_dir, 'data', 'bad', 'memman.db').write_bytes(b'')
        env_file(config.BACKEND_FOR('bad'), 'mongo')

        out = check_per_store_keys(data_dir)
        assert out['status'] == 'fail'
        bad = next(s for s in out['detail']['stores']
                   if s['store'] == 'bad')
        assert 'unknown backend' in bad.get('error', '').lower()

    def test_warns_when_postgres_dsn_missing(self, tmp_path, env_file):
        """`MEMMAN_BACKEND_<store>=postgres` without DSN -> fail."""
        from memman import config
        from memman.doctor import check_per_store_keys

        data_dir = str(tmp_path / 'memman')
        Path(data_dir, 'data', 'pg_one').mkdir(parents=True, exist_ok=True)
        Path(data_dir, 'data', 'pg_one', 'memman.db').write_bytes(b'')
        env_file(config.BACKEND_FOR('pg_one'), 'postgres')

        out = check_per_store_keys(data_dir)
        assert out['status'] == 'fail'
        pg = next(s for s in out['detail']['stores']
                  if s['store'] == 'pg_one')
        assert 'dsn' in pg.get('error', '').lower()

    def test_postgres_default_dsn_satisfies(self, tmp_path, env_file):
        """`MEMMAN_DEFAULT_POSTGRES_DSN` covers a postgres store without a per-store DSN.
        """
        from memman import config
        from memman.doctor import check_per_store_keys

        data_dir = str(tmp_path / 'memman')
        Path(data_dir, 'data', 'pg_two').mkdir(parents=True, exist_ok=True)
        Path(data_dir, 'data', 'pg_two', 'memman.db').write_bytes(b'')
        env_file(config.BACKEND_FOR('pg_two'), 'postgres')
        env_file(config.DEFAULT_PG_DSN, 'postgresql://x@y/z')

        out = check_per_store_keys(data_dir)
        pg = next(s for s in out['detail']['stores']
                  if s['store'] == 'pg_two')
        assert pg.get('error') is None
        assert pg['backend'] == 'postgres'

    def test_no_warn_when_dsns_differ(self, tmp_path, env_file):
        """Per-store DSN differs from default DSN -> pass (canonical
        rotation-pinning state, not a typo).

        The 0.14.1 doctor warned on this divergence; F.2 drops the
        warn because per-store routing pins each store to its own
        DSN, and an explicit per-store DSN is the documented way to
        keep a store on a stable cluster while the default rotates.
        """
        from memman import config
        from memman.doctor import check_per_store_keys

        data_dir = str(tmp_path / 'memman')
        Path(data_dir, 'data', 'pg_pinned').mkdir(parents=True, exist_ok=True)
        Path(data_dir, 'data', 'pg_pinned', 'memman.db').write_bytes(b'')
        env_file(config.BACKEND_FOR('pg_pinned'), 'postgres')
        env_file(config.env_key_for('postgres', 'DSN', 'pg_pinned'), 'postgresql://pinned@host/db')
        env_file(config.DEFAULT_PG_DSN, 'postgresql://default@host/db')

        out = check_per_store_keys(data_dir)
        assert out['status'] == 'pass'
        pg = next(s for s in out['detail']['stores']
                  if s['store'] == 'pg_pinned')
        assert pg.get('warning') is None
        assert pg.get('error') is None

    def test_check_per_store_keys_includes_declared_but_not_created_store(
            self, tmp_path, env_file):
        """A store declared via `MEMMAN_BACKEND_<name>` but missing
        on disk still appears in the doctor enumeration so the
        operator notices the mismatch.
        """
        from memman import config
        from memman.doctor import check_per_store_keys

        data_dir = str(tmp_path / 'memman')
        env_file(config.BACKEND_FOR('declared_only'), 'sqlite')

        out = check_per_store_keys(data_dir)
        names = [s['store'] for s in out['detail']['stores']
                 if s['store'] is not None]
        assert 'declared_only' in names


def _started_scheduler_status(interval=900):
    """Test helper: pretend the scheduler is installed + started."""
    return {
        'interval_seconds': interval,
        'state': 'started',
        'installed': True,
        }


class TestHardening:
    """B12 doctor checks: schema, env perms, scheduler, worker runs."""

    @pytest.mark.parametrize(('mode', 'expected_status', 'assert_issue'), [
        (None, 'pass', False),
        (0o644, 'fail', True),
        (0o600, 'pass', False),
    ])
    def test_env_permissions(
            self, tmp_path, monkeypatch, mode, expected_status, assert_issue):
        """Permissions check passes for missing or 0600, fails for 0644."""
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        if mode is not None:
            mm = tmp_path / '.memman'
            mm.mkdir(mode=0o700)
            env = mm / 'env'
            env.write_text('OPENROUTER_API_KEY=fake\n')
            env.chmod(mode)
        result = check_env_permissions()
        assert result['status'] == expected_status
        if assert_issue:
            assert any('env file' in issue
                       for issue in result['detail']['issues'])

    def test_scheduler_state_warn_when_uninstalled(self, monkeypatch):
        """Scheduler-not-installed is a warn, not a fail.

        Mutation: the `not installed` branch dropped or its status
            flipped to `pass` or `fail`.
        Oracle: the check's own status against a stubbed
            not-installed `status()`.
        """
        from memman.setup import scheduler as sch
        monkeypatch.setattr(
            sch, 'status',
            lambda: {'installed': False, 'active': False,
                     'state': 'stopped', 'interval_seconds': None})
        result = check_scheduler_state()
        assert result['status'] == 'warn'

    def test_scheduler_state_pass_when_installed(self, monkeypatch):
        """An installed, active scheduler passes.

        Mutation: the `installed` branch reporting `warn` or `fail`
            instead of `pass`.
        Oracle: the check's own status against a stubbed installed,
            active `status()`.
        """
        from memman.setup import scheduler as sch
        monkeypatch.setattr(
            sch, 'status',
            lambda: {'installed': True, 'active': True,
                     'state': 'started', 'interval_seconds': 900})
        result = check_scheduler_state()
        assert result['status'] == 'pass'

    def test_scheduler_heartbeat_fail_when_no_drains_and_started(self, tmp_path, monkeypatch):
        """Scheduler started + installed but no worker_runs row yet -> fail."""
        from memman.setup import scheduler as sch
        monkeypatch.setattr(sch, 'status', _started_scheduler_status)
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'fail'
        assert 'no drains recorded' in result['detail']['reason']

    @pytest.mark.parametrize(('status', 'reason_snippet'), [
        ({'interval_seconds': 900, 'state': 'stopped',
          'installed': True}, "'stopped'"),
        ({'interval_seconds': None, 'state': 'stopped',
          'installed': False}, None),
    ])
    def test_scheduler_heartbeat_pass_when_inactive(
            self, tmp_path, monkeypatch, status, reason_snippet):
        """Scheduler stopped or uninstalled -> pass (no drain expected)."""
        from memman.setup import scheduler as sch
        monkeypatch.setattr(sch, 'status', lambda: status)
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'pass'
        if reason_snippet is not None:
            assert reason_snippet in result['detail']['reason']

    def test_scheduler_heartbeat_pass_on_recent_drain(self, tmp_path, monkeypatch):
        """A drain within the interval window passes."""
        from memman.queue import finish_worker_run, open_queue_db
        from memman.queue import start_worker_run
        from memman.setup import scheduler as sch

        monkeypatch.setattr(sch, 'status', _started_scheduler_status)
        conn = open_queue_db(str(tmp_path))
        try:
            run_id = start_worker_run(conn, worker_pid=1)
            finish_worker_run(conn, run_id, 0, 0, 0)
        finally:
            conn.close()
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'pass'

    def test_scheduler_heartbeat_threshold_floors_at_180s(self, tmp_path, monkeypatch):
        """At interval=0 (serve continuous), the threshold floors at 180s.

        Without the floor, `3 * 0 = 0` would fail every heartbeat check.
        With the floor (max(3*interval, 180s)), serve mode is robust to
        sub-minute intervals -- the rate-limited heartbeat writes 1/min so
        a 180s window allows two-miss tolerance.
        """
        from memman.queue import finish_worker_run, open_queue_db
        from memman.queue import start_worker_run
        from memman.setup import scheduler as sch

        monkeypatch.setattr(sch, 'status',
                            lambda: _started_scheduler_status(interval=0))
        conn = open_queue_db(str(tmp_path))
        try:
            run_id = start_worker_run(conn, worker_pid=1)
            finish_worker_run(conn, run_id, 0, 0, 0)
            conn.execute(
                'UPDATE worker_runs SET started_at = started_at - 90'
                ' WHERE id = ?', (run_id,))
            conn.commit()
        finally:
            conn.close()
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'pass', (
            f'90s old heartbeat at interval=0 should PASS under 180s floor;'
            f' got {result}')
        assert result['detail']['threshold_fail_seconds'] == 180

    def test_scheduler_heartbeat_fails_at_interval_zero_when_stale(
            self, tmp_path, monkeypatch):
        """At interval=0, a heartbeat older than 180s fails.

        Validates that the `interval and` truthiness guard is removed --
        interval=0 must reach the threshold comparison, not short-circuit
        to PASS.
        """
        from memman.queue import finish_worker_run, open_queue_db
        from memman.queue import start_worker_run
        from memman.setup import scheduler as sch

        monkeypatch.setattr(sch, 'status',
                            lambda: _started_scheduler_status(interval=0))
        conn = open_queue_db(str(tmp_path))
        try:
            run_id = start_worker_run(conn, worker_pid=1)
            finish_worker_run(conn, run_id, 0, 0, 0)
            conn.execute(
                'UPDATE worker_runs SET started_at = started_at - 200'
                ' WHERE id = ?', (run_id,))
            conn.commit()
        finally:
            conn.close()
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'fail', (
            f'200s old heartbeat at interval=0 should FAIL (180s floor);'
            f' got {result}')

    def test_scheduler_heartbeat_fail_on_recorded_error(self, tmp_path, monkeypatch):
        """A finished run with an error string flips the check to fail."""
        from memman.queue import finish_worker_run, open_queue_db
        from memman.queue import start_worker_run
        from memman.setup import scheduler as sch

        monkeypatch.setattr(sch, 'status', _started_scheduler_status)
        conn = open_queue_db(str(tmp_path))
        try:
            run_id = start_worker_run(conn, worker_pid=1)
            finish_worker_run(
                conn, run_id, 1, 0, 1, error='RuntimeError: boom')
        finally:
            conn.close()
        result = check_scheduler_heartbeat(str(tmp_path))
        assert result['status'] == 'fail'

    @pytest.fixture
    def runner(self, mm_runner):
        return mm_runner

    def test_doctor_text_mode_emits_colored_summary(self, runner):
        """`memman doctor --text` produces a human-readable report.

        Exit code may be 0 (pass/warn) or 1 (fail) depending on environment.
        """
        r, data_dir = runner
        result = r.invoke(cli, ['--data-dir', data_dir, 'doctor', '--text'])
        assert result.exit_code in {0, 1}, result.output
        assert 'memman doctor' in result.output
        assert ('sqlite_integrity' in result.output
                or 'env_permissions' in result.output)

    def test_doctor_json_default(self, runner):
        """`memman doctor` emits JSON by default.

        Exit code may be 0 (pass/warn) or 1 (fail) depending on environment.
        """
        r, data_dir = runner
        result = r.invoke(cli, ['--data-dir', data_dir, 'doctor'])
        assert result.exit_code in {0, 1}, result.output
        payload = json.loads(result.output)
        assert 'checks' in payload
        assert 'status' in payload

    def test_doctor_reports_llm_probe_failure(self, runner, monkeypatch):
        """`memman doctor` surfaces an LLM ConfigError and exits non-zero.

        Replaces the prior `keys test` surface; doctor's check_llm_probe
        is now the canonical key-validity gate.
        """
        from memman.exceptions import ConfigError

        r, data_dir = runner
        monkeypatch.delenv('MEMMAN_OPENROUTER_API_KEY', raising=False)

        def _raise():
            raise ConfigError('MEMMAN_OPENROUTER_API_KEY must be set')
        monkeypatch.setattr(
            'memman.llm.client.get_llm_client', _raise)

        result = r.invoke(cli, ['--data-dir', data_dir, 'doctor'])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload['status'] == 'fail'
        llm_check = next(
            (c for c in payload['checks'] if c['name'] == 'llm_probe'),
            None)
        assert llm_check is not None
        assert llm_check['status'] == 'fail'
        assert 'MEMMAN_OPENROUTER_API_KEY' in llm_check['detail']['error']

    def test_doctor_reports_probes_pass_under_mocks(self, runner):
        """With the autouse mocks both LLM and embed probes pass."""
        r, data_dir = runner
        result = r.invoke(cli, ['--data-dir', data_dir, 'doctor'])
        payload = json.loads(result.output)
        llm_check = next(
            c for c in payload['checks'] if c['name'] == 'llm_probe')
        embed_check = next(
            c for c in payload['checks'] if c['name'] == 'embed_probe')
        assert llm_check['status'] == 'pass'
        assert embed_check['status'] == 'pass'


class TestDrainHeartbeat:
    """check_drain_heartbeat: per-store drain-heartbeat consumer."""

    pytestmark = pytest.mark.postgres

    def test_skips_when_no_postgres_stores(self, tmp_path):
        """No postgres-backed stores -> pass with skipped_reason."""
        result = check_drain_heartbeat(str(tmp_path))
        assert result['name'] == 'drain_heartbeat'
        assert result['status'] == 'pass'
        assert 'skipped_reason' in result['detail']

    def test_passes_when_no_in_progress_runs(self, env_file, pg_dsn):
        """Postgres-backed store with no in-progress runs: status pass."""
        import os

        from memman.store.postgres import _store_schema, drop_postgres_store
        from memman.store.postgres import open_postgres_backend

        store = 'hb_doctor_setup'
        env_file(f'MEMMAN_BACKEND_{store}', 'postgres')
        env_file(f'MEMMAN_POSTGRES_DSN_{store}', pg_dsn)
        try:
            drop_postgres_store(store, pg_dsn)
        except Exception:
            pass
        backend = open_postgres_backend(store, pg_dsn)
        backend.close()

        try:
            data_dir = os.environ['MEMMAN_DATA_DIR']
            schema = _store_schema(store)
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f'update {schema}.worker_runs set ended_at = now()'
                        f' where ended_at is null')

            result = check_drain_heartbeat(data_dir)
            assert result['status'] == 'pass'
            assert result['detail']['in_progress'] == 0
            assert result['detail']['stale_runs'] == []
            assert store in result['detail']['stores_checked']
        finally:
            try:
                drop_postgres_store(store, pg_dsn)
            except Exception:
                pass

    def test_warns_no_drain_heartbeat_in_5m(self, env_file, pg_dsn):
        """In-progress per-store run with stale heartbeat -> warn."""
        import os

        from memman.store.postgres import _store_schema, drop_postgres_store
        from memman.store.postgres import open_postgres_backend

        store = 'hb_doctor_stale'
        env_file(f'MEMMAN_BACKEND_{store}', 'postgres')
        env_file(f'MEMMAN_POSTGRES_DSN_{store}', pg_dsn)
        try:
            drop_postgres_store(store, pg_dsn)
        except Exception:
            pass
        backend = open_postgres_backend(store, pg_dsn)
        backend.close()

        try:
            data_dir = os.environ['MEMMAN_DATA_DIR']
            schema = _store_schema(store)
            stale = datetime.now(timezone.utc) - timedelta(minutes=10)
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f'insert into {schema}.worker_runs'
                        f' (started_at, ended_at, last_heartbeat_at)'
                        f' values (%s, null, %s) returning id',
                        (stale, stale))
                    stale_id = cur.fetchone()[0]

            result = check_drain_heartbeat(data_dir)
            assert result['status'] == 'warn'
            stale_runs = result['detail']['stale_runs']
            assert any(s['run_id'] == stale_id for s in stale_runs)
            match = next(
                s for s in stale_runs if s['run_id'] == stale_id)
            assert match['age_seconds'] >= 5 * 60
            assert match['store'] == store
        finally:
            try:
                drop_postgres_store(store, pg_dsn)
            except Exception:
                pass

    def test_no_warn_for_fresh_heartbeat(self, env_file, pg_dsn):
        """Per-store in-progress run with recent heartbeat does NOT warn."""
        import os

        from memman.store.postgres import _store_schema, drop_postgres_store
        from memman.store.postgres import open_postgres_backend

        store = 'hb_doctor_fresh'
        env_file(f'MEMMAN_BACKEND_{store}', 'postgres')
        env_file(f'MEMMAN_POSTGRES_DSN_{store}', pg_dsn)
        try:
            drop_postgres_store(store, pg_dsn)
        except Exception:
            pass
        backend = open_postgres_backend(store, pg_dsn)
        backend.close()

        try:
            data_dir = os.environ['MEMMAN_DATA_DIR']
            schema = _store_schema(store)
            fresh = datetime.now(timezone.utc) - timedelta(seconds=30)
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f'insert into {schema}.worker_runs'
                        f' (started_at, ended_at, last_heartbeat_at)'
                        f' values (%s, null, %s) returning id',
                        (fresh, fresh))

            result = check_drain_heartbeat(data_dir)
            assert result['status'] == 'pass'
            assert result['detail']['stale_runs'] == []
            assert result['detail']['in_progress'] >= 1
        finally:
            try:
                drop_postgres_store(store, pg_dsn)
            except Exception:
                pass


class TestDrainHeartbeatSeverity:
    """check_drain_heartbeat severity ladder when failures and stale combine."""

    def test_failures_outrank_stale(self, tmp_path, monkeypatch):
        """Both failures and stale present -> fail (failures wins)."""
        from contextlib import contextmanager

        from memman import doctor as doctor_mod

        @contextmanager
        def _fake_open_backend(store, data_dir, *, read_only=False):
            if store == 'broken':
                raise RuntimeError('connection refused')
            yield _StaleRunsBackend()

        class _StaleRunsBackend:

            def recent_runs(self, *, limit):
                from datetime import datetime, timedelta, timezone

                from memman.store.model import WorkerRun
                stale = datetime.now(timezone.utc) - timedelta(minutes=10)
                return [WorkerRun(
                    id=42, started_at=stale, ended_at=None,
                    last_heartbeat_at=stale)]

        monkeypatch.setattr(
            'memman.store.factory.list_stores',
            lambda data_dir: ['broken', 'has_stale'])
        monkeypatch.setattr(
            'memman.store.factory.resolve_store_backend',
            lambda store, data_dir: 'postgres')
        monkeypatch.setattr(
            'memman.store.factory.open_backend', _fake_open_backend)

        result = doctor_mod.check_drain_heartbeat(str(tmp_path))
        assert result['status'] == 'fail'
        assert len(result['detail']['failures']) == 1
        assert result['detail']['failures'][0]['store'] == 'broken'
        assert len(result['detail']['stale_runs']) == 1
        assert result['detail']['stale_runs'][0]['store'] == 'has_stale'


class TestDoctorBackendDispatch:
    """`memman doctor` runs against the active backend, not always SQLite."""

    pytestmark = pytest.mark.postgres

    def test_doctor_dispatches_to_postgres(
            self, tmp_path, env_file, pg_dsn, monkeypatch):
        """`db_path` reports the redacted DSN, not a filesystem path."""
        store = 'doctor_dispatch'
        env_file(f'MEMMAN_BACKEND_{store}', 'postgres')
        env_file(f'MEMMAN_POSTGRES_DSN_{store}', pg_dsn)
        monkeypatch.setenv('MEMMAN_STORE', store)

        from memman.store.postgres import drop_postgres_store
        from memman.store.postgres import open_postgres_backend
        try:
            drop_postgres_store(store, pg_dsn)
        except Exception:
            pass
        b = open_postgres_backend(store, pg_dsn)
        b.close()

        try:
            runner = CliRunner()
            result = runner.invoke(
                cli, ['--data-dir', str(tmp_path / 'memman'), 'doctor'])
            assert result.exit_code in {0, 1}, result.output
            data = json.loads(result.output)
            assert '#store_doctor_dispatch' in data['db_path']
        finally:
            try:
                drop_postgres_store(store, pg_dsn)
            except Exception:
                pass


class TestClaudeHooksCheck:
    """`check_claude_hooks` compares live registrations to the installer."""

    def _install(self, home, *, matcher='Agent|Task', drop=None,
                 extra_stop=False, dangle=False):
        """Write a settings.json and hook files under a fake home."""
        import json as _json
        hooks_dir = home / '.claude' / 'hooks' / 'memman'
        hooks_dir.mkdir(parents=True)
        scripts = ['prime.sh', 'user_prompt.sh', 'compact.sh',
                   'task_recall.sh', 'exit_plan.sh']
        for name in scripts:
            if dangle and name == 'task_recall.sh':
                continue
            (hooks_dir / name).write_text('#!/bin/bash\n')

        def cmd(name):
            return f'~/.claude/hooks/memman/{name}'

        hooks = {
            'SessionStart': [{'hooks': [
                {'type': 'command', 'command': cmd('prime.sh')}]}],
            'UserPromptSubmit': [{'hooks': [
                {'type': 'command', 'command': cmd('user_prompt.sh')}]}],
            'PreCompact': [{'hooks': [
                {'type': 'command', 'command': cmd('compact.sh')}]}],
            'PreToolUse': [
                {'hooks': [{'type': 'command',
                            'command': cmd('task_recall.sh')}],
                 'matcher': matcher},
                {'hooks': [{'type': 'command',
                            'command': cmd('exit_plan.sh')}],
                 'matcher': 'ExitPlanMode'},
                ],
            }
        if extra_stop:
            hooks['Stop'] = [{'hooks': [
                {'type': 'command', 'command': cmd('stop.sh')}]}]
            (hooks_dir / 'stop.sh').write_text('#!/bin/bash\n')
        if drop:
            hooks.pop(drop)
        settings = home / '.claude' / 'settings.json'
        settings.write_text(_json.dumps({'hooks': hooks}))

    def test_clean_install_passes(self, tmp_path, monkeypatch):
        """Verify a settings file the installer would write reports pass.

        Mutation: a check that compares the wrong shape and flags every
            healthy install, which would make the report unreadable.
        Oracle: a settings.json built to match what
            add_claude_hooks_selective emits.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        self._install(tmp_path)
        from memman.doctor import check_claude_hooks
        assert check_claude_hooks()['status'] == 'pass'

    def test_no_claude_config_passes(self, tmp_path, monkeypatch):
        """Verify a machine with no Claude Code install is not a failure.

        Mutation: treating a missing settings.json as drift, which
            would fail doctor on a machine with no Claude Code install.
        Oracle: a home directory with no .claude at all.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        from memman.doctor import check_claude_hooks
        assert check_claude_hooks()['status'] == 'pass'

    def test_dangling_command_fails(self, tmp_path, monkeypatch):
        """Verify a registration whose script is gone reports fail.

        Mutation: a check that compares registrations but never probes
            the command path, so a pipx upgrade that removed a hook
            script leaves Claude Code running exit 127 unreported.
        Oracle: a settings entry naming a script absent from the hooks
            directory.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        self._install(tmp_path, dangle=True)
        from memman.doctor import check_claude_hooks
        result = check_claude_hooks()
        assert result['status'] == 'fail'
        assert any('task_recall.sh' in c
                   for c in result['detail']['dangling'])

    def test_retired_stop_entry_warns(self, tmp_path, monkeypatch):
        """Verify a hook event memman no longer registers is reported.

        Mutation: comparing only the events the installer writes, so a
            retired registration left by an older install stays
            invisible.
        Oracle: a Stop entry, which no memman version at HEAD writes.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        self._install(tmp_path, extra_stop=True)
        from memman.doctor import check_claude_hooks
        result = check_claude_hooks()
        assert result['status'] == 'warn'
        assert any('Stop' in e for e in result['detail']['extra'])

    def test_stale_matcher_warns(self, tmp_path, monkeypatch):
        """Verify a matcher the installer no longer writes is reported.

        Mutation: comparing event and command but dropping the matcher,
            so a pre-0.40.1 registration keeps a narrower matcher with
            no warning.
        Oracle: the replaced matcher value Task against the
            installer's own current value.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        self._install(tmp_path, matcher='Task')
        from memman.doctor import check_claude_hooks
        result = check_claude_hooks()
        assert result['status'] == 'warn'
        assert result['detail']['missing']
        assert result['detail']['extra']

    def test_missing_event_warns(self, tmp_path, monkeypatch):
        """Verify a hook the installer writes but settings lacks is seen.

        Mutation: comparing live against expected in one direction, so
            a registration dropped by hand is never noticed.
        Oracle: a settings.json with the PreCompact entry removed.
        """
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        self._install(tmp_path, drop='PreCompact')
        from memman.doctor import check_claude_hooks
        result = check_claude_hooks()
        assert result['status'] == 'warn'
        assert any('compact.sh' in m for m in result['detail']['missing'])
