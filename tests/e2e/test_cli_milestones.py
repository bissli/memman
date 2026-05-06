"""End-to-end CLI milestone tests (subset against the queue API).

V4.1 trimmed this file from 68 to 17 tests. The deletions removed
tests that:
- asserted on `facts.0.edges_created.X` synchronous-API artifacts that
  no longer exist under the queue model
- duplicated unit-suite coverage with weaker `status.edge_count`
  assertions
- coupled to LLM-output stability (specific entity strings, causal
  edge counts) and would re-break on every model swap

The 17 surviving tests exercise unique end-to-end behavior the unit
suite cannot: CLI subprocess + real env file + drain pipeline. They
split across 4 milestones (M0 store ops, M1 CRUD read paths, M3
search, M11 validation) and are surgically gated on
`requires_live_keys` only where drain is exercised.
"""

import os
import sqlite3
import uuid
from pathlib import Path

import pytest

from .helpers import assert_contains, assert_jq, assert_jq_gte
from .helpers import find_insight_by_recall, json_out, run_cli

pytestmark = pytest.mark.e2e_cli


_BASE_ENV_LINES = (
    'MEMMAN_LLM_PROVIDER=openrouter',
    'MEMMAN_LLM_MODEL_FAST=anthropic/claude-haiku-4.5',
    'MEMMAN_LLM_MODEL_SLOW_CANONICAL=anthropic/claude-sonnet-4.6',
    'MEMMAN_LLM_MODEL_SLOW_METADATA=anthropic/claude-sonnet-4.6',
    'MEMMAN_EMBED_PROVIDER=voyage',
    'MEMMAN_RERANK_PROVIDER=voyage',
    'MEMMAN_OPENROUTER_ENDPOINT=https://openrouter.ai/api/v1',
    'MEMMAN_VOYAGE_RERANK_MODEL=rerank-2.5-lite',
    'MEMMAN_BACKEND=sqlite',
    )


# ---------------------------------------------------------------------
# Module-scoped fixtures: HOME with env file seeded; data dirs that
# pre-seed the embed_fingerprint to skip the seed_if_fresh Voyage
# probe (except `store_dir`, which test_store_list_empty needs empty).
# ---------------------------------------------------------------------

@pytest.fixture(scope='module')
def home_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One HOME for the whole module.

    Writes scheduler-state + an env file so memman's runtime config
    resolver finds providers, endpoints, and (when present) real API
    keys. Without the env file, every store open raises
    `MEMMAN_EMBED_PROVIDER is not set`.
    """
    home = tmp_path_factory.mktemp('e2e_home')
    dot = home / '.memman'
    dot.mkdir(parents=True, exist_ok=True)
    (dot / 'scheduler.state').write_text('started\n')
    (dot / 'scheduler.state').chmod(0o600)
    (dot / 'cache').mkdir(exist_ok=True)
    lines = list(_BASE_ENV_LINES)
    for k in ('OPENROUTER_API_KEY', 'VOYAGE_API_KEY'):
        v = os.environ.get(k) or 'placeholder-for-non-live-tests'
        lines.append(f'{k}={v}')
    (dot / 'env').write_text('\n'.join(lines) + '\n')
    return home


def _seed_fingerprint(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            'create table if not exists meta '
            '(key text primary key, value text not null)')
        conn.execute(
            "insert or replace into meta (key, value) values "
            "('embed_fingerprint', "
            '\'{"provider":"voyage","model":"voyage-3-lite","dim":512}\')')
        conn.commit()
    finally:
        conn.close()


def _data_dir(home: Path, name: str, seed: bool = True) -> Path:
    """Per-test data dir, optionally pre-seeded with a fingerprint.

    Pre-seeding lets `seed_if_fresh` short-circuit on the existing
    fingerprint, avoiding the Voyage availability probe on every
    store open. Tests that explicitly need an empty store list
    (M0 `test_store_list_empty`) pass `seed=False`.
    """
    d = home / 'data' / name
    d.mkdir(parents=True, exist_ok=True)
    if seed:
        _seed_fingerprint(d / 'data' / 'default' / 'memman.db')
    return d


@pytest.fixture(scope='module')
def store_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'store_test', seed=False)


@pytest.fixture(scope='module')
def m1_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'm1')


@pytest.fixture(scope='module')
def m3_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'm3')


# ---------------------------------------------------------------------
# M0: Store Management
# ---------------------------------------------------------------------

class TestM0Stores:

    def test_store_list_empty(self, home_dir: Path, store_dir: Path):
        out = run_cli(['store', 'list'], home_dir, store_dir)
        assert_jq(json_out(out), 'stores', [], 'empty stores array')

    @pytest.mark.requires_live_keys
    def test_store_create_default(self, home_dir: Path, store_dir: Path,
                                  live_keys):
        out = run_cli(['store', 'create', 'default'], home_dir, store_dir)
        assert_jq(json_out(out), 'action', 'created', 'created default')

    @pytest.mark.requires_live_keys
    def test_store_create_work(self, home_dir: Path, store_dir: Path,
                               live_keys):
        out = run_cli(['store', 'create', 'work'], home_dir, store_dir)
        assert_jq(json_out(out), 'store', 'work', 'created work')

    def test_store_create_reject_duplicate(self, home_dir: Path,
                                           store_dir: Path):
        out = run_cli(['store', 'create', 'work'], home_dir, store_dir,
                      check=False)
        assert_contains(out.stdout + out.stderr, 'already exists',
                        'rejects duplicate')

    def test_store_create_reject_invalid_name(self, home_dir: Path,
                                              store_dir: Path):
        out = run_cli(['store', 'create', '.bad'], home_dir, store_dir,
                      check=False)
        assert_contains(out.stdout + out.stderr, 'invalid store name',
                        'rejects invalid')

    @pytest.mark.requires_live_keys
    def test_store_list_shows_created(self, home_dir: Path,
                                      store_dir: Path, live_keys):
        data = json_out(run_cli(['store', 'list'], home_dir, store_dir))
        assert 'default' in data['stores'], 'lists default'
        assert 'work' in data['stores'], 'lists work'

    @pytest.mark.requires_live_keys
    def test_store_use_switch_active(self, home_dir: Path,
                                     store_dir: Path, live_keys):
        run_cli(['store', 'use', 'work'], home_dir, store_dir)
        data = json_out(run_cli(['store', 'list'], home_dir, store_dir))
        assert_jq(data, 'active', 'work', 'work is active')

    def test_store_use_reject_nonexistent(self, home_dir: Path,
                                          store_dir: Path):
        out = run_cli(['store', 'use', 'nonexistent'], home_dir,
                      store_dir, check=False)
        assert_contains(out.stdout + out.stderr, 'does not exist',
                        'rejects missing')

    def test_store_remove_reject_active(self, home_dir: Path,
                                        store_dir: Path):
        out = run_cli(['store', 'remove', 'work'], home_dir, store_dir,
                      check=False)
        assert_contains(out.stdout + out.stderr,
                        'cannot remove the active store',
                        'rejects active removal')

    @pytest.mark.requires_live_keys
    def test_store_remove_inactive(self, home_dir: Path,
                                   store_dir: Path, live_keys):
        run_cli(['store', 'create', 'temp'], home_dir, store_dir)
        out = run_cli(['store', 'remove', 'temp', '--yes'], home_dir,
                      store_dir)
        assert_jq(json_out(out), 'action', 'removed', 'removed temp')


# ---------------------------------------------------------------------
# M1: Basic CRUD read paths (write -> drain -> read)
# ---------------------------------------------------------------------

class TestM1CRUD:

    @pytest.mark.requires_live_keys
    def test_recall_keyword(self, home_dir: Path, m1_dir: Path,
                            live_keys):
        unique = uuid.uuid4().hex[:8]
        run_cli(
            ['remember', '--no-reconcile',
             f'User prefers Qdrant for vector DB recall-keyword-{unique}',
             '--cat', 'preference', '--imp', '4'],
            home_dir, m1_dir)
        run_cli(['scheduler', 'serve', '--once'], home_dir, m1_dir)
        out = run_cli(['recall', '--basic', f'recall-keyword-{unique}'],
                      home_dir, m1_dir)
        assert_contains(out.stdout, 'Qdrant',
                        'recall returns the stored insight')

    @pytest.mark.requires_live_keys
    def test_recall_no_match_sparse(self, home_dir: Path, m1_dir: Path,
                                    live_keys):
        out = run_cli(['recall', 'nonexistent_xyz_no_match_token'],
                      home_dir, m1_dir)
        assert_jq(json_out(out), 'meta.sparse', True, 'sparse flag')

    @pytest.mark.requires_live_keys
    def test_status_statistics(self, home_dir: Path, m1_dir: Path,
                               live_keys):
        data = json_out(run_cli(['status'], home_dir, m1_dir))
        assert_jq_gte(data, 'total_insights', 1, 'total >= 1 after writes')
        assert_jq_gte(data, 'by_category.preference', 1,
                      'preference count >= 1')

    @pytest.mark.requires_live_keys
    def test_forget_soft_delete(self, home_dir: Path, m1_dir: Path,
                                live_keys):
        unique = uuid.uuid4().hex[:8]
        run_cli(
            ['remember', '--no-reconcile',
             f'User prefers PostgreSQL for forget-test-{unique}',
             '--cat', 'preference', '--imp', '3'],
            home_dir, m1_dir)
        run_cli(['scheduler', 'serve', '--once'], home_dir, m1_dir)
        insight_id = find_insight_by_recall(
            home_dir, m1_dir, f'forget-test-{unique}')
        out = run_cli(['forget', insight_id], home_dir, m1_dir)
        assert_jq(json_out(out), 'status', 'deleted', 'soft-deleted')
        data = json_out(run_cli(['status'], home_dir, m1_dir))
        assert_jq_gte(data, 'deleted_insights', 1, 'deleted count >= 1')


# ---------------------------------------------------------------------
# M3: Basic search (--basic recall)
# ---------------------------------------------------------------------

class TestM3Search:

    @pytest.mark.requires_live_keys
    def test_recall_basic_token_only(self, home_dir: Path, m3_dir: Path,
                                     live_keys):
        unique = uuid.uuid4().hex[:8]
        run_cli(
            ['remember', '--no-reconcile',
             f'Chose Qdrant because of Rust performance basic-{unique}',
             '--cat', 'decision', '--imp', '5'],
            home_dir, m3_dir)
        run_cli(['scheduler', 'serve', '--once'], home_dir, m3_dir)
        out = run_cli(['recall', '--basic', f'basic-{unique}'],
                      home_dir, m3_dir)
        assert_contains(out.stdout, 'Chose Qdrant',
                        'finds decision insight')
        assert_contains(out.stdout, '"results"', 'results envelope')

    @pytest.mark.requires_live_keys
    def test_recall_basic_no_match(self, home_dir: Path, m3_dir: Path,
                                   live_keys):
        out = run_cli(['recall', '--basic', 'zzz_no_match_zzz'],
                      home_dir, m3_dir)
        assert_jq(json_out(out), 'results', [], 'empty results array')


# ---------------------------------------------------------------------
# M11: CLI validation (no keys needed)
# ---------------------------------------------------------------------

class TestM11Reranking:

    def test_invalid_intent_rejected(self, home_dir: Path, m3_dir: Path):
        out = run_cli(
            ['recall', 'test', '--intent', 'INVALID'],
            home_dir, m3_dir, check=False)
        assert_contains(out.stdout + out.stderr, 'unknown intent',
                        'rejects invalid intent')
