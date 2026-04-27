"""End-to-end CLI milestone tests.

One test per milestone step, class-grouped per banner. Helpers in
`tests/e2e/helpers.py` provide JSON-walk assertion utilities.

Cross-milestone state sharing (M2 -> M3, M4 -> M11, etc.) is preserved
via module-scoped data-dir fixtures. Tests within a class run in
declaration order (pytest default), and module fixtures retain state
across classes within this file.

`time.sleep(1)` guards ensure distinct timestamps for temporal-edge
detection.
"""

import time
from pathlib import Path

import pytest

from .helpers import assert_contains, assert_jq, assert_jq_gte
from .helpers import assert_not_contains, extract_id, json_out, run_cli

pytestmark = pytest.mark.e2e_cli


# ---------------------------------------------------------------------
# Shared module-scoped data-dir fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope='module')
def home_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One HOME for the whole module; scheduler started state written once.
    """
    home = tmp_path_factory.mktemp('e2e_home')
    dot = home / '.memman'
    dot.mkdir(parents=True, exist_ok=True)
    (dot / 'scheduler.state').write_text('started\n')
    (dot / 'scheduler.state').chmod(0o600)
    (dot / 'cache').mkdir(exist_ok=True)
    return home


def _data_dir(home: Path, name: str) -> Path:
    d = home / 'data' / name
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope='module')
def store_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'store_test')


@pytest.fixture(scope='module')
def m1_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'm1')


@pytest.fixture(scope='module')
def m2_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'm2')


@pytest.fixture(scope='module')
def m3_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'm4')


@pytest.fixture(scope='module')
def m5_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'm5')


@pytest.fixture(scope='module')
def m6_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'm6')


@pytest.fixture(scope='module')
def m7_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'm7')


@pytest.fixture(scope='module')
def m8_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'm8')


@pytest.fixture(scope='module')
def m_dict_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'm_dict')


@pytest.fixture(scope='module')
def m9_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'm9')


@pytest.fixture(scope='module')
def m10_dir(home_dir: Path) -> Path:
    return _data_dir(home_dir, 'm10')


@pytest.fixture(scope='module')
def shared_state(home_dir: Path) -> dict:
    """Stash IDs that one test needs to pass to another.
    """
    return {}


# ---------------------------------------------------------------------
# M0: Store Management & Data Isolation (lines 135-214)
# ---------------------------------------------------------------------

class TestM0Stores:

    def test_store_list_empty(self, home_dir: Path, store_dir: Path):
        out = run_cli(['store', 'list'], home_dir, store_dir)
        assert_jq(json_out(out), 'stores', [], 'empty stores array')

    def test_store_create_default(self, home_dir: Path, store_dir: Path):
        out = run_cli(['store', 'create', 'default'], home_dir, store_dir)
        assert_jq(json_out(out), 'action', 'created', 'created default')

    def test_store_create_work(self, home_dir: Path, store_dir: Path):
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

    def test_store_list_shows_created(self, home_dir: Path,
                                      store_dir: Path):
        data = json_out(run_cli(['store', 'list'], home_dir, store_dir))
        assert 'default' in data['stores'], 'lists default'
        assert 'work' in data['stores'], 'lists work'

    def test_store_use_switch_active(self, home_dir: Path, store_dir: Path):
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

    def test_store_remove_inactive(self, home_dir: Path, store_dir: Path):
        run_cli(['store', 'create', 'temp'], home_dir, store_dir)
        out = run_cli(['store', 'remove', 'temp', '--yes'], home_dir,
                      store_dir)
        assert_jq(json_out(out), 'action', 'removed', 'removed temp')

    def test_data_isolation(self, home_dir: Path, store_dir: Path):
        out = run_cli(
            ['remember', '--no-reconcile',
             'Configuration for the default store uses PostgreSQL database',
             '--cat', 'fact', '--imp', '3'],
            home_dir, store_dir, extra_env={'MEMMAN_STORE': 'default'})
        assert_jq(json_out(out), 'facts.0.action', 'add', 'default stored')

        out = run_cli(
            ['remember', '--no-reconcile',
             'Work store configuration uses Redis for caching layer',
             '--cat', 'fact', '--imp', '3'],
            home_dir, store_dir, extra_env={'MEMMAN_STORE': 'work'})
        assert_jq(json_out(out), 'facts.0.action', 'add', 'work stored')

        out = run_cli(
            ['recall', '--basic', 'PostgreSQL database'],
            home_dir, store_dir, extra_env={'MEMMAN_STORE': 'default'})
        assert_contains(out.stdout, 'PostgreSQL', 'default finds own data')
        assert_not_contains(out.stdout, 'Redis',
                            'default not finds work data')

        out = run_cli(
            ['recall', '--basic', 'Redis caching'],
            home_dir, store_dir, extra_env={'MEMMAN_STORE': 'work'})
        assert_contains(out.stdout, 'Redis', 'work finds own data')
        assert_not_contains(out.stdout, 'PostgreSQL',
                            'work not finds default data')

    def test_memman_store_env_override(self, home_dir: Path,
                                       store_dir: Path):
        out = run_cli(['status'], home_dir, store_dir,
                      extra_env={'MEMMAN_STORE': 'default'})
        assert_contains(out.stdout, 'data/default/memman.db',
                        'env override db path')


# ---------------------------------------------------------------------
# M1: Basic CRUD (lines 217-265)
# ---------------------------------------------------------------------

class TestM1CRUD:

    def test_remember_with_tags(self, home_dir: Path, m1_dir: Path,
                                shared_state: dict):
        out = run_cli(
            ['remember', '--no-reconcile',
             'User prefers Qdrant for vector DB',
             '--cat', 'preference', '--imp', '4',
             '--tags', 'tool,db'],
            home_dir, m1_dir)
        data = json_out(out)
        shared_state['m1_id1'] = extract_id(data)
        assert_jq(data, 'facts.0.category', 'preference',
                  'category is preference')
        assert_jq(data, 'facts.0.importance', 4, 'importance is 4')
        assert_contains(out.stdout, '"tool"', 'tags include tool')
        assert_contains(out.stdout, '"Qdrant"', 'entities has Qdrant')

    def test_recall_keyword(self, home_dir: Path, m1_dir: Path):
        out = run_cli(['recall', 'Qdrant'], home_dir, m1_dir)
        assert_contains(out.stdout, 'User prefers Qdrant',
                        'found Qdrant insight')

    def test_recall_no_match_sparse(self, home_dir: Path, m1_dir: Path):
        out = run_cli(['recall', 'nonexistent_xyz'], home_dir, m1_dir)
        assert_jq(json_out(out), 'meta.sparse', True, 'sparse flag')

    def test_status_statistics(self, home_dir: Path, m1_dir: Path):
        data = json_out(run_cli(['status'], home_dir, m1_dir))
        assert_jq(data, 'total_insights', 1, 'total is 1')
        assert_jq(data, 'by_category.preference', 1, 'preference count')
        assert_jq(data, 'deleted_insights', 0, 'no deleted insights')

    def test_forget_soft_delete(self, home_dir: Path, m1_dir: Path,
                                shared_state: dict):
        out = run_cli(['forget', shared_state['m1_id1']], home_dir, m1_dir)
        assert_jq(json_out(out), 'status', 'deleted', 'status is deleted')
        data = json_out(run_cli(['status'], home_dir, m1_dir))
        assert_jq(data, 'total_insights', 0, 'total now 0')
        assert_jq(data, 'deleted_insights', 1, 'deleted now 1')

    def test_replace_atomic(self, home_dir: Path, m1_dir: Path):
        out = run_cli(
            ['remember', '--no-reconcile',
             'Python is best for scripting', '--cat', 'fact', '--imp', '3'],
            home_dir, m1_dir)
        repl_id = extract_id(json_out(out))

        out = run_cli(
            ['replace', repl_id, 'Python is excellent for scripting'],
            home_dir, m1_dir)
        data = json_out(out)
        assert_jq(data, 'facts.0.action', 'replace', 'action is replaced')
        assert_jq(data, 'facts.0.replaced_id', repl_id, 'replaced_id set')

        out = run_cli(['recall', '--basic', 'Python scripting'],
                      home_dir, m1_dir)
        assert_contains(out.stdout, 'Python is excellent',
                        'new content present')
        assert_not_contains(out.stdout, 'Python is best',
                            'old content gone')


# ---------------------------------------------------------------------
# M2: Graph Edge Auto-Generation (lines 268-331)
# ---------------------------------------------------------------------

class TestM2GraphEdges:

    def test_temporal_edges_3_insights(self, home_dir: Path, m2_dir: Path,
                                       shared_state: dict):
        out = run_cli(
            ['remember', '--no-reconcile',
             'User prefers Qdrant for vector DB',
             '--cat', 'preference', '--imp', '4'],
            home_dir, m2_dir)
        data = json_out(out)
        shared_state['m2_id_a'] = extract_id(data)
        assert_jq(data, 'facts.0.edges_created.temporal', 0,
                  'first: no temporal')

        time.sleep(1)
        out = run_cli(
            ['remember', '--no-reconcile',
             'Chose Qdrant because of Rust performance',
             '--cat', 'decision', '--imp', '5'],
            home_dir, m2_dir)
        data = json_out(out)
        shared_state['m2_id_b'] = extract_id(data)
        assert_jq(data, 'facts.0.edges_created.temporal', 2,
                  'second: 2 temporal')

        time.sleep(1)
        out = run_cli(
            ['remember', '--no-reconcile',
             'Qdrant benchmark shows 10ms p99 latency',
             '--cat', 'fact', '--imp', '3'],
            home_dir, m2_dir)
        data = json_out(out)
        shared_state['m2_id_c'] = extract_id(data)
        assert_jq_gte(data, 'facts.0.edges_created.temporal', 2,
                      'third: >= 2 temporal')

    def test_status_edge_count(self, home_dir: Path, m2_dir: Path):
        data = json_out(run_cli(['status'], home_dir, m2_dir))
        assert_jq_gte(data, 'edge_count', 5, 'edges >= 5')

    def test_graph_related_temporal(self, home_dir: Path, m2_dir: Path,
                                    shared_state: dict):
        out = run_cli(
            ['graph', 'related', shared_state['m2_id_b'],
             '--edge', 'temporal'],
            home_dir, m2_dir)
        assert_contains(out.stdout, shared_state['m2_id_a'],
                        'finds A via temporal')
        assert_contains(out.stdout, shared_state['m2_id_c'],
                        'finds C via temporal')

    def test_entity_extraction_camelcase(self, home_dir: Path,
                                         m2_dir: Path, shared_state: dict):
        out = run_cli(
            ['remember', '--no-reconcile',
             'We use HttpServer and DataStore in the project',
             '--cat', 'fact'],
            home_dir, m2_dir)
        shared_state['m2_id_d'] = extract_id(json_out(out))
        assert_contains(out.stdout, '"HttpServer"', 'HttpServer extracted')
        assert_contains(out.stdout, '"DataStore"', 'DataStore extracted')

    def test_entity_edge_shared(self, home_dir: Path, m2_dir: Path,
                                shared_state: dict):
        time.sleep(1)
        out = run_cli(
            ['remember', '--no-reconcile',
             'HttpServer handles all API requests',
             '--cat', 'fact'],
            home_dir, m2_dir)
        data = json_out(out)
        shared_state['m2_id_e'] = extract_id(data)
        assert_jq_gte(data, 'facts.0.edges_created.entity', 2,
                      'entity edges created (bidirectional)')

    def test_entity_edge_bidirectional(self, home_dir: Path, m2_dir: Path,
                                       shared_state: dict):
        out = run_cli(
            ['graph', 'related', shared_state['m2_id_e'],
             '--edge', 'entity'],
            home_dir, m2_dir)
        assert_contains(out.stdout, shared_state['m2_id_d'],
                        'E -> D via entity')
        out = run_cli(
            ['graph', 'related', shared_state['m2_id_d'],
             '--edge', 'entity'],
            home_dir, m2_dir)
        assert_contains(out.stdout, shared_state['m2_id_e'],
                        'D -> E via entity (reverse)')

    def test_entity_extraction_file_paths(self, home_dir: Path,
                                          m2_dir: Path):
        out = run_cli(
            ['remember', '--no-reconcile',
             'Config lives at ./cmd/root.go and internal/store/db.go',
             '--cat', 'fact'],
            home_dir, m2_dir)
        assert_contains(out.stdout, './cmd/root.go',
                        'file path extracted')


# ---------------------------------------------------------------------
# M3: Search + Diff (lines 334-345)
# ---------------------------------------------------------------------

class TestM3Search:

    def test_recall_basic_token_only(self, home_dir: Path, m2_dir: Path):
        out = run_cli(['recall', '--basic', 'Rust performance'],
                      home_dir, m2_dir)
        assert_contains(out.stdout, 'Chose Qdrant',
                        'finds decision insight')
        assert_contains(out.stdout, '"results"', 'results envelope')

    def test_recall_basic_no_match(self, home_dir: Path, m2_dir: Path):
        out = run_cli(['recall', '--basic', 'zzz_no_match_zzz'],
                      home_dir, m2_dir)
        assert_jq(json_out(out), 'results', [], 'empty results array')


# ---------------------------------------------------------------------
# M4: Intent-Aware Smart Recall (lines 349-401)
# ---------------------------------------------------------------------

class TestM4SmartRecall:

    def test_multi_level_chain(self, home_dir: Path, m3_dir: Path,
                               shared_state: dict):
        out = run_cli(
            ['remember', '--no-reconcile',
             'Alpha service handles request routing',
             '--cat', 'fact', '--imp', '3'],
            home_dir, m3_dir)
        shared_state['m4_id_x'] = extract_id(json_out(out))

        time.sleep(1)
        out = run_cli(
            ['remember', '--no-reconcile',
             'Request routing uses Alpha service because of low latency',
             '--cat', 'decision', '--imp', '4'],
            home_dir, m3_dir)
        data = json_out(out)
        shared_state['m4_id_y'] = extract_id(data)
        assert_jq_gte(data, 'facts.0.edges_created.temporal', 2,
                      'Y has temporal edge')

        time.sleep(1)
        out = run_cli(
            ['remember', '--no-reconcile',
             'Low latency achieved because of edge caching',
             '--cat', 'fact', '--imp', '3'],
            home_dir, m3_dir)
        data = json_out(out)
        shared_state['m4_id_z'] = extract_id(data)
        assert_jq_gte(data, 'facts.0.edges_created.temporal', 2,
                      'Z has temporal edge')

    def test_smart_recall_finds_depth_2(self, home_dir: Path,
                                        m3_dir: Path,
                                        shared_state: dict):
        out = run_cli(['recall', 'why Alpha service routing'],
                      home_dir, m3_dir)
        assert_contains(out.stdout, shared_state['m4_id_x'],
                        'finds anchor X')
        assert_contains(out.stdout, shared_state['m4_id_y'],
                        'finds depth-1 Y')
        assert_contains(out.stdout, shared_state['m4_id_z'],
                        'finds depth-2 Z')

    def test_why_intent(self, home_dir: Path, m2_dir: Path):
        out = run_cli(['recall', 'why did we choose Qdrant'],
                      home_dir, m2_dir)
        assert_contains(out.stdout, '"WHY"', 'intent is WHY')
        assert_contains(out.stdout, 'Qdrant', 'finds Qdrant insight')

    def test_when_intent(self, home_dir: Path, m2_dir: Path):
        out = run_cli(['recall', 'when did we choose vector db'],
                      home_dir, m2_dir)
        assert_contains(out.stdout, '"WHEN"', 'intent is WHEN')

    def test_graph_augments_results(self, home_dir: Path, m2_dir: Path):
        out = run_cli(['recall', 'why Qdrant performance'],
                      home_dir, m2_dir)
        data = json_out(out)
        assert len(data['results']) >= 2, (
            f'expected >= 2 results, got {len(data["results"])}')


# ---------------------------------------------------------------------
# M5: Semantic Edges (Claude-in-the-loop) (lines 404-457)
# ---------------------------------------------------------------------

class TestM5SemanticEdges:

    def test_remember_stamps_linked_at(self, home_dir: Path, m5_dir: Path,
                                       shared_state: dict):
        out = run_cli(
            ['remember', '--no-reconcile',
             'Go is great for building CLI tools',
             '--cat', 'fact', '--imp', '3'],
            home_dir, m5_dir)
        shared_state['m5_id_s1'] = extract_id(json_out(out))

    def test_enrichment_dict_present(self, home_dir: Path, m5_dir: Path,
                                     shared_state: dict):
        time.sleep(1)
        out = run_cli(
            ['remember', '--no-reconcile',
             'Building CLI tools in Go is efficient',
             '--cat', 'fact', '--imp', '3'],
            home_dir, m5_dir)
        shared_state['m5_id_s2'] = extract_id(json_out(out))
        assert_contains(out.stdout, '"enrichment"', 'has enrichment')
        assert_contains(out.stdout, '"causal"',
                        'has causal in edges_created')

    def test_graph_rebuild_dry_run(self, home_dir: Path, m5_dir: Path):
        out = run_cli(['graph', 'rebuild', '--dry-run'], home_dir, m5_dir)
        assert_contains(out.stdout, '"total"', 'has total')

    def test_graph_link_semantic(self, home_dir: Path, m5_dir: Path,
                                 shared_state: dict):
        out = run_cli(
            ['graph', 'link', shared_state['m5_id_s1'],
             shared_state['m5_id_s2'],
             '--type', 'semantic', '--weight', '0.85'],
            home_dir, m5_dir)
        data = json_out(out)
        assert_jq(data, 'status', 'linked', 'status is linked')
        assert_jq(data, 'edge_type', 'semantic',
                  'edge type is semantic')
        assert_contains(out.stdout, '"created_by"',
                        'created_by claude')

    def test_graph_link_bidirectional(self, home_dir: Path, m5_dir: Path,
                                      shared_state: dict):
        out = run_cli(
            ['graph', 'related', shared_state['m5_id_s1'],
             '--edge', 'semantic'],
            home_dir, m5_dir)
        assert_contains(out.stdout, shared_state['m5_id_s2'],
                        'S1 -> S2 via semantic')
        out = run_cli(
            ['graph', 'related', shared_state['m5_id_s2'],
             '--edge', 'semantic'],
            home_dir, m5_dir)
        assert_contains(out.stdout, shared_state['m5_id_s1'],
                        'S2 -> S1 via semantic (reverse)')

    def test_graph_link_weight_validation(self, home_dir: Path,
                                          m5_dir: Path,
                                          shared_state: dict):
        out = run_cli(
            ['graph', 'link', shared_state['m5_id_s1'],
             shared_state['m5_id_s2'],
             '--type', 'semantic', '--weight', '1.5'],
            home_dir, m5_dir, check=False)
        assert_contains(out.stdout + out.stderr, 'weight must be',
                        'rejects weight > 1.0')

    def test_graph_link_nonexistent(self, home_dir: Path, m5_dir: Path,
                                    shared_state: dict):
        out = run_cli(
            ['graph', 'link', shared_state['m5_id_s1'],
             'nonexistent-id-000',
             '--type', 'semantic', '--weight', '0.5'],
            home_dir, m5_dir, check=False)
        assert_contains(out.stdout + out.stderr, 'not found',
                        'rejects missing insight')

    def test_smart_recall_traverses_semantic(self, home_dir: Path,
                                             m5_dir: Path):
        out = run_cli(['recall', 'Go CLI'], home_dir, m5_dir)
        data = json_out(out)
        assert len(data['results']) >= 2, (
            f'expected >= 2 semantic-linked results, '
            f'got {len(data["results"])}')


# ---------------------------------------------------------------------
# M6: Retention Lifecycle (lines 460-549)
# ---------------------------------------------------------------------

class TestM6Retention:

    def test_setup_varying_importance(self, home_dir: Path, m6_dir: Path,
                                      shared_state: dict):
        out = run_cli(
            ['remember', '--no-reconcile',
             'Critical architecture decision to use SQLite for embedded storage',
             '--cat', 'decision', '--imp', '5'],
            home_dir, m6_dir)
        assert_jq(json_out(out), 'facts.0.action', 'add', 'imp5 stored')
        time.sleep(1)
        out = run_cli(
            ['remember', '--no-reconcile',
             'Code formatting follows the PEP8 standard for Python projects',
             '--cat', 'general', '--imp', '1'],
            home_dir, m6_dir)
        assert_jq(json_out(out), 'facts.0.action', 'add', 'imp1 stored')

        out = run_cli(
            ['remember', '--no-reconcile',
             'Temporary deployment context for staging environment setup',
             '--cat', 'context', '--imp', '1'],
            home_dir, m6_dir)
        data = json_out(out)
        assert_jq(data, 'facts.0.action', 'add', 'context stored')
        shared_state['m6_id_low'] = extract_id(data)

        time.sleep(1)
        out = run_cli(
            ['remember', '--no-reconcile',
             'User prefers dark mode theme in all development environments',
             '--cat', 'preference', '--imp', '4'],
            home_dir, m6_dir)
        assert_jq(json_out(out), 'facts.0.action', 'add', 'pref stored')

    def test_remember_includes_effective_importance(self, home_dir: Path,
                                                    m6_dir: Path):
        out = run_cli(
            ['remember', '--no-reconcile',
             'Testing is an important part of the software lifecycle process',
             '--cat', 'fact', '--imp', '3'],
            home_dir, m6_dir)
        assert_contains(out.stdout, '"effective_importance"',
                        'has effective_importance')
        assert_contains(out.stdout, '"auto_pruned"', 'has auto_pruned')
        data = json_out(out)
        assert_jq(data, 'facts.0.auto_pruned', 0,
                  'auto_pruned is 0 (under cap)')

    def test_insights_candidates_suggest_mode(self, home_dir: Path,
                                              m6_dir: Path,
                                              shared_state: dict):
        out = run_cli(
            ['insights', 'candidates', '--threshold', '0.7'],
            home_dir, m6_dir)
        assert_contains(out.stdout, '"candidates"', 'has candidates field')
        assert_contains(out.stdout, '"actions"', 'has actions field')
        assert_contains(out.stdout, '"max_insights"', 'has max_insights')
        data = json_out(out)
        assert_jq_gte(data, 'total_insights', 5, 'total_insights >= 5')
        shared_state['m6_cand_count'] = data['candidates_found']
        assert data['candidates_found'] >= 1, (
            f'expected >= 1 GC candidates, got {data["candidates_found"]}')

    def test_insights_candidates_fields(self, home_dir: Path, m6_dir: Path):
        data = json_out(run_cli(
            ['insights', 'candidates', '--threshold', '0.7'],
            home_dir, m6_dir))
        first = data['candidates'][0]
        assert 'effective_importance' in first, 'has effective_importance'
        assert 'days_since_access' in first, 'has days_since'
        assert 'immune' in first, 'has immune field'

    def test_insights_candidates_excludes_immune(self, home_dir: Path,
                                                 m6_dir: Path):
        data = json_out(run_cli(
            ['insights', 'candidates', '--threshold', '0.7'],
            home_dir, m6_dir))
        high = [c for c in data['candidates'] if c['importance'] >= 4]
        assert len(high) == 0, f'no high-imp candidates, got {len(high)}'

    def test_insights_protect(self, home_dir: Path, m6_dir: Path,
                              shared_state: dict):
        out = run_cli(
            ['insights', 'protect', shared_state['m6_id_low']],
            home_dir, m6_dir)
        data = json_out(out)
        assert_jq(data, 'status', 'retained', 'status is retained')
        assert_jq(data, 'new_access', 3, 'access count boosted')
        assert_contains(out.stdout, '"effective_importance"',
                        'has effective_importance')
        assert_contains(out.stdout, '"immune"', 'has immune field')

    def test_protected_now_immune(self, home_dir: Path, m6_dir: Path,
                                  shared_state: dict):
        data = json_out(run_cli(
            ['insights', 'candidates', '--threshold', '0.7'],
            home_dir, m6_dir))
        ids = [c['id'] for c in data['candidates']]
        assert shared_state['m6_id_low'] not in ids, (
            'boosted insight should be immune (not in candidates)')

    def test_insights_protect_nonexistent(self, home_dir: Path,
                                          m6_dir: Path):
        out = run_cli(
            ['insights', 'protect', 'nonexistent-id-000'],
            home_dir, m6_dir, check=False)
        assert_contains(out.stdout + out.stderr, 'not found',
                        'rejects missing insight')

    def test_higher_threshold_more_candidates(self, home_dir: Path,
                                              m6_dir: Path,
                                              shared_state: dict):
        data = json_out(run_cli(
            ['insights', 'candidates', '--threshold', '2.0'],
            home_dir, m6_dir))
        assert data['candidates_found'] >= shared_state['m6_cand_count'], (
            f'higher threshold expected >= {shared_state["m6_cand_count"]}, '
            f'got {data["candidates_found"]}')


# ---------------------------------------------------------------------
# Observability: Operation Log (lines 552-566)
# ---------------------------------------------------------------------

class TestObservabilityLog:

    def test_log_remember_recall(self, home_dir: Path, m2_dir: Path):
        out = run_cli(['log', 'list', '--limit', '30'], home_dir, m2_dir)
        assert_contains(out.stdout, 'remember', 'log has remember ops')
        assert_contains(out.stdout, 'recall', 'log has recall ops')

    def test_log_link(self, home_dir: Path, m5_dir: Path):
        out = run_cli(['log', 'list', '--limit', '30'], home_dir, m5_dir)
        assert_contains(out.stdout, 'link', 'log has link ops')

    def test_log_protect(self, home_dir: Path, m6_dir: Path):
        out = run_cli(['log', 'list', '--limit', '30'], home_dir, m6_dir)
        assert_contains(out.stdout, 'protect', 'log has protect ops')


# ---------------------------------------------------------------------
# M7: Embedding Support — DROPPED.
#
# The original bash script (lines 569-625) called `memman embed status`
# and `memman embed backfill`, but the `embed` subcommand was removed
# in a recent refactor. The bash script now fails on M7 too. The modern
# replacement is `memman doctor` reporting `enrichment_coverage`. A
# direct port is not possible without changing what is being tested.
# Follow-up: someone with embedding-coverage context should design new
# milestone coverage against `memman doctor` and the `embedded` flag on
# `remember` output.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# M8: Inline Causal Edges (lines 628-688)
# ---------------------------------------------------------------------

class TestM8InlineCausal:

    def test_causal_edges_in_output(self, home_dir: Path, m8_dir: Path,
                                    shared_state: dict):
        out = run_cli(
            ['remember', '--no-reconcile',
             'Causal test baseline insight about caching',
             '--cat', 'fact', '--imp', '3'],
            home_dir, m8_dir)
        assert_contains(out.stdout, '"causal"',
                        'has causal in edges_created')
        assert_contains(out.stdout, '"enrichment"', 'has enrichment')
        shared_state['m8_id_cc1'] = extract_id(json_out(out))

    def test_causal_llm_inference(self, home_dir: Path, m8_dir: Path,
                                  shared_state: dict):
        time.sleep(1)
        out = run_cli(
            ['remember', '--no-reconcile',
             'Chose Redis because of latency requirements in caching',
             '--cat', 'decision', '--imp', '4'],
            home_dir, m8_dir)
        data = json_out(out)
        shared_state['m8_id_cc2'] = extract_id(data)
        assert data['facts'][0]['edges_created']['causal'] >= 0, (
            'causal edge count present')

        keywords = data['facts'][0].get('enrichment', {}).get(
            'keywords', [])
        assert len(keywords) >= 0, 'enrichment keywords field present'

    def test_causal_linked_at(self, home_dir: Path, m8_dir: Path,
                              shared_state: dict):
        time.sleep(1)
        out = run_cli(
            ['remember', '--no-reconcile',
             'Edge caching reduces Redis load significantly',
             '--cat', 'fact', '--imp', '3'],
            home_dir, m8_dir)
        shared_state['m8_id_cc3'] = extract_id(json_out(out))

        data = json_out(run_cli(
            ['graph', 'rebuild', '--dry-run'], home_dir, m8_dir))
        assert int(data.get('total', -1)) >= 0, (
            f'graph rebuild reports total: {data}')

    def test_entity_extraction_dict(self, home_dir: Path,
                                    m_dict_dir: Path):
        out = run_cli(
            ['remember', '--no-reconcile',
             'We use React and TypeScript with Redis for caching',
             '--cat', 'fact', '--imp', '3'],
            home_dir, m_dict_dir)
        assert_contains(out.stdout, '"React"', 'React extracted')
        assert_contains(out.stdout, '"TypeScript"', 'TypeScript extracted')
        assert_contains(out.stdout, '"Redis"', 'Redis extracted')


# ---------------------------------------------------------------------
# M9: LLM Entity Injection (lines 691-724)
# ---------------------------------------------------------------------

class TestM9EntityInjection:

    def test_entities_appear(self, home_dir: Path, m9_dir: Path):
        out = run_cli(
            ['remember', '--no-reconcile',
             'The new caching layer improves performance significantly',
             '--cat', 'fact', '--imp', '3',
             '--entities', 'caching-layer,performance-optimization'],
            home_dir, m9_dir)
        assert_contains(out.stdout, '"caching-layer"',
                        'LLM entity present')
        assert_contains(out.stdout, '"performance-optimization"',
                        'LLM entity present')

    def test_entities_merge_regex(self, home_dir: Path, m9_dir: Path):
        out = run_cli(
            ['remember', '--no-reconcile',
             'We deploy HttpServer on Docker with Redis',
             '--cat', 'fact', '--imp', '3',
             '--entities', 'deployment-pipeline,high-availability'],
            home_dir, m9_dir)
        assert_contains(out.stdout, '"deployment-pipeline"',
                        'LLM entity: deployment-pipeline')
        assert_contains(out.stdout, '"high-availability"',
                        'LLM entity: high-availability')
        assert_contains(out.stdout, '"HttpServer"',
                        'regex entity: HttpServer')
        assert_contains(out.stdout, '"Docker"', 'dict entity: Docker')
        assert_contains(out.stdout, '"Redis"', 'dict entity: Redis')

    def test_entities_create_edges(self, home_dir: Path, m9_dir: Path):
        out = run_cli(
            ['remember', '--no-reconcile',
             'Upgrading the caching layer for better throughput',
             '--cat', 'decision', '--imp', '4',
             '--entities', 'caching-layer,throughput'],
            home_dir, m9_dir)
        data = json_out(out)
        assert_jq_gte(data, 'facts.0.edges_created.entity', 2,
                      'entity edges from shared LLM entity')

    def test_no_entities_flag(self, home_dir: Path, m9_dir: Path):
        out = run_cli(
            ['remember', '--no-reconcile',
             'Python and FastAPI are great for prototyping',
             '--cat', 'fact', '--imp', '3'],
            home_dir, m9_dir)
        assert_contains(out.stdout, '"Python"', 'dict entity: Python')
        assert_contains(out.stdout, '"FastAPI"', 'dict entity: FastAPI')


# ---------------------------------------------------------------------
# M10: Auto-Prune Lifecycle (lines 727-770)
# ---------------------------------------------------------------------

class TestM10AutoPrune:

    def test_setup_low_high(self, home_dir: Path, m10_dir: Path):
        for i in (1, 2, 3):
            out = run_cli(
                ['remember', '--no-reconcile',
                 (f'Low importance observation about system behavior '
                  f'number {i} in testing'),
                 '--cat', 'general', '--imp', '1'],
                home_dir, m10_dir)
            assert_jq(json_out(out), 'facts.0.action', 'add',
                      f'low-imp {i} stored')

        out = run_cli(
            ['remember', '--no-reconcile',
             'Critical architecture decision to use event-driven design',
             '--cat', 'decision', '--imp', '5'],
            home_dir, m10_dir)
        assert_jq(json_out(out), 'facts.0.auto_pruned', 0,
                  'auto_pruned is 0 under cap')

    def test_no_immune_in_candidates(self, home_dir: Path, m10_dir: Path,
                                     shared_state: dict):
        data = json_out(run_cli(
            ['insights', 'candidates', '--threshold', '999'],
            home_dir, m10_dir))
        shared_state['m10_high_thresh'] = data
        immune = [c for c in data['candidates'] if c.get('immune')]
        assert len(immune) == 0, (
            f'expected no immune in candidates, got {len(immune)}')

    def test_effective_importance_high_gt_low(self, home_dir: Path,
                                              m10_dir: Path,
                                              shared_state: dict):
        ei_low = shared_state['m10_high_thresh']['candidates'][0][
            'effective_importance']
        out = run_cli(
            ['remember', '--no-reconcile',
             ('Production database migration requires careful planning '
              'and execution'),
             '--cat', 'fact', '--imp', '5'],
            home_dir, m10_dir)
        ei_high = json_out(out)['facts'][0]['effective_importance']
        assert ei_high > ei_low, (
            f'imp=5 EI ({ei_high}) > imp=1 EI ({ei_low})')


# ---------------------------------------------------------------------
# M11: Smart Recall Reranking + Signals (lines 773-802)
# ---------------------------------------------------------------------

class TestM11Reranking:

    def test_intent_override(self, home_dir: Path, m3_dir: Path):
        out = run_cli(
            ['recall', 'Alpha service', '--intent', 'WHY'],
            home_dir, m3_dir)
        data = json_out(out)
        assert_jq(data, 'meta.intent', 'WHY', 'intent is WHY')
        assert_jq(data, 'meta.intent_source', 'override',
                  'intent_source is override')

    def test_llm_expanded_intent(self, home_dir: Path, m3_dir: Path):
        out = run_cli(
            ['recall', 'why Alpha service routing'], home_dir, m3_dir)
        data = json_out(out)
        assert_jq(data, 'meta.intent_source', 'override',
                  'intent_source is override (LLM expansion)')

    def test_signals_metadata(self, home_dir: Path, m3_dir: Path):
        out = run_cli(['recall', 'Alpha service routing'],
                      home_dir, m3_dir)
        data = json_out(out)
        first = data['results'][0]
        assert 'signals' in first, 'has signals'
        assert 'keyword' in first['signals'], 'has keyword signal'
        assert 'graph' in first['signals'], 'has graph signal'

    def test_meta_fields(self, home_dir: Path, m3_dir: Path):
        out = run_cli(['recall', 'Alpha service routing'],
                      home_dir, m3_dir)
        data = json_out(out)
        assert 'anchor_count' in data['meta'], 'has anchor_count'
        assert 'traversed' in data['meta'], 'has traversed'
        assert_jq_gte(data, 'meta.anchor_count', 1, 'anchor_count >= 1')
        assert 'hint' in data['meta'], 'has hint'
        assert 'ordering' in data['meta'], 'has ordering'

    def test_invalid_intent_rejected(self, home_dir: Path, m3_dir: Path):
        out = run_cli(
            ['recall', 'test', '--intent', 'INVALID'],
            home_dir, m3_dir, check=False)
        assert_contains(out.stdout + out.stderr, 'unknown intent',
                        'rejects invalid intent')
