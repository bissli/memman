"""Tests for mnemon.cli — Click CLI commands via CliRunner.

All tests use real Haiku LLM and Voyage embedding APIs.
Requires ANTHROPIC_API_KEY and VOYAGE_API_KEY in environment.
"""

import json
import pathlib

import pytest
from click.testing import CliRunner
from mnemon.cli import cli


@pytest.fixture
def runner(tmp_path):
    """CliRunner with --data-dir pointing to temp directory."""
    r = CliRunner()
    data_dir = str(tmp_path / 'mnemon_data')
    pathlib.Path(data_dir).mkdir(exist_ok=True, parents=True)
    return r, data_dir


def invoke(runner_tuple, args):
    """Helper to invoke CLI with data-dir."""
    r, data_dir = runner_tuple
    return r.invoke(cli, ['--data-dir', data_dir] + args)


def parse_remember(result):
    """Parse remember output, returning first fact dict."""
    raw = json.loads(result.output)
    if 'facts' in raw and raw['facts']:
        fact = dict(raw['facts'][0])
        fact['_raw'] = raw
        return fact
    return raw


def test_remember_basic(runner):
    """Store a basic insight."""
    result = invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    assert result.exit_code == 0
    data = parse_remember(result)
    assert data['action'] in {'add', 'added', 'update', 'updated'}
    assert 'sqlite' in data['content'].lower()


def test_remember_with_flags(runner):
    """Store with category, importance, tags."""
    result = invoke(runner, [
        'remember',
        'Chose Docker for container orchestration in production',
        '--no-reconcile',
        '--cat', 'decision', '--imp', '4',
        '--tags', 'docker,deployment'])
    assert result.exit_code == 0
    data = parse_remember(result)
    assert 'id' in data

    result = invoke(runner, ['search', 'Docker container'])
    hits = json.loads(result.output)
    match = [h for h in hits if h['id'] == data['id']][0]
    assert match['category'] == 'decision'
    assert match['importance'] == 4


def test_remember_invalid_category(runner):
    """Invalid category is rejected."""
    result = invoke(runner, [
        'remember', 'Go uses SQLite for storage', '--cat', 'bogus'])
    assert result.exit_code != 0


def test_remember_invalid_importance(runner):
    """Importance outside 1-5 is rejected."""
    result = invoke(runner, [
        'remember', 'Go uses SQLite for storage', '--imp', '0'])
    assert result.exit_code != 0


def test_recall_basic(runner):
    """Recall after remembering."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    result = invoke(runner, ['recall', 'Go SQLite storage'])
    assert result.exit_code == 0


def test_recall_does_not_call_link_pending(runner, monkeypatch):
    """Recall path must not call link_pending (performance regression guard)."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])

    from unittest.mock import patch
    with patch('mnemon.graph.engine.link_pending',
               side_effect=AssertionError('link_pending called')) as mock_lp:
        result = invoke(runner, ['recall', 'Go SQLite storage'])
        assert result.exit_code == 0
        mock_lp.assert_not_called()


def test_remember_does_not_link_old_pending_insights(runner, monkeypatch):
    """Remember does inline enrichment, never calls link_pending."""
    invoke(runner, [
        'remember', 'Redis cache eviction uses LRU algorithm',
        '--no-reconcile'])

    from unittest.mock import patch
    with patch('mnemon.graph.engine.link_pending',
               side_effect=AssertionError(
                   'link_pending called from remember')) as mock_lp:
        result = invoke(runner, [
            'remember',
            'PostgreSQL MVCC provides snapshot isolation',
            '--no-reconcile'])
        assert result.exit_code == 0
        mock_lp.assert_not_called()


def test_recall_basic_mode(runner):
    """Basic recall returns array."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    result = invoke(runner, ['recall', 'Go SQLite', '--basic'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)


def test_search_basic(runner):
    """Search returns scored results."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    result = invoke(runner, ['search', 'Go SQLite'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)


def test_forget_basic(runner):
    """Forget an insight by ID."""
    result = invoke(runner, [
        'remember',
        'Redis cache eviction policy uses LRU by default',
        '--no-reconcile'])
    data = parse_remember(result)
    iid = data['id']
    result = invoke(runner, ['forget', iid])
    assert result.exit_code == 0
    fdata = json.loads(result.output)
    assert fdata['status'] == 'deleted'


def test_forget_writes_oplog(runner):
    """Forget command writes an oplog entry atomically."""
    result = invoke(runner, [
        'remember',
        'PostgreSQL uses MVCC for transaction isolation',
        '--no-reconcile'])
    data = parse_remember(result)
    iid = data['id']
    invoke(runner, ['forget', iid])
    result = invoke(runner, ['log', '--stats'])
    assert result.exit_code == 0
    log_data = json.loads(result.output)
    assert 'forget' in log_data['operation_counts']


def test_forget_nonexistent_fails(runner):
    """Forget with nonexistent ID returns error."""
    result = invoke(runner, ['forget', 'nonexistent-id-12345'])
    assert result.exit_code != 0


def test_store_list(runner):
    """Store list shows stores."""
    result = invoke(runner, ['store', 'list'])
    assert result.exit_code == 0


def test_store_create(runner):
    """Create a new store."""
    result = invoke(runner, ['store', 'create', 'test-store'])
    assert result.exit_code == 0
    assert 'Created' in result.output


def test_store_create_duplicate(runner):
    """Duplicate store name is rejected."""
    invoke(runner, ['store', 'create', 'dup'])
    result = invoke(runner, ['store', 'create', 'dup'])
    assert result.exit_code != 0


def test_store_set(runner):
    """Set active store."""
    invoke(runner, ['store', 'create', 'work'])
    result = invoke(runner, ['store', 'set', 'work'])
    assert result.exit_code == 0
    assert 'Active store' in result.output


def test_store_remove(runner):
    """Remove a non-active store."""
    invoke(runner, ['store', 'create', 'temp'])
    result = invoke(runner, ['store', 'remove', 'temp'])
    assert result.exit_code == 0
    assert 'Removed' in result.output


def test_status_basic(runner):
    """Status returns JSON."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    result = invoke(runner, ['status'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert 'total_insights' in data


def test_log_basic(runner):
    """Log shows recent operations."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    result = invoke(runner, ['log'])
    assert result.exit_code == 0


def test_gc_suggest(runner):
    """GC suggest mode returns JSON."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    result = invoke(runner, ['gc'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert 'total_insights' in data


def test_viz_dot(runner):
    """Viz dot format renders."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    result = invoke(runner, ['viz'])
    assert result.exit_code == 0
    assert 'digraph' in result.output


def test_js_str_escapes_script_close():
    """_js_str prevents XSS via </script> in content."""
    from mnemon.cli import _js_str
    result = _js_str('</script><script>alert(1)</script>')
    assert '</script>' not in result
    assert '<\\/' in result


def test_js_str_normal_string():
    """_js_str handles regular strings correctly."""
    from mnemon.cli import _js_str
    result = _js_str('hello world')
    assert result == '"hello world"'


def test_remember_quality_warnings(runner):
    """Content with 2+ quality warnings is rejected."""
    result = invoke(runner, [
        'remember', 'i-0c220c2402a5245bc deployed via Terraform',
        '--no-reconcile'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['action'] == 'rejected'
    assert 'AWS instance ID' in data['quality_warnings']
    assert 'deployment receipt' in data['quality_warnings']


def test_remember_no_quality_warnings(runner):
    """Durable content produces empty quality_warnings."""
    result = invoke(runner, [
        'remember',
        'SQLite chosen for single-node simplicity and embedded operation',
        '--no-reconcile'])
    assert result.exit_code == 0
    raw = json.loads(result.output)
    assert raw['quality_warnings'] == []


def test_gc_review(runner):
    """GC --review flags transient content with 1 warning (stored)."""
    invoke(runner, [
        'remember',
        'Production outage traced to instance i-0c220c2402a5245bc running out of memory causing cascading failure',
        '--no-reconcile'])
    invoke(runner, [
        'remember',
        'SQLite chosen for simplicity and embedded operation',
        '--no-reconcile', '--imp', '5'])
    result = invoke(runner, ['gc', '--review'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['total_flagged'] >= 1
    flagged = [r['content'] for r in data['review_results']]
    assert any('i-0c220c2402a5245bc' in c.lower() for c in flagged)


def test_remember_quality_rejection(runner):
    """2+ warnings rejected, 1 warning stored, quality-reject in oplog."""
    result = invoke(runner, [
        'remember', 'Stack deployed via Terraform. 32 resources total.',
        '--no-reconcile'])
    data = json.loads(result.output)
    assert data['action'] == 'rejected'
    assert len(data['quality_warnings']) >= 2

    result = invoke(runner, [
        'remember', 'Production outage traced to instance i-0c220c2402a5245bc running out of memory causing cascading failure',
        '--no-reconcile'])
    data = parse_remember(result)
    assert data['action'] == 'add'
    raw = json.loads(result.output)
    assert len(raw['quality_warnings']) == 1

    result = invoke(runner, ['log', '--limit', '10'])
    assert 'quality-reject' in result.output


def test_replace_quality_rejection(runner):
    """Replace path also rejects content with 2+ quality warnings."""
    result = invoke(runner, [
        'remember',
        'Kafka chosen for event streaming due to partition tolerance',
        '--no-reconcile'])
    old_id = parse_remember(result)['id']

    result = invoke(runner, [
        'replace', old_id,
        'Stack deployed via Terraform. 32 resources total.'])
    data = json.loads(result.output)
    assert data['action'] == 'rejected'
    assert len(data['quality_warnings']) >= 2


def test_gc_review_empty(runner):
    """GC --review with clean store returns zero flagged."""
    invoke(runner, [
        'remember',
        'SQLite chosen for simplicity and embedded operation',
        '--no-reconcile'])
    result = invoke(runner, ['gc', '--review'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['total_flagged'] == 0


def test_replace_basic(runner):
    """Replace an insight, verify old soft-deleted, new exists."""
    result = invoke(runner, [
        'remember',
        'Redis cache configured with 512MB memory limit',
        '--no-reconcile', '--cat', 'fact', '--imp', '3'])
    old_id = parse_remember(result)['id']

    result = invoke(runner, [
        'replace', old_id,
        'Redis cache configured with 1GB memory limit for production'])
    assert result.exit_code == 0
    data = parse_remember(result)
    assert data['action'] == 'replace'
    assert data['replaced_id'] == old_id
    assert 'redis' in data['content'].lower()


def test_replace_inherits_metadata(runner):
    """Replace without flags inherits cat/imp/tags from original."""
    result = invoke(runner, [
        'remember',
        'Chose PostgreSQL over MySQL for JSONB support',
        '--no-reconcile',
        '--cat', 'decision', '--imp', '5',
        '--tags', 'arch,design'])
    old_id = parse_remember(result)['id']

    result = invoke(runner, [
        'replace', old_id,
        'Chose PostgreSQL over MySQL for JSONB and CTE support'])
    assert result.exit_code == 0
    data = parse_remember(result)
    assert 'id' in data

    result = invoke(runner, ['search', 'PostgreSQL JSONB'])
    hits = json.loads(result.output)
    match = [h for h in hits if h['id'] == data['id']][0]
    assert match['category'] == 'decision'
    assert match['importance'] == 5
    assert 'arch' in match['tags']
    assert 'design' in match['tags']


def test_replace_overrides_metadata(runner):
    """Replace with explicit flags uses new values."""
    result = invoke(runner, [
        'remember',
        'Nginx configured as reverse proxy for API gateway',
        '--no-reconcile', '--cat', 'fact', '--imp', '2'])
    old_id = parse_remember(result)['id']

    result = invoke(runner, [
        'replace', old_id,
        'Switched from Nginx to Envoy for service mesh integration',
        '--cat', 'decision', '--imp', '5'])
    assert result.exit_code == 0
    data = parse_remember(result)
    assert 'id' in data

    result = invoke(runner, ['search', 'Envoy service mesh'])
    hits = json.loads(result.output)
    match = [h for h in hits if h['id'] == data['id']][0]
    assert match['category'] == 'decision'
    assert match['importance'] == 5


def test_replace_preserves_access_count(runner):
    """Replace carries over access_count from original."""
    result = invoke(runner, [
        'remember',
        'Terraform modules organized by environment and region',
        '--no-reconcile'])
    old_id = parse_remember(result)['id']
    invoke(runner, ['recall', 'Terraform modules', '--basic'])
    invoke(runner, ['recall', 'Terraform modules', '--basic'])

    result = invoke(runner, [
        'replace', old_id,
        'Terraform modules organized by service and environment'])
    assert result.exit_code == 0
    new_id = parse_remember(result)['id']

    result = invoke(runner, ['recall', 'Terraform modules', '--basic'])
    hits = json.loads(result.output)
    match = [h for h in hits if h['id'] == new_id]
    assert match
    assert match[0]['access_count'] >= 2


def test_replace_nonexistent_id(runner):
    """Replace a nonexistent ID produces error."""
    result = invoke(runner, [
        'replace', 'nonexistent-id',
        'Redis configured for cluster mode replication'])
    assert result.exit_code != 0
    assert 'not found' in result.output


def test_replace_already_deleted(runner):
    """Replace an already-deleted insight produces error."""
    result = invoke(runner, [
        'remember',
        'Kafka consumer group rebalance strategy uses cooperative',
        '--no-reconcile'])
    old_id = parse_remember(result)['id']
    invoke(runner, ['forget', old_id])

    result = invoke(runner, [
        'replace', old_id,
        'Kafka consumer group rebalance uses eager strategy'])
    assert result.exit_code != 0
    assert 'not found' in result.output


def test_replace_oplog_entries(runner):
    """Replace logs both replace and remember ops."""
    result = invoke(runner, [
        'remember',
        'Prometheus alerting rules configured for SLO monitoring',
        '--no-reconcile'])
    old_id = parse_remember(result)['id']

    result = invoke(runner, [
        'replace', old_id,
        'Prometheus alerting rules with Grafana dashboards for SLO'])
    parse_remember(result)

    result = invoke(runner, ['log', '--limit', '10'])
    assert result.exit_code == 0
    assert 'replace' in result.output
    assert 'remember' in result.output


def test_link_creates_both_directions(runner):
    """Link creates edges in both directions atomically."""
    r1 = invoke(runner, [
        'remember', 'FastAPI chosen for async API development',
        '--no-reconcile'])
    id1 = parse_remember(r1)['id']
    r2 = invoke(runner, [
        'remember', 'Uvicorn configured as ASGI server for FastAPI',
        '--no-reconcile'])
    id2 = parse_remember(r2)['id']

    result = invoke(runner, [
        'link', id1, id2, '--type', 'causal'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['status'] == 'linked'

    fwd = invoke(runner, ['related', id1, '--edge', 'causal'])
    assert fwd.exit_code == 0
    fwd_data = json.loads(fwd.output)
    assert any(e['id'] == id2 for e in fwd_data)

    rev = invoke(runner, ['related', id2, '--edge', 'causal'])
    assert rev.exit_code == 0
    rev_data = json.loads(rev.output)
    assert any(e['id'] == id1 for e in rev_data)


def test_link_meta_non_dict_fails(runner):
    """Non-dict JSON metadata is rejected."""
    r1 = invoke(runner, [
        'remember', 'Elasticsearch configured for full-text search',
        '--no-reconcile'])
    id1 = parse_remember(r1)['id']
    r2 = invoke(runner, [
        'remember', 'Kibana dashboards visualize Elasticsearch data',
        '--no-reconcile'])
    id2 = parse_remember(r2)['id']

    result = invoke(runner, [
        'link', id1, id2, '--type', 'semantic',
        '--meta', '[1, 2]'])
    assert result.exit_code != 0
    assert 'object' in result.output.lower() or 'dict' in result.output.lower()


def test_link_self_edge_rejected(runner):
    """Linking an insight to itself is rejected."""
    r1 = invoke(runner, [
        'remember',
        'GraphQL schema stitching combines microservice APIs',
        '--no-reconcile'])
    id1 = parse_remember(r1)['id']

    result = invoke(runner, ['link', id1, id1, '--type', 'semantic'])
    assert result.exit_code != 0
    assert 'itself' in result.output.lower()


def test_recall_source_filter_smart(runner):
    """Smart recall respects --source filter."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile', '--source', 'agent'])
    invoke(runner, [
        'remember', 'Python uses PostgreSQL for web application storage',
        '--no-reconcile', '--source', 'human'])

    result = invoke(runner, [
        'recall', 'database storage', '--source', 'agent'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for r in data['results']:
        assert r['insight']['source'] == 'agent'


def test_replace_creates_background_edges(runner):
    """Replace passes store context so background edges are created."""
    r1 = invoke(runner, [
        'remember',
        'Celery task queue configured for async job processing',
        '--no-reconcile'])
    orig_id = parse_remember(r1)['id']

    r2 = invoke(runner, [
        'replace', orig_id,
        'Celery with Redis broker for distributed task processing'])
    assert r2.exit_code == 0
    new_id = parse_remember(r2)['id']

    result = invoke(runner, ['related', new_id])
    assert result.exit_code == 0


def test_link_warns_when_lower_weight(runner):
    """Link output includes warning when requested weight < existing."""
    r1 = invoke(runner, [
        'remember',
        'Consul service discovery enables dynamic routing',
        '--no-reconcile'])
    id1 = parse_remember(r1)['id']
    r2 = invoke(runner, [
        'remember',
        'Vault secrets management integrates with Consul',
        '--no-reconcile'])
    id2 = parse_remember(r2)['id']

    invoke(runner, ['link', id1, id2, '--weight', '0.9'])
    result = invoke(runner, ['link', id1, id2, '--weight', '0.3'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert 'warning' in data
    assert '0.9' in data['warning']


class TestSingleTierEnrichment:
    """Remember runs enrichment + causal inline via ThreadPoolExecutor."""

    def test_output_has_enrichment_dict(self, runner):
        """Remember output includes enrichment metadata."""
        result = invoke(runner, [
            'remember',
            'Redis cache configured with LRU eviction policy',
            '--no-reconcile'])
        assert result.exit_code == 0
        data = parse_remember(result)
        assert 'enrichment' in data
        enr = data['enrichment']
        assert 'keywords' in enr
        assert 'summary' in enr
        assert 'entities' in enr
        assert 'semantic_facts' in enr

    def test_output_has_causal_count(self, runner):
        """edges_created includes causal count."""
        result = invoke(runner, [
            'remember',
            'PostgreSQL chosen for JSONB support in API layer',
            '--no-reconcile'])
        assert result.exit_code == 0
        data = parse_remember(result)
        assert 'causal' in data['edges_created']
        assert isinstance(data['edges_created']['causal'], int)

    def test_no_link_pending_in_output(self, runner):
        """Output no longer includes link_pending field."""
        result = invoke(runner, [
            'remember',
            'Docker containers orchestrated via Kubernetes',
            '--no-reconcile'])
        assert result.exit_code == 0
        raw = json.loads(result.output)
        assert 'link_pending' not in raw

    def test_no_causal_candidates_in_output(self, runner):
        """Output no longer includes causal_candidates field."""
        result = invoke(runner, [
            'remember',
            'Terraform modules organized by service boundaries',
            '--no-reconcile'])
        assert result.exit_code == 0
        data = parse_remember(result)
        assert 'causal_candidates' not in data

    def test_linked_at_stamped_after_remember(self, runner):
        """linked_at is non-NULL after remember returns."""
        from mnemon.store.db import open_read_only

        result = invoke(runner, [
            'remember',
            'Consul service mesh enables secure service communication',
            '--no-reconcile'])
        assert result.exit_code == 0
        data = parse_remember(result)
        iid = data['id']

        _, data_dir = runner
        ro = open_read_only(data_dir + '/data/default')
        row = ro._conn.execute(
            'SELECT linked_at FROM insights WHERE id = ?',
            (iid,)).fetchone()
        ro.close()
        assert row is not None
        assert row[0] is not None

    def test_graph_link_zero_pending_after_remember(self, runner):
        """Graph link finds nothing pending after single-tier remember."""
        invoke(runner, [
            'remember',
            'Kafka event streaming configured for microservices',
            '--no-reconcile'])
        result = invoke(runner, ['graph', 'link'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['processed'] == 0

    def test_enriched_at_stamped_after_remember(self, runner):
        """enriched_at is non-NULL after remember returns."""
        from mnemon.store.db import open_read_only

        result = invoke(runner, [
            'remember',
            'Elasticsearch full-text search with custom analyzers',
            '--no-reconcile'])
        assert result.exit_code == 0
        data = parse_remember(result)
        iid = data['id']

        _, data_dir = runner
        ro = open_read_only(data_dir + '/data/default')
        row = ro._conn.execute(
            'SELECT enriched_at FROM insights WHERE id = ?',
            (iid,)).fetchone()
        ro.close()
        assert row is not None
        assert row[0] is not None
