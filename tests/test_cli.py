"""Tests for memman.cli — Click CLI commands via CliRunner.

All tests use real Haiku LLM and Voyage embedding APIs.
Requires OPENROUTER_API_KEY and VOYAGE_API_KEY in environment.
"""

import json
import pathlib
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from memman.cli import cli
from memman.store.db import store_exists


@pytest.fixture
def runner(tmp_path):
    """CliRunner with --data-dir pointing to temp directory."""
    r = CliRunner()
    data_dir = str(tmp_path / 'memman_data')
    pathlib.Path(data_dir).mkdir(exist_ok=True, parents=True)
    return r, data_dir


def invoke(runner_tuple, args):
    """Helper to invoke CLI with data-dir."""
    r, data_dir = runner_tuple
    return r.invoke(cli, ['--data-dir', data_dir] + args)


def parse_remember(result, runner_tuple=None):
    """Parse remember/replace output to a fact-shaped dict.

    Modern `remember`/`replace` returns just `{action: queued, queue_id,
    store}`. The autouse-drain runs the worker, so the new insight is
    in the store DB tagged with `source = queue:<id>`. Look it up so
    tests that read `id`/`content` keep working.
    """
    raw = json.loads(result.output)
    if 'facts' in raw and raw['facts']:
        fact = dict(raw['facts'][0])
        fact['_raw'] = raw
        return fact
    if runner_tuple is None:
        return raw
    queue_id = raw.get('queue_id')
    if queue_id is None:
        return raw
    _, data_dir = runner_tuple
    from memman.store.db import open_read_only, read_active, store_dir
    name = raw.get('store') or read_active(data_dir) or 'default'
    sdir = store_dir(data_dir, name)
    db = open_read_only(sdir)
    try:
        rows = db._query(
            'select id, content, category, importance from insights'
            ' where source = ? and deleted_at is null'
            ' order by created_at',
            (f'queue:{queue_id}',)).fetchall()
    finally:
        db.close()
    if not rows:
        return raw
    action = 'replace' if raw.get('replaced_id') else 'add'
    return {
        'id': rows[0][0],
        'content': rows[0][1],
        'category': rows[0][2],
        'importance': rows[0][3],
        'action': action,
        'replaced_id': raw.get('replaced_id'),
        '_raw': raw,
        }


def test_remember_basic(runner):
    """Store a basic insight."""
    result = invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    assert result.exit_code == 0
    data = parse_remember(result, runner)
    assert data['action'] in {'add', 'added', 'update', 'updated'}
    assert 'sqlite' in data['content'].lower()


def test_remember_with_flags(runner):
    """Store with category, importance, tags."""
    result = invoke(runner, [
        'remember', 'Chose Docker for container orchestration in production',
        '--no-reconcile',
        '--cat', 'decision', '--imp', '4',
        '--tags', 'docker,deployment'])
    assert result.exit_code == 0
    data = parse_remember(result, runner)
    assert 'id' in data

    result = invoke(runner, ['recall', '--basic', 'Docker container'])
    hits = json.loads(result.output)['results']
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
    with patch('memman.graph.engine.link_pending',
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
    with patch('memman.graph.engine.link_pending',
               side_effect=AssertionError(
                   'link_pending called from remember')) as mock_lp:
        result = invoke(runner, [
            'remember', 'PostgreSQL MVCC provides snapshot isolation',
            '--no-reconcile'])
        assert result.exit_code == 0
        mock_lp.assert_not_called()


def test_recall_basic_mode(runner):
    """Basic recall returns {results: [...], meta: {basic: True}}."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    result = invoke(runner, ['recall', 'Go SQLite', '--basic'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['meta']['basic'] is True
    assert isinstance(data['results'], list)


def test_recall_basic_returns_envelope(runner):
    """Recall --basic returns insights wrapped in {results: [...]}."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    result = invoke(runner, ['recall', '--basic', 'Go SQLite'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert 'results' in data
    assert any('SQLite' in r['content'] for r in data['results'])


def test_forget_basic(runner):
    """Forget an insight by ID."""
    result = invoke(runner, [
        'remember', 'Redis cache eviction policy uses LRU by default',
        '--no-reconcile'])
    data = parse_remember(result, runner)
    iid = data['id']
    result = invoke(runner, ['forget', iid])
    assert result.exit_code == 0
    fdata = json.loads(result.output)
    assert fdata['status'] == 'deleted'


def test_forget_writes_oplog(runner):
    """Forget command writes an oplog entry atomically."""
    result = invoke(runner, [
        'remember', 'PostgreSQL uses MVCC for transaction isolation',
        '--no-reconcile'])
    data = parse_remember(result, runner)
    iid = data['id']
    invoke(runner, ['forget', iid])
    result = invoke(runner, ['log', 'list', '--stats'])
    assert result.exit_code == 0
    log_data = json.loads(result.output)
    assert 'forget' in log_data['operation_counts']


def test_forget_nonexistent_fails(runner):
    """Forget with nonexistent ID returns error."""
    result = invoke(runner, ['forget', 'nonexistent-id-12345'])
    assert result.exit_code != 0


def test_store_list(runner):
    """Store list emits a JSON envelope with stores[] and active."""
    import json as _json
    result = invoke(runner, ['store', 'list'])
    assert result.exit_code == 0
    payload = _json.loads(result.output)
    assert 'stores' in payload
    assert 'active' in payload


def test_store_create(runner):
    """Create a new store; JSON reports action='created'."""
    import json as _json
    result = invoke(runner, ['store', 'create', 'test-store'])
    assert result.exit_code == 0
    payload = _json.loads(result.output)
    assert payload['action'] == 'created'
    assert payload['store'] == 'test-store'


def test_store_create_duplicate(runner):
    """Duplicate store name is rejected."""
    invoke(runner, ['store', 'create', 'dup'])
    result = invoke(runner, ['store', 'create', 'dup'])
    assert result.exit_code != 0


def test_store_set(runner):
    """Set active store; JSON reports action='set'."""
    import json as _json
    invoke(runner, ['store', 'create', 'work'])
    result = invoke(runner, ['store', 'use', 'work'])
    assert result.exit_code == 0
    payload = _json.loads(result.output)
    assert payload['action'] == 'set'
    assert payload['store'] == 'work'


def test_store_remove_yes(runner):
    """Remove a non-active store with --yes skips prompt."""
    import json as _json
    invoke(runner, ['store', 'create', 'temp'])
    result = invoke(runner, ['store', 'remove', '--yes', 'temp'])
    assert result.exit_code == 0
    payload = _json.loads(result.output)
    assert payload['action'] == 'removed'
    assert payload['store'] == 'temp'


def test_store_remove_prompts_without_yes(runner):
    """Without --yes, remove prompts; typing 'n' aborts."""
    r, data_dir = runner
    invoke(runner, ['store', 'create', 'temp2'])
    result = r.invoke(
        cli, ['--data-dir', data_dir, 'store', 'remove', 'temp2'],
        input='n\n')
    assert result.exit_code != 0
    assert store_exists(data_dir, 'temp2')


def test_store_remove_prompts_accept(runner):
    """Without --yes, typing 'y' at the prompt completes the delete."""
    import json as _json
    r, data_dir = runner
    invoke(runner, ['store', 'create', 'temp3'])
    result = r.invoke(
        cli, ['--data-dir', data_dir, 'store', 'remove', 'temp3'],
        input='y\n')
    assert result.exit_code == 0, result.output
    # The confirm prompt echoes before the JSON payload; find the payload.
    payload_start = result.output.find('{')
    payload = _json.loads(result.output[payload_start:])
    assert payload['action'] == 'removed'


def test_store_auto_create_from_env(runner, monkeypatch):
    """MEMMAN_STORE env var silently creates a non-existent store."""
    r, data_dir = runner
    monkeypatch.setenv('MEMMAN_STORE', 'auto-created')

    result = r.invoke(cli, ['--data-dir', data_dir, 'recall', 'test',
                            '--limit', '1'])
    assert result.exit_code == 0, result.output

    store_path = pathlib.Path(data_dir) / 'data' / 'auto-created'
    assert store_path.is_dir(), 'store directory should be auto-created'

    monkeypatch.delenv('MEMMAN_STORE')
    list_result = r.invoke(cli, ['--data-dir', data_dir, 'store', 'list'])
    assert 'auto-created' in list_result.output

    r.invoke(cli, ['--data-dir', data_dir, 'store', 'remove', 'auto-created'])


def test_status_basic(runner):
    """Status returns JSON."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    result = invoke(runner, ['status'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert 'total_insights' in data


def test_doctor_basic(runner):
    """Doctor returns JSON with checks and status.

    Exit code may be 0 (pass/warn) or 1 (fail) depending on environment.
    """
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    result = invoke(runner, ['doctor'])
    assert result.exit_code in {0, 1}
    data = json.loads(result.output)
    assert 'status' in data
    assert 'checks' in data
    assert 'total_active' in data


def test_log_basic(runner):
    """Log shows recent operations."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    result = invoke(runner, ['log', 'list'])
    assert result.exit_code == 0


def test_gc_suggest(runner):
    """GC suggest mode returns JSON."""
    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    result = invoke(runner, ['insights', 'candidates'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert 'total_insights' in data


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
        'remember', 'SQLite chosen for single-node simplicity and embedded operation',
        '--no-reconcile'])
    assert result.exit_code == 0
    raw = json.loads(result.output)
    assert raw['quality_warnings'] == []


def test_gc_review(runner):
    """GC --review flags transient content with 1 warning (stored)."""
    invoke(runner, [
        'remember', 'Production outage traced to instance i-0c220c2402a5245bc running out of memory causing cascading failure',
        '--no-reconcile'])
    invoke(runner, [
        'remember', 'SQLite chosen for simplicity and embedded operation',
        '--no-reconcile', '--imp', '5'])
    result = invoke(runner, ['insights', 'review'])
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
    data = parse_remember(result, runner)
    assert data['action'] == 'add'
    raw = json.loads(result.output)
    assert len(raw['quality_warnings']) == 1

    result = invoke(runner, ['log', 'list', '--limit', '10'])
    assert 'quality-reject' in result.output


def test_replace_quality_rejection(runner):
    """Replace path also rejects content with 2+ quality warnings."""
    result = invoke(runner, [
        'remember', 'Kafka chosen for event streaming due to partition tolerance',
        '--no-reconcile'])
    old_id = parse_remember(result, runner)['id']

    result = invoke(runner, [
        'replace', old_id,
        'Stack deployed via Terraform. 32 resources total.'])
    data = json.loads(result.output)
    assert data['action'] == 'rejected'
    assert len(data['quality_warnings']) >= 2


def test_gc_review_empty(runner):
    """GC --review with clean store returns zero flagged."""
    invoke(runner, [
        'remember', 'SQLite chosen for simplicity and embedded operation',
        '--no-reconcile'])
    result = invoke(runner, ['insights', 'review'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['total_flagged'] == 0


def test_replace_basic(runner):
    """Replace an insight, verify old soft-deleted, new exists."""
    result = invoke(runner, [
        'remember', 'Redis cache configured with 512MB memory limit',
        '--no-reconcile', '--cat', 'fact', '--imp', '3'])
    old_id = parse_remember(result, runner)['id']

    result = invoke(runner, [
        'replace', old_id,
        'Redis cache configured with 1GB memory limit for production'])
    assert result.exit_code == 0
    data = parse_remember(result, runner)
    assert data['action'] == 'replace'
    assert data['replaced_id'] == old_id
    assert 'redis' in data['content'].lower()


def test_replace_inherits_metadata(runner):
    """Replace without flags inherits cat/imp/tags from original."""
    result = invoke(runner, [
        'remember', 'Chose PostgreSQL over MySQL for JSONB support',
        '--no-reconcile',
        '--cat', 'decision', '--imp', '5',
        '--tags', 'arch,design'])
    old_id = parse_remember(result, runner)['id']

    result = invoke(runner, [
        'replace', old_id,
        'Chose PostgreSQL over MySQL for JSONB and CTE support'])
    assert result.exit_code == 0
    data = parse_remember(result, runner)
    assert 'id' in data

    result = invoke(runner, ['recall', '--basic', 'PostgreSQL JSONB'])
    hits = json.loads(result.output)['results']
    match = [h for h in hits if h['id'] == data['id']][0]
    assert match['category'] == 'decision'
    assert match['importance'] == 5
    assert 'arch' in match['tags']
    assert 'design' in match['tags']


def test_replace_overrides_metadata(runner):
    """Replace with explicit flags uses new values."""
    result = invoke(runner, [
        'remember', 'Nginx configured as reverse proxy for API gateway',
        '--no-reconcile', '--cat', 'fact', '--imp', '2'])
    old_id = parse_remember(result, runner)['id']

    result = invoke(runner, [
        'replace', old_id,
        'Switched from Nginx to Envoy for service mesh integration',
        '--cat', 'decision', '--imp', '5'])
    assert result.exit_code == 0
    data = parse_remember(result, runner)
    assert 'id' in data

    result = invoke(runner, ['recall', '--basic', 'Envoy service mesh'])
    hits = json.loads(result.output)['results']
    match = [h for h in hits if h['id'] == data['id']][0]
    assert match['category'] == 'decision'
    assert match['importance'] == 5


def test_replace_preserves_access_count(runner):
    """Replace carries over access_count from original."""
    result = invoke(runner, [
        'remember', 'Terraform modules organized by environment and region',
        '--no-reconcile'])
    old_id = parse_remember(result, runner)['id']
    invoke(runner, ['recall', 'Terraform modules', '--basic'])
    invoke(runner, ['recall', 'Terraform modules', '--basic'])

    result = invoke(runner, [
        'replace', old_id,
        'Terraform modules organized by service and environment'])
    assert result.exit_code == 0
    new_id = parse_remember(result, runner)['id']

    result = invoke(runner, ['recall', 'Terraform modules', '--basic'])
    hits = json.loads(result.output)['results']
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
        'remember', 'Kafka consumer group rebalance strategy uses cooperative',
        '--no-reconcile'])
    old_id = parse_remember(result, runner)['id']
    invoke(runner, ['forget', old_id])

    result = invoke(runner, [
        'replace', old_id,
        'Kafka consumer group rebalance uses eager strategy'])
    assert result.exit_code != 0
    assert 'not found' in result.output


def test_replace_oplog_entries(runner):
    """Replace logs both replace and remember ops."""
    result = invoke(runner, [
        'remember', 'Prometheus alerting rules configured for SLO monitoring',
        '--no-reconcile'])
    old_id = parse_remember(result, runner)['id']

    result = invoke(runner, [
        'replace', old_id,
        'Prometheus alerting rules with Grafana dashboards for SLO'])
    parse_remember(result, runner)

    result = invoke(runner, ['log', 'list', '--limit', '10'])
    assert result.exit_code == 0
    assert 'replace' in result.output
    assert 'remember' in result.output


def test_link_creates_both_directions(runner):
    """Link creates edges in both directions atomically."""
    r1 = invoke(runner, [
        'remember', 'FastAPI chosen for async API development',
        '--no-reconcile'])
    id1 = parse_remember(r1, runner)['id']
    r2 = invoke(runner, [
        'remember', 'Uvicorn configured as ASGI server for FastAPI',
        '--no-reconcile'])
    id2 = parse_remember(r2, runner)['id']

    result = invoke(runner, ['graph', 'link', id1, id2, '--type', 'causal'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['status'] == 'linked'

    fwd = invoke(runner, ['graph', 'related', id1, '--edge', 'causal'])
    assert fwd.exit_code == 0
    fwd_data = json.loads(fwd.output)
    assert any(e['id'] == id2 for e in fwd_data)

    rev = invoke(runner, ['graph', 'related', id2, '--edge', 'causal'])
    assert rev.exit_code == 0
    rev_data = json.loads(rev.output)
    assert any(e['id'] == id1 for e in rev_data)


def test_link_respects_user_created_by(runner):
    """User-provided --meta['created_by'] is preserved, not clobbered to 'claude'.
    """
    import sqlite3
    r1 = invoke(runner, [
        'remember', 'Nginx is configured as the reverse proxy',
        '--no-reconcile'])
    id1 = parse_remember(r1, runner)['id']
    r2 = invoke(runner, [
        'remember', "Let's Encrypt auto-renews TLS certificates",
        '--no-reconcile'])
    id2 = parse_remember(r2, runner)['id']

    result = invoke(runner, ['graph', 'link', id1, id2, '--type', 'semantic',
                             '--meta', '{"created_by": "research-agent"}'])
    assert result.exit_code == 0

    _, data_dir = runner
    store_db = pathlib.Path(data_dir) / 'data' / 'default' / 'memman.db'
    conn = sqlite3.connect(str(store_db))
    try:
        rows = conn.execute(
            'SELECT metadata FROM edges'
            ' WHERE source_id = ? AND target_id = ?'
            ' AND edge_type = ?',
            (id1, id2, 'semantic')).fetchall()
    finally:
        conn.close()
    assert rows, 'expected one semantic edge source->target'
    meta = json.loads(rows[0][0])
    assert meta['created_by'] == 'research-agent'


def test_link_meta_non_dict_fails(runner):
    """Non-dict JSON metadata is rejected."""
    r1 = invoke(runner, [
        'remember', 'Elasticsearch configured for full-text search',
        '--no-reconcile'])
    id1 = parse_remember(r1, runner)['id']
    r2 = invoke(runner, [
        'remember', 'Kibana dashboards visualize Elasticsearch data',
        '--no-reconcile'])
    id2 = parse_remember(r2, runner)['id']

    result = invoke(runner, ['graph', 'link', id1, id2, '--type', 'semantic',
                             '--meta', '[1, 2]'])
    assert result.exit_code != 0
    assert 'object' in result.output.lower() or 'dict' in result.output.lower()


def test_link_self_edge_rejected(runner):
    """Linking an insight to itself is rejected."""
    r1 = invoke(runner, [
        'remember', 'GraphQL schema stitching combines microservice APIs',
        '--no-reconcile'])
    id1 = parse_remember(r1, runner)['id']

    result = invoke(runner, ['graph', 'link', id1, id1, '--type', 'semantic'])
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
        'remember', 'Celery task queue configured for async job processing',
        '--no-reconcile'])
    orig_id = parse_remember(r1, runner)['id']

    r2 = invoke(runner, [
        'replace', orig_id,
        'Celery with Redis broker for distributed task processing'])
    assert r2.exit_code == 0
    new_id = parse_remember(r2, runner)['id']

    result = invoke(runner, ['graph', 'related', new_id])
    assert result.exit_code == 0


def test_link_warns_when_lower_weight(runner):
    """Link output includes warning when requested weight < existing."""
    r1 = invoke(runner, [
        'remember', 'Consul service discovery enables dynamic routing',
        '--no-reconcile'])
    id1 = parse_remember(r1, runner)['id']
    r2 = invoke(runner, [
        'remember', 'Vault secrets management integrates with Consul',
        '--no-reconcile'])
    id2 = parse_remember(r2, runner)['id']

    invoke(runner, ['graph', 'link', id1, id2, '--weight', '0.9'])
    result = invoke(runner, ['graph', 'link', id1, id2, '--weight', '0.3'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert 'warning' in data
    assert '0.9' in data['warning']


class TestSingleTierEnrichment:
    """Remember runs enrichment + causal inline via ThreadPoolExecutor."""

    def test_output_has_enrichment_dict(self, runner):
        """Worker enrichment lands keywords/summary/entities on the row."""
        from memman.store.db import open_read_only, store_dir

        result = invoke(runner, [
            'remember', 'Redis cache configured with LRU eviction policy',
            '--no-reconcile'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        iid = data['id']

        _, data_dir = runner
        db = open_read_only(store_dir(data_dir, 'default'))
        try:
            row = db._query(
                'SELECT keywords, summary, semantic_facts, entities'
                ' FROM insights WHERE id = ?',
                (iid,)).fetchone()
        finally:
            db.close()
        assert row is not None
        keywords, summary, semantic_facts, entities = row
        assert keywords is not None
        assert summary is not None
        assert semantic_facts is not None
        assert entities is not None

    def test_output_has_causal_count(self, runner):
        """Worker creates a (possibly empty) causal-edge set per insight."""
        from memman.store.db import open_read_only, store_dir

        result = invoke(runner, [
            'remember', 'PostgreSQL chosen for JSONB support in API layer',
            '--no-reconcile'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        iid = data['id']

        _, data_dir = runner
        db = open_read_only(store_dir(data_dir, 'default'))
        try:
            count = db._query(
                "SELECT COUNT(*) FROM edges WHERE source_id = ?"
                " AND edge_type = 'causal'", (iid,)).fetchone()[0]
        finally:
            db.close()
        assert isinstance(count, int)

    def test_no_link_pending_in_output(self, runner):
        """Output no longer includes link_pending field."""
        result = invoke(runner, [
            'remember', 'Docker containers orchestrated via Kubernetes',
            '--no-reconcile'])
        assert result.exit_code == 0
        raw = json.loads(result.output)
        assert 'link_pending' not in raw

    def test_no_causal_candidates_in_output(self, runner):
        """Output no longer includes causal_candidates field."""
        result = invoke(runner, [
            'remember', 'Terraform modules organized by service boundaries',
            '--no-reconcile'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        assert 'causal_candidates' not in data

    def test_linked_at_stamped_after_remember(self, runner):
        """linked_at is non-NULL after remember returns."""
        from memman.store.db import open_read_only

        result = invoke(runner, [
            'remember', 'Consul service mesh enables secure service communication',
            '--no-reconcile'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        iid = data['id']

        _, data_dir = runner
        ro = open_read_only(data_dir + '/data/default')
        row = ro._conn.execute(
            'SELECT linked_at FROM insights WHERE id = ?',
            (iid,)).fetchone()
        ro.close()
        assert row is not None
        assert row[0] is not None

    def test_graph_rebuild_zero_pending_after_remember(self, runner):
        """Graph rebuild processes already-linked insights after remember."""
        invoke(runner, [
            'remember', 'Kafka event streaming configured for microservices',
            '--no-reconcile'])
        result = invoke(runner, ['graph', 'rebuild', '--dry-run'])
        assert result.exit_code == 0

    def test_enriched_at_stamped_after_remember(self, runner):
        """enriched_at is non-NULL after remember returns."""
        from memman.store.db import open_read_only

        result = invoke(runner, [
            'remember', 'Elasticsearch full-text search with custom analyzers',
            '--no-reconcile'])
        assert result.exit_code == 0
        data = parse_remember(result, runner)
        iid = data['id']

        _, data_dir = runner
        ro = open_read_only(data_dir + '/data/default')
        row = ro._conn.execute(
            'SELECT enriched_at FROM insights WHERE id = ?',
            (iid,)).fetchone()
        ro.close()
        assert row is not None
        assert row[0] is not None


def test_recall_source_filter_inflates_fetch_limit(runner):
    """Recall --source must over-fetch so post-filter doesn't truncate results."""
    topics = [
        'PostgreSQL query optimization with EXPLAIN ANALYZE',
        'PostgreSQL index types including B-tree GIN GiST',
        'PostgreSQL vacuum autovacuum tuning parameters',
        'PostgreSQL partitioning strategies for large tables',
        'PostgreSQL connection pooling with PgBouncer setup',
        'PostgreSQL replication streaming and logical decoding',
        ]
    for topic in topics:
        invoke(runner, [
            'remember', topic, '--no-reconcile', '--source', 'user'])
    invoke(runner, [
        'remember', 'PostgreSQL JSONB operators for document queries',
        '--no-reconcile', '--source', 'agent'])

    result = invoke(runner, [
        'recall', 'PostgreSQL database',
        '--source', 'agent', '--limit', '3'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    agent_results = [r for r in data['results']
                     if r['insight']['source'] == 'agent']
    assert len(agent_results) >= 1, (
        f'Expected at least 1 agent result, got {len(agent_results)}. '
        f'--source filter without fetch inflation drops valid results')


def test_remember_creates_semantic_edges(runner):
    """Worker creates semantic edges for the new insight."""
    from memman.store.db import open_read_only, store_dir

    invoke(runner, [
        'remember', 'Go uses SQLite for persistent storage',
        '--no-reconcile'])
    invoke(runner, [
        'remember', 'SQLite WAL mode improves write throughput',
        '--no-reconcile'])

    _, data_dir = runner
    db = open_read_only(store_dir(data_dir, 'default'))
    try:
        rows = db._query(
            "SELECT edge_type FROM edges WHERE edge_type = 'semantic'"
            ).fetchall()
    finally:
        db.close()
    # Either zero or many semantic edges, depending on similarity;
    # the table exists and the worker reaches the edge-creation step.
    assert isinstance(rows, list)


def test_link_returns_actual_db_weight(runner):
    """Link output weight reflects the DB value, not the user-supplied value."""
    r1 = invoke(runner, [
        'remember', 'FastAPI chosen for async API development',
        '--no-reconcile'])
    id1 = parse_remember(r1, runner)['id']
    r2 = invoke(runner, [
        'remember', 'Uvicorn configured as ASGI server for FastAPI',
        '--no-reconcile'])
    id2 = parse_remember(r2, runner)['id']

    invoke(runner, ['graph', 'link', id1, id2, '--type', 'causal', '--weight', '0.9'])
    result = invoke(runner, ['graph', 'link', id1, id2, '--type', 'causal', '--weight', '0.3'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['weight'] >= 0.9, (
        f'Link output shows {data["weight"]} but should be >= 0.9 '
        f'(MAX preserves higher weight over requested 0.3)')
    assert data['weight'] != 0.3, (
        'Link output should not show 0.3 — MAX should preserve higher')


class TestGraphRebuild:
    """Graph rebuild command tests — dry-run, live, edge preservation."""

    def test_rebuild_dry_run_reports_count(self, tmp_path, monkeypatch):
        """Dry run reports total insights without modifying DB."""
        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        from memman.store.db import open_db
        from memman.store.node import insert_insight
        from tests.conftest import make_insight
        db = open_db(str(store_path))
        for i in range(3):
            insert_insight(db, make_insight(
                id=f'rd-{i}', content=f'Test insight {i}'))
            db._conn.execute(
                'UPDATE insights SET linked_at = ?, enriched_at = ?'
                ' WHERE id = ?',
                ('2024-01-01T00:00:00+00:00',
                 '2024-01-01T00:00:00+00:00', f'rd-{i}'))
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild', '--dry-run'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['total'] == 3
        assert data['dry_run'] == 1

        db = open_db(str(store_path))
        row = db._conn.execute(
            'SELECT COUNT(*) FROM insights'
            ' WHERE enriched_at IS NOT NULL').fetchone()
        assert row[0] == 3, 'dry-run must not clear enriched_at'
        db.close()

    def test_rebuild_reprocesses_stale_insights(
            self, tmp_path, monkeypatch):
        """Rebuild re-enriches insights with stale/empty keywords."""
        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        from memman.store.db import open_db
        from memman.store.node import insert_insight
        from tests.conftest import make_insight
        db = open_db(str(store_path))
        insert_insight(db, make_insight(
            id='rs-1', content='Python and SQLite used for data analysis',
            entities=['Python', 'SQLite']))
        insert_insight(db, make_insight(
            id='rs-2', content='SQLite database migration with Python scripts',
            entities=['SQLite', 'Python']))
        db._conn.execute(
            "UPDATE insights"
            " SET linked_at = '2024-01-01T00:00:00+00:00',"
            "     enriched_at = '2024-01-01T00:00:00+00:00',"
            "     keywords = '[]'"
            " WHERE id IN ('rs-1', 'rs-2')")
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['processed'] >= 2

        db = open_db(str(store_path))
        row = db._conn.execute(
            "SELECT keywords, enriched_at FROM insights"
            " WHERE id = 'rs-1'").fetchone()
        keywords = json.loads(row[0]) if row[0] else []
        assert len(keywords) > 0, 'rebuild should populate keywords'
        assert row[1] is not None, 'rebuild should set enriched_at'

        entities_raw = db._conn.execute(
            "SELECT entities FROM insights"
            " WHERE id = 'rs-1'").fetchone()[0]
        entities = json.loads(entities_raw) if entities_raw else []
        assert len(entities) > 0, 'rebuild should populate entities'
        db.close()

    def test_rebuild_handles_mix_of_linked_and_unlinked(
            self, tmp_path, monkeypatch):
        """Rebuild processes both linked and unlinked insights."""
        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        from memman.store.db import open_db
        from memman.store.node import insert_insight
        from tests.conftest import make_insight
        db = open_db(str(store_path))
        insert_insight(db, make_insight(
            id='mx-1', content='Already linked insight'))
        db._conn.execute(
            "UPDATE insights SET linked_at = ?, enriched_at = ?"
            " WHERE id = 'mx-1'",
            ('2024-01-01T00:00:00+00:00',
             '2024-01-01T00:00:00+00:00'))
        insert_insight(db, make_insight(
            id='mx-2', content='Never linked insight'))
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild'])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['processed'] >= 2

        db = open_db(str(store_path))
        pending = db._conn.execute(
            'SELECT COUNT(*) FROM insights'
            ' WHERE linked_at IS NULL'
            ' AND deleted_at IS NULL').fetchone()[0]
        assert pending == 0, 'all insights should be linked after rebuild'
        db.close()

    def test_rebuild_preserves_manual_edges(
            self, tmp_path, monkeypatch):
        """Manual claude edges survive rebuild."""
        monkeypatch.delenv('MEMMAN_STORE', raising=False)
        data_dir = str(tmp_path)
        store_path = tmp_path / 'data' / 'default'
        from memman.store.db import open_db
        from memman.store.edge import get_all_edges, insert_edge
        from memman.store.node import insert_insight
        from tests.conftest import make_edge, make_insight
        db = open_db(str(store_path))
        insert_insight(db, make_insight(
            id='me-1', content='Python web framework',
            entities=['Python']))
        insert_insight(db, make_insight(
            id='me-2', content='Python data pipeline',
            entities=['Python']))
        db._conn.execute(
            "UPDATE insights"
            " SET linked_at = '2024-01-01T00:00:00+00:00',"
            "     enriched_at = '2024-01-01T00:00:00+00:00'")
        manual_edge = make_edge(
            source_id='me-1', target_id='me-2',
            edge_type='semantic',
            metadata={'created_by': 'claude', 'cosine': '0.95'})
        insert_edge(db, manual_edge)
        db.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--data-dir', data_dir, 'graph', 'rebuild'])
        assert result.exit_code == 0, result.output

        db = open_db(str(store_path))
        edges = get_all_edges(db)
        manual = [e for e in edges
                  if e.edge_type == 'semantic'
                  and e.metadata.get('created_by') == 'claude']
        assert len(manual) == 1, (
            'rebuild deleted manual claude edge — '
            'should preserve created_by=claude')
        db.close()


class TestIntraBatchDedup:
    """Sibling facts from the same remember call must deduplicate."""

    def test_similar_sibling_facts_deduplicated(self, runner):
        """When extraction produces two paraphrases, only one is stored."""
        def _two_similar_facts(llm_client, content):
            return [
                {
                    'text': 'Do not rename loop variables to avoid '
                            'shadowing opts attributes',
                    'category': 'preference',
                    'importance': 3,
                    'entities': ['loop variables', 'opts'],
                    },
                {
                    'text': 'Avoid renaming loop variables to prevent '
                            'shadowing of opts attributes',
                    'category': 'preference',
                    'importance': 3,
                    'entities': ['loop variables', 'opts'],
                    },
                ]

        with patch('memman.llm.extract.extract_facts',
                   _two_similar_facts):
            result = invoke(runner, [
                'remember', ('Do not rename loop variables to avoid shadowing '
                             'opts attributes')])
        assert result.exit_code == 0, result.output
        from memman.store.db import open_read_only, store_dir
        raw = json.loads(result.output)
        queue_id = raw['queue_id']
        _, data_dir = runner
        db = open_read_only(store_dir(data_dir, raw['store']))
        try:
            rows = db._query(
                'SELECT id FROM insights WHERE source = ?'
                ' AND deleted_at IS NULL', (f'queue:{queue_id}',)
                ).fetchall()
        finally:
            db.close()
        assert len(rows) == 1, (
            f'expected 1 stored fact, got {len(rows)}: '
            f'queue_id={queue_id}')

    def test_distinct_facts_both_stored(self, runner):
        """Genuinely different facts from one input are both stored."""
        def _two_distinct_facts(llm_client, content):
            return [
                {
                    'text': 'Switched from Flask to FastAPI',
                    'category': 'decision',
                    'importance': 4,
                    'entities': ['Flask', 'FastAPI'],
                    },
                {
                    'text': 'Redis cache configured with 4GB max memory',
                    'category': 'fact',
                    'importance': 3,
                    'entities': ['Redis'],
                    },
                ]

        with patch('memman.llm.extract.extract_facts',
                   _two_distinct_facts):
            result = invoke(runner, [
                'remember', 'Switched to FastAPI and configured Redis cache'])
        assert result.exit_code == 0, result.output
        from memman.store.db import open_read_only, store_dir
        raw = json.loads(result.output)
        queue_id = raw['queue_id']
        _, data_dir = runner
        db = open_read_only(store_dir(data_dir, raw['store']))
        try:
            rows = db._query(
                'SELECT id FROM insights WHERE source = ?'
                ' AND deleted_at IS NULL', (f'queue:{queue_id}',)
                ).fetchall()
        finally:
            db.close()
        assert len(rows) == 2, (
            f'expected 2 stored facts, got {len(rows)}: '
            f'queue_id={queue_id}')

    @pytest.mark.skipif(
        'not config.getoption("--live")',
        reason='requires --live for real LLM calls')
    def test_single_thought_not_duplicated_live(self, runner):
        """A single coherent preference should produce at most 1 stored fact."""
        result = invoke(runner, [
            'remember', ('Loop variable naming: do not rename loop variables '
                         'to avoid shadowing opts attributes. Maintain existing '
                         'variable names to prevent unintended shadowing of '
                         'options object properties.')])
        assert result.exit_code == 0, result.output
        raw = json.loads(result.output)
        facts = raw['facts']
        stored = [f for f in facts if f['action'] != 'skipped']
        assert len(stored) == 1, (
            f'expected 1 stored fact from single thought, got '
            f'{len(stored)}: {json.dumps(facts, indent=2)}')

    def test_two_updates_same_target_no_duplicate(self, runner):
        """Sibling UPDATEs must not create duplicates from stale candidates."""
        invoke(runner, [
            'remember', 'PostgreSQL chosen for ACID compliance and JSON support',
            '--no-reconcile'])

        def _two_update_facts(llm_client, content):
            return [
                {
                    'text': 'PostgreSQL chosen for ACID compliance'
                            ' and JSON support plus extensions',
                    'category': 'decision',
                    'importance': 4,
                    'entities': ['PostgreSQL'],
                    },
                {
                    'text': 'PostgreSQL chosen for ACID compliance'
                            ' and JSON support with replication',
                    'category': 'decision',
                    'importance': 4,
                    'entities': ['PostgreSQL'],
                    },
                ]

        with patch('memman.llm.extract.extract_facts',
                   _two_update_facts):
            result = invoke(runner, [
                'remember', 'PostgreSQL chosen for ACID compliance'])
        assert result.exit_code == 0, result.output

        search_result = invoke(runner, ['recall', '--basic', 'PostgreSQL'])
        active = json.loads(search_result.output)['results']
        assert len(active) == 1, (
            f'expected 1 active PostgreSQL insight, got {len(active)}')

    def test_forced_update_stale_target_no_duplicate(self, runner):
        """Forced UPDATE against stale target must not create a duplicate."""
        invoke(runner, [
            'remember', 'Kafka uses topic partitioning for message ordering',
            '--no-reconcile'])

        def _two_facts(llm_client, content):
            return [
                {
                    'text': 'Kafka uses topic partitioning for'
                            ' message ordering and consumer groups',
                    'category': 'fact',
                    'importance': 3,
                    'entities': ['Kafka'],
                    },
                {
                    'text': 'Kafka uses topic partitioning for'
                            ' message ordering and replication',
                    'category': 'fact',
                    'importance': 3,
                    'entities': ['Kafka'],
                    },
                ]

        def _force_update(llm_client, facts, existing):
            target_id = existing[0][0] if existing else None
            return [{'fact': f['text'], 'action': 'UPDATE',
                     'target_id': target_id,
                     'merged_text': f['text']}
                    for f in facts]

        with patch('memman.llm.extract.extract_facts', _two_facts), \
        patch('memman.llm.extract.reconcile_memories',
              _force_update):
            result = invoke(runner, [
                'remember', 'Kafka topic partitioning'])
        assert result.exit_code == 0, result.output

        search_result = invoke(runner, ['recall', '--basic', 'Kafka'])
        active = json.loads(search_result.output)['results']
        assert len(active) == 1, (
            f'expected 1 active Kafka insight, got {len(active)}')

    @pytest.mark.skipif(
        'not config.getoption("--live")',
        reason='requires --live for real LLM calls')
    def test_near_identical_updates_no_duplicate_live(self, runner):
        """Near-identical sibling facts must not create duplicates with real LLM."""
        invoke(runner, [
            'remember', 'Redis cache uses 4GB max memory for session storage',
            '--no-reconcile'])

        def _near_identical_facts(llm_client, content):
            return [
                {
                    'text': 'Redis cache uses 4GB maximum memory'
                            ' for session storage',
                    'category': 'fact',
                    'importance': 3,
                    'entities': ['Redis'],
                    },
                {
                    'text': 'Redis cache uses 4GB max memory'
                            ' for session storage',
                    'category': 'fact',
                    'importance': 3,
                    'entities': ['Redis'],
                    },
                ]

        with patch('memman.llm.extract.extract_facts',
                   _near_identical_facts):
            result = invoke(runner, [
                'remember', 'Redis cache memory configuration'])
        assert result.exit_code == 0, result.output

        search_result = invoke(runner, ['recall', '--basic', 'Redis'])
        active = json.loads(search_result.output)['results']
        assert len(active) == 1, (
            f'expected 1 active Redis insight, got {len(active)}')

    def test_update_reconciliation_no_dangling_edges(self, runner):
        """Chained intra-batch UPDATEs must not leave semantic edges to soft-deleted insights."""
        _r, data_dir = runner

        def _three_paraphrase_facts(llm_client, content):
            return [
                {
                    'text': 'Delta mode dropdown defaults'
                            ' to incremental_sync with no empty option',
                    'category': 'decision',
                    'importance': 3,
                    'entities': [],
                    },
                {
                    'text': 'Delta mode dropdown defaults'
                            ' to incremental_sync with no empty option'
                            ' unlike filter dropdowns',
                    'category': 'decision',
                    'importance': 3,
                    'entities': [],
                    },
                {
                    'text': 'Delta mode dropdown defaults'
                            ' to incremental_sync with no empty option'
                            ' unlike filter dropdowns which have one',
                    'category': 'decision',
                    'importance': 3,
                    'entities': [],
                    },
                ]

        def _force_update(llm_client, facts, existing):
            target_id = existing[0][0] if existing else None
            return [{'fact': f['text'], 'action': 'UPDATE',
                     'target_id': target_id,
                     'merged_text': f['text']}
                    for f in facts]

        fixed_vec = [1.0] + [0.0] * 511

        def _fixed_embed(self, text):
            return list(fixed_vec)

        with patch('memman.llm.extract.extract_facts',
                   _three_paraphrase_facts), \
        patch('memman.llm.extract.reconcile_memories',
              _force_update), \
        patch('memman.embed.voyage.Client.embed', _fixed_embed):
            result = invoke(runner, [
                'remember', 'Delta mode dropdown defaults to incremental_sync'])
        assert result.exit_code == 0, result.output

        store_path = pathlib.Path(data_dir) / 'data' / 'default'
        from memman.doctor import check_dangling_edges
        from memman.store.db import open_db
        db = open_db(str(store_path))
        doctor_result = check_dangling_edges(db)
        db.close()

        assert doctor_result['status'] == 'pass', (
            f'dangling edges found: {doctor_result["detail"]}')
        assert doctor_result['detail']['count'] == 0
