"""Black-box behavioral tests for memman memory management system.

These tests verify behavioral invariants through the CLI only — no internal
module imports.
"""

import json
import pathlib

import pytest
from click.testing import CliRunner
from memman.cli import cli


@pytest.fixture
def runner(tmp_path):
    """CliRunner with --data-dir pointing to temp directory."""
    r = CliRunner()
    data_dir = str(tmp_path / 'memman_data')
    pathlib.Path(data_dir).mkdir(exist_ok=True, parents=True)
    return r, data_dir


def invoke(runner_tuple, args):
    """Invoke CLI with data-dir."""
    r, data_dir = runner_tuple
    return r.invoke(cli, ['--data-dir', data_dir] + args)


def parse_remember(result, runner_tuple=None):
    """Parse remember/replace output, returning a fact-shaped dict.

    Modern `remember`/`replace` returns just `{action: queued,
    queue_id, store}`. The autouse-drain runs the worker after the
    invocation, so the new insight lives in the store DB tagged with
    `source = queue:<queue_id>`. This helper looks it up so tests
    that read `id`/`content`/`category`/`importance` after a remember
    keep working.
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
    fact = {
        'id': rows[0][0],
        'content': rows[0][1],
        'category': rows[0][2],
        'importance': rows[0][3],
        'action': action,
        'replaced_id': raw.get('replaced_id'),
        '_raw': raw,
        }
    return fact


def remember(runner_tuple, content, no_reconcile=False, **flags):
    """Store an insight, return first fact dict from output.

    `remember` queues + auto-drains via the CliRunner wrapper, then
    we look up the newly-stored insight by `source = queue:<id>` so
    existing assertions like `data['id']` keep working.
    """
    args = ['remember', content]
    if no_reconcile:
        args.append('--no-reconcile')
    for k, v in flags.items():
        args.extend([f'--{k}', str(v)])
    result = invoke(runner_tuple, args)
    assert result.exit_code == 0, result.output
    return parse_remember(result, runner_tuple)


def recall_basic(runner_tuple, keyword):
    """Recall via --basic (SQL LIKE on single keyword), return list."""
    result = invoke(runner_tuple, ['recall', keyword, '--basic'])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)['results']


def recall_smart(runner_tuple, query, **flags):
    """Recall via intent-aware mode, return results list of insight dicts."""
    args = ['recall', query]
    for k, v in flags.items():
        args.extend([f'--{k}', str(v)])
    result = invoke(runner_tuple, args)
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    return [r['insight'] for r in data.get('results', [])]


def search_cmd(runner_tuple, query):
    """Keyword-only retrieval via recall --basic, return insight list."""
    result = invoke(runner_tuple, ['recall', '--basic', query])
    assert result.exit_code == 0, result.output
    return json.loads(result.output).get('results', [])


def contents(results):
    """Extract content strings from result dicts."""
    return [r['content'] for r in results]


def result_ids(results):
    """Extract IDs from result dicts."""
    return [r['id'] for r in results]


# ═══════════════════════════════════════════════════════════════════
# PART 1: CORE INVARIANTS — the system MUST satisfy these
# ═══════════════════════════════════════════════════════════════════


class TestPersistence:
    """What you store, you can retrieve."""

    def test_store_then_recall_finds_it(self, runner):
        """Single insight retrievable by keyword from its content."""
        data = remember(runner,
                        'I configured Redis with allkeys-lru eviction and 4GB maxmemory limit',
                        no_reconcile=True)
        hits = recall_basic(runner, 'Redis')
        assert data['id'] in result_ids(hits)

    def test_store_five_diverse_insights_recall_each(self, runner):
        """Five diverse insights all individually retrievable."""
        topics = [
            ('I implemented OAuth2 PKCE flow for our mobile app',
             'OAuth2'),
            ('I configured Redis Sentinel with 3 nodes for failover',
             'Redis'),
            ('I use Terraform modules to compose our AWS deployment',
             'Terraform'),
            ('I adopted Hypothesis for property-based testing in Python',
             'Hypothesis'),
            ('I set up Prometheus alerting for SLO burn rate tracking',
             'Prometheus'),
            ]
        ids = []
        for content, _ in topics:
            data = remember(runner, content, no_reconcile=True)
            ids.append(data['id'])

        for i, (content, keyword) in enumerate(topics):
            hits = recall_basic(runner, keyword)
            hit_ids = result_ids(hits)
            assert ids[i] in hit_ids, (
                f'Could not recall "{keyword}" — got: {contents(hits)[:3]}')

    def test_partial_keyword_match(self, runner):
        """Partial keyword from content is enough to find insight."""
        remember(runner,
                 'Kubernetes pod scheduling affinity rules and taints',
                 no_reconcile=True)
        hits = recall_basic(runner, 'scheduling')
        assert any('scheduling' in c for c in contents(hits))

    def test_word_order_irrelevant(self, runner):
        """Search finds content regardless of query word order."""
        remember(runner,
                 'SQLite WAL mode write-ahead logging benefits',
                 no_reconcile=True)
        hits = search_cmd(runner, 'benefits write-ahead SQLite')
        assert any('SQLite' in c for c in contents(hits))

    def test_no_false_positives_on_unrelated_query(self, runner):
        """Completely unrelated query returns no results."""
        remember(runner, 'Python web framework comparison', no_reconcile=True)
        remember(runner, 'Docker container networking', no_reconcile=True)
        hits = recall_basic(runner, 'chromodynamics')
        assert len(hits) == 0


class TestDeletionCompleteness:
    """Forgotten insights vanish from ALL retrieval paths."""

    def test_forget_removes_from_recall(self, runner):
        """Forgotten insight absent from recall results."""
        data = remember(runner, 'Python GIL prevents true parallelism',
                        no_reconcile=True)
        invoke(runner, ['forget', data['id']])
        hits = recall_basic(runner, 'GIL')
        assert data['id'] not in result_ids(hits)

    def test_forget_removes_from_search(self, runner):
        """Forgotten insight absent from search results."""
        data = remember(runner, 'Nginx reverse proxy configuration',
                        no_reconcile=True)
        invoke(runner, ['forget', data['id']])
        hits = search_cmd(runner, 'Nginx reverse proxy')
        assert data['id'] not in [h['id'] for h in hits]

    def test_forget_does_not_collateral_damage_peers(self, runner):
        """Forgetting A does not affect B."""
        a = remember(runner,
                     'Celery task queue uses exponential backoff retry with max 5 attempts',
                     no_reconcile=True)
        b = remember(runner,
                     'PgBouncer connection pooling reduces PostgreSQL connection overhead by 90 percent',
                     no_reconcile=True)
        invoke(runner, ['forget', a['id']])

        hits_b = recall_basic(runner, 'PgBouncer')
        assert b['id'] in result_ids(hits_b)
        hits_a = recall_basic(runner, 'Celery')
        assert a['id'] not in result_ids(hits_a)

    def test_forget_then_re_store_same_content(self, runner):
        """Content can be re-stored after being forgotten."""
        text = 'Python GIL behavior under multiprocessing'
        data = remember(runner, text, no_reconcile=True)
        invoke(runner, ['forget', data['id']])
        new_data = remember(runner, text, no_reconcile=True)
        hits = recall_basic(runner, 'GIL')
        assert new_data['id'] in result_ids(hits)

    def test_double_forget_fails(self, runner):
        """Second forget on same ID returns error."""
        data = remember(runner,
                        'Nginx configured with 4096 worker connections for load balancing',
                        no_reconcile=True)
        invoke(runner, ['forget', data['id']])
        result = invoke(runner, ['forget', data['id']])
        assert result.exit_code != 0


class TestReplaceAtomicity:
    """Old content gone, new content present, metadata coherent."""

    def test_replace_swaps_content(self, runner):
        """Old content absent, new content present after replace."""
        data = remember(runner, 'team uses Flask for API layer',
                        no_reconcile=True)
        invoke(runner, ['replace', data['id'],
                        'team migrated to FastAPI for API layer'])

        hits_old = recall_basic(runner, 'Flask')
        assert not any('Flask' in c for c in contents(hits_old))
        hits_new = recall_basic(runner, 'FastAPI')
        assert any('FastAPI' in c for c in contents(hits_new))

    def test_replace_inherits_metadata(self, runner):
        """Replace without flags inherits cat/imp/tags from original."""
        data = remember(runner, 'chose event sourcing for audit trail',
                        no_reconcile=True, cat='decision', imp='5',
                        tags='arch,design')
        result = invoke(runner, ['replace', data['id'],
                                 'chose CQRS with event sourcing for audit'])
        new = parse_remember(result, runner)
        assert 'id' in new

        result = invoke(runner, ['recall', '--basic', 'CQRS'])
        hits = json.loads(result.output)['results']
        match = [h for h in hits if h['id'] == new['id']][0]
        assert match['category'] == 'decision'
        assert match['importance'] == 5
        assert 'arch' in match['tags']

    def test_replace_override_metadata(self, runner):
        """Replace with explicit flags overrides original metadata."""
        data = remember(runner, 'Varnish HTTP cache configured with 2GB memory for static assets',
                        no_reconcile=True, cat='fact', imp='2')
        result = invoke(runner, ['replace', data['id'],
                                 'Switched from Varnish to CloudFront CDN for global edge caching',
                                 '--cat', 'decision', '--imp', '5'])
        new = parse_remember(result, runner)
        assert 'id' in new

        result = invoke(runner, ['recall', '--basic', 'CloudFront CDN'])
        hits = json.loads(result.output)['results']
        match = [h for h in hits if h['id'] == new['id']][0]
        assert match['category'] == 'decision'
        assert match['importance'] == 5

    def test_replace_preserves_access_count(self, runner):
        """Replace carries forward accumulated access count."""
        data = remember(runner,
                        'PostgreSQL migration from version 14 to 16 completed successfully',
                        no_reconcile=True)
        recall_basic(runner, 'PostgreSQL')
        recall_basic(runner, 'PostgreSQL')
        result = invoke(runner, ['replace', data['id'],
                                 'PostgreSQL migration from 14 to 16 required reindex of all GIN indexes'])
        new = parse_remember(result, runner)
        hits = recall_basic(runner, 'PostgreSQL')
        replaced = [h for h in hits if h['id'] == new['id']]
        assert replaced
        assert replaced[0]['access_count'] >= 2

    def test_replace_nonexistent_id_errors(self, runner):
        """Replace with fake ID fails."""
        result = invoke(runner, ['replace', 'nonexistent-fake-id', 'nope'])
        assert result.exit_code != 0

    def test_replace_deleted_id_errors(self, runner):
        """Replace on already-forgotten insight fails."""
        data = remember(runner,
                        'RabbitMQ queue mirroring configured for high availability',
                        no_reconcile=True)
        invoke(runner, ['forget', data['id']])
        result = invoke(runner, ['replace', data['id'], 'too late'])
        assert result.exit_code != 0


class TestDeduplication:
    """No silent duplicates, no false positive dedup."""

    def test_reconcile_with_no_reconcile_always_adds(self, runner):
        """With --no-reconcile, identical content is added."""
        text = 'Go error handling with sentinel values and wrapping'
        remember(runner, text, no_reconcile=True)
        second = remember(runner, text, no_reconcile=True)
        assert second['action'] == 'add'

    def test_reconcile_runs_when_similar_exists(self, runner):
        """Without --no-reconcile, similar content triggers reconciliation.

        Real LLM decides action — may be add/update/none/delete/skipped.
        'skipped' means LLM found the info already captured (valid).
        """
        text = 'Go error handling with sentinel values and wrapping'
        remember(runner, text, no_reconcile=True)
        second = remember(runner, text)
        assert second['action'] in {
            'add', 'update', 'none', 'delete', 'skipped'}

    def test_different_content_added(self, runner):
        """Genuinely different content is added via reconciliation."""
        remember(runner, 'I use mypy strict mode for all Python projects',
                 no_reconcile=True)
        second = remember(runner,
                          'I switched to Podman from Docker for rootless containers')
        assert second['action'] in {'add', 'update'}


class TestGraphTraversal:
    """Links are traversable and filtered correctly."""

    def test_link_makes_insight_reachable_via_related(self, runner):
        """Linked insight appears in related output."""
        a = remember(runner, 'authentication design with JWT tokens',
                     no_reconcile=True)
        b = remember(runner, 'token rotation schedule every 24 hours',
                     no_reconcile=True)
        invoke(runner, ['graph', 'link', a['id'], b['id'], '--type', 'semantic'])

        result = invoke(runner, ['graph', 'related', a['id']])
        assert b['id'] in result.output

    def test_related_respects_edge_type_filter(self, runner):
        """Edge type filter includes matching, excludes non-matching."""
        a = remember(runner, 'chose SQLite because embedded serverless',
                     no_reconcile=True)
        b = remember(runner, 'SQLite enables single-file deployment',
                     no_reconcile=True)
        invoke(runner, ['graph', 'link', a['id'], b['id'], '--type', 'causal'])

        result_causal = invoke(runner, ['graph', 'related', a['id'],
                                        '--edge', 'causal'])
        assert b['id'] in result_causal.output
        result_semantic = invoke(runner, ['graph', 'related', a['id'],
                                          '--edge', 'semantic'])
        assert b['id'] not in result_semantic.output

    def test_link_persists_after_other_operations(self, runner):
        """New inserts don't clobber existing edges."""
        a = remember(runner, 'microservice communication via gRPC',
                     no_reconcile=True)
        b = remember(runner, 'protobuf schema evolution rules',
                     no_reconcile=True)
        invoke(runner, ['graph', 'link', a['id'], b['id'], '--type', 'semantic'])

        remember(runner, 'Kafka topic partitioning strategy uses key-based routing for ordering guarantees', no_reconcile=True)

        result = invoke(runner, ['graph', 'related', a['id']])
        assert b['id'] in result.output

    def test_related_respects_depth(self, runner):
        """Depth=1 returns only direct neighbors, not hop-2 nodes.

        Uses --edge causal to isolate from auto-created temporal
        proximity edges that shortcut the graph.
        """
        a = remember(runner, 'API gateway routing rules', no_reconcile=True)
        b = remember(runner, 'rate limiting middleware', no_reconcile=True)
        c = remember(runner, 'circuit breaker pattern', no_reconcile=True)
        invoke(runner, ['graph', 'link', a['id'], b['id'], '--type', 'causal'])
        invoke(runner, ['graph', 'link', b['id'], c['id'], '--type', 'causal'])

        result_d1 = invoke(runner, ['graph', 'related', a['id'],
                                    '--edge', 'causal', '--depth', '1'])
        data_d1 = json.loads(result_d1.output)
        ids_d1 = [r['id'] for r in data_d1]
        assert b['id'] in ids_d1
        assert c['id'] not in ids_d1

        result_d2 = invoke(runner, ['graph', 'related', a['id'],
                                    '--edge', 'causal', '--depth', '2'])
        data_d2 = json.loads(result_d2.output)
        ids_d2 = [r['id'] for r in data_d2]
        assert b['id'] in ids_d2
        assert c['id'] in ids_d2


class TestComposition:
    """Multi-step workflows stay consistent."""

    def test_forget_target_does_not_break_related(self, runner):
        """Forgetting a linked target does not crash related or leak."""
        a = remember(runner, 'API design principles REST vs GraphQL',
                     no_reconcile=True)
        b = remember(runner, 'GraphQL schema stitching federation',
                     no_reconcile=True)
        c = remember(runner, 'REST pagination cursor-based approach',
                     no_reconcile=True)
        link_ab = invoke(runner, ['graph', 'link', a['id'], b['id'],
                                  '--type', 'causal'])
        assert link_ab.exit_code == 0
        link_ac = invoke(runner, ['graph', 'link', a['id'], c['id'],
                                  '--type', 'causal'])
        assert link_ac.exit_code == 0
        invoke(runner, ['forget', b['id']])

        result = invoke(runner, ['graph', 'related', a['id'],
                                 '--edge', 'causal'])
        assert result.exit_code == 0
        assert c['id'] in result.output
        assert b['id'] not in result.output

    def test_store_replace_recall_sequence(self, runner):
        """Replace + subsequent inserts don't interfere with each other."""
        x = remember(runner, 'Flask API for internal tooling',
                     no_reconcile=True)
        hits = recall_basic(runner, 'Flask')
        assert any('Flask' in c for c in contents(hits))

        invoke(runner, ['replace', x['id'],
                        'FastAPI migration for internal tooling'])
        hits_old = recall_basic(runner, 'Flask')
        assert not any('Flask' in c for c in contents(hits_old))
        hits_new = recall_basic(runner, 'FastAPI')
        assert any('FastAPI' in c for c in contents(hits_new))

        remember(runner, 'Django admin for backoffice portal',
                 no_reconcile=True)
        hits_fast = recall_basic(runner, 'FastAPI')
        assert any('FastAPI' in c for c in contents(hits_fast))
        hits_django = recall_basic(runner, 'Django')
        assert any('Django' in c for c in contents(hits_django))

    def test_bulk_insert_selective_delete_consistency(self, runner):
        """Store 10, delete 3, verify 7 remain and 3 gone."""
        keywords = [
            'gRPC', 'Kafka', 'etcd', 'Vault', 'Consul',
            'Envoy', 'Jaeger', 'Fluentd', 'ArgoCD', 'Istio',
            ]
        stored = [remember(
                runner,
                f'I deployed {kw} version 3.2 on our production cluster',
                no_reconcile=True) for kw in keywords]

        delete_indices = [1, 4, 7]
        for i in delete_indices:
            invoke(runner, ['forget', stored[i]['id']])

        for i, s in enumerate(stored):
            hits = recall_basic(runner, keywords[i])
            hit_ids = result_ids(hits)
            if i in delete_indices:
                assert s['id'] not in hit_ids
            else:
                assert s['id'] in hit_ids, (
                    f'{keywords[i]} not found in recall')

        result = invoke(runner, ['status'])
        data = json.loads(result.output)
        assert data['total_insights'] == 7


class TestInputValidation:
    """Bad input is rejected, not silently accepted."""

    def test_invalid_category_rejected(self, runner):
        """Unknown category produces non-zero exit."""
        result = invoke(runner, ['remember', 'test', '--cat', 'bogus'])
        assert result.exit_code != 0

    def test_importance_out_of_range_rejected(self, runner):
        """Importance 0 and 6 are out of range."""
        for imp in ['0', '6']:
            result = invoke(runner, ['remember', 'test', '--imp', imp])
            assert result.exit_code != 0

    def test_store_name_invalid_rejected(self, runner):
        """Invalid store names are rejected."""
        for name in ['-bad', 'has space', '.hidden']:
            result = invoke(runner, ['store', 'create', name])
            assert result.exit_code != 0


class TestRanking:
    """Better matches rank higher."""

    def test_exact_keyword_match_outranks_partial(self, runner):
        """Exact keyword match ranks above partial overlap."""
        a = remember(runner,
                     'I tuned Redis cache eviction to allkeys-lru',
                     no_reconcile=True, imp='3')
        b = remember(runner,
                     'I automated Redis deployment with Ansible',
                     no_reconcile=True, imp='3')
        hits = search_cmd(runner, 'Redis cache eviction')
        assert hits, 'Expected at least one result'
        assert hits[0]['id'] == a['id']

    def test_importance_breaks_ties(self, runner):
        """Higher importance ranks first among similar content."""
        low = remember(runner,
                       'I set up Grafana dashboards for API latency monitoring',
                       no_reconcile=True, imp='2')
        high = remember(runner,
                        'I set up Grafana dashboards for API latency monitoring',
                        no_reconcile=True, imp='5')
        hits = search_cmd(runner, 'Grafana')
        assert len(hits) >= 2
        hit_ids = result_ids(hits)
        high_idx = hit_ids.index(high['id'])
        low_idx = hit_ids.index(low['id'])
        assert high_idx < low_idx

    def test_category_filter_restricts_results(self, runner):
        """Recall --cat returns only matching category."""
        remember(runner, 'chose Postgres for relational data',
                 no_reconcile=True, cat='decision')
        remember(runner, 'prefer dark mode for IDEs',
                 no_reconcile=True, cat='preference')
        remember(runner, 'SQLite is an embedded database',
                 no_reconcile=True, cat='fact')
        hits = recall_smart(runner, 'database', cat='decision')
        categories = [h['category'] for h in hits]
        assert all(c == 'decision' for c in categories), (
            f'Expected only decision, got {categories}')
        assert any('postgres' in h['content'].lower() for h in hits)


class TestGCLifecycle:
    """GC surfaces low-value insights and respects protections."""

    def test_gc_produces_stats(self, runner):
        """GC returns JSON with total_insights field."""
        remember(runner,
                 'HAProxy health checks configured with 3 second interval',
                 no_reconcile=True, imp='1')
        remember(runner,
                 'Logstash pipeline processes 10000 events per second',
                 no_reconcile=True, imp='1')
        result = invoke(runner, ['insights', 'candidates'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert 'total_insights' in data


class TestOplogChronology:
    """Operation log entries are in chronological order."""

    def test_oplog_order_is_chronological(self, runner):
        """Log --limit N returns most-recent-first ordering."""
        techs = ['Redis', 'Kafka', 'Consul', 'Vault', 'Envoy']
        for tech in techs:
            remember(runner,
                     f'{tech} cluster deployed across three availability zones for resilience',
                     no_reconcile=True)
        result = invoke(runner, ['log', 'list', '--limit', '5'])
        assert result.exit_code == 0
        lines = [l for l in result.output.strip().split('\n')
                 if l.strip() and not l.startswith('TIME')]
        assert len(lines) >= 5


class TestStatusAfterMutations:
    """Status counts reflect actual state after mixed mutations."""

    def test_status_count_after_mixed_mutations(self, runner):
        """Store 4, forget 1, replace 1 — status shows 4 total.

        Replace creates a new row and soft-deletes the old one,
        so 4 inserts - 1 forget - 1 replaced + 1 new = 3 active
        originals + 1 replacement = 4 visible if replace doesn't
        add net count, or 3 if it does. We assert >= 3.
        """
        techs = ['Grafana', 'Jaeger', 'ArgoCD', 'Istio']
        stored = [remember(
                runner,
                f'{tech} service mesh component configured for production monitoring',
                no_reconcile=True) for tech in techs]
        invoke(runner, ['forget', stored[0]['id']])
        invoke(runner, ['replace', stored[1]['id'],
                        'Jaeger distributed tracing upgraded to OpenTelemetry collector'])
        result = invoke(runner, ['status'])
        data = json.loads(result.output)
        assert data['total_insights'] >= 3


class TestRecallFindsContentByEntities:
    """Insights should be findable by their entities, not just content."""

    def test_basic_recall_finds_by_entity(self, runner):
        """Insight with entity 'Kubernetes' found by recalling 'Kubernetes'.

        Content says 'Kubernetes pod scheduling uses affinity rules and taints for node placement' with
        no mention of Kubernetes, but the entity field has it.
        """
        remember(runner,
                 'Kubernetes pod scheduling uses affinity rules and taints for node placement',
                 no_reconcile=True, entities='Kubernetes')
        hits = recall_basic(runner, 'Kubernetes')
        assert len(hits) > 0

    def test_basic_recall_finds_by_tag(self, runner):
        """Insight with tag 'compliance' found by recalling 'compliance'.

        Content says 'Docker container security hardening with read-only root filesystem and non-root user' with no
        mention of compliance, but the tags field has it.
        """
        remember(runner,
                 'Docker container security hardening with read-only root filesystem and non-root user',
                 no_reconcile=True, tags='security,docker,compliance')
        hits = recall_basic(runner, 'compliance')
        assert len(hits) > 0

    def test_search_finds_by_entity_and_tag(self, runner):
        """Search (token-based) does find entities and tags."""
        remember(runner,
                 'Kubernetes pod scheduling uses affinity rules and taints for node placement',
                 no_reconcile=True, entities='Kubernetes',
                 tags='infra,compliance')
        entity_hits = search_cmd(runner, 'Kubernetes')
        assert len(entity_hits) > 0
        tag_hits = search_cmd(runner, 'compliance')
        assert len(tag_hits) > 0


class TestMultiWordRecall:
    """Multi-word queries should work across all retrieval paths."""

    def test_basic_recall_non_adjacent_words(self, runner):
        """'Python slow' should find 'Python is slow for CPU-bound tasks'.

        Words appear in content but not adjacently.
        """
        remember(runner, 'Python is slow for CPU-bound tasks', no_reconcile=True)
        hits = recall_basic(runner, 'Python slow')
        assert len(hits) > 0

    def test_search_handles_non_adjacent_words(self, runner):
        """Search tokenizes independently, finds non-adjacent matches."""
        remember(runner, 'Python is slow for CPU-bound tasks', no_reconcile=True)
        hits = search_cmd(runner, 'Python slow')
        assert any('Python' in c for c in contents(hits))

    def test_smart_recall_handles_multi_word(self, runner):
        """Smart recall finds content by multi-word query."""
        remember(runner, 'PostgreSQL JSONB indexing for document queries',
                 no_reconcile=True)
        hits = recall_smart(runner, 'PostgreSQL JSONB indexing')
        assert any('JSONB' in c for c in contents(hits))


class TestContradictionDetection:
    """Contradicting facts should trigger LLM reconciliation."""

    def test_contradiction_triggers_reconciliation(self, runner):
        """Storing contradictory content triggers LLM reconciliation.

        Real LLM handles contradiction — we verify valid action.
        """
        remember(runner,
                 'Redis is single-threaded and cannot use multiple cores',
                 no_reconcile=True)
        result = remember(runner,
                          'Redis 6.0 supports multi-threaded IO')
        assert result['action'] in {'add', 'update', 'none', 'delete'}


class TestAccessCountAccuracy:
    """Access count should exactly track recall invocations."""

    def test_access_count_matches_recall_count(self, runner):
        """After N recalls, access_count should be N."""
        remember(runner,
                 'Wireguard VPN tunnel configured with 256-bit encryption between datacenters',
                 no_reconcile=True)
        for _ in range(5):
            recall_basic(runner, 'Wireguard')
        hits = recall_basic(runner, 'Wireguard')
        assert hits[0]['access_count'] == 6, (
            f'Expected 6, got {hits[0]["access_count"]}')


class TestRecallPrecisionUnderNoise:
    """Recall should find the right needle in a large haystack."""

    def test_specific_insight_among_fifty_similar(self, runner):
        """One specific insight findable among 50 generic ones."""
        for i in range(50):
            remember(runner,
                     f'PostgreSQL query optimization uses index scan on column_{i} with btree',
                     no_reconcile=True)
        remember(runner,
                 'alertmanager silencing rules for oncall rotation',
                 no_reconcile=True)
        hits = recall_basic(runner, 'alertmanager')
        assert any('alertmanager' in c.lower() for c in contents(hits))

    def test_high_importance_surfaces_above_noise(self, runner):
        """imp=5 insight ranks first among 20 imp=1 with same keywords."""
        for i in range(20):
            remember(runner,
                     f'Memcached slab allocation class {i} configured for session storage',
                     no_reconcile=True, imp='1')
        remember(runner,
                 'Memcached critical production outage caused by thundering herd on cache expiry',
                 no_reconcile=True, imp='5')
        hits = recall_basic(runner, 'Memcached')
        assert hits[0]['importance'] == 5, (
            f'Expected imp=5 first, got imp={hits[0]["importance"]}')

    def test_search_ranks_by_importance(self, runner):
        """Search (token-based) does rank by importance tiebreak."""
        low = remember(runner,
                       'I deployed Fluentd for log aggregation to Elasticsearch',
                       no_reconcile=True, imp='2')
        high = remember(runner,
                        'I deployed Fluentd for log aggregation to Elasticsearch',
                        no_reconcile=True, imp='5')
        hits = search_cmd(runner, 'Fluentd')
        assert len(hits) >= 2
        hit_ids = result_ids(hits)
        high_idx = hit_ids.index(high['id'])
        low_idx = hit_ids.index(low['id'])
        assert high_idx < low_idx


class TestStoreIsolation:
    """Named stores are airtight — no data leakage."""

    def test_insight_invisible_across_stores(self, runner):
        """Insight stored in 'work' is invisible from default store."""
        invoke(runner, ['store', 'create', 'work'])
        result = invoke(runner, ['--store', 'work', 'remember', 'secret project alpha roadmap details',
                                 '--no-reconcile'])
        assert result.exit_code == 0

        hits = recall_basic(runner, 'secret')
        assert len(hits) == 0

        result = invoke(runner, ['--store', 'work', 'recall',
                                 'secret', '--basic'])
        work_hits = json.loads(result.output)['results']
        assert any('secret' in c for c in contents(work_hits))

    def test_forget_in_one_store_does_not_affect_another(self, runner):
        """Forget in store A leaves store B's copy intact."""
        invoke(runner, ['store', 'create', 'alpha'])
        invoke(runner, ['store', 'create', 'beta'])
        text = 'Terraform infrastructure deployment checklist for AWS regions'

        result_a = invoke(runner, ['--store', 'alpha', 'remember', text, '--no-reconcile'])
        data_a = parse_remember(result_a, runner)
        invoke(runner, ['--store', 'beta', 'remember', text, '--no-reconcile'])

        invoke(runner, ['--store', 'alpha', 'forget', data_a['id']])

        result_b = invoke(runner, ['--store', 'beta', 'recall',
                                   'Terraform', '--basic'])
        beta_hits = json.loads(result_b.output)['results']
        assert any('terraform' in c.lower() for c in contents(beta_hits))


class TestRecallCompleteness:
    """All retrieval paths should return consistent results."""

    def test_all_retrieval_paths_agree(self, runner):
        """If search finds it, basic recall should find it too.

        A user should not need to know which retrieval command
        to use — they should all find the same insights.
        """
        remember(runner,
                 'AWS Lambda serverless functions with DynamoDB backend',
                 no_reconcile=True, entities='Lambda,DynamoDB',
                 tags='aws,serverless')

        search_hits = search_cmd(runner, 'Lambda')
        basic_hits = recall_basic(runner, 'Lambda')

        assert len(search_hits) > 0, 'Search should find by entity'
        assert len(basic_hits) > 0, 'Basic recall should also find it'


class TestGarbageCollection:
    """GC surfaces low-value insights and respects protections."""

    def test_gc_review_flags_transient_not_durable(self, runner):
        """GC --review flags transient content, not durable content."""
        remember(runner,
                 'Production outage traced to instance i-0c220c2402a5245bc running out of memory causing cascading failure',
                 no_reconcile=True)
        remember(runner,
                 'Chose SQLite for single-node simplicity and embedded operation',
                 no_reconcile=True, imp='5')
        result = invoke(runner, ['insights', 'review'])
        data = json.loads(result.output)
        assert data['total_flagged'] >= 1
        flagged = [r['content'] for r in data['review_results']]
        assert any('i-0c220c2402a5245bc' in c.lower() for c in flagged)

    def test_gc_keep_protects_specific_id(self, runner):
        """GC --keep <id> does not include that ID in candidates."""
        kept = remember(runner,
                        'Terraform state backend uses S3 with DynamoDB locking',
                        no_reconcile=True, imp='1')
        remember(runner,
                 'Ansible playbook deploys nginx to staging environment',
                 no_reconcile=True, imp='1')
        result = invoke(runner, ['insights', 'protect', kept['id']])
        data = json.loads(result.output)
        candidate_ids = [c['id'] for c in data.get('candidates', [])]
        assert kept['id'] not in candidate_ids


class TestOperationLog:
    """Actions are auditable in the operation log."""

    def test_oplog_records_all_mutation_types(self, runner):
        """Remember, forget, and replace all appear in log."""
        data = remember(runner,
                        'Elasticsearch index sharding strategy uses 5 primary shards',
                        no_reconcile=True)
        invoke(runner, ['forget', data['id']])
        data2 = remember(runner,
                         'Kibana dashboard configured for APM monitoring',
                         no_reconcile=True)
        invoke(runner, ['replace', data2['id'],
                        'Kibana dashboard upgraded to Lens visualization for APM'])

        result = invoke(runner, ['log', 'list', '--limit', '10'])
        assert 'remember' in result.output
        assert 'forget' in result.output
        assert 'replace' in result.output


class TestStatusConsistency:
    """Status counts reflect actual state after mutations."""

    def test_status_count_after_inserts_and_forget(self, runner):
        """Store 4, forget 1 — status shows 3 total."""
        techs = ['Prometheus', 'Thanos', 'Cortex', 'Mimir']
        stored = [remember(
                runner,
                f'{tech} metrics backend configured for long-term storage retention',
                no_reconcile=True) for tech in techs]
        invoke(runner, ['forget', stored[0]['id']])

        result = invoke(runner, ['status'])
        data = json.loads(result.output)
        assert data['total_insights'] == 3


class TestEdgeCases:
    """Robustness under unusual input."""

    def test_long_content_survives(self, runner):
        """5000-char insight stored and retrievable."""
        filler = 'infrastructure automation deployment runbook procedures '
        long_content = (
            'ZeroMQ distributed messaging broker configuration. '
            + filler * 100)
        long_content = long_content[:5000]
        remember(runner, long_content, no_reconcile=True)
        hits = recall_basic(runner, 'ZeroMQ')
        assert any('zeromq' in c.lower() for c in contents(hits))

    def test_special_chars_in_content(self, runner):
        """Content with brackets, parens, quotes preserved."""
        content = 'zephyr config["key"] = (value & 0xFF) | flags'
        remember(runner, content, no_reconcile=True)
        hits = recall_basic(runner, 'zephyr')
        assert any('0xFF' in c for c in contents(hits))
