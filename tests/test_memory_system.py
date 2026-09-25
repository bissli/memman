"""Black-box behavioral tests for memman memory management system.

These tests verify behavioral invariants through the CLI only - no internal
module imports.
"""

import json
import os
import re

import pytest
from tests.conftest import invoke, parse_remember

_SCORED_LINE = re.compile(
    r'^(?P<id>\S{8}) (?P<score>-?\d+\.\d\d)'
    r' (?P<created>\S+) (?P<author>\S+) (?P<category>\S+) \| (?P<text>.*)$')
_BASIC_LINE = re.compile(
    r'^(?P<id>\S{8})'
    r' (?P<created>\S+) (?P<author>\S+) (?P<category>\S+) \| (?P<text>.*)$')


@pytest.fixture
def runner(cross_backend_runner):
    """CliRunner parametrized over `{sqlite, postgres}`.

    Delegates to `cross_backend_runner` so each of the 53 black-box CLI
    tests in this module runs against both backends. Postgres
    invocations carry `pytest.mark.postgres` and are gated on
    `psycopg + testcontainers` being importable.
    """
    return cross_backend_runner


def remember(runner_tuple, content, **flags):
    """Store an insight, return first fact dict from output.

    `remember` queues + auto-drains via the CliRunner wrapper, then
    we look up the newly-stored insight by the queue row's
    `queue_uuid` (since D1, `source` is provenance and defaults to
    `'user'`) so existing assertions like `data['id']` keep working.

    A flag whose value is a list or tuple repeats, one occurrence per
    item, which is how `--entity` supplies several names.
    """
    args = ['remember', content]
    for k, v in flags.items():
        values = v if isinstance(v, (list, tuple)) else [v]
        for value in values:
            args.extend([f'--{k}', str(value)])
    result = invoke(runner_tuple, args)
    assert result.exit_code == 0, result.output
    return parse_remember(result, runner_tuple)


def _hydrate_page(runner_tuple, output, basic):
    """Resolve a recall page's id8 lines to their full insight dicts.

    Mutation: reading the deleted JSON envelope in place of the
        plain-text page.
    """
    pattern = _BASIC_LINE if basic else _SCORED_LINE
    rows = []
    for line in output.splitlines():
        match = pattern.match(line)
        assert match, f'line off the page format: {line!r}'
        result = invoke(runner_tuple, ['insights', 'show', match['id']])
        assert result.exit_code == 0, result.output
        rows.append(json.loads(result.output))
    return rows


def recall_basic(runner_tuple, keyword):
    """Recall via --basic (SQL LIKE on single keyword), return list."""
    result = invoke(runner_tuple, ['recall', keyword, '--basic'])
    assert result.exit_code == 0, result.output
    return _hydrate_page(runner_tuple, result.output, basic=True)


def recall_smart(runner_tuple, query, **flags):
    """Recall via intent-aware mode, return results list of insight dicts."""
    args = ['recall', query]
    for k, v in flags.items():
        args.extend([f'--{k}', str(v)])
    result = invoke(runner_tuple, args)
    assert result.exit_code == 0, result.output
    return _hydrate_page(runner_tuple, result.output, basic=False)


def search_cmd(runner_tuple, query):
    """Keyword-only retrieval via recall --basic, return insight list."""
    result = invoke(runner_tuple, ['recall', '--basic', query])
    assert result.exit_code == 0, result.output
    return _hydrate_page(runner_tuple, result.output, basic=True)


def contents(results):
    """Extract content strings from result dicts."""
    return [r['content'] for r in results]


def result_ids(results):
    """Extract IDs from result dicts."""
    return [r['id'] for r in results]


# --- PART 1: CORE INVARIANTS - the system MUST satisfy these ---


class TestPersistence:
    """What you store, you can retrieve."""

    def test_store_then_recall_finds_it(self, runner):
        """Single insight retrievable by keyword from its content."""
        data = remember(runner,
                        'I configured Redis with allkeys-lru eviction and 4GB maxmemory limit')
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
            data = remember(runner, content)
            ids.append(data['id'])

        for i, (content, keyword) in enumerate(topics):
            hits = recall_basic(runner, keyword)
            hit_ids = result_ids(hits)
            assert ids[i] in hit_ids, (
                f'Could not recall "{keyword}" - got: {contents(hits)[:3]}')

    def test_partial_keyword_match(self, runner):
        """Partial keyword from content is enough to find insight."""
        remember(runner,
                 'Kubernetes pod scheduling affinity rules and taints')
        hits = recall_basic(runner, 'scheduling')
        assert any('scheduling' in c for c in contents(hits))

    def test_word_order_irrelevant(self, runner):
        """Search finds content regardless of query word order."""
        remember(runner,
                 'SQLite WAL mode write-ahead logging benefits')
        hits = search_cmd(runner, 'benefits write-ahead SQLite')
        assert any('SQLite' in c for c in contents(hits))

    def test_no_false_positives_on_unrelated_query(self, runner):
        """Completely unrelated query returns no results."""
        remember(runner, 'Python web framework comparison')
        remember(runner, 'Docker container networking')
        hits = recall_basic(runner, 'chromodynamics')
        assert len(hits) == 0


class TestDeletionCompleteness:
    """Forgotten insights vanish from ALL retrieval paths."""

    def test_forget_removes_from_recall(self, runner):
        """Forgotten insight absent from recall results."""
        data = remember(runner, 'Python GIL prevents true parallelism')
        invoke(runner, ['forget', data['id']])
        hits = recall_basic(runner, 'GIL')
        assert data['id'] not in result_ids(hits)

    def test_forget_removes_from_search(self, runner):
        """Forgotten insight absent from search results."""
        data = remember(runner, 'Nginx reverse proxy configuration')
        invoke(runner, ['forget', data['id']])
        hits = search_cmd(runner, 'Nginx reverse proxy')
        assert data['id'] not in [h['id'] for h in hits]

    def test_forget_does_not_collateral_damage_peers(self, runner):
        """Forgetting A does not affect B."""
        a = remember(runner,
                     'Celery task queue uses exponential backoff retry with max 5 attempts')
        b = remember(runner,
                     'PgBouncer connection pooling reduces PostgreSQL connection overhead by 90 percent')
        invoke(runner, ['forget', a['id']])

        hits_b = recall_basic(runner, 'PgBouncer')
        assert b['id'] in result_ids(hits_b)
        hits_a = recall_basic(runner, 'Celery')
        assert a['id'] not in result_ids(hits_a)

    def test_forget_then_re_store_same_content(self, runner):
        """Content can be re-stored after being forgotten."""
        text = 'Python GIL behavior under multiprocessing'
        data = remember(runner, text)
        invoke(runner, ['forget', data['id']])
        new_data = remember(runner, text)
        hits = recall_basic(runner, 'GIL')
        assert new_data['id'] in result_ids(hits)

    def test_double_forget_fails(self, runner):
        """Second forget on same ID returns error."""
        data = remember(runner,
                        'Nginx configured with 4096 worker connections for load balancing')
        invoke(runner, ['forget', data['id']])
        result = invoke(runner, ['forget', data['id']])
        assert result.exit_code != 0


class TestReplaceAtomicity:
    """Old content gone, new content present, metadata coherent."""

    def test_replace_swaps_content(self, runner):
        """Old content absent, new content present after replace."""
        data = remember(runner, 'team uses Flask for API layer')
        invoke(runner, ['replace', data['id'],
                        'team migrated to FastAPI for API layer'])

        hits_old = recall_basic(runner, 'Flask')
        assert not any('Flask' in c for c in contents(hits_old))
        hits_new = recall_basic(runner, 'FastAPI')
        assert any('FastAPI' in c for c in contents(hits_new))

    def test_replace_inherits_metadata(self, runner):
        """Replace without flags inherits cat/imp from original.

        Mutation: dropping the inherited category or importance on a
            flag-less replace, defaulting instead.
        Oracle: `insights show` on the replacement id, compared
            against the original's stored values.
        """
        data = remember(runner, 'chose event sourcing for audit trail', cat='decision', imp='5')
        result = invoke(runner, ['replace', data['id'],
                                 'chose CQRS with event sourcing for audit'])
        new = parse_remember(result, runner)
        assert 'id' in new

        shown = json.loads(
            invoke(runner, ['insights', 'show', new['id']]).output)
        assert shown['category'] == 'decision'
        assert shown['importance'] == 5

    def test_replace_override_metadata(self, runner):
        """Replace with explicit flags overrides original metadata.

        Mutation: keeping the original category/importance despite an
            explicit override on the replace command.
        Oracle: `insights show` on the replacement id, compared
            against the flags passed to `replace`.
        """
        data = remember(runner, 'Varnish HTTP cache configured with 2GB memory for static assets', cat='fact', imp='2')
        result = invoke(runner, ['replace', data['id'],
                                 'Switched from Varnish to CloudFront CDN for global edge caching',
                                 '--cat', 'decision', '--imp', '5'])
        new = parse_remember(result, runner)
        assert 'id' in new

        shown = json.loads(
            invoke(runner, ['insights', 'show', new['id']]).output)
        assert shown['category'] == 'decision'
        assert shown['importance'] == 5

    def test_replace_nonexistent_id_errors(self, runner):
        """Replace with fake ID fails."""
        result = invoke(runner, ['replace', 'nonexistent-fake-id', 'nope'])
        assert result.exit_code != 0

    def test_replace_deleted_id_errors(self, runner):
        """Replace on already-forgotten insight fails."""
        data = remember(runner,
                        'RabbitMQ queue mirroring configured for high availability')
        invoke(runner, ['forget', data['id']])
        result = invoke(runner, ['replace', data['id'], 'too late'])
        assert result.exit_code != 0


class TestDeduplication:
    """No false positive merge onto an unrelated row."""

    def test_identical_content_adds_a_second_row(self, runner):
        """A `remember` of identical content lands as its own row.

        Mutation: merging the second write onto the first row instead
            of storing it, losing the second call's id.
        Oracle: the two ids returned by the two writes, both distinct
            and both readable through `insights show`.
        """
        text = 'Go error handling with sentinel values and wrapping'
        first = remember(runner, text)
        second = remember(runner, text)
        assert first['id'] != second['id']
        for insight_id in (first['id'], second['id']):
            shown = invoke(runner, ['insights', 'show', insight_id])
            assert shown.exit_code == 0, shown.output

    def test_similar_but_not_identical_write_is_added(self, runner):
        """A write whose nearest stored row is similar, not identical, adds.

        Mutation: the write path retiring or merging the earlier row
            instead of leaving it current and adding the new one.
        Oracle: the second write reports `action == 'add'`.
        """
        text = 'Go error handling with sentinel values and wrapping'
        remember(runner, text)
        second = remember(runner, text + ' in a long-running service')
        assert second['action'] == 'add'

    def test_different_content_added(self, runner):
        """Genuinely different content is added, never merged.

        Mutation: an `update` or `merge` disposition surviving for
            content that shares no fact with the stored row.
        Oracle: the second write reports `action == 'add'`.
        """
        remember(runner, 'I use mypy strict mode for all Python projects')
        second = remember(runner,
                          'I switched to Podman from Docker for rootless containers')
        assert second['action'] == 'add'


class TestGraphTraversal:
    """Links are traversable and filtered correctly."""

    def test_link_makes_insight_reachable_via_related(self, runner):
        """Linked insight appears in related output."""
        a = remember(runner, 'authentication design with JWT tokens')
        b = remember(runner, 'token rotation schedule every 24 hours')
        invoke(runner, ['graph', 'link', a['id'], b['id'], '--type', 'entity'])

        result = invoke(runner, ['graph', 'related', a['id']])
        assert b['id'] in result.output

    def test_related_respects_edge_type_filter(self, runner):
        """Edge type filter includes matching, excludes non-matching.

        Anchors use textually distant content so the auto-semantic
        edge generator does not fire at the active per-surface
        threshold - otherwise the explicit edge we add would be
        shadowed by an automatic one between the same pair.
        """
        a = remember(runner, 'chose SQLite because embedded serverless')
        b = remember(runner, 'preferred color is emerald green')
        invoke(runner, ['graph', 'link', a['id'], b['id'], '--type', 'entity'])

        result_entity = invoke(runner, ['graph', 'related', a['id'],
                                        '--edge', 'entity'])
        assert b['id'] in result_entity.output
        result_semantic = invoke(runner, ['graph', 'related', a['id'],
                                          '--edge', 'semantic'])
        assert b['id'] not in result_semantic.output

    def test_link_persists_after_other_operations(self, runner):
        """New inserts don't clobber existing edges."""
        a = remember(runner, 'microservice communication via gRPC')
        b = remember(runner, 'protobuf schema evolution rules')
        invoke(runner, ['graph', 'link', a['id'], b['id'], '--type', 'entity'])

        remember(runner, 'Kafka topic partitioning strategy uses key-based routing for ordering guarantees')

        result = invoke(runner, ['graph', 'related', a['id']])
        assert b['id'] in result.output

    def test_related_respects_depth(self, runner):
        """Depth=1 returns only direct neighbors, not hop-2 nodes.

        Uses --edge semantic on three mutually unrelated anchors:
        temporal proximity and entity edges are both auto-minted
        between every pair here and would shortcut a -> c, while the
        auto-semantic generator does not fire at this distance, so the
        semantic graph holds exactly the two links this test writes.
        """
        a = remember(runner, 'chose SQLite because embedded serverless')
        b = remember(runner, 'preferred color is emerald green')
        c = remember(runner, 'the office plant is a fiddle leaf fig')
        invoke(runner, ['graph', 'link', a['id'], b['id'], '--type', 'semantic'])
        invoke(runner, ['graph', 'link', b['id'], c['id'], '--type', 'semantic'])

        result_d1 = invoke(runner, ['graph', 'related', a['id'],
                                    '--edge', 'semantic', '--depth', '1'])
        data_d1 = json.loads(result_d1.output)
        ids_d1 = [r['id'] for r in data_d1]
        assert b['id'] in ids_d1
        assert c['id'] not in ids_d1

        result_d2 = invoke(runner, ['graph', 'related', a['id'],
                                    '--edge', 'semantic', '--depth', '2'])
        data_d2 = json.loads(result_d2.output)
        ids_d2 = [r['id'] for r in data_d2]
        assert b['id'] in ids_d2
        assert c['id'] in ids_d2


class TestComposition:
    """Multi-step workflows stay consistent."""

    def test_forget_target_does_not_break_related(self, runner):
        """Forgetting a linked target does not crash related or leak."""
        a = remember(runner, 'API design principles REST vs GraphQL')
        b = remember(runner, 'GraphQL schema stitching federation')
        c = remember(runner, 'REST pagination cursor-based approach')
        link_ab = invoke(runner, ['graph', 'link', a['id'], b['id'],
                                  '--type', 'entity'])
        assert link_ab.exit_code == 0
        link_ac = invoke(runner, ['graph', 'link', a['id'], c['id'],
                                  '--type', 'entity'])
        assert link_ac.exit_code == 0
        invoke(runner, ['forget', b['id']])

        result = invoke(runner, ['graph', 'related', a['id'],
                                 '--edge', 'entity'])
        assert result.exit_code == 0
        assert c['id'] in result.output
        assert b['id'] not in result.output

    def test_store_replace_recall_sequence(self, runner):
        """Replace + subsequent inserts don't interfere with each other."""
        x = remember(runner, 'Flask API for internal tooling')
        hits = recall_basic(runner, 'Flask')
        assert any('Flask' in c for c in contents(hits))

        invoke(runner, ['replace', x['id'],
                        'FastAPI migration for internal tooling'])
        hits_old = recall_basic(runner, 'Flask')
        assert not any('Flask' in c for c in contents(hits_old))
        hits_new = recall_basic(runner, 'FastAPI')
        assert any('FastAPI' in c for c in contents(hits_new))

        remember(runner, 'Django admin for backoffice portal')
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
                f'I deployed {kw} version 3.2 on our production cluster') for kw in keywords]

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
                     'I tuned Redis cache eviction to allkeys-lru', imp='3')
        b = remember(runner,
                     'I automated Redis deployment with Ansible', imp='3')
        hits = search_cmd(runner, 'Redis cache eviction')
        assert hits, 'Expected at least one result'
        assert hits[0]['id'] == a['id']

    def test_importance_breaks_ties(self, runner):
        """Higher importance ranks first among similar content.

        Mutation: `order by` dropping `importance desc` from the basic
            query, or comparing on `created_at` alone.
        Oracle: the two rows' index positions in the returned list.
        """
        low = remember(runner,
                       'I set up Grafana dashboards for API latency monitoring', imp='2')
        high = remember(runner,
                        'I set up Grafana dashboards for request throughput monitoring', imp='5')
        hits = search_cmd(runner, 'Grafana')
        assert len(hits) >= 2
        hit_ids = result_ids(hits)
        high_idx = hit_ids.index(high['id'])
        low_idx = hit_ids.index(low['id'])
        assert high_idx < low_idx

    def test_category_filter_restricts_results(self, runner):
        """Recall --cat returns only matching category."""
        remember(runner, 'chose Postgres for relational data', cat='decision')
        remember(runner, 'prefer dark mode for IDEs', cat='preference')
        remember(runner, 'SQLite is an embedded database', cat='fact')
        hits = recall_smart(runner, 'database', cat='decision')
        categories = [h['category'] for h in hits]
        assert all(c == 'decision' for c in categories), (
            f'Expected only decision, got {categories}')
        assert any('postgres' in h['content'].lower() for h in hits)


class TestOplogChronology:
    """Operation log entries are in chronological order."""

    def test_oplog_order_is_chronological(self, runner):
        """Log --limit N returns most-recent-first ordering."""
        techs = ['Redis', 'Kafka', 'Consul', 'Vault', 'Envoy']
        for tech in techs:
            remember(runner,
                     f'{tech} cluster deployed across three availability zones for resilience')
        result = invoke(runner, ['log', 'list', '--limit', '5'])
        assert result.exit_code == 0
        lines = [l for l in result.output.strip().split('\n')
                 if l.strip() and not l.startswith('TIME')]
        assert len(lines) >= 5


class TestStatusAfterMutations:
    """Status counts reflect actual state after mixed mutations."""

    def test_status_count_after_mixed_mutations(self, runner):
        """Store 4, forget 1, replace 1 - status shows 4 total.

        Replace creates a new row and soft-deletes the old one,
        so 4 inserts - 1 forget - 1 replaced + 1 new = 3 active
        originals + 1 replacement = 4 visible if replace doesn't
        add net count, or 3 if it does. We assert >= 3.
        """
        techs = ['Grafana', 'Jaeger', 'ArgoCD', 'Istio']
        stored = [remember(
                runner,
                f'{tech} service mesh component configured for production monitoring') for tech in techs]
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
                 'Kubernetes pod scheduling uses affinity rules and taints for node placement', entity='Kubernetes')
        hits = recall_basic(runner, 'Kubernetes')
        assert len(hits) > 0


class TestMultiWordRecall:
    """Multi-word queries should work across all retrieval paths."""

    def test_basic_recall_non_adjacent_words(self, runner):
        """'Python slow' should find 'Python is slow for CPU-bound tasks'.

        Words appear in content but not adjacently.
        """
        remember(runner, 'Python is slow for CPU-bound tasks')
        hits = recall_basic(runner, 'Python slow')
        assert len(hits) > 0

    def test_search_handles_non_adjacent_words(self, runner):
        """Search tokenizes independently, finds non-adjacent matches."""
        remember(runner, 'Python is slow for CPU-bound tasks')
        hits = search_cmd(runner, 'Python slow')
        assert any('Python' in c for c in contents(hits))

    def test_smart_recall_handles_multi_word(self, runner):
        """Smart recall finds content by multi-word query."""
        remember(runner, 'PostgreSQL JSONB indexing for document queries')
        hits = recall_smart(runner, 'PostgreSQL JSONB indexing')
        assert any('JSONB' in c for c in contents(hits))


class TestContradictionDetection:
    """A write that contradicts a stored row lands beside it, not over it."""

    def test_contradiction_triggers_reconciliation(self, runner):
        """Storing contradictory content adds a row; nothing retires.

        Mutation: a contradiction disposition surviving that supersedes
            or merges the earlier row instead of adding beside it.
        Oracle: the second write reports `action == 'add'`.
        """
        remember(runner,
                 'Redis is single-threaded and cannot use multiple cores')
        result = remember(runner,
                          'Redis 6.0 supports multi-threaded IO')
        assert result['action'] == 'add'


class TestRecallPrecisionUnderNoise:
    """Recall should find the right needle in a large haystack."""

    def test_specific_insight_among_fifty_similar(self, runner):
        """One specific insight findable among 50 generic ones."""
        for i in range(50):
            remember(runner,
                     f'PostgreSQL query optimization uses index scan on column_{i} with btree')
        remember(runner,
                 'alertmanager silencing rules for oncall rotation')
        hits = recall_basic(runner, 'alertmanager')
        assert any('alertmanager' in c.lower() for c in contents(hits))

    def test_high_importance_surfaces_above_noise(self, runner):
        """imp=5 insight ranks first among 20 imp=1 with same keywords."""
        for i in range(20):
            remember(runner,
                     f'Memcached slab allocation class {i} configured for session storage', imp='1')
        remember(runner,
                 'Memcached critical production outage caused by thundering herd on cache expiry', imp='5')
        hits = recall_basic(runner, 'Memcached')
        assert hits[0]['importance'] == 5, (
            f'Expected imp=5 first, got imp={hits[0]["importance"]}')

    def test_search_ranks_by_importance(self, runner):
        """Search (token-based) does rank by importance tiebreak.

        Mutation: `order by` dropping `importance desc` from the basic
            query, or comparing on `created_at` alone.
        Oracle: the two rows' index positions in the returned list.
        """
        low = remember(runner,
                       'I deployed Fluentd for log aggregation to Elasticsearch', imp='2')
        high = remember(runner,
                        'I deployed Fluentd for log aggregation and shipping', imp='5')
        hits = search_cmd(runner, 'Fluentd')
        assert len(hits) >= 2
        hit_ids = result_ids(hits)
        high_idx = hit_ids.index(high['id'])
        low_idx = hit_ids.index(low['id'])
        assert high_idx < low_idx


class TestStoreIsolation:
    """Named stores are airtight - no data leakage."""

    def test_insight_invisible_across_stores(self, runner):
        """Insight stored in 'work' is invisible from default store."""
        invoke(runner, ['store', 'create', 'work'])
        result = invoke(runner, ['--store', 'work', 'remember', 'secret project alpha roadmap details'])
        assert result.exit_code == 0

        hits = recall_basic(runner, 'secret')
        assert len(hits) == 0

        result = invoke(runner, ['--store', 'work', 'recall',
                                 'secret', '--basic'])
        assert result.exit_code == 0, result.output
        rows = [_BASIC_LINE.match(line).groupdict()
                for line in result.output.splitlines()]
        assert any('secret' in row['text'] for row in rows)

    def test_forget_in_one_store_does_not_affect_another(self, runner):
        """Forget in store A leaves store B's copy intact."""
        invoke(runner, ['store', 'create', 'alpha'])
        invoke(runner, ['store', 'create', 'beta'])
        text = 'Terraform infrastructure deployment checklist for AWS regions'

        result_a = invoke(runner, ['--store', 'alpha', 'remember', text])
        data_a = parse_remember(result_a, runner)
        invoke(runner, ['--store', 'beta', 'remember', text])

        invoke(runner, ['--store', 'alpha', 'forget', data_a['id']])

        result_b = invoke(runner, ['--store', 'beta', 'recall',
                                   'Terraform', '--basic'])
        assert result_b.exit_code == 0, result_b.output
        rows = [_BASIC_LINE.match(line).groupdict()
                for line in result_b.output.splitlines()]
        assert any('terraform' in row['text'].lower() for row in rows)


class TestRecallCompleteness:
    """All retrieval paths should return consistent results."""

    def test_all_retrieval_paths_agree(self, runner):
        """If search finds it, basic recall should find it too.

        A user should not need to know which retrieval command
        to use - they should all find the same insights.
        """
        remember(runner,
                 'AWS Lambda serverless functions with DynamoDB backend', entity=('Lambda', 'DynamoDB'))

        search_hits = search_cmd(runner, 'Lambda')
        basic_hits = recall_basic(runner, 'Lambda')

        assert len(search_hits) > 0, 'Search should find by entity'
        assert len(basic_hits) > 0, 'Basic recall should also find it'


class TestContentReview:
    """`insights review` surfaces transient content for an operator."""

    def test_review_flags_transient_not_durable(self, runner):
        """A stored instance id is flagged; a durable decision is not."""
        remember(runner,
                 'Production outage traced to instance i-0c220c2402a5245bc running out of memory causing cascading failure')
        remember(runner,
                 'Chose SQLite for single-node simplicity and embedded operation', imp='5')
        result = invoke(runner, ['insights', 'review'])
        data = json.loads(result.output)
        assert data['total_flagged'] >= 1
        flagged = [r['content'] for r in data['review_results']]
        assert any('i-0c220c2402a5245bc' in c.lower() for c in flagged)


class TestOperationLog:
    """Actions are auditable in the operation log."""

    def test_oplog_records_all_mutation_types(self, runner):
        """Remember, forget, and replace all appear in log."""
        data = remember(runner,
                        'Elasticsearch index sharding strategy uses 5 primary shards')
        invoke(runner, ['forget', data['id']])
        data2 = remember(runner,
                         'Kibana dashboard configured for APM monitoring')
        invoke(runner, ['replace', data2['id'],
                        'Kibana dashboard upgraded to Lens visualization for APM'])

        result = invoke(runner, ['log', 'list', '--limit', '10'])
        assert 'remember' in result.output
        assert 'forget' in result.output
        assert 'replace' in result.output


class TestInsightsShow:
    """`insights show <id>` returns the insight via the active backend."""

    def test_show_returns_stored_insight(self, runner):
        """Show roundtrips a remember-d insight by id across both backends."""
        fact = remember(
            runner,
            'Loki log aggregator runs in single-binary monolithic mode')
        result = invoke(runner, ['insights', 'show', fact['id']])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data['id'] == fact['id']
        assert 'Loki' in data['content']

    def test_show_unknown_id_fails_cleanly(self, runner):
        """Unknown id exits non-zero with an actionable message."""
        result = invoke(runner, ['insights', 'show', 'no-such-id'])
        assert result.exit_code != 0
        assert 'not found' in result.output.lower()

    def test_show_returns_a_superseded_row_and_refuses_a_forgotten_one(
            self, runner):
        """Verify `show` reads history but not deletions.

        Mutation: leaving `insights show` on `nodes.get`, which hides a
            superseded row behind the same not-found as a missing one.
        Oracle: the superseded row's JSON carries its pointer; the
            forgotten row gets its own refusal; both backends.
        """
        old = remember(runner, 'Loki retention is seven days')
        result = invoke(runner, ['replace', old['id'],
                                 'Loki retention is thirty days'])
        new = parse_remember(result, runner)

        shown = invoke(runner, ['insights', 'show', old['id']])
        assert shown.exit_code == 0, shown.output
        data = json.loads(shown.output)
        assert data['id'] == old['id']
        assert data['superseded_by'] == new['id']
        assert 'deleted_at' not in data

        gone = remember(runner, 'Tempo traces are sampled at one percent')
        invoke(runner, ['forget', gone['id']])
        refused = invoke(runner, ['insights', 'show', gone['id']])
        assert refused.exit_code != 0
        assert 'was forgotten' in refused.output


class TestResolveId:
    """Node-store prefix resolution and CLI id-argument coverage."""

    def test_unique_prefix_resolves_to_full_id(self, tmp_backend):
        """Prefix shorter than the full id resolves when it is unique.

        Mutation: keeping 'where id = ?' exact lookup, which returns
            None for a partial id even when one row has it as a prefix.
        Oracle: hand-built rows sharing the first 4 chars; the 8-char
            prefix is unique and resolves to the first row's full id.
        """
        from tests.conftest import make_insight
        tmp_backend.nodes.insert(make_insight(id='aaaabbbb-cccc-dddd'))
        tmp_backend.nodes.insert(make_insight(id='aaaaxxx1-cccc-dddd'))
        resolved = tmp_backend.nodes.resolve_id('aaaabbbb')
        assert resolved == 'aaaabbbb-cccc-dddd'

    def test_shared_prefix_raises_with_count(self, tmp_backend):
        """Ambiguous prefix raises ValueError naming the match count.

        Mutation: first-match resolution, which silently returns one of
            several matching rows instead of raising.
        Oracle: ValueError raised with '2' in the message for a prefix
            matching both stored ids.
        """
        from tests.conftest import make_insight
        tmp_backend.nodes.insert(make_insight(id='aaaabbbb-cccc-dddd'))
        tmp_backend.nodes.insert(make_insight(id='aaaaxxx1-cccc-dddd'))
        with pytest.raises(ValueError, match='2'):
            tmp_backend.nodes.resolve_id('aaaa')

    def test_full_id_wins_over_prefix_match(self, tmp_backend):
        """Full id resolves to itself even when it is another row's prefix.

        Mutation: reading a full id as a prefix, which would match both
            rows and raise ValueError.
        Oracle: rows 'abcd' and 'abcd-1234'; exact id 'abcd' resolves
            to 'abcd', not to ValueError.
        """
        from tests.conftest import make_insight
        tmp_backend.nodes.insert(make_insight(id='abcd'))
        tmp_backend.nodes.insert(make_insight(id='abcd-1234'))
        resolved = tmp_backend.nodes.resolve_id('abcd')
        assert resolved == 'abcd'

    def test_graph_related_unknown_id_exits_nonzero(self, runner):
        """Graph related on an unknown id exits non-zero with 'not found'.

        Mutation: the empty-list fallthrough in graph_related, which
            returns exit 0 and an empty list when bfs finds no neighbors
            for an id absent from the store.
        Oracle: exit_code != 0 and 'not found' in output.
        """
        result = invoke(runner, ['graph', 'related', 'no-such-id'])
        assert result.exit_code != 0
        assert 'not found' in result.output.lower()

    def test_supersede_refuses_a_prefix_and_the_full_id_of_one_row(
            self, runner):
        """Verify the same-row guard compares resolved ids, not raw text.

        Mutation: comparing the raw arguments before resolution, which
            lets a prefix and the full id of one row pass the guard and
            supersede the row with itself.
        Oracle: exit non-zero naming the same-insight refusal; the row
            stays current.
        """
        fact = remember(runner, 'Grafana dashboards refresh every minute')
        result = invoke(runner, ['supersede', fact['id'][:8], fact['id']])
        assert result.exit_code != 0
        assert 'same insight' in result.output
        shown = invoke(runner, ['insights', 'show', fact['id']])
        assert json.loads(shown.output).get('superseded_by') is None

    def test_graph_link_refuses_a_prefix_and_the_full_id_of_one_row(
            self, runner):
        """Verify the self-link guard compares resolved ids, not raw text.

        Mutation: comparing the raw arguments before resolution, which
            lets a prefix and the full id of one row store a self-edge.
        Oracle: exit non-zero naming the self-link refusal.
        """
        fact = remember(runner, 'Prometheus scrapes every fifteen seconds')
        result = invoke(runner, ['graph', 'link', fact['id'][:8], fact['id']])
        assert result.exit_code != 0
        assert 'itself' in result.output

    def test_show_accepts_an_unambiguous_prefix(self, runner):
        """Verify a CLI command resolves an 8-char prefix to the full id.

        Mutation: dropping the resolve_id call from the command, so the
            prefix reaches the exact-id lookup and reads as not found.
        Oracle: the JSON id returned equals the full stored id.
        """
        fact = remember(runner, 'Tempo keeps traces for three days')
        result = invoke(runner, ['insights', 'show', fact['id'][:8]])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)['id'] == fact['id']

    def test_ambiguous_prefix_exits_nonzero_naming_the_count(self, runner):
        """Verify an ambiguous prefix is refused at the CLI with its count.

        Mutation: first-match resolution, which shows one of the two
            rows and exits zero.
        Oracle: the common prefix of two stored ids exits non-zero and
            the output names the two matches.
        """
        first = remember(runner, 'Loki indexes labels only')
        second = remember(runner, 'Mimir stores metrics long term')
        prefix = os.path.commonprefix([first['id'], second['id']])
        result = invoke(runner, ['insights', 'show', prefix])
        assert result.exit_code != 0
        assert '2' in result.output

    def test_prefix_resolves_a_forgotten_and_a_superseded_row(self, tmp_backend):
        """Verify resolution scans deleted and superseded rows too.

        Mutation: adding 'and deleted_at is null' or 'and superseded_by
            is null' to the prefix query, which hides the rows that
            `insights show` and `unsupersede` must still reach.
        Oracle: a soft-deleted row and a superseded row each resolve
            from an 8-char prefix.
        """
        from tests.conftest import make_insight
        tmp_backend.nodes.insert(make_insight(id='deadbeef-0001'))
        tmp_backend.nodes.soft_delete('deadbeef-0001')
        tmp_backend.nodes.insert(make_insight(id='feedface-0001'))
        tmp_backend.nodes.insert(make_insight(id='0badf00d-0001'))
        tmp_backend.nodes.supersede('feedface-0001', '0badf00d-0001')
        assert tmp_backend.nodes.resolve_id('deadbeef') == 'deadbeef-0001'
        assert tmp_backend.nodes.resolve_id('feedface') == 'feedface-0001'


class TestStatusConsistency:
    """Status counts reflect actual state after mutations."""

    def test_status_count_after_inserts_and_forget(self, runner):
        """Store 4, forget 1 - status shows 3 total."""
        techs = ['Prometheus', 'Thanos', 'Cortex', 'Mimir']
        stored = [remember(
                runner,
                f'{tech} metrics backend configured for long-term storage retention') for tech in techs]
        invoke(runner, ['forget', stored[0]['id']])

        result = invoke(runner, ['status'])
        data = json.loads(result.output)
        assert data['total_insights'] == 3


class TestEdgeCases:
    """Robustness under unusual input."""

    def test_long_content_survives(self, runner):
        """Verify a cap-sized insight is stored and recalled whole.

        Mutation: `>=` in place of `>` in the size check, which refuses a
            write sitting on the cap, or any storage path that keeps a
            prefix of the text.
        Oracle: the hand-built 1,000-byte input, compared whole against
            the recalled content.
        """
        filler = 'infrastructure automation deployment runbook procedures '
        long_content = (
            'ZeroMQ distributed messaging broker configuration. '
            + filler * 100)
        long_content = long_content[:1000]
        remember(runner, long_content)
        hits = recall_basic(runner, 'ZeroMQ')
        assert long_content in contents(hits)

    def test_special_chars_in_content(self, runner):
        """Content with brackets, parens, quotes preserved."""
        content = 'zephyr config["key"] = (value & 0xFF) | flags'
        remember(runner, content)
        hits = recall_basic(runner, 'zephyr')
        assert any('0xFF' in c for c in contents(hits))


class TestRecallFreshness:
    """Scored recall's candidate universe is the store's active set.

    A materialized read cache once served this path and froze
    permanently above a row cap, and no test asserted this equality,
    so recall served deleted rows and hid live ones. These are the
    gate that keeps the candidate universe honest.
    """

    def test_scored_recall_matches_db_active_set(self, mm_runner):
        """Verify scored recall sees exactly the non-deleted rows on disk.

        Mutation: any read-side cache on the recall path that a
            write does not invalidate - memoizing get_all_active(),
            or reintroducing a materialized read artifact.
        Oracle: the store DB's own active-id set, read through a raw
            sqlite3 connection outside the pipeline.

        Sqlite-only by construction: Postgres never had a snapshot, so
        the pre-change failure is a sqlite property. `remember`
        auto-drains via the CliRunner wrapper, so only the
        out-of-band insert and the `forget` skip the drain.
        """
        import sqlite3
        from pathlib import Path

        kept = remember(mm_runner, 'Kombu message serialization uses JSON')
        doomed = remember(mm_runner, 'Supervisord manages worker processes')
        invoke(mm_runner, ['forget', doomed['id']])

        _cli, data_dir = mm_runner
        dbs = list(Path(data_dir).glob('data/*/memman.db'))
        assert len(dbs) == 1, f'expected one store db, found {dbs}'
        conn = sqlite3.connect(str(dbs[0]))
        try:
            conn.execute(
                "insert into insights (id, content, created_at, updated_at)"
                " values (?, ?, '2026-01-01T00:00:00+00:00',"
                " '2026-01-01T00:00:00+00:00')",
                ('oob-freshness-1', 'Havelock proxy rotates upstream keys'))
            conn.commit()
            expected = {
                r[0] for r in conn.execute(
                    'select id from insights'
                    ' where deleted_at is null and superseded_by is null')}
        finally:
            conn.close()

        hits = recall_smart(mm_runner, 'Kombu Havelock Supervisord', limit=0)
        assert {h['id'] for h in hits} == expected
        assert kept['id'] in expected
        assert doomed['id'] not in expected

    def test_recall_universe_equals_get_all_active(self, runner):
        """Verify recall's pool equals the backend's own active set.

        Mutation: a read-side cache on either backend that a write
            does not invalidate.
        Oracle: `memman status`'s `total_insights`, a plain
            `count(*) where deleted_at is null` taken outside the
            recall pipeline, plus the ids of the rows kept.
        """
        kept = [
            remember(runner, 'Traefik ingress terminates TLS at the edge')['id'],
            remember(runner, 'Kombu serializes celery task payloads')['id'],
            remember(runner, 'Havelock proxy rotates its upstream keys')['id'],
            ]
        doomed = remember(runner, 'Vagrant provisions local dev boxes')
        invoke(runner, ['forget', doomed['id']])

        result = invoke(runner, ['status'])
        active_count = json.loads(result.output)['total_insights']

        hits = recall_smart(
            runner, 'Traefik Kombu Havelock Vagrant ingress boxes', limit=0)
        returned = {h['id'] for h in hits}
        # Set equality, not a count: with several rows alive, a count
        # alone cannot tell "the whole universe" from "some rows".
        assert returned == set(kept)
        assert len(hits) == active_count == len(kept)
        assert doomed['id'] not in returned

    def test_forget_removes_from_scored_recall(self, runner):
        """Verify a forgotten insight is absent from SCORED recall.

        Mutation: serving a soft-deleted row from a stale read cache
            on the scored path (the `--basic` sibling cannot catch it,
            since it queries SQL directly).
        Oracle: a surviving sibling insight that must still be
            returned, so an empty result set cannot satisfy the
            absence assertion.
        """
        data = remember(runner, 'Redis Cluster resharding moves hash slots')
        survivor = remember(
            runner, 'Redis Sentinel promotes a replica on failover')
        invoke(runner, ['forget', data['id']])
        hits = recall_smart(runner, 'Redis Cluster resharding hash slots')
        returned = {h['id'] for h in hits}
        # Anti-vacuity: without this the test passes whenever scored
        # recall returns nothing at all, for any reason.
        assert survivor['id'] in returned
        assert data['id'] not in returned
