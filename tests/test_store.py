"""Store layer tests ported from Go store_test.go."""


from memman.store.db import DEFAULT_STORE_NAME, list_local_store_dirs, open_db
from memman.store.db import read_active, store_dir, store_exists
from memman.store.db import valid_store_name, write_active
from memman.store.node import count_active_insights, get_all_active_insights
from memman.store.node import get_insight_by_id
from memman.store.node import get_insight_by_id_include_deleted
from memman.store.node import insert_insight, query_insights
from memman.store.node import soft_delete_insight
from memman.store.oplog import get_oplog, log_op
from tests.conftest import make_insight

# --- Insight CRUD ---


class TestInsertAndGetInsight:
    """Insert and verify round-trip."""

    def test_insert_and_get(self, tmp_db):
        """Insert insight, retrieve by id, verify content round-trips.

        Mutation: `get_insight_by_id` reading a stale or truncated
            content column.
        Oracle: the content string the insight was built with.
        """
        ins = make_insight(id='ins-1', content='Go uses SQLite for storage')
        insert_insight(tmp_db, ins)

        got = get_insight_by_id(tmp_db, 'ins-1')
        assert got is not None
        assert got.content == ins.content


class TestGetInsightByIDNotFound:
    """Nonexistent ID returns None."""

    def test_not_found(self, tmp_db):
        """get_insight_by_id returns None for missing id."""
        got = get_insight_by_id(tmp_db, 'nonexistent')
        assert got is None


class TestSoftDeleteInsight:
    """Soft delete hides an insight from the active reads."""

    def test_soft_delete(self, tmp_db):
        """Verify not found via get, found via include_deleted."""
        ins = make_insight(id='del-1', content='to be deleted')
        insert_insight(tmp_db, ins)

        soft_delete_insight(tmp_db, 'del-1')

        assert get_insight_by_id(tmp_db, 'del-1') is None

        got = get_insight_by_id_include_deleted(tmp_db, 'del-1')
        assert got is not None
        assert got.deleted_at is not None


# --- Query ---


class TestQueryInsightsFilters:
    """Keyword filter."""

    def test_keyword_filter(self, tmp_db):
        """Keyword filter matches content via LIKE."""
        insert_insight(tmp_db, make_insight(
            id='q-1', content='Go language features', category='fact'))
        insert_insight(tmp_db, make_insight(
            id='q-2', content='Python web framework', category='decision'))
        insert_insight(tmp_db, make_insight(
            id='q-3', content='Go concurrency patterns', category='fact'))

        results = query_insights(tmp_db, keyword='Go')
        assert len(results) == 2


# --- Edges ---


# --- Oplog ---


class TestOplog:
    """Operation log insert and retrieval."""

    def test_log_and_get(self, tmp_db):
        """Log two operations, verify order and fields."""
        log_op(tmp_db, 'remember', 'ins-1', 'test detail')
        log_op(tmp_db, 'recall', '', 'query: test')

        entries = get_oplog(tmp_db, 10)
        assert len(entries) == 2
        assert entries[0]['operation'] == 'recall'
        assert entries[1]['operation'] == 'remember'


# --- Embedding ---


# --- GetAllActiveInsights ---


class TestGetAllActiveInsights:
    """Active insights excludes soft-deleted."""

    def test_excludes_deleted(self, tmp_db):
        """Soft-deleted insight is not returned."""
        insert_insight(tmp_db, make_insight(id='all-1', content='a'))
        insert_insight(tmp_db, make_insight(id='all-2', content='b'))
        insert_insight(tmp_db, make_insight(id='all-3', content='c'))
        soft_delete_insight(tmp_db, 'all-2')

        all_active = get_all_active_insights(tmp_db)
        assert len(all_active) == 2


# --- Store management ---


class TestValidStoreName:
    """Regex-based store name validation."""

    def test_valid_names(self):
        """Accepted name patterns."""
        assert valid_store_name('default') is True
        assert valid_store_name('my-store') is True
        assert valid_store_name('work_2024') is True
        assert valid_store_name('A') is True
        assert valid_store_name('a1') is True

    def test_invalid_names(self):
        """Rejected name patterns."""
        assert valid_store_name('') is False
        assert valid_store_name('-bad') is False
        assert valid_store_name('_bad') is False
        assert valid_store_name('has space') is False
        assert valid_store_name('has/slash') is False
        assert valid_store_name('has.dot') is False
        assert valid_store_name('.hidden') is False


class TestReadWriteActive:
    """Active store name persistence."""

    def test_default_when_missing(self, tmp_path):
        """No active file returns default store name."""
        got = read_active(str(tmp_path))
        assert got == DEFAULT_STORE_NAME

    def test_write_and_read(self, tmp_path):
        """Written name is read back correctly."""
        base = str(tmp_path)
        write_active(base, 'work')
        got = read_active(base)
        assert got == 'work'


class TestListStores:
    """Enumerate store directories."""

    def test_empty(self, tmp_path):
        """No data dir returns empty list."""
        names = list_local_store_dirs(str(tmp_path))
        assert len(names) == 0

    def test_two_stores(self, tmp_path):
        """Two created stores returned sorted."""
        base = str(tmp_path)
        db1 = open_db(store_dir(base, 'alpha'))
        db1.close()
        db2 = open_db(store_dir(base, 'beta'))
        db2.close()

        names = list_local_store_dirs(base)
        assert len(names) == 2
        assert names[0] == 'alpha'
        assert names[1] == 'beta'


class TestStoreExists:
    """Check existence of named store directory."""

    def test_does_not_exist(self, tmp_path):
        """Missing store returns False."""
        assert store_exists(str(tmp_path), 'nope') is False

    def test_exists_after_open(self, tmp_path):
        """Store exists after open_db creates it."""
        base = str(tmp_path)
        db = open_db(store_dir(base, 'yes'))
        db.close()
        assert store_exists(base, 'yes') is True


# --- CountActiveInsights ---


class TestCountActiveInsights:
    """Count non-deleted insights."""

    def test_count(self, tmp_db):
        """Returns correct count excluding deleted."""
        insert_insight(tmp_db, make_insight(id='cnt-1', content='a'))
        insert_insight(tmp_db, make_insight(id='cnt-2', content='b'))
        insert_insight(tmp_db, make_insight(id='cnt-3', content='c'))
        soft_delete_insight(tmp_db, 'cnt-2')

        assert count_active_insights(tmp_db) == 2

    def test_empty(self, tmp_db):
        """Empty DB returns zero."""
        assert count_active_insights(tmp_db) == 0


# --- CountInsightsWithEntity ---


class TestEnrichmentSchema:
    """Verify enrichment columns exist in fresh databases."""

    def test_new_columns_in_schema(self, tmp_db):
        """Fresh DB has summary, and no semantic_facts.

        Mutation: leaving `semantic_facts` in `_BASELINE_SCHEMA`
            after the enrichment prompt stops populating it, so a
            fresh store still carries a column no write or read
            touches.
        Oracle: `PRAGMA table_info`, read straight off a fresh store.
        """
        cols = tmp_db._conn.execute(
            'PRAGMA table_info(insights)').fetchall()
        col_names = {row[1] for row in cols}
        assert 'summary' in col_names
        assert 'semantic_facts' not in col_names


class TestPendingEnrichIndex:
    """Partial index on (enrich_attempted_at, created_at) for the
    pending-enrich scan.
    """

    def test_pending_enrich_query_uses_index(self, tmp_path):
        """Verify the scheduler's pending-enrich scan is served by its
        index.

        Mutation: dropping `created_at` from `idx_insights_pending_enrich`
            (the planner then takes `idx_insights_current_listing` and
            sorts every current row per tick), or dropping
            `superseded_by is null` from its predicate (the partial
            index no longer matches the query and is unusable).
        Oracle: sqlite's own `explain query plan` naming the partial
            index and reporting no sort step.
        """
        from memman.store.db import open_db
        db = open_db(str(tmp_path))
        try:
            plan = db._conn.execute(
                'explain query plan select id from insights'
                ' where enrich_attempted_at is null and deleted_at is null'
                ' and superseded_by is null'
                ' order by created_at asc limit 10'
                ).fetchall()
        finally:
            db.close()
        steps = ' | '.join(row[3] for row in plan)
        assert 'idx_insights_pending_enrich' in steps, plan
        assert 'ORDER BY' not in steps, plan


class TestSchemaColumnRename:
    """The insights table carries `enrich_attempted_at`, never `linked_at`,
    on a freshly opened store of either backend.
    """

    def test_schema_has_enrich_attempted_at(self, backend, backend_kind):
        """A freshly opened store has the renamed column and index.

        Mutation: reverting `enrich_attempted_at` to `linked_at` or
            `idx_insights_pending_enrich` to `idx_insights_pending_link`
            in either backend's baseline, or keeping the old name
            beside the new one.
        Oracle: the backend's own catalog (`pragma table_info` /
            `sqlite_master` on SQLite, `information_schema` /
            `pg_indexes` on Postgres), read directly rather than
            through the ORM-style accessors in `store/node.py`.
        """
        if backend_kind == 'sqlite':
            db = backend.nodes._db
            cols = {row[1] for row in db._conn.execute(
                'pragma table_info(insights)').fetchall()}
            index_names = {row[0] for row in db._conn.execute(
                "select name from sqlite_master where type = 'index'"
                ).fetchall()}
            assert 'enrich_attempted_at' in cols
            assert 'linked_at' not in cols
            assert 'idx_insights_pending_enrich' in index_names
            assert 'idx_insights_pending_link' not in index_names
        else:
            schema = backend.nodes._schema
            with backend.nodes._conn.cursor() as cur:
                cur.execute(
                    'select column_name from information_schema.columns'
                    ' where table_schema = %s and table_name = %s',
                    (schema, 'insights'))
                cols = {row[0] for row in cur.fetchall()}
                cur.execute(
                    'select indexname from pg_indexes'
                    ' where schemaname = %s', (schema,))
                index_names = {row[0] for row in cur.fetchall()}
            assert 'enrich_attempted_at' in cols
            assert 'linked_at' not in cols
            assert any(
                n.startswith('idx_insights_pending_enrich')
                for n in index_names)
            assert not any(
                n.startswith('idx_insights_pending_link')
                for n in index_names)
