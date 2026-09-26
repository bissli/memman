"""update_entities deduplication of case variants."""

from memman.store.node import insert_insight, update_entities
from tests.conftest import make_insight


class TestUpdateEntitiesDedup:
    """update_entities deduplicates case variants."""

    def test_dedup_case_variants(self, tmp_db):
        """Storing ['Thesis', 'thesis'] keeps only the first."""
        insert_insight(tmp_db, make_insight(
            id='ud-1', content='test'))
        update_entities(tmp_db, 'ud-1', ['Thesis', 'thesis', 'THESIS'])

        from memman.store.node import get_insight_by_id
        ins = get_insight_by_id(tmp_db, 'ud-1')
        assert len(ins.entities) == 1
        assert ins.entities[0] == 'Thesis'

    def test_dedup_preserves_distinct(self, tmp_db):
        """Distinct entities are all kept."""
        insert_insight(tmp_db, make_insight(
            id='ud-2', content='test'))
        update_entities(tmp_db, 'ud-2', ['Python', 'Go', 'Rust'])

        from memman.store.node import get_insight_by_id
        ins = get_insight_by_id(tmp_db, 'ud-2')
        assert len(ins.entities) == 3
