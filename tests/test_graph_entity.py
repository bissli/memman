"""Entity IDF weighting tests."""

from memman.graph.entity import entity_idf_weight


class TestEntityIdfWeightRare:
    """Rare entity (doc_freq=1) gets maximum weight."""

    def test_rare_entity(self):
        """Entity in 1 of 100 docs yields weight ~1.0."""
        w = entity_idf_weight(1, 100)
        assert abs(w - 1.0) < 0.01


class TestEntityIdfWeightCommon:
    """Common entity gets low weight."""

    def test_common_entity(self):
        """Entity in 80 of 100 docs yields weight < 0.2."""
        w = entity_idf_weight(80, 100)
        assert w < 0.2


class TestEntityIdfWeightUniversal:
    """Universal entity (in all docs) gets zero weight."""

    def test_universal_entity(self):
        """Entity in all docs yields weight 0.0."""
        w = entity_idf_weight(100, 100)
        assert w == 0.0


class TestEntityIdfWeightEdgeCases:
    """Edge cases for IDF weight computation."""

    def test_single_doc(self):
        """Only 1 total doc yields 0.0."""
        w = entity_idf_weight(1, 1)
        assert w == 0.0

    def test_zero_doc_freq(self):
        """doc_freq=0 (brand new entity) yields maximum weight."""
        w = entity_idf_weight(0, 100)
        assert w > 0.9

    def test_floor(self):
        """Weight never drops below 0.1 for non-universal entities."""
        w = entity_idf_weight(90, 100)
        assert w >= 0.1

    def test_monotonic(self):
        """Weight decreases as doc_freq increases."""
        w1 = entity_idf_weight(2, 100)
        w2 = entity_idf_weight(50, 100)
        w3 = entity_idf_weight(90, 100)
        assert w1 > w2 > w3
