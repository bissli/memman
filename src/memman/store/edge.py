"""Edge CRUD and traversal queries."""

import logging

from memman.model import Edge, format_timestamp, parse_timestamp

logger = logging.getLogger('memman')


def insert_edge(db: 'DB', e: Edge) -> None:
    """Insert or update an edge, keeping the higher weight.

    Edges with created_by 'claude' or 'manual' are protected: their
    metadata is never overwritten by auto-generated edges.
    """
    db._exec(
        'INSERT INTO edges'
        ' (source_id, target_id, edge_type, weight, metadata, created_at)'
        ' VALUES (?, ?, ?, ?, ?, ?)'
        ' ON CONFLICT(source_id, target_id, edge_type)'
        ' DO UPDATE SET metadata = CASE'
        "                  WHEN json_extract(metadata, '$.created_by')"
        "                       IN ('claude', 'manual')"
        '                  THEN metadata'
        '                  WHEN excluded.weight >= weight'
        '                  THEN excluded.metadata'
        '                  ELSE metadata END,'
        '              weight = MAX(weight, excluded.weight)',
        (e.source_id, e.target_id, e.edge_type, e.weight,
         e.metadata_json(), format_timestamp(e.created_at)))


def get_edges_by_node(db: 'DB', node_id: str) -> list[Edge]:
    """Return all edges where the given node is source or target."""
    rows = db._query(
        'SELECT source_id, target_id, edge_type, weight,'
        ' metadata, created_at'
        ' FROM edges WHERE source_id = ?'
        ' UNION ALL'
        ' SELECT source_id, target_id, edge_type, weight,'
        ' metadata, created_at'
        ' FROM edges WHERE target_id = ? AND source_id != ?',
        (node_id, node_id, node_id)).fetchall()
    return [_scan_edge(r) for r in rows]


def get_edges_by_node_and_type(
        db: 'DB', node_id: str, edge_type: str) -> list[Edge]:
    """Return edges for a node filtered by edge type."""
    rows = db._query(
        'SELECT source_id, target_id, edge_type, weight,'
        ' metadata, created_at'
        ' FROM edges WHERE source_id = ? AND edge_type = ?'
        ' UNION ALL'
        ' SELECT source_id, target_id, edge_type, weight,'
        ' metadata, created_at'
        ' FROM edges WHERE target_id = ? AND edge_type = ?'
        ' AND source_id != ?',
        (node_id, edge_type, node_id, edge_type, node_id)).fetchall()
    return [_scan_edge(r) for r in rows]


def get_edges_by_source_and_type(
        db: 'DB', source_id: str, edge_type: str) -> list[Edge]:
    """Return edges where the given node is source, filtered by type."""
    rows = db._query(
        'SELECT source_id, target_id, edge_type, weight,'
        ' metadata, created_at'
        ' FROM edges WHERE source_id = ? AND edge_type = ?',
        (source_id, edge_type)).fetchall()
    return [_scan_edge(r) for r in rows]


def find_insights_with_entity(
        db: 'DB', entity: str, exclude_id: str,
        limit: int) -> list[str]:
    """Return insight IDs that have the given entity."""
    rows = db._query(
        'SELECT DISTINCT i.id FROM insights i, json_each(i.entities) je'
        ' WHERE i.deleted_at IS NULL AND i.id != ?'
        ' AND LOWER(TRIM(je.value)) = ?'
        ' ORDER BY i.created_at DESC LIMIT ?',
        (exclude_id, entity.strip().lower(), limit)).fetchall()
    return [r[0] for r in rows]


def count_insights_with_entity(
        db: 'DB', entity: str, exclude_id: str) -> int:
    """Count distinct insights that contain the given entity."""
    row = db._query(
        'SELECT COUNT(DISTINCT i.id)'
        ' FROM insights i, json_each(i.entities) je'
        ' WHERE i.deleted_at IS NULL AND i.id != ?'
        ' AND LOWER(TRIM(je.value)) = ?',
        (exclude_id, entity.strip().lower())).fetchone()
    return row[0] if row else 0


def get_all_edges(db: 'DB') -> list[Edge]:
    """Return all edges in the graph."""
    rows = db._query(
        'SELECT source_id, target_id, edge_type, weight,'
        ' metadata, created_at FROM edges').fetchall()
    return [_scan_edge(r) for r in rows]


def delete_edges_by_node(db: 'DB', node_id: str) -> None:
    """Remove all edges referencing a node."""
    db._exec(
        'DELETE FROM edges WHERE source_id = ? OR target_id = ?',
        (node_id, node_id))


_PER_NODE_CREATED_BY_FILTER = {
    'entity': ("(json_extract(metadata, '$.created_by') IS NULL"
               " OR json_extract(metadata, '$.created_by')"
               " NOT IN ('claude', 'manual'))"),
    'semantic': ("(json_extract(metadata, '$.created_by') IS NULL"
                 " OR json_extract(metadata, '$.created_by') = 'auto')"),
    'causal': "json_extract(metadata, '$.created_by') = 'llm'",
    }

_REINDEX_CREATED_BY_FILTER = {
    'semantic': "json_extract(metadata, '$.created_by') = 'auto'",
    'entity': ("(json_extract(metadata, '$.created_by') IS NULL"
               " OR json_extract(metadata, '$.created_by')"
               " NOT IN ('claude', 'manual'))"),
    'causal': ("(json_extract(metadata, '$.created_by') IS NULL"
               " OR json_extract(metadata, '$.created_by')"
               " NOT IN ('llm', 'claude', 'manual'))"),
    }


def delete_auto_edges_for_node(
        db: 'DB', node_id: str, edge_type: str) -> None:
    """Delete auto-generated edges for a node, preserving manual/claude.

    Filter varies by edge_type:
    - entity: deletes null + non-claude/manual
    - semantic: deletes null + auto
    - causal: deletes llm only
    """
    filt = _PER_NODE_CREATED_BY_FILTER[edge_type]
    db._exec(
        f'DELETE FROM edges WHERE (source_id = ? OR target_id = ?)'
        f' AND edge_type = ? AND {filt}',
        (node_id, node_id, edge_type))


def delete_auto_edges_by_type(db: 'DB', edge_type: str) -> None:
    """Delete auto-generated edges globally for reindex.

    Filter varies by edge_type:
    - semantic: deletes auto only
    - entity: deletes null + non-claude/manual
    - causal: deletes null + non-llm/claude/manual (heuristic)
    """
    filt = _REINDEX_CREATED_BY_FILTER[edge_type]
    db._exec(
        f'DELETE FROM edges WHERE edge_type = ? AND {filt}',
        (edge_type,))


def count_auto_edges_by_type(db: 'DB', edge_type: str) -> int:
    """Count auto-generated edges by type using reindex filters."""
    filt = _REINDEX_CREATED_BY_FILTER[edge_type]
    row = db._query(
        f'SELECT COUNT(*) FROM edges WHERE edge_type = ? AND {filt}',
        (edge_type,)).fetchone()
    return row[0] if row else 0


def delete_low_weight_temporal_proximity(
        db: 'DB', min_weight: float) -> None:
    """Delete temporal proximity edges below min_weight threshold."""
    db._exec(
        "DELETE FROM edges WHERE edge_type = 'temporal'"
        " AND json_extract(metadata, '$.sub_type') = 'proximity'"
        ' AND weight < ?',
        (min_weight,))


def count_low_weight_temporal_proximity(
        db: 'DB', min_weight: float) -> int:
    """Count temporal proximity edges below min_weight threshold."""
    row = db._query(
        "SELECT COUNT(*) FROM edges WHERE edge_type = 'temporal'"
        " AND json_extract(metadata, '$.sub_type') = 'proximity'"
        ' AND weight < ?',
        (min_weight,)).fetchone()
    return row[0] if row else 0


def _scan_edge(row: tuple) -> Edge:
    """Parse a database row into an Edge dataclass."""
    e = Edge()
    e.source_id = row[0]
    e.target_id = row[1]
    e.edge_type = row[2]
    e.weight = row[3]
    e.parse_metadata(row[4])
    e.created_at = parse_timestamp(row[5])
    return e
