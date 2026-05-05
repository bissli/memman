"""Edge CRUD and traversal queries."""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from memman.store.model import Edge, format_timestamp, parse_timestamp

if TYPE_CHECKING:
    from memman.store.db import DB

logger = logging.getLogger('memman')


def insert_edge(db: 'DB', e: Edge) -> None:
    """Insert or update an edge, keeping the higher weight.

    Edges with created_by 'claude' or 'manual' are protected: their
    metadata is never overwritten by auto-generated edges.

    Stamps `created_at` server-side: caller-passed `e.created_at`
    is IGNORED. Mirrors `PostgresEdgeStore.upsert` which relies on
    `DEFAULT now()`.
    """
    created_at = datetime.now(timezone.utc)
    sql = """
insert into edges
    (source_id, target_id, edge_type, weight, metadata, created_at)
values (?, ?, ?, ?, ?, ?)
on conflict(source_id, target_id, edge_type) do update set
    metadata = case
        when json_extract(metadata, '$.created_by') in ('claude', 'manual')
            then metadata
        when excluded.weight >= weight
            then excluded.metadata
        else metadata
    end,
    weight = max(weight, excluded.weight)
"""
    db._exec(sql, (
        e.source_id, e.target_id, e.edge_type, e.weight,
        e.metadata_json(), format_timestamp(created_at)))


_EDGE_COLUMNS = 'source_id, target_id, edge_type, weight, metadata, created_at'


def get_edges_by_node(db: 'DB', node_id: str) -> list[Edge]:
    """Return all edges where the given node is source or target."""
    sql = f"""
select {_EDGE_COLUMNS}
from edges
where source_id = ?
union all
select {_EDGE_COLUMNS}
from edges
where target_id = ? and source_id != ?
"""
    rows = db._query(sql, (node_id, node_id, node_id)).fetchall()
    return [_scan_edge(r) for r in rows]


def get_edges_by_node_and_type(
        db: 'DB', node_id: str, edge_type: str) -> list[Edge]:
    """Return edges for a node filtered by edge type."""
    sql = f"""
select {_EDGE_COLUMNS}
from edges
where source_id = ? and edge_type = ?
union all
select {_EDGE_COLUMNS}
from edges
where target_id = ? and edge_type = ? and source_id != ?
"""
    rows = db._query(
        sql, (node_id, edge_type, node_id, edge_type, node_id)
        ).fetchall()
    return [_scan_edge(r) for r in rows]


def get_edges_by_source_and_type(
        db: 'DB', source_id: str, edge_type: str) -> list[Edge]:
    """Return edges where the given node is source, filtered by type."""
    sql = f"""
select {_EDGE_COLUMNS}
from edges
where source_id = ? and edge_type = ?
"""
    rows = db._query(sql, (source_id, edge_type)).fetchall()
    return [_scan_edge(r) for r in rows]


def find_insights_with_entity(
        db: 'DB', entity: str, exclude_id: str,
        limit: int) -> list[str]:
    """Return insight IDs that have the given entity."""
    sql = """
select distinct i.id
from insights i, json_each(i.entities) je
where i.deleted_at is null
  and i.id != ?
  and lower(trim(je.value)) = ?
order by i.created_at desc
limit ?
"""
    rows = db._query(
        sql, (exclude_id, entity.strip().lower(), limit)).fetchall()
    return [r[0] for r in rows]


def count_insights_with_entity(
        db: 'DB', entity: str, exclude_id: str) -> int:
    """Count distinct insights that contain the given entity."""
    sql = """
select count(distinct i.id)
from insights i, json_each(i.entities) je
where i.deleted_at is null
  and i.id != ?
  and lower(trim(je.value)) = ?
"""
    row = db._query(sql, (exclude_id, entity.strip().lower())).fetchone()
    return row[0] if row else 0


def get_all_edges(db: 'DB') -> list[Edge]:
    """Return all edges in the graph."""
    sql = f'select {_EDGE_COLUMNS} from edges'
    rows = db._query(sql).fetchall()
    return [_scan_edge(r) for r in rows]


def delete_edges_by_node(db: 'DB', node_id: str) -> None:
    """Remove all edges referencing a node."""
    db._exec(
        'delete from edges where source_id = ? or target_id = ?',
        (node_id, node_id))


_PER_NODE_CREATED_BY_FILTER = {
    'entity': ("(json_extract(metadata, '$.created_by') is null"
               " or json_extract(metadata, '$.created_by')"
               " not in ('claude', 'manual'))"),
    'semantic': ("(json_extract(metadata, '$.created_by') is null"
                 " or json_extract(metadata, '$.created_by') = 'auto')"),
    'causal': "json_extract(metadata, '$.created_by') = 'llm'",
    }

_REINDEX_CREATED_BY_FILTER = {
    'semantic': "json_extract(metadata, '$.created_by') = 'auto'",
    'entity': ("(json_extract(metadata, '$.created_by') is null"
               " or json_extract(metadata, '$.created_by')"
               " not in ('claude', 'manual'))"),
    'causal': ("(json_extract(metadata, '$.created_by') is null"
               " or json_extract(metadata, '$.created_by')"
               " not in ('llm', 'claude', 'manual'))"),
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
    sql = f"""
delete from edges
where (source_id = ? or target_id = ?)
  and edge_type = ?
  and {filt}
"""
    db._exec(sql, (node_id, node_id, edge_type))


def delete_auto_edges_by_type(db: 'DB', edge_type: str) -> None:
    """Delete auto-generated edges globally for reindex.

    Filter varies by edge_type:
    - semantic: deletes auto only
    - entity: deletes null + non-claude/manual
    - causal: deletes null + non-llm/claude/manual (heuristic)
    """
    filt = _REINDEX_CREATED_BY_FILTER[edge_type]
    db._exec(
        f'delete from edges where edge_type = ? and {filt}',
        (edge_type,))


def count_auto_edges_by_type(db: 'DB', edge_type: str) -> int:
    """Count auto-generated edges by type using reindex filters."""
    filt = _REINDEX_CREATED_BY_FILTER[edge_type]
    row = db._query(
        f'select count(*) from edges where edge_type = ? and {filt}',
        (edge_type,)).fetchone()
    return row[0] if row else 0


def delete_low_weight_temporal_proximity(
        db: 'DB', min_weight: float) -> None:
    """Delete temporal proximity edges below min_weight threshold."""
    sql = """
delete from edges
where edge_type = 'temporal'
  and json_extract(metadata, '$.sub_type') = 'proximity'
  and weight < ?
"""
    db._exec(sql, (min_weight,))


def count_low_weight_temporal_proximity(
        db: 'DB', min_weight: float) -> int:
    """Count temporal proximity edges below min_weight threshold."""
    sql = """
select count(*) from edges
where edge_type = 'temporal'
  and json_extract(metadata, '$.sub_type') = 'proximity'
  and weight < ?
"""
    row = db._query(sql, (min_weight,)).fetchone()
    return row[0] if row else 0


def get_edge_weight(
        db: 'DB', source_id: str, target_id: str,
        edge_type: str) -> float | None:
    """Return the weight of one directed edge, or None if absent.

    Used by the `memman link` CLI command for the
    "what was the existing weight before / after my upsert" probe.
    """
    sql = """
select weight from edges
where source_id = ? and target_id = ? and edge_type = ?
"""
    row = db._query(sql, (source_id, target_id, edge_type)).fetchone()
    return row[0] if row else None


def count_dangling_by_type(db: 'DB') -> dict[str, int]:
    """Return {edge_type: count} for edges referencing missing or deleted nodes.

    Used by `doctor.check_dangling_edges`. Keeps the set-difference
    inside the database via a `not exists` subquery.
    """
    sql = """
select e.edge_type, count(*)
from edges e
where not exists (
    select 1 from insights i
    where i.id = e.source_id and i.deleted_at is null
)
   or not exists (
    select 1 from insights i
    where i.id = e.target_id and i.deleted_at is null
)
group by e.edge_type
"""
    rows = db._query(sql).fetchall()
    return {r[0]: r[1] for r in rows}


def degree_distribution(db: 'DB') -> dict[str, int]:
    """Return {insight_id: total_degree} for all active insights.

    Active insights with zero degree are included with value 0.
    Used by `doctor.check_edge_degree`.
    """
    id_rows = db._query(
        'select id from insights where deleted_at is null').fetchall()
    if not id_rows:
        return {}
    degree_sql = """
select id, sum(cnt) as degree
from (
    select source_id as id, count(*) as cnt
    from edges
    group by source_id
    union all
    select target_id as id, count(*) as cnt
    from edges
    group by target_id
)
group by id
"""
    degree_rows = db._query(degree_sql).fetchall()
    by_id = {r[0]: r[1] for r in degree_rows}
    return {r[0]: by_id.get(r[0], 0) for r in id_rows}


def _scan_edge(row: tuple[Any, ...]) -> Edge:
    """Parse a database row into an Edge dataclass."""
    e = Edge()
    e.source_id = row[0]
    e.target_id = row[1]
    e.edge_type = row[2]
    e.weight = row[3]
    e.parse_metadata(row[4])
    e.created_at = parse_timestamp(row[5])
    return e
