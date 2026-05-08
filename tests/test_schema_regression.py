"""Schema-coverage regressions for the migration payload.

Pins:
- `MigrateInsight` field set covers the postgres baseline DDL
  columns (so a new `insights` column added to `PG_BASELINE_SCHEMA`
  cannot drift past gather/apply silently).
- `MigrateInsight` field set covers the sqlite baseline schema
  columns identically.
- `PAYLOAD_VERSION` reminds reviewers to bump the wire format
  version when adding `MigrationPayload` fields.

These act as canaries against the class of bug v3 was built to
prevent: a future schema column change that compiles but silently
drops data through the migrator's apply path.
"""
from __future__ import annotations

import re

from memman.migrate import MigrateInsight, MigrationPayload, PAYLOAD_VERSION
from memman.store.db import _BASELINE_SCHEMA
from memman.store.postgres import PG_BASELINE_SCHEMA


_MIGRATE_INSIGHT_FIELDS = set(MigrateInsight.__dataclass_fields__.keys())

_MIGRATION_PAYLOAD_FIELDS = set(
    MigrationPayload.__dataclass_fields__.keys())


def _columns_under(ddl: str, table: str) -> set[str]:
    """Pull bare column identifiers from a `create table {schema}.<table>`
    DDL block. Best-effort regex; the assertions below tolerate the
    handful of pseudo-columns the migrator deliberately omits.
    """
    pattern = rf'create table[^(]*?{re.escape(table)}\s*\((.*?)\)\s*;'
    match = re.search(pattern, ddl, re.IGNORECASE | re.DOTALL)
    if not match:
        return set()
    body = match.group(1)
    cols: set[str] = set()
    for raw in body.split('\n'):
        line = raw.strip().rstrip(',')
        if not line or line.startswith('--'):
            continue
        head = line.split()[0]
        if head.lower() in (
                'primary', 'foreign', 'constraint', 'check', 'unique'):
            continue
        cols.add(head.lower())
    return cols


def test_migrate_insight_fields_cover_pg_baseline_schema_columns():
    """Every postgres `insights` DDL column has a `MigrateInsight` field.

    Excludes `embedding_pending` (added on demand by the swap path,
    not a payload-time field — the gather path probes the column
    list at runtime).
    """
    cols = _columns_under(
        PG_BASELINE_SCHEMA.replace('{schema}', 'store_x')
        .replace('{dim}', '512'), 'insights')
    cols -= {'embedding_pending'}
    missing = cols - _MIGRATE_INSIGHT_FIELDS
    assert not missing, (
        f'Postgres baseline insights columns missing from'
        f' MigrateInsight: {sorted(missing)}.'
        f' Add the field or extend the exclusion list and bump'
        f' PAYLOAD_VERSION.')


def test_migrate_insight_fields_cover_sqlite_baseline_schema_columns():
    """Every sqlite `insights` DDL column has a `MigrateInsight` field.

    Excludes `embedding_pending` (carried as a separate
    `PendingReembed` list in the payload).
    """
    cols = _columns_under(_BASELINE_SCHEMA, 'insights')
    cols -= {'embedding_pending'}
    missing = cols - _MIGRATE_INSIGHT_FIELDS
    assert not missing, (
        f'SQLite baseline insights columns missing from'
        f' MigrateInsight: {sorted(missing)}.'
        f' Add the field or extend the exclusion list and bump'
        f' PAYLOAD_VERSION.')


def test_payload_version_pinned():
    """`PAYLOAD_VERSION` is a positive int and matches MigrationPayload.

    Bump this constant whenever a `MigrationPayload` field is
    added/removed/repurposed -- it is the wire-format compat key
    backends use to refuse stale payloads.
    """
    assert isinstance(PAYLOAD_VERSION, int)
    assert PAYLOAD_VERSION >= 1
    assert 'payload_version' in _MIGRATION_PAYLOAD_FIELDS
