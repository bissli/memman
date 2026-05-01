"""Phase 1b: oplog.log is INSERT-only; trim moves to maintenance_step.

Verifies the cadence change: per-call writes never trim. The cap
applies once per drain via `Backend.oplog.maintenance_step()`.
"""

from memman.store.oplog import MAX_OPLOG_ENTRIES


def test_log_does_not_trim(tmp_db, tmp_backend):
    """Stuffing the oplog past the cap does NOT trim during log()."""
    over_cap = MAX_OPLOG_ENTRIES + 50
    for i in range(over_cap):
        tmp_backend.oplog.log(
            operation='probe', insight_id=str(i), detail='')
    row = tmp_db._query('SELECT COUNT(*) FROM oplog').fetchone()
    assert row[0] == over_cap


def test_maintenance_step_trims(tmp_db, tmp_backend):
    """maintenance_step() caps the oplog at MAX_OPLOG_ENTRIES."""
    over_cap = MAX_OPLOG_ENTRIES + 50
    for i in range(over_cap):
        tmp_backend.oplog.log(
            operation='probe', insight_id=str(i), detail='')

    tmp_backend.oplog.maintenance_step()

    row = tmp_db._query('SELECT COUNT(*) FROM oplog').fetchone()
    assert row[0] == MAX_OPLOG_ENTRIES
