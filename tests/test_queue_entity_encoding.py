"""The queue's `hint_entities` column is a JSON array of entity names.

An entity name may contain a comma. An LDAP distinguished name always
does. A comma-joined column silently cut such a name into fragments on
the one command that sends a STORED list back through the queue,
`memman replace` without `--entities`. Every fragment then became a
named entity of its own, spending the entity-edge budget and growing
the row's list on each pass.

These tests pin the wire contract, not the caller-facing `--entities`
grammar, which stays comma-separated.
"""

import json
import os
import sqlite3

from memman.cli import _entities_json
from memman.store.db import read_active, store_dir
from memman.store.factory import open_backend
from tests.conftest import force_drain, invoke, parse_remember

DN = 'OU=Servers,DC=example,DC=com'
FRAGMENTS = ('OU=Servers', 'DC=example', 'DC=com')


def _stored_entities(data_dir, store, queue_uuid):
    """Return the entity list of the insight carrying `queue_uuid`."""
    db = os.path.join(store_dir(data_dir, store), 'memman.db')
    conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    try:
        row = conn.execute(
            'select entities from insights where queue_uuid = ?'
            ' and deleted_at is null order by created_at',
            (queue_uuid,)).fetchone()
    finally:
        conn.close()
    return json.loads(row[0]) if row and row[0] else []


def test_replace_keeps_a_comma_bearing_inherited_entity_whole(mm_runner):
    """Verify `replace` does not cut an inherited LDAP DN into fragments.

    Mutation: encoding the inherited list as `','.join(old.entities)`
        and splitting it back in the drain, which cuts one DN into
        three entities -- the defect itself. Also catches a fix
        applied to the caller-typed branch of `replace` and not to
        the inherited branch.
    Oracle: the target's own stored entity list, the single name
        `OU=Servers,DC=example,DC=com`, against the successor's. The
        teeth are on the ABSENCE of fragments: the DN itself is
        present on the broken code too, re-added by the predecessor
        union at `pipeline/remember.py:1153`, so asserting only that
        the DN survives passes on the defect.
    """
    _, data_dir = mm_runner
    first = invoke(mm_runner, [
        'remember', 'a note about a directory container',
        '--no-reconcile'])
    old = parse_remember(first, mm_runner)
    name = read_active(data_dir) or 'default'
    open_backend(name, data_dir).nodes.update_entities(old['id'], [DN])

    result = invoke(mm_runner, [
        'replace', old['id'], 'a corrected note'])

    assert result.exit_code == 0, result.output
    successor = parse_remember(result, mm_runner)
    assert 'id' in successor, result.output
    stored = open_backend(name, data_dir).nodes.get(successor['id']).entities
    assert DN in stored
    assert [e for e in stored if e in FRAGMENTS] == []


def test_drain_decodes_the_queue_entity_list_verbatim(mm_runner):
    """Verify the drain decodes `hint_entities` rather than splitting it.

    Pins both ends of the wire contract with no CLI command in
    between: the row is written by the production encoder and read by
    the production drain, so a delimited encoding reintroduced at
    either end fails here.

    Mutation: the drain splitting `row.hint_entities` on a comma
        instead of decoding it, or `_entities_json` joining on one
        instead of encoding.
    Oracle: the list handed to `_entities_json`, hand-built as
        `['OU=Servers,DC=example,DC=com', 'plain entity']`, against
        the entity list of the insight the drain stored.
    """
    _, data_dir = mm_runner
    wanted = [DN, 'plain entity']
    invoke(mm_runner, ['remember', 'seed so the store exists',
                       '--no-reconcile'])
    name = read_active(data_dir) or 'default'

    from memman.queue import enqueue, queue_db
    with queue_db(data_dir) as conn:
        _row_id, queue_uuid = enqueue(
            conn, store=name, content='a note enqueued by hand',
            hint_entities=_entities_json(wanted),
            hint_no_reconcile=True)
    force_drain(data_dir)

    stored = _stored_entities(data_dir, name, queue_uuid)
    assert [e for e in stored if e in wanted] == wanted
    assert [e for e in stored if e in FRAGMENTS] == []
    assert [e for e in stored if e.startswith('["') or e.endswith('"]')] == []


def test_a_delimited_entity_value_fails_the_row_instead_of_splitting(
        mm_runner):
    """Verify the drain refuses a non-JSON `hint_entities` value.

    The strict reader is the design decision this change rests on: a
    tolerant one that fell back to a comma split would be the
    backward-compatible shim the project forbids, and it would
    silently restore fragment-splitting for every legacy value.

    Mutation: a `startswith('[')` fallback reader in the drain, which
        leaves every valid-JSON test green while splitting any other
        value into fragments.
    Oracle: the queue row's own terminal state and the store. A
        refused row stores no insight and records an attempt; a split
        one would store the two fragments `a` and `b`.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['remember', 'seed so the store exists',
                       '--no-reconcile'])
    name = read_active(data_dir) or 'default'

    from memman.queue import enqueue, queue_db
    with queue_db(data_dir) as conn:
        row_id, queue_uuid = enqueue(
            conn, store=name, content='a row carrying a legacy value',
            hint_entities='a,b',
            hint_no_reconcile=True)
    force_drain(data_dir)

    assert _stored_entities(data_dir, name, queue_uuid) == []
    with queue_db(data_dir) as conn:
        status, attempts, last_error = conn.execute(
            'select status, attempts, last_error from queue where id = ?',
            (row_id,)).fetchone()
    assert status != 'done'
    assert attempts > 0
    assert 'JSONDecodeError' in (last_error or '')
