"""Item 7: a caller can find out where its write landed.

`remember` and `replace` hand back the `queue_uuid` stamped on every
insight the write produces, and `memman insights by-queue` resolves
that key to the rows themselves. The queue row is purged about a
minute after the drain; the uuid is what outlives it.
"""

import json
import sqlite3
from datetime import datetime, timezone

from memman.store.db import store_dir
from tests.conftest import invoke, make_insight, set_created_at


def _queue_uuid_of(data_dir, queue_id):
    """Read a queue row's own uuid, independent of the CLI response."""
    from memman.queue import queue_db
    with queue_db(data_dir) as conn:
        return conn.execute(
            'select queue_uuid from queue where id = ?',
            (queue_id,)).fetchone()[0]


def _stored_ids(data_dir, store, queue_uuid, active_only=True):
    """Insight ids carrying `queue_uuid`, read by raw SQL."""
    clause = ' and deleted_at is null' if active_only else ''
    path = f'{store_dir(data_dir, store)}/memman.db'
    with sqlite3.connect(path) as conn:
        return [r[0] for r in conn.execute(
            'select id from insights where queue_uuid = ?'
            f'{clause} order by created_at',
            (queue_uuid,)).fetchall()]


def test_remember_returns_the_uuid_its_insight_carries(mm_runner):
    """`remember`'s `queue_uuid` is the key on the row it produced.

    Mutation: minting a fresh uuid for the response instead of
        returning the one `enqueue` wrote, or echoing `queue_id`
        under the `queue_uuid` name.
    Oracle: the same uuid read twice independently -- off the queue
        row, and off the stored insight -- both by raw SQL.
    """
    _, data_dir = mm_runner
    r = invoke(mm_runner, [
        'remember', 'sqlite pages are 4096 bytes', '--no-reconcile'])
    assert r.exit_code == 0, r.output
    raw = json.loads(r.output)

    assert raw['queue_uuid'] == _queue_uuid_of(data_dir, raw['queue_id'])
    ids = _stored_ids(data_dir, raw['store'], raw['queue_uuid'])
    assert len(ids) == 1

    # The key must sit on the row this write CREATED, not on some
    # other row it touched: resolve it and read the content back.
    resolved = json.loads(invoke(
        mm_runner, ['insights', 'by-queue', raw['queue_uuid']]).output)
    assert [r['id'] for r in resolved['results']] == ids
    assert resolved['results'][0]['content'] == (
        'sqlite pages are 4096 bytes')


def test_replace_returns_its_own_uuid_not_the_originals(mm_runner):
    """`replace` carries a `queue_uuid` distinct from the write it supersedes.

    Mutation: omitting the key from the `replace` envelope, or
        reusing the replaced insight's uuid so both writes resolve
        to one set of rows.
    Oracle: the two responses' uuids compared, and each resolved
        against the store by raw SQL.
    """
    _, data_dir = mm_runner
    first = json.loads(invoke(mm_runner, [
        'remember', 'redis evicts on maxmemory', '--no-reconcile']).output)
    old_id = _stored_ids(data_dir, first['store'], first['queue_uuid'])[0]

    r = invoke(mm_runner, ['replace', old_id, 'redis evicts lru first'])
    assert r.exit_code == 0, r.output
    second = json.loads(r.output)

    assert second['queue_uuid'] != first['queue_uuid']
    assert second['queue_uuid'] == _queue_uuid_of(
        data_dir, second['queue_id'])
    assert _stored_ids(data_dir, second['store'], second['queue_uuid'])
    # The superseded write's key must now resolve to nothing: its only
    # row is a tombstone, which is the other half of "its own uuid".
    assert _stored_ids(data_dir, first['store'], first['queue_uuid']) == []


def test_by_queue_returns_exactly_the_rows_that_write_stored(mm_runner):
    """`insights by-queue` resolves the uuid to that write's rows alone.

    Mutation: dropping the `queue_uuid = ?` predicate so every row
        comes back, or resolving against a different write's key.
    Oracle: the id set read straight from the store DB by SQL, and a
        second unrelated write whose row must not appear.
    """
    _, data_dir = mm_runner
    mine = json.loads(invoke(mm_runner, [
        'remember', 'postgres toasts values over 2kb',
        '--no-reconcile']).output)
    other = json.loads(invoke(mm_runner, [
        'remember', 'kafka retains by segment age',
        '--no-reconcile']).output)

    expected = _stored_ids(data_dir, mine['store'], mine['queue_uuid'])
    other_ids = _stored_ids(data_dir, other['store'], other['queue_uuid'])
    assert expected
    assert other_ids
    assert not set(expected) & set(other_ids)

    r = invoke(mm_runner, ['insights', 'by-queue', mine['queue_uuid']])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output)

    assert [row['id'] for row in payload['results']] == expected
    assert payload['count'] == len(expected)
    assert payload['queue_uuid'] == mine['queue_uuid']


def test_by_queue_omits_a_soft_deleted_row(mm_runner):
    """A write whose insight was deleted resolves to nothing.

    Mutation: dropping `and deleted_at is null` from the backend
        query, which would report a tombstone as where the write
        landed.
    Oracle: the same uuid resolving to one row before the delete and
        zero after, with the tombstone still present in raw SQL.
    """
    _, data_dir = mm_runner
    raw = json.loads(invoke(mm_runner, [
        'remember', 'etcd compacts revisions', '--no-reconcile']).output)
    stored_id = _stored_ids(data_dir, raw['store'], raw['queue_uuid'])[0]

    before = json.loads(invoke(
        mm_runner, ['insights', 'by-queue', raw['queue_uuid']]).output)
    assert before['count'] == 1
    assert len(before['results']) == 1

    assert invoke(mm_runner, ['forget', stored_id]).exit_code == 0

    after = json.loads(invoke(
        mm_runner, ['insights', 'by-queue', raw['queue_uuid']]).output)
    assert after['count'] == 0
    assert after['results'] == []
    assert _stored_ids(
        data_dir, raw['store'], raw['queue_uuid'], active_only=False)


def test_by_queue_on_an_unknown_uuid_is_empty_not_an_error(mm_runner):
    """An unresolvable uuid answers `count: 0` and exits clean.

    Mutation: raising ClickException on an empty result, which would
        make "the write is still queued" indistinguishable from a
        malformed key and break a caller that polls for its write.
    Oracle: exit code 0 against a well-formed uuid that was never
        enqueued.
    """
    r = invoke(mm_runner, [
        'insights', 'by-queue', '7f3c1e00-0d1a-4f7e-9c2b-2a1d5b8e4c60'])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output)
    assert payload['count'] == 0
    assert payload['results'] == []


def test_by_queue_rejects_a_malformed_uuid(mm_runner):
    """A `queue_id` passed where a `queue_uuid` belongs fails loudly.

    Mutation: dropping the shape check, so `by-queue 42` answers a
        clean `count: 0` -- indistinguishable from "your write stored
        nothing", which is the confusion this command exists to end.
        `remember` returns the two keys adjacent, so the mix-up is
        the likely one.
    Oracle: non-zero exit naming the argument, against a well-formed
        uuid that also resolves to nothing but exits 0.
    """
    bad = invoke(mm_runner, ['insights', 'by-queue', '42'])
    assert bad.exit_code != 0
    assert 'queue_id' in bad.output

    good = invoke(mm_runner, [
        'insights', 'by-queue', '7f3c1e00-0d1a-4f7e-9c2b-2a1d5b8e4c60'])
    assert good.exit_code == 0
    assert json.loads(good.output)['count'] == 0


def test_get_by_queue_uuid_orders_siblings_deterministically(backend):
    """Rows sharing one write's key come back ordered, on both backends.

    Both backends stamp `created_at` server-side and IGNORE the value
    carried on the Insight, so the siblings are tied by an explicit
    update instead. That tie is what leaves the `id` tiebreak as the
    only thing deciding order; without it the rows sort by time and
    the ORDER BY under test is never exercised.

    Mutation: dropping `, id` from the ORDER BY, which leaves the
        query plan deciding. Also catches dropping the `deleted_at`
        or `queue_uuid` predicate.
    Oracle: ids inserted in reverse-sorted order, so plan order and
        sorted order disagree; expected list is hand-written.
    """
    siblings = ('qu-c', 'qu-b', 'qu-a', 'qu-dead')
    with backend.transaction():
        for row_id in siblings:
            backend.nodes.insert(make_insight(
                id=row_id, content=f'sibling {row_id}', queue_uuid='w-1'))
    # Notes:
    # - `nodes.insert` ignores `Insight.created_at` on both backends,
    #   so passing a shared stamp to `make_insight` would be inert.
    #   Postgres takes `default now()`, constant across the
    #   transaction; SQLite writes a per-row `datetime.now()` cut to
    #   whole seconds, which ties only when the inserts happen to land
    #   inside one second.
    # - `set_created_at` commits on its Postgres branch, and psycopg
    #   forbids an explicit commit inside `backend.transaction()`, so
    #   it runs after the block rather than in it.
    for row_id in siblings:
        set_created_at(backend, row_id, datetime(
            2026, 1, 1, tzinfo=timezone.utc))
    backend.nodes.insert(make_insight(
        id='qu-other', content='another write', queue_uuid='w-2'))
    backend.nodes.soft_delete('qu-dead')

    stamps = {r.created_at for r in backend.nodes.get_by_queue_uuid('w-1')}
    assert len(stamps) == 1, (
        'siblings must share one timestamp or the tiebreak is untested')

    rows = backend.nodes.get_by_queue_uuid('w-1')
    assert [r.id for r in rows] == ['qu-a', 'qu-b', 'qu-c']
    assert [r.id for r in backend.nodes.get_by_queue_uuid('w-2')] == [
        'qu-other']
    assert backend.nodes.get_by_queue_uuid('w-none') == []
