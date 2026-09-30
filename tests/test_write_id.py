"""A write's printed id is the handle an agent corrects it by.

`remember` and `replace` print the id the drain stores the row under,
so an agent can replace its own write before the drain has run.
"""

import json
import sqlite3

import pytest
from memman.queue import queue_db
from memman.store.db import store_dir
from tests.conftest import force_drain, invoke


def _row(data_dir, id):
    """Read `(content, replaced_by)` for one id by raw SQL.
    """
    path = f"{store_dir(data_dir, 'default')}/memman.db"
    with sqlite3.connect(path) as conn:
        return conn.execute(
            'select content, replaced_by from insights'
            ' where id = ?', (id,)).fetchone()


def test_remember_prints_the_id_its_row_is_stored_under(mm_runner):
    """The `id` `remember` prints names the row the drain stored.

    Mutation: the drain minting a fresh uuid for the row instead of
        the write's own id, or the response omitting the id.
    Oracle: the stored row read back by that id with raw SQL.
    """
    _, data_dir = mm_runner
    r = invoke(mm_runner, ['remember', 'sqlite pages are 4096 bytes'])
    assert r.exit_code == 0, r.output

    printed = json.loads(r.output)['id']

    assert _row(data_dir, printed)[0] == 'sqlite pages are 4096 bytes'


def test_replace_prints_the_id_of_the_replacement(mm_runner):
    """The `id` `replace` prints names the row that retired the target.

    Mutation: printing the target's id, or the queue row id, in
        place of the replacement's own id.
    Oracle: the target's `replaced_by` read by raw SQL.
    """
    _, data_dir = mm_runner
    first = json.loads(invoke(
        mm_runner, ['remember', 'redis evicts on maxmemory']).output)

    r = invoke(mm_runner, [
        'replace', first['id'], 'redis evicts lru keys first'])
    assert r.exit_code == 0, r.output

    assert _row(data_dir, first['id'])[1] == json.loads(r.output)['id']


def test_remember_and_replace_print_the_id_on_one_line(mm_runner):
    """Verify the printed id survives `| tail -1`.

    Mutation: `_json_out` indenting its output, so the last line is
        `}` and an agent that tails the reply loses the id.
    Oracle: the reply's last line parsed alone, against the id the
        whole reply carries.
    """
    first = invoke(mm_runner, ['remember', 'kafka keeps seven days'])
    second = invoke(mm_runner, [
        'replace', json.loads(first.output)['id'],
        'kafka keeps three days'])

    for r in (first, second):
        last_line = r.output.strip().splitlines()[-1]
        assert json.loads(last_line)['id'] == json.loads(r.output)['id']


@pytest.mark.no_auto_drain
def test_replace_of_a_queued_write_retires_it_once_it_lands(mm_runner):
    """A replace aimed at a write still in the queue links on one drain.

    Mutation: refusing a still-queued id as not found, or the drain
        storing the replacement before its target.
    Oracle: both rows read by raw SQL after a single drain.
    """
    _, data_dir = mm_runner
    first = json.loads(invoke(
        mm_runner, ['remember', 'the sign-in limit is 24 hours']).output)

    r = invoke(mm_runner, [
        'replace', first['id'], 'the sign-in limit is 30 days'])
    assert r.exit_code == 0, r.output
    force_drain(data_dir)

    assert _row(data_dir, first['id'])[1] == json.loads(r.output)['id']


@pytest.mark.no_auto_drain
def test_replace_resolves_a_queued_write_by_prefix(mm_runner):
    """An unambiguous prefix of a queued write's id resolves like a stored one.

    Mutation: matching queued writes by the full id only, so the id8
        an agent copies is refused.
    Oracle: the target's `replaced_by` read by raw SQL after a drain.
    """
    _, data_dir = mm_runner
    first = json.loads(invoke(
        mm_runner, ['remember', 'kafka retains by segment age']).output)

    r = invoke(mm_runner, [
        'replace', first['id'][:8], 'kafka retains by size and age'])
    assert r.exit_code == 0, r.output
    force_drain(data_dir)

    assert _row(data_dir, first['id'])[1] == json.loads(r.output)['id']


@pytest.mark.no_auto_drain
def test_replace_refuses_a_write_queued_for_another_store(mm_runner):
    """A queued write in one store is not a replace target in another.

    Mutation: dropping the store predicate from the queued-write
        lookup, which links rows across stores.
    Oracle: the refusal, with the other store's write still queued.
    """
    first = json.loads(invoke(mm_runner, [
        '--store', 'shop', 'remember', 'carts expire after a week']).output)

    r = invoke(mm_runner, [
        'replace', first['id'], 'carts expire after a day'])

    assert r.exit_code != 0
    assert 'not found' in r.output


@pytest.mark.no_auto_drain
def test_show_of_a_queued_write_says_it_is_queued(mm_runner):
    """`insights show` on a write the drain has not stored names the queue.

    Mutation: answering `not found` for a write still in the queue.
    Oracle: the refusal text, with the write known to be queued.
    """
    first = json.loads(invoke(
        mm_runner, ['remember', 'etcd compacts revisions']).output)

    r = invoke(mm_runner, ['insights', 'show', first['id']])

    assert r.exit_code != 0
    assert 'still queued' in r.output


@pytest.mark.no_auto_drain
def test_forget_of_a_queued_write_says_it_is_queued(mm_runner):
    """`forget` on a write the drain has not stored names the queue.

    Mutation: `forget` resolving only stored rows, so it answers `not
        found` for the id `remember` just printed and the write lands
        anyway.
    Oracle: the refusal text, with the write known to be queued.
    """
    first = json.loads(invoke(
        mm_runner, ['remember', 'etcd compacts revisions']).output)

    r = invoke(mm_runner, ['forget', first['id']])

    assert r.exit_code != 0
    assert 'still queued' in r.output


def _share_prefix_with_a_stored_row(mm_runner):
    """Store one write, queue a second under an id sharing its prefix.

    Returns the four-character prefix the two ids share.
    """
    _, data_dir = mm_runner
    stored = json.loads(invoke(
        mm_runner, ['remember', 'sqlite pages are 4096 bytes']).output)
    force_drain(data_dir)
    queued = json.loads(invoke(
        mm_runner, ['remember', 'postgres pages are 8192 bytes']).output)
    with queue_db(data_dir) as conn:
        conn.execute(
            'update queue set queue_uuid = ? where queue_uuid = ?',
            (stored['id'][:4] + queued['id'][4:], queued['id']))
    return stored['id'][:4]


@pytest.mark.no_auto_drain
def test_replace_refuses_a_prefix_a_queued_and_a_stored_row_share(
        mm_runner):
    """A prefix matching a queued write and a stored row is ambiguous.

    Mutation: `replace` taking the queued match without reading the
        store, so it retires a different row than `insights show`
        reads for the same prefix.
    Oracle: two ids forced to share a prefix; the refusal names it.
    """
    prefix = _share_prefix_with_a_stored_row(mm_runner)

    r = invoke(mm_runner, ['replace', prefix, 'pages are 16384 bytes'])

    assert r.exit_code != 0
    assert f'prefix {prefix!r} matches a queued write and a stored row' \
        in r.output


@pytest.mark.no_auto_drain
def test_show_refuses_a_prefix_a_queued_and_a_stored_row_share(mm_runner):
    """`insights show` refuses the prefix `replace` refuses.

    Mutation: `insights show` reading the store alone, so it shows the
        stored row for a prefix a queued write also matches.
    Oracle: two ids forced to share a prefix; the refusal names it.
    """
    prefix = _share_prefix_with_a_stored_row(mm_runner)

    r = invoke(mm_runner, ['insights', 'show', prefix])

    assert r.exit_code != 0
    assert f'prefix {prefix!r} matches a queued write and a stored row' \
        in r.output
