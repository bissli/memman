"""`remember` lists the current rows its text most likely corrects.

The write is queued before the store is read, so no failure of the
read costs the write.
"""

import json
import os
from pathlib import Path

import pytest
from memman.queue import queue_db
from memman.store import factory
from memman.store.db import store_dir
from memman.store.factory import open_backend
from tests.conftest import invoke, make_insight, queued_contents

_SHORT_STALE = 'The retry cap for batch jobs is three.'
_LONG_SIBLING = (
    'The worker queue drains batch jobs with a retry cap, and the'
    ' scheduler, the enrichment pass, the embedding pass, the backup'
    ' job, the heartbeat file, the drain lock, the trace log, and the'
    ' doctor checks all run on the same timer every sixty seconds.')
_UNRELATED = 'Postgres stores run on RDS in us-east-1.'
_CORRECTION = (
    'The retry cap for batch jobs is five, and the worker queue holds'
    ' them.')


def _remember_id(runner, text):
    """Store `text` and return the id `remember` printed.
    """
    result = invoke(runner, ['remember', text])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)['id']


def test_related_ranks_a_short_stale_row_above_a_longer_one(mm_runner):
    """Verify related rows rank by shared words over row length.

    Mutation: ranking by raw shared words, which puts the long sibling
        (six shared of 27 words) above the short stale row (four of
        five), or listing a row that shares no word.
    Oracle: hand-computed scores, 4/sqrt(5) = 1.79 for the stale row
        and 6/sqrt(27) = 1.15 for the sibling, and the whole text of
        each listed row.
    """
    stale_id = _remember_id(mm_runner, _SHORT_STALE)
    sibling_id = _remember_id(mm_runner, _LONG_SIBLING)
    _remember_id(mm_runner, _UNRELATED)

    result = invoke(mm_runner, ['remember', _CORRECTION])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['related'] == [
        f'{stale_id[:8]} {_SHORT_STALE}',
        f'{sibling_id[:8]} {_LONG_SIBLING}',
        ]


def test_related_lists_three_rows_with_whitespace_folded(mm_runner):
    """Verify related stops at three rows and prints each on one line.

    Mutation: the three-row cap dropped, which lists the fourth
        matching row, or the content printed unfolded, which keeps the
        double space.
    Oracle: four rows sharing words with the new text, the weakest
        (two of four words) left out, and a stored double space read
        back as one.
    """
    doubled_id = _remember_id(mm_runner, 'The retry cap  for batch jobs is three.')
    _remember_id(mm_runner, 'Batch jobs retry on the worker queue.')
    _remember_id(mm_runner, 'The worker queue holds batch jobs.')
    weakest_id = _remember_id(mm_runner, 'The retry cap stays at three.')

    related = json.loads(invoke(
        mm_runner, ['remember', _CORRECTION]).output)['related']

    assert len(related) == 3
    assert f'{doubled_id[:8]} The retry cap for batch jobs is three.' in related
    assert not any(row.startswith(weakest_id[:8]) for row in related)


def test_related_never_lists_a_row_over_the_cap(mm_runner):
    """Verify a row over 1,000 bytes stays out of related.

    Mutation: no size filter on the candidates, so the oversized row,
        which shares every word of the new text, ranks first.
    Oracle: the one short row that shares words with the new text,
        listed alone.
    """
    _, data_dir = mm_runner
    stale_id = _remember_id(mm_runner, _SHORT_STALE)
    oversized = ' '.join([_CORRECTION] * 20)
    assert len(oversized.encode()) > 1000
    with open_backend('default', data_dir) as backend:
        backend.nodes.insert(make_insight(id='oversized-row', content=oversized))

    result = invoke(mm_runner, ['remember', _CORRECTION])

    assert json.loads(result.output)['related'] == [
        f'{stale_id[:8]} {_SHORT_STALE}',
        ]


@pytest.mark.no_auto_drain
def test_related_leaves_out_the_write_itself(mm_runner, monkeypatch):
    """Verify a write a drain lands mid-command is not its own related row.

    Mutation: no guard on the new write's id, so a drain that lands the
        row before the read lists it first, since it shares every word
        with itself, and pushes a real candidate out.
    Oracle: a store open that first lands the queued write, as the
        timer drain does, and the one stale row listed alone.
    """
    _, data_dir = mm_runner
    stale_id = _remember_id(mm_runner, _SHORT_STALE)
    assert invoke(mm_runner, ['scheduler', 'drain']).exit_code == 0
    real_open = factory.open_backend

    def open_after_the_drain_lands_the_write(name, dir_arg, *, read_only=False):
        with queue_db(data_dir) as conn:
            queue_uuid, content = conn.execute(
                'select queue_uuid, content from queue'
                " where status = 'pending' order by id desc limit 1"
                ).fetchone()
        with real_open(name, dir_arg) as backend:
            backend.nodes.insert(make_insight(
                id=queue_uuid, content=content, queue_uuid=queue_uuid))
        return real_open(name, dir_arg, read_only=read_only)

    monkeypatch.setattr(
        factory, 'open_backend', open_after_the_drain_lands_the_write)

    result = invoke(mm_runner, ['remember', _CORRECTION])

    assert json.loads(result.output)['related'] == [
        f'{stale_id[:8]} {_SHORT_STALE}',
        ]


@pytest.mark.no_auto_drain
def test_first_remember_on_a_new_store_lists_nothing(mm_runner):
    """Verify a store with no database yet gives an empty related list.

    Mutation: opening a store that has no database, which reports
        `database not found` as `related_error` on every first write.
    Oracle: an empty `related` and no `related_error` key.
    """
    result = invoke(mm_runner, ['remember', _CORRECTION])

    reply = json.loads(result.output)
    assert reply['related'] == []
    assert 'related_error' not in reply


@pytest.mark.no_auto_drain
def test_remember_queues_and_reports_a_corrupt_store(mm_runner):
    """Verify a store that cannot be read still takes the write.

    Mutation: letting the read's error propagate after the enqueue, so
        the command exits 1 on a write that is already queued and the
        agent writes it again.
    Oracle: exit 0, a `related_error`, and the queue holding the text.
    """
    _, data_dir = mm_runner
    database = Path(store_dir(data_dir, 'default')) / 'memman.db'
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_bytes(b'not a sqlite database' * 64)

    result = invoke(mm_runner, ['remember', _CORRECTION])

    assert result.exit_code == 0, result.output
    reply = json.loads(result.output)
    assert reply['related_error']
    assert 'related' not in reply
    assert queued_contents(data_dir) == [_CORRECTION]


@pytest.mark.no_auto_drain
def test_remember_bounds_the_postgres_connect_and_still_queues(
        mm_runner, env_file, monkeypatch):
    """Verify an unreachable Postgres store fails fast and keeps the write.

    Mutation: dropping the `PGCONNECT_TIMEOUT` default, which leaves
        psycopg's two-minute connect timeout in place, or letting the
        connect error propagate after the enqueue.
    Oracle: a spy on `psycopg.connect` that records the timeout the
        connect saw and refuses it, plus exit 0 and the queued text.
    """
    import psycopg

    _, data_dir = mm_runner
    env_file('MEMMAN_BACKEND_default', 'postgres')
    env_file('MEMMAN_POSTGRES_DSN_default',
             'postgresql://memman@127.0.0.1:1/memman')
    seen_timeouts = []

    def refuse(*args, **kwargs):
        seen_timeouts.append(os.environ.get('PGCONNECT_TIMEOUT'))
        raise psycopg.OperationalError('connection refused')

    monkeypatch.setattr(psycopg, 'connect', refuse)

    result = invoke(mm_runner, ['remember', _CORRECTION])

    assert result.exit_code == 0, result.output
    assert seen_timeouts
    assert set(seen_timeouts) == {'3'}
    assert json.loads(result.output)['related_error']
    assert queued_contents(data_dir) == [_CORRECTION]
