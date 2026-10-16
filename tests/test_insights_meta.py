"""`insights meta`: in-place meta edits, the store methods under them, refusals.
"""

import dataclasses
import json
import os

import pytest
from memman import drain_lock
from memman import queue as queue_mod
from memman.queue import enqueue, queue_db
from memman.store import factory
from memman.store.errors import BackendError
from memman.store.model import format_meta
from tests.conftest import _mock_embed, _vec, invoke
from tests.test_store_branch import CREATED, _branch, _remember, _replace
from tests.test_store_branch import _row, _set_meta, _snapshot

_SELECTOR_CLASH = (
    '{tok} is read as an id; give ids or --where KEY=VALUE, one selector'
    ' per call, and quote a value that holds spaces')
_BAD_KEY = ' is not key=value with a key of lower-case letters and _'


def _store():
    """Name of the store the test's runner writes to.
    """
    return os.environ.get('MEMMAN_STORE', 'default')


def _remember_with(runner, text, *pairs):
    """Store text with `--meta` pairs, drained, and return its id.
    """
    args = ['remember']
    for pair in pairs:
        args += ['--meta', pair]
    result = invoke(runner, args + [text])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)['id']


def _meta(runner, *args):
    """Run `insights meta` with args.
    """
    return invoke(runner, ['insights', 'meta', *args])


def _changed(result):
    """The `changed` list of a successful `insights meta` reply.
    """
    assert result.exit_code == 0, result.output
    return json.loads(result.output)['changed']


def _meta_of(runner, row_id):
    """The stored meta of a row in the runner's store.
    """
    _, data_dir = runner
    with factory.open_backend(_store(), data_dir, read_only=True) as backend:
        return backend.nodes.get_raw(row_id).meta


def _meta_ops(runner):
    """Oplog entries of operation `meta` in the runner's store, newest first.
    """
    _, data_dir = runner
    with factory.open_backend(_store(), data_dir, read_only=True) as backend:
        return [entry for entry in backend.oplog.recent(limit=10_000)
                if entry.operation == 'meta']


def _basic_hits(runner, word):
    """Id8 of each row `recall --basic word` prints.
    """
    result = invoke(runner, ['recall', '--basic', word])
    assert result.exit_code == 0, result.output
    return {line.split()[0] for line in result.output.splitlines()
            if line.strip()}


def _stored_row(row_id, **fields):
    """A current `insert_raw` row with an embedding and fixed timestamps.
    """
    return _row(row_id, f'content of {row_id}', _vec(1.0), **fields)


def _raw_sql(backend, sqlite_sql, postgres_sql, params=()):
    """First column of the first row, by raw SQL on either backend.

    `postgres_sql` takes `{s}` for the store's schema.
    """
    if hasattr(backend, '_schema'):
        with backend._conn.cursor() as cur:
            cur.execute(postgres_sql.format(s=backend._schema), params)
            row = cur.fetchone() if cur.description else None
    else:
        row = backend._db._conn.execute(
            sqlite_sql.replace('%s', '?'), params).fetchone()
    return None if row is None else row[0]


def _meta_is_null(backend, row_id):
    """True when the row's meta column is SQL NULL.
    """
    return bool(_raw_sql(
        backend, 'select meta is null from insights where id = %s',
        'select meta is null from {s}.insights where id = %s', (row_id,)))


# --- Store layer ---

def test_update_meta_changes_only_meta_and_updated_at(backend):
    """Verify update_meta leaves every column but meta and updated_at alone.

    Mutation: the update also nulls summary or embedding.
    Oracle: `get_raw` before and after, equal apart from meta and
        `updated_at`.
    """
    backend.nodes.insert_raw(_stored_row(
        'um-1', summary='a summary', enriched_at=CREATED,
        embedding_model='embed-a', summary_model='llm-a',
        meta={'doc': 'docs/old.md', 'ticket': 'T1'}))
    before = backend.nodes.get_raw('um-1')

    written = backend.nodes.update_meta('um-1', {'doc': 'docs/new.md'})

    after = backend.nodes.get_raw('um-1')
    assert written is True
    assert after.meta == {'doc': 'docs/new.md'}
    assert after.updated_at > before.updated_at
    assert dataclasses.replace(
        after, meta=before.meta, updated_at=before.updated_at) == before


def test_update_meta_refuses_replaced_and_forgotten_rows(
        backend, backend_kind):
    """Verify update_meta returns False and writes nothing on a retired row.

    Mutation: `replaced_by is null` dropped from the guard.
    Oracle: rows retired by `mark_replaced` and `soft_delete` return
        False and keep their meta; Postgres `kw_tokens` stays empty on
        the forgotten row.
    """
    for row_id in ('cur', 'rep', 'gone'):
        backend.nodes.insert_raw(_stored_row(row_id, meta={'doc': 'old'}))
    assert backend.nodes.mark_replaced('rep', 'cur')
    assert backend.nodes.soft_delete('gone')

    results = [
        backend.nodes.update_meta('rep', {'doc': 'new'}),
        backend.nodes.update_meta('gone', {'doc': 'new'}),
        ]

    assert results == [False, False]
    assert backend.nodes.get_raw('rep').meta == {'doc': 'old'}
    assert backend.nodes.get_raw('gone').meta == {'doc': 'old'}
    if backend_kind == 'postgres':
        with backend._conn.cursor() as cur:
            cur.execute(
                f'select kw_tokens from {backend._schema}.insights'
                ' where id = %s', ('gone',))
            assert cur.fetchone()[0] == []


def test_keyword_channel_finds_the_new_value_not_the_old(backend):
    """Verify keyword search follows an update_meta edit.

    Mutation: Postgres leaves `kw_tokens` stale, or the FTS trigger
        loses the meta column.
    Oracle: a hand-picked token counted 1 for the new value and absent
        for the old.
    """
    backend.nodes.insert_raw(_stored_row(
        'kw-1', meta={'doc': 'docs/quokka-plan/'}))

    backend.nodes.update_meta('kw-1', {'doc': 'docs/wombat-plan/'})

    with backend.recall_session() as session:
        assert session.keyword_counts({'wombat'}) == {'kw-1': 1}
        assert session.keyword_counts({'quokka'}) == {}


@pytest.mark.parametrize('empty', [{}, None], ids=['dict', 'none'])
def test_update_meta_empty_stores_null(backend, empty):
    """Verify an empty meta stores SQL NULL, never an empty object.

    Mutation: `'{}'` stored, or the JSON text `null` stored.
    Oracle: `meta is null` by raw SQL on the column.
    """
    backend.nodes.insert_raw(_stored_row('nul-1', meta={'doc': 'old'}))

    written = backend.nodes.update_meta('nul-1', empty)

    assert written is True
    assert _meta_is_null(backend, 'nul-1')


def test_a_failed_oplog_insert_keeps_the_meta_write(backend):
    """Verify a meta write commits when the transaction's last oplog insert fails.

    Mutation: the Postgres oplog insert run without a savepoint, so the
        aborted transaction rolls back at commit and raises nothing.
    Oracle: the stored meta after a commit whose last statement is an
        insert into a renamed oplog table.
    """
    backend.nodes.insert_raw(_stored_row('op-1', meta={'doc': 'old'}))
    _raw_sql(
        backend, 'alter table oplog rename to oplog_gone',
        'alter table {s}.oplog rename to oplog_gone')

    with backend.transaction():
        assert backend.nodes.update_meta('op-1', {'doc': 'new'})
        backend.oplog.log(operation='meta', insight_id='op-1', detail='{}')

    assert backend.nodes.get_raw('op-1').meta == {'doc': 'new'}


def test_overlay_update_meta_refuses_a_parent_only_row(mm_runner):
    """Verify a branch never copies or writes a parent row on update_meta.

    Mutation: the overlay method missing (returns None) or copying the
        row.
    Oracle: BackendError, the branch holds no ids, parent meta
        unchanged.
    """
    _, data_dir = mm_runner
    parent_row = _remember_with(mm_runner, 'The retry cap is three.', 'doc=a')
    branch = _branch(data_dir)

    with factory.open_backend(branch, data_dir) as overlay:
        with pytest.raises(BackendError):
            overlay.nodes.update_meta(parent_row, {'doc': 'b'})
        assert overlay.branch.nodes.get_all_ids() == set()

    assert _meta_of(mm_runner, parent_row) == {'doc': 'a'}


def test_overlay_update_meta_refuses_while_merging(mm_runner):
    """Verify a merging branch refuses update_meta on its own row.

    Mutation: `_require_unmerged` dropped.
    Oracle: `branch_merging` set by `_set_meta`; the branch row keeps
        its meta.
    """
    _, data_dir = mm_runner
    branch = _branch(data_dir)
    branch_row = _remember(mm_runner, 'The gateway retries twice.', branch)
    with factory.open_backend(branch, data_dir) as overlay:
        overlay.nodes.update_meta(branch_row, {'doc': 'a'})
    _set_meta(data_dir, branch, 'branch_merging', '2026-10-06T00:00:00Z')

    with factory.open_backend(branch, data_dir) as overlay:
        with pytest.raises(BackendError, match='is merging'):
            overlay.nodes.update_meta(branch_row, {'doc': 'b'})

    with factory.open_backend(branch, data_dir, read_only=True) as overlay:
        assert overlay.branch.nodes.get_raw(branch_row).meta == {'doc': 'a'}


def test_list_pending_returns_pending_and_claimed_rows_of_named_stores(
        queue_conn):
    """Verify list_pending keeps pending rows, claimed or not, of named stores.

    Mutation: `claimed_at is null` added, or the store filter dropped.
    Oracle: statuses and claims set by hand across `p`, `p__b` and
        `other`.
    """
    plain, _ = enqueue(queue_conn, 'p', 'one', meta={'doc': 'a'})
    claimed, _ = enqueue(queue_conn, 'p', 'two')
    branch_row, _ = enqueue(queue_conn, 'p__b', 'three')
    failed, _ = enqueue(queue_conn, 'p', 'four')
    done, _ = enqueue(queue_conn, 'p', 'five')
    enqueue(queue_conn, 'other', 'six')
    queue_conn.execute(
        'update queue set claimed_at = 1000 where id = ?', (claimed,))
    queue_conn.execute(
        "update queue set status = 'failed' where id = ?", (failed,))
    queue_conn.execute(
        "update queue set status = 'done' where id = ?", (done,))
    queue_conn.commit()

    rows = queue_mod.list_pending(queue_conn, ['p', 'p__b'])

    assert [row.id for row in rows] == [plain, claimed, branch_row]
    assert [row.meta for row in rows] == [{'doc': 'a'}, None, None]


# --- Command behavior ---

def test_meta_sets_one_key_and_keeps_the_row(
        cross_backend_runner, monkeypatch):
    """Verify an id-form edit sets one key, queues nothing and calls no model.

    Mutation: routing through the queue or `replace`, or a whole-set
        swap.
    Oracle: values read before, the queue row count, and the recorded
        LLM and embedding calls.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    row_id = _remember_with(
        runner, 'The retry cap is three.', 'doc=docs/old.md', 'ticket=T1')
    with factory.open_backend(_store(), data_dir, read_only=True) as backend:
        before = backend.nodes.get_raw(row_id)
    with queue_db(data_dir) as conn:
        queued_before = conn.execute('select count(*) from queue').fetchone()[0]
    calls = []

    def record_llm(self, system, user, *, stage):
        calls.append(('llm', stage))
        return json.dumps({'ok': True})

    def record_embed(self, text):
        calls.append(('embed', text))
        return _mock_embed(self, text)

    monkeypatch.setattr('memman.llm.client.MemmanLLMClient.complete', record_llm)
    monkeypatch.setattr('memman.embed.client.Client.embed', record_embed)

    result = _meta(runner, row_id[:8], 'doc=docs/new.md')

    with factory.open_backend(_store(), data_dir, read_only=True) as backend:
        after = backend.nodes.get_raw(row_id)
    with queue_db(data_dir) as conn:
        queued_after = conn.execute('select count(*) from queue').fetchone()[0]
    assert _changed(result) == [row_id[:8]]
    assert after.meta == {'doc': 'docs/new.md', 'ticket': 'T1'}
    assert dataclasses.replace(
        after, meta=before.meta, updated_at=before.updated_at) == before
    assert queued_after == queued_before
    assert calls == []


@pytest.mark.parametrize(('start', 'args', 'expected'), [
    pytest.param(
        ['doc=a', 'ticket=T1'], ['ticket='], {'doc': 'a'}, id='one-of-two'),
    pytest.param(['doc=a'], ['doc='], None, id='last-key'),
    ])
def test_meta_empty_value_removes_the_key(
        cross_backend_runner, start, args, expected):
    """Verify an empty value removes its key and the last key leaves NULL.

    Mutation: `{}` or the JSON text `null` stored, or an empty value read
        as "set nothing".
    Oracle: the stored meta is the surviving pair, or SQL NULL by raw
        SQL; `insights show` has no `meta` for NULL.
    """
    runner = cross_backend_runner
    _, data_dir = runner
    row_id = _remember_with(runner, 'The retry cap is three.', *start)

    result = _meta(runner, row_id[:8], *args)

    shown = json.loads(invoke(runner, ['insights', 'show', row_id]).output)
    with factory.open_backend(_store(), data_dir, read_only=True) as backend:
        is_null = _meta_is_null(backend, row_id)
    assert _changed(result) == [row_id[:8]]
    assert _meta_of(runner, row_id) == expected
    assert is_null == (expected is None)
    assert shown.get('meta') == expected


def test_meta_where_rename_moves_basic_hits_from_old_to_new(
        cross_backend_runner):
    """Verify `--where` renames a value on every holder and spares the rest.

    Mutation: matching by key alone, or the write landing on another
        row.
    Oracle: hand-picked rows and `recall --basic` hits before and
        after.
    """
    runner = cross_backend_runner
    first = _remember_with(
        runner, 'The zebra cache evicts entries after an hour.',
        'doc=docs/old.md')
    second = _remember_with(
        runner, 'The giraffe scheduler polls every minute.',
        'doc=docs/old.md')
    other = _remember_with(
        runner, 'The okapi gateway refreshes its auth token.',
        'doc=docs/other.md')
    before = _basic_hits(runner, 'docs/old.md')

    result = _meta(runner, '--where', 'doc=docs/old.md', 'doc=docs/new.md')

    assert before == {first[:8], second[:8]}
    assert sorted(_changed(result)) == sorted([first[:8], second[:8]])
    assert _basic_hits(runner, 'docs/old.md') == set()
    assert _basic_hits(runner, 'docs/new.md') == {first[:8], second[:8]}
    assert _basic_hits(runner, 'docs/other.md') == {other[:8]}


def test_meta_where_selects_only_the_exact_pair(cross_backend_runner):
    """Verify `--where` takes exact current holders and no lookalike or retired row.

    Mutation: LIKE or case-folded match, or retired rows taken.
    Oracle: hand-listed ids.
    """
    runner = cross_backend_runner
    exact = _remember_with(runner, 'The retry cap is three.', 'doc=x/y')
    plus = _remember_with(
        runner, 'The backup job runs at midnight.', 'doc=x/y', 'ticket=T1')
    upper = _remember_with(runner, 'The grackle colony nests.', 'doc=X/Y')
    slash = _remember_with(runner, 'The kiwi rollout is staged.', 'doc=x/y/')
    other_key = _remember_with(
        runner, 'The okapi gateway refreshes tokens.', 'ticket=x/y')
    replaced = _remember_with(
        runner, 'The zebra cache evicts after an hour.', 'doc=x/y')
    _replace_with_meta(
        runner, replaced, 'The zebra cache evicts after two hours.',
        'doc=', 'ticket=T2')
    forgotten = _remember_with(
        runner, 'The giraffe scheduler polls every minute.', 'doc=x/y')
    assert invoke(runner, ['forget', forgotten]).exit_code == 0
    untouched = {
        upper: {'doc': 'X/Y'}, slash: {'doc': 'x/y/'},
        other_key: {'ticket': 'x/y'}, replaced: {'doc': 'x/y'},
        forgotten: {'doc': 'x/y'},
        }

    result = _meta(runner, '--where', 'doc=x/y', 'doc=z/w')

    assert sorted(_changed(result)) == sorted([exact[:8], plus[:8]])
    assert _meta_of(runner, plus) == {'doc': 'z/w', 'ticket': 'T1'}
    assert {row: _meta_of(runner, row) for row in untouched} == untouched


def test_meta_where_held_only_by_retired_rows_is_a_zero_match(
        cross_backend_runner):
    """Verify a pair only replaced and forgotten rows hold matches nothing.

    Mutation: `--where` selecting from retired rows, which then exits 0
        with an empty `changed`.
    Oracle: the zero-match text of spec section 5 step 15.
    """
    runner = cross_backend_runner
    replaced = _remember_with(
        runner, 'The zebra cache evicts after an hour.', 'doc=x/y')
    _replace_with_meta(
        runner, replaced, 'The zebra cache evicts after two hours.',
        'doc=', 'ticket=T2')
    forgotten = _remember_with(
        runner, 'The giraffe scheduler polls every minute.', 'doc=x/y')
    assert invoke(runner, ['forget', forgotten]).exit_code == 0

    result = _meta(runner, '--where', 'doc=x/y', 'doc=z/w')

    assert result.exit_code == 1
    assert f'no current row holds doc=x/y in {_store()};' in result.output


def _replace_with_meta(runner, target, text, *pairs):
    """Replace target with text and explicit `--meta` pairs, drained.
    """
    args = ['replace', target, text]
    for pair in pairs:
        args += ['--meta', pair]
    result = invoke(runner, args)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)['id']


def test_meta_two_tokens_for_one_row_edit_it_once(cross_backend_runner):
    """Verify a prefix and an id8 naming one row give one edit and one entry.

    Mutation: selections not deduplicated by full id.
    Oracle: oplog count before and after.
    """
    runner = cross_backend_runner
    row_id = _remember_with(runner, 'The retry cap is three.')
    before = len(_meta_ops(runner))

    result = _meta(runner, row_id[:4], row_id[:8], 'doc=a')

    assert _changed(result) == [row_id[:8]]
    assert len(_meta_ops(runner)) - before == 1


def test_meta_rerun_of_the_id_form_changes_nothing(cross_backend_runner):
    """Verify a repeated id-form call replies with no change and no oplog row.

    Mutation: writing every row.
    Oracle: oplog count after the first and the second call.
    """
    runner = cross_backend_runner
    row_id = _remember_with(runner, 'The retry cap is three.')
    first = _meta(runner, row_id[:8], 'doc=a')
    after_first = len(_meta_ops(runner))

    second = _meta(runner, row_id[:8], 'doc=a')

    assert _changed(first) == [row_id[:8]]
    assert _changed(second) == []
    assert len(_meta_ops(runner)) == after_first == 1


def test_meta_where_rerun_refuses_with_the_zero_match_text(mm_runner):
    """Verify a `--where` call whose pair no row holds exits 1 with the text.

    Mutation: exit 0 on no match.
    Oracle: the text in spec section 5 step 15.
    """
    _remember_with(mm_runner, 'The retry cap is three.', 'doc=docs/old.md')
    args = ['--where', 'doc=docs/old.md', 'doc=docs/new.md']
    assert _meta(mm_runner, *args).exit_code == 0

    second = _meta(mm_runner, *args)

    assert second.exit_code == 1
    assert 'no current row holds doc=docs/old.md in default;' in second.output
    assert (' it may already be changed, or the stored value differs;'
            ' recall --basic docs/old.md finds it') in second.output


def test_meta_logs_one_oplog_entry_per_changed_row(cross_backend_runner):
    """Verify each changed row gets one `meta` entry holding before and after.

    Mutation: `insight_to_delta_dict` used, or `detail` left empty.
    Oracle: `json.loads` of each entry's detail against the hand-set
        pairs.
    """
    runner = cross_backend_runner
    stamped = _remember_with(runner, 'The retry cap is three.', 'doc=a')
    bare = _remember_with(runner, 'The backup job runs at midnight.')

    result = _meta(runner, stamped[:8], bare[:8], 'doc=b')

    entries = {entry.insight_id: json.loads(entry.detail)
               for entry in _meta_ops(runner)}
    assert sorted(_changed(result)) == sorted([stamped[:8], bare[:8]])
    assert entries[stamped] == {'before': {'doc': 'a'}, 'after': {'doc': 'b'}}
    assert entries[bare]['before'] is None
    assert entries[bare]['after'] == {'doc': 'b'}
    assert len(entries) == 2


def test_meta_accepts_a_result_at_the_byte_cap(cross_backend_runner):
    """Verify a result of exactly 200 bytes is stored.

    Mutation: `>=` in place of `>` in the cap check on results.
    Oracle: hand-computed `format_meta` byte length of 200.
    """
    runner = cross_backend_runner
    row_id = _remember_with(
        runner, 'The retry cap is three.', f'ticket={"y" * 176}')
    result_meta = {'ticket': 'y' * 176, 'doc': 'x' * 10}
    assert len(format_meta(result_meta).encode()) == 200

    result = _meta(runner, row_id[:8], f'doc={"x" * 10}')

    assert _changed(result) == [row_id[:8]]
    assert _meta_of(runner, row_id) == result_meta


def test_meta_refuses_a_result_over_the_byte_cap(cross_backend_runner):
    """Verify a result of 201 bytes is refused and nothing is written.

    Mutation: cap checked on the given pairs only.
    Oracle: hand-computed `format_meta` byte length of 201.
    """
    runner = cross_backend_runner
    row_id = _remember_with(
        runner, 'The retry cap is three.', f'ticket={"y" * 177}')
    assert len(format_meta(
        {'ticket': 'y' * 177, 'doc': 'x' * 10}).encode()) == 201

    result = _meta(runner, row_id[:8], f'doc={"x" * 10}')

    assert result.exit_code == 1
    assert f'meta of {row_id[:8]}' in result.output
    assert ('would be 201 bytes, max 200; nothing was written. Keep each'
            ' value a short pointer such as a path or ticket'
            ) in result.output
    assert _meta_of(runner, row_id) == {'ticket': 'y' * 177}


@pytest.mark.parametrize(('args', 'expected'), [
    pytest.param(
        ['{id8}', '--where', 'doc=a', 'doc=b'],
        _SELECTOR_CLASH.replace('{tok}', '{id8}'), id='both-selectors'),
    pytest.param(
        ['doc=b'],
        'name the rows: give their ids, or --where KEY=VALUE for every'
        ' current row holding that pair', id='neither-selector'),
    pytest.param(
        ['{id8}'],
        'give at least one KEY=VALUE to set; an empty value removes the'
        " key; insights show <id> prints a row's pairs", id='no-pair'),
    pytest.param(
        ['--where', 'doc=a'],
        'give at least one KEY=VALUE to set; an empty value removes the'
        " key; insights show <id> prints a row's pairs", id='where-without-pair'),
    pytest.param(
        ['--where', 'doc=', 'doc=b'],
        '--where doc= has an empty value; give the value the rows hold now',
        id='empty-where-value'),
    pytest.param(
        ['--where', 'doc=a', '--where', 'doc=b', 'doc=c'],
        'give --where once; it selects every current row holding one'
        ' KEY=VALUE pair', id='where-twice'),
    pytest.param(
        ['{id8}', 'doc=a', 'doc=b'],
        "--meta key 'doc' given twice", id='duplicate-key'),
    pytest.param(
        ['{id8}', 'Bad=x'], "--meta 'Bad=x'" + _BAD_KEY, id='bad-key'),
    pytest.param(
        ['', 'doc=b'],
        'an empty id matches every row; give an id or a prefix of one',
        id='empty-id-token'),
    pytest.param(
        ['--where', 'doc=a', 'doc=my', 'notes.md'],
        _SELECTOR_CLASH.replace('{tok}', 'notes.md'), id='unquoted-space'),
    pytest.param(
        ['--where', 'Bad=x', 'doc=b'], "--where 'Bad=x'" + _BAD_KEY,
        id='bad-where-key'),
    ])
def test_meta_refuses_a_malformed_call_and_writes_nothing(
        mm_runner, args, expected):
    """Verify each grammar guard refuses with its text before any write.

    Mutation: each guard removed; last-wins `--where`.
    Oracle: the texts in spec section 5.
    """
    row_id = _remember_with(mm_runner, 'The retry cap is three.', 'doc=a')
    argv = [arg.replace('{id8}', row_id[:8]) for arg in args]

    result = _meta(mm_runner, *argv)

    assert result.exit_code == 1, result.output
    assert expected.replace('{id8}', row_id[:8]) in result.output
    assert _meta_of(mm_runner, row_id) == {'doc': 'a'}
    assert _meta_ops(mm_runner) == []


def test_meta_unmatched_prefix_is_not_found_and_logs_nothing(mm_runner):
    """Verify a prefix no row holds is refused as not found.

    Mutation: trusting `resolve_id`'s unchanged return.
    Oracle: the absent prefix `zzz`.
    """
    _remember_with(mm_runner, 'The retry cap is three.', 'doc=a')

    result = _meta(mm_runner, 'zzz', 'doc=b')

    assert result.exit_code == 1
    assert 'insight zzz not found' in result.output
    assert _meta_ops(mm_runner) == []


def test_meta_refuses_a_forgotten_id(cross_backend_runner):
    """Verify a forgotten row is refused by id.

    Mutation: editing a retired row.
    Oracle: `replace`'s refusal wording for a forgotten row.
    """
    runner = cross_backend_runner
    row_id = _remember_with(runner, 'The retry cap is three.', 'doc=a')
    assert invoke(runner, ['forget', row_id]).exit_code == 0

    result = _meta(runner, row_id, 'doc=b')

    assert result.exit_code == 1
    assert f'insight {row_id} was forgotten' in result.output
    assert _meta_of(runner, row_id) == {'doc': 'a'}


def test_meta_refuses_a_replaced_id_and_names_the_chain_head(
        cross_backend_runner):
    """Verify a replaced row is refused with its successor and the chain head.

    Mutation: editing a retired row, or naming the direct successor in
        place of the head.
    Oracle: `replace`'s refusal wording, with a hand-built two-step
        chain.
    """
    runner = cross_backend_runner
    first = _remember_with(runner, 'The retry cap is three.', 'doc=a')
    second = _replace(runner, first, 'The retry cap is four.')
    head = _replace(runner, second, 'The retry cap is five.')

    result = _meta(runner, first, 'doc=b')

    assert result.exit_code == 1
    assert (f'insight {first} was replaced by {second}; set meta on {head},'
            f' or run insights show {first} --history') in result.output
    assert _meta_of(runner, first) == {'doc': 'a'}


@pytest.mark.scheduler_stopped
def test_meta_refuses_while_the_scheduler_is_stopped(mm_runner):
    """Verify the command is a write and refuses when the scheduler is stopped.

    Mutation: `_require_started` missing.
    Oracle: its message.
    """
    result = _meta(mm_runner, 'no-such-id', 'doc=a')

    assert result.exit_code == 1
    assert ("Scheduler is stopped; cannot write. Run 'memman scheduler start'"
            ' to enable.') in result.output


def test_meta_refuses_at_once_while_the_drain_lock_is_held(mm_runner):
    """Verify a held drain lock refuses the edit and writes nothing.

    Mutation: no lock.
    Oracle: the test holds `acquire(data_dir)`; the store snapshot is
        unchanged.
    """
    _, data_dir = mm_runner
    row_id = _remember_with(mm_runner, 'The retry cap is three.', 'doc=a')
    before = _snapshot(data_dir, 'default')
    fd = drain_lock.acquire(data_dir)
    try:
        result = _meta(mm_runner, row_id[:8], 'doc=b')
    finally:
        drain_lock.release(fd)

    assert result.exit_code == 1
    assert ('a drain, store merge, store drop, migrate, backup restore or'
            ' insights meta is running; re-run when it ends'
            ) in result.output
    assert _snapshot(data_dir, 'default') == before


@pytest.mark.parametrize(('args', 'expected'), [
    pytest.param(['zzz', 'doc=b'], 'insight zzz not found', id='not-found'),
    pytest.param(
        ['--where', 'doc=nope', 'doc=b'], 'no current row holds doc=nope',
        id='zero-match'),
    pytest.param(
        ['{id8}', f'doc={"x" * 100}'], 'nothing was written', id='over-cap'),
    ])
def test_meta_frees_the_drain_lock_after_a_refusal(mm_runner, args, expected):
    """Verify a refusal past the lock step releases the drain lock.

    Mutation: missing `finally: release`.
    Oracle: a non-blocking `acquire` in the test succeeds.
    """
    _, data_dir = mm_runner
    row_id = _remember_with(
        mm_runner, 'The retry cap is three.', 'doc=a', f'ticket={"y" * 100}')

    result = _meta(mm_runner, *[a.replace('{id8}', row_id[:8]) for a in args])

    assert result.exit_code == 1
    assert expected in result.output
    drain_lock.release(drain_lock.acquire(data_dir))
