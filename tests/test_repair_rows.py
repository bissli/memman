"""Hard delete and in-place rewrite of stored rows by scripts/repair_rows.py.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from memman.embed.fingerprint import META_KEY, seed_default_fingerprint
from memman.embed.swap import META_STATE as EMBED_SWAP_STATE
from memman.queue import enqueue, queue_db
from memman.store import factory
from memman.store.sqlite import SqliteBackend
from tests.conftest import EMBEDDING_DIM, make_insight

_SPEC = importlib.util.spec_from_file_location(
    'repair_rows',
    Path(__file__).resolve().parents[1] / 'scripts' / 'repair_rows.py')
repair_rows = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(repair_rows)


def _insert(backend, row_id, content=None, **overrides):
    backend.nodes.insert(make_insight(
        id=row_id, content=content or f'claim {row_id} holds', **overrides))


def _chain(backend, *row_ids):
    """Insert rows oldest first, each replaced by the next.
    """
    for row_id in row_ids:
        _insert(backend, row_id)
    for old, new in zip(row_ids, row_ids[1:]):
        assert backend.nodes.mark_replaced(old, new)


def _rows(backend):
    """`{id: (replaced_by, forgotten)}` for every stored row.
    """
    return {
        row_id: (replaced_by, deleted_at is not None)
        for row_id, replaced_by, deleted_at in repair_rows.fetch(
            backend, 'select id, replaced_by, deleted_at from {t}insights')
        }


def _oplog_ids(backend):
    return sorted(
        row[0] for row in repair_rows.fetch(
            backend, 'select insight_id from {t}oplog'))


def _run(backend, op_list):
    with backend.transaction():
        return repair_rows.run_ops(backend, op_list)


def _integrity_clean(backend):
    return all(not ids for ids in backend.nodes.replacement_integrity().values())


def _keyword_hits(backend, word):
    """Ids the keyword channel finds for one word.
    """
    if isinstance(backend, SqliteBackend):
        sql = """
select i.id
from insights_fts f
join insights i on i.rowid = f.rowid
where insights_fts match ?
"""
    else:
        sql = 'select id from {t}insights where ? = any(kw_tokens)'
    return sorted(row[0] for row in repair_rows.fetch(backend, sql, (word,)))


def test_delete_head_removes_whole_chain_and_its_oplog(backend):
    """Verify deleting a current head deletes every row it replaced.

    Mutation: deleting only the head, which leaves its predecessors
    pointing at a missing row; or keeping their oplog rows, which hold
    the deleted text.
    Oracle: hand-built chain a <- b <- c beside an unrelated row x.
    """
    _chain(backend, 'a', 'b', 'c')
    _insert(backend, 'x')
    backend.oplog.log(
        operation='replace', insight_id='a', detail='replaced by b')
    backend.oplog.log(operation='remember', insight_id='x', detail='x')

    report = _run(backend, [{'action': 'delete', 'id': 'c'}])

    assert _rows(backend) == {'x': (None, False)}
    assert _oplog_ids(backend) == ['c', 'x']
    assert sorted(report['deleted']) == ['a', 'b', 'c']
    assert report['spliced'] == []
    assert _integrity_clean(backend)


def test_delete_middle_row_splices_chain(backend):
    """Verify deleting a replaced row repoints its predecessors.

    Mutation: deleting the row without repointing, so a dangles; or
    deleting the predecessors as for a head.
    Oracle: chain a <- b <- c, deleting b must leave a <- c.
    """
    _chain(backend, 'a', 'b', 'c')

    report = _run(backend, [{'action': 'delete', 'id': 'b'}])

    assert _rows(backend) == {'a': ('c', False), 'c': (None, False)}
    assert report['spliced'] == ['b']
    assert _integrity_clean(backend)


def test_delete_oldest_row_leaves_rest_of_chain(backend):
    """Verify deleting the first version touches no other row.

    Mutation: treating a row with a successor as a head and deleting
    the successor's side of the chain.
    Oracle: chain a <- b <- c, deleting a must leave b <- c.
    """
    _chain(backend, 'a', 'b', 'c')

    _run(backend, [{'action': 'delete', 'id': 'a'}])

    assert _rows(backend) == {'b': ('c', False), 'c': (None, False)}


def test_delete_forgotten_head_removes_its_history(backend):
    """Verify a forgotten head counts as a head.

    Mutation: splicing on any non-current row, which leaves a pointing
    at the deleted b.
    Oracle: chain a <- b with b forgotten.
    """
    _chain(backend, 'a', 'b')
    backend.nodes.soft_delete('b')

    _run(backend, [{'action': 'delete', 'id': 'b'}])

    assert _rows(backend) == {}


def test_delete_dangling_row_removes_its_history(backend):
    """Verify a row pointing at a missing row counts as a head.

    Mutation: splicing to the missing successor, which keeps p pointing
    at an absent id.
    Oracle: p <- d, with d pointing at an id no row holds.
    """
    _chain(backend, 'p', 'd')
    repair_rows.execute(
        backend, "update {t}insights set replaced_by = 'gone' where id = 'd'")

    _run(backend, [{'action': 'delete', 'id': 'd'}])

    assert _rows(backend) == {}


def test_delete_refuses_an_absent_id(backend):
    """Verify an unknown id refuses rather than passing silently.

    Mutation: skipping an absent id, so a caller's typo reports success.
    Oracle: an id never inserted.
    """
    with pytest.raises(repair_rows.Refused, match='nope'):
        _run(backend, [{'action': 'delete', 'id': 'nope'}])


def test_refusal_rolls_back_the_whole_batch(backend):
    """Verify a refused op undoes the ops before it.

    Mutation: committing per op, so the first delete survives the
    refusal of the second.
    Oracle: a valid delete of a followed by a delete of an absent id.
    """
    _insert(backend, 'a')

    with pytest.raises(repair_rows.Refused):
        _run(backend, [
            {'action': 'delete', 'id': 'a'},
            {'action': 'delete', 'id': 'nope'},
            ])

    assert _rows(backend) == {'a': (None, False)}


def test_refuses_a_row_on_a_cycle(backend):
    """Verify a row whose chain never ends is refused.

    Mutation: walking predecessors without the cycle check, which loops
    or deletes rows outside the target's history.
    Oracle: a and b point at each other.
    """
    _insert(backend, 'a')
    _insert(backend, 'b')
    repair_rows.execute(
        backend, "update {t}insights set replaced_by = 'b' where id = 'a'")
    repair_rows.execute(
        backend, "update {t}insights set replaced_by = 'a' where id = 'b'")

    with pytest.raises(repair_rows.Refused, match='cycle'):
        _run(backend, [{'action': 'delete', 'id': 'a'}])


def test_rewrite_current_row_stages_enrichment_in_place(backend):
    """Verify a rewrite keeps identity and clears all derived state.

    Mutation: leaving a stamp, the summary, the vector or the SQLite
    swap vector of the old text; restamping created_at; or rebuilding
    Postgres `kw_tokens` from the content alone, without the meta.
    Oracle: an enriched row; its pre-rewrite created_at, author and meta.
    """
    _insert(
        backend, 'a', content='zebra grazing', author='alice',
        meta={'ticket': 'quokka'})
    backend.nodes.update_enrichment('a', summary='zebra', summary_model='m')
    backend.nodes.update_embedding('a', [0.1] * EMBEDDING_DIM, 'e')
    backend.nodes.stamp_enrich_attempted('a')
    backend.nodes.stamp_enriched('a')
    if isinstance(backend, SqliteBackend):
        repair_rows.execute(
            backend,
            "update {t}insights set embedding_pending = x'00' where id = 'a'")
    before = backend.nodes.get_include_deleted('a')

    _run(backend, [{
        'action': 'rewrite', 'id': 'a',
        'content': ['The walrus basks on the ice floe.'],
        }])

    raw = backend.nodes.get_raw('a')
    assert raw.content == 'The walrus basks on the ice floe.'
    assert (raw.summary, raw.summary_model) == (None, None)
    assert (raw.embedding, raw.embedding_model) == (None, None)
    assert (raw.enrich_attempted_at, raw.enriched_at) == (None, None)
    assert (raw.created_at, raw.author, raw.meta) == (
        before.created_at, 'alice', {'ticket': 'quokka'})
    assert 'a' in backend.nodes.get_pending_enrich_ids(limit=10)
    assert _keyword_hits(backend, 'walrus') == ['a']
    assert _keyword_hits(backend, 'quokka') == ['a']
    assert _keyword_hits(backend, 'zebra') == []
    if isinstance(backend, SqliteBackend):
        assert repair_rows.fetch(
            backend, "select embedding_pending from {t}insights where id = 'a'"
            ) == [(None,)]


def test_rewrite_split_keeps_date_author_and_meta(backend):
    """Verify each extra text becomes a current row of the same origin.

    Mutation: inserting the split rows with today's date, no author or
    no meta, or stamped as already enriched.
    Oracle: the source row's own created_at, author and meta.
    """
    _insert(
        backend, 'a', author='alice', meta={'ticket': 'T1'})
    source = backend.nodes.get_include_deleted('a')

    report = _run(backend, [{
        'action': 'rewrite', 'id': 'a',
        'content': [
            'The retry cap stays at three.',
            'The queue drains every minute.',
            ],
        }])

    (new_id,) = report['inserted']
    new = backend.nodes.get_raw(new_id)
    assert new.content == 'The queue drains every minute.'
    assert (new.created_at, new.author, new.meta) == (
        source.created_at, 'alice', {'ticket': 'T1'})
    assert (new.replaced_by, new.deleted_at, new.enrich_attempted_at) == (
        None, None, None)
    assert backend.nodes.get(new_id) is not None
    assert backend.nodes.get('a').content == 'The retry cap stays at three.'


def test_rewrite_replaced_row_detaches_it_from_its_successor(backend):
    """Verify a rewritten history row becomes current with its own history.

    Mutation: keeping the successor pointer, which hides the row; or
    moving the row's predecessors to the successor.
    Oracle: chain a <- b <- c, rewriting b.
    """
    _chain(backend, 'a', 'b', 'c')

    _run(backend, [{
        'action': 'rewrite', 'id': 'b',
        'content': ['The broker is redis.'],
        }])

    assert _rows(backend) == {
        'a': ('b', False), 'b': (None, False), 'c': (None, False)}
    assert backend.nodes.get('b').content == 'The broker is redis.'
    assert _integrity_clean(backend)


def test_rewrite_forgotten_row_revives_it(backend):
    """Verify a rewrite clears deleted_at.

    Mutation: rewriting the text but leaving the row forgotten.
    Oracle: a forgotten row a.
    """
    _insert(backend, 'a')
    backend.nodes.soft_delete('a')

    _run(backend, [{
        'action': 'rewrite', 'id': 'a',
        'content': ['The broker is redis.'],
        }])

    assert backend.nodes.get('a').content == 'The broker is redis.'


def test_ops_run_in_plan_order(backend):
    """Verify a later op sees the state an earlier op left.

    Mutation: running deletes before rewrites, or validating every op
    against the starting state, which deletes a with its old head.
    Oracle: chain a <- b <- c; reviving a then deleting c leaves a alone.
    """
    _chain(backend, 'a', 'b', 'c')

    _run(backend, [
        {'action': 'rewrite', 'id': 'a', 'content': ['The broker is redis.']},
        {'action': 'delete', 'id': 'c'},
        ])

    assert _rows(backend) == {'a': (None, False)}


@pytest.mark.parametrize(('op', 'message'), [
    ({'action': 'drop', 'id': 'a'}, 'action'),
    ({'action': 'delete'}, 'id'),
    ({'action': 'rewrite', 'id': 'a', 'content': []}, 'content'),
    ({'action': 'rewrite', 'id': 'a', 'content': 'one'}, 'content'),
    ({'action': 'rewrite', 'id': 'a', 'content': ['Note: x']}, 'label'),
    ({'action': 'rewrite', 'id': 'a', 'content': ['one\ntwo']}, 'several lines'),
    ])
def test_check_ops_refuses_bad_shapes(op, message):
    """Verify an op the CLI write rules refuse never reaches the store.

    Mutation: skipping the memman refusal checks on rewrite text, or
    accepting a bare string as a one-item content list.
    Oracle: one bad field per case, against the remember refusals.
    """
    with pytest.raises(repair_rows.Refused, match=message):
        repair_rows.check_ops([op])


@pytest.fixture
def sqlite_store(tmp_path):
    """A seeded sqlite store `default` in a fresh data dir.
    """
    data_dir = str(tmp_path / 'memman')
    backend = factory.open_backend('default', data_dir, create=True)
    backend.meta.set(META_KEY, seed_default_fingerprint().to_json())
    _insert(backend, 'a')
    backend.close()
    return data_dir


def _stored_ids(data_dir):
    with factory.open_backend('default', data_dir, read_only=True) as backend:
        return sorted(_rows(backend))


@pytest.mark.parametrize(('op', 'status'), [
    ({'action': 'delete', 'id': 'a'}, 'pending'),
    ({'action': 'rewrite', 'id': 'a', 'content': ['The broker is redis.']},
     'failed'),
    ])
def test_apply_refuses_a_row_a_queued_replace_names(
        sqlite_store, tmp_path, op, status):
    """Verify a pending or failed replace of a touched row blocks the batch.

    Mutation: dropping the queue check, guarding deleted ids only, or
    reading pending rows only, so the drain later replaces a row this
    run deleted or rewrote.
    Oracle: a replace of a enqueued and not yet drained, in each status
    the drain still acts on.
    """
    with queue_db(sqlite_store) as conn:
        row_id, _ = enqueue(
            conn, 'default', 'The broker is kafka.', replaced_id='a')
        conn.execute('update queue set status = ? where id = ?', (status, row_id))
        conn.commit()

    with pytest.raises(repair_rows.Refused, match='queued'):
        repair_rows.apply(
            'default', [op], sqlite_store,
            dry_run=False, backup_dir=str(tmp_path / 'bk'))

    assert _stored_ids(sqlite_store) == ['a']


def test_apply_dry_run_writes_nothing(sqlite_store):
    """Verify a dry run reports the batch and rolls it back.

    Mutation: committing the dry run, or reporting nothing.
    Oracle: the one row a, which a real run would delete.
    """
    report = repair_rows.apply(
        'default', [{'action': 'delete', 'id': 'a'}], sqlite_store,
        dry_run=True, backup_dir=None)

    assert report['deleted'] == ['a']
    assert _stored_ids(sqlite_store) == ['a']


def test_apply_backs_up_before_writing(sqlite_store, tmp_path):
    """Verify a real run writes a bundle and refuses without a target.

    Mutation: writing with no backup target, or skipping the bundle.
    Oracle: the bundle path the report names, and the row left in place
    by the refused run.
    """
    with pytest.raises(repair_rows.Refused, match='backup'):
        repair_rows.apply(
            'default', [{'action': 'delete', 'id': 'a'}], sqlite_store,
            dry_run=False, backup_dir=None)
    assert _stored_ids(sqlite_store) == ['a']

    report = repair_rows.apply(
        'default', [{'action': 'delete', 'id': 'a'}], sqlite_store,
        dry_run=False, backup_dir=str(tmp_path / 'bk'))

    assert Path(report['backup']).is_file()
    assert _stored_ids(sqlite_store) == []


def test_apply_refuses_a_branch_parent(sqlite_store):
    """Verify a store holding a branch token is refused before any write.

    Mutation: dropping the branch-token check, so a delete in the parent
    lets the branch's copy of the row show through.
    Oracle: the token key `memman store branch` writes into the parent.
    """
    with factory.open_backend('default', sqlite_store) as backend:
        backend.meta.set('branch_token:feature', 'tok')

    with pytest.raises(repair_rows.Refused, match='branch'):
        repair_rows.apply(
            'default', [{'action': 'delete', 'id': 'a'}], sqlite_store,
            dry_run=True, backup_dir=None)

    assert _stored_ids(sqlite_store) == ['a']


def _oplog_entries(backend):
    return sorted(
        (operation, insight_id) for operation, insight_id in repair_rows.fetch(
            backend, 'select operation, insight_id from {t}oplog'))


def test_delete_purges_every_oplog_entry_holding_the_rows_text(backend):
    """Verify a splice leaves no oplog entry with any text the row held.

    Mutation: matching oplog entries by id or by the 'replaced by <id>'
    detail alone, which keeps a merge's `replace` entry under the
    surviving predecessor and an entry holding the row's older text; or
    purging the successor's text that b's own `replace` entry holds.
    Oracle: hand-written entries for chain p <- b <- s, one per way an
    entry can hold b's current or earlier text, and s's own entry.
    """
    _insert(backend, 'p', content='p text')
    _insert(backend, 'b', content='b text')
    _insert(backend, 's', content='s text')
    _insert(backend, 'q', content='q text')
    assert backend.nodes.mark_replaced('p', 'b')
    assert backend.nodes.mark_replaced('b', 's')
    for operation, insight_id, detail, before, after in [
            ('remember', 'p', 'p text', None, 'p text'),
            ('replace', 'p', 'merged from br', 'p text', 'b text'),
            ('remember', 'b', 'b old text', None, 'b old text'),
            ('rewrite', 'b', '{}', 'b old text', 'b text'),
            ('replace', 'b', 'replaced by s', 'b text', 's text'),
            ('replace', 'q', 'replaced by x', 'q text', 'b old text'),
            ('remember', 's', 's text', None, 's text'),
            ]:
        backend.oplog.log(
            operation=operation, insight_id=insight_id, detail=detail,
            before={'content': before, 'summary': ''} if before else None,
            after={'content': after, 'summary': ''})

    _run(backend, [{'action': 'delete', 'id': 'b'}])

    assert _oplog_entries(backend) == [
        ('purge', 'b'), ('remember', 'p'), ('remember', 's')]


def test_delete_splices_every_predecessor(backend):
    """Verify a row with two predecessors hands both to its successor.

    Mutation: repointing only the first predecessor found.
    Oracle: a1 and a2 both replaced by b, b replaced by c.
    """
    for row_id in ('a1', 'a2', 'b', 'c'):
        _insert(backend, row_id)
    assert backend.nodes.mark_replaced('a1', 'b')
    assert backend.nodes.mark_replaced('b', 'c')
    repair_rows.execute(
        backend, "update {t}insights set replaced_by = 'b' where id = 'a2'")

    _run(backend, [{'action': 'delete', 'id': 'b'}])

    assert _rows(backend) == {
        'a1': ('c', False), 'a2': ('c', False), 'c': (None, False)}


def test_delete_of_a_row_this_run_deleted_is_a_no_op(backend):
    """Verify naming a head and its history in one batch, in any order, works.

    Mutation: refusing an id an earlier op of the run already deleted,
    which fails a batch of every bad row an audit lists.
    Oracle: chain a <- b <- c deleted as c then a.
    """
    _chain(backend, 'a', 'b', 'c')

    report = _run(backend, [
        {'action': 'delete', 'id': 'c'},
        {'action': 'delete', 'id': 'a'},
        ])

    assert _rows(backend) == {}
    assert sorted(report['deleted']) == ['a', 'b', 'c']


def test_integrity_guard_refuses_a_run_that_breaks_a_chain(backend, monkeypatch):
    """Verify the closing integrity check rolls back a corrupting op.

    Mutation: skipping the post-run integrity check, so a defect in an
    op commits a dangling pointer.
    Oracle: a stand-in delete that removes b without repointing a.
    """
    _chain(backend, 'a', 'b', 'c')

    def delete_without_splice(backend, row_id, report):
        repair_rows.execute(
            backend, 'delete from {t}insights where id = ?', (row_id,))

    monkeypatch.setattr(repair_rows, '_delete', delete_without_splice)

    with pytest.raises(repair_rows.Refused, match='dangling'):
        _run(backend, [{'action': 'delete', 'id': 'b'}])

    assert sorted(_rows(backend)) == ['a', 'b', 'c']


def test_apply_refuses_during_an_embedding_swap(sqlite_store):
    """Verify a recorded swap blocks the run.

    Mutation: dropping the swap check, so the cutover hands a rewritten
    row the vector of its old text.
    Oracle: the swap state key `memman embed swap` records.
    """
    with factory.open_backend('default', sqlite_store) as backend:
        backend.meta.set(EMBED_SWAP_STATE, 'backfilling')

    with pytest.raises(repair_rows.Refused, match='swap'):
        repair_rows.apply(
            'default', [{'action': 'delete', 'id': 'a'}], sqlite_store,
            dry_run=True, backup_dir=None)


def test_apply_keeps_every_runs_backup(sqlite_store, tmp_path):
    """Verify two runs in the same second write two bundles.

    Mutation: building every bundle straight into the backup directory,
    where a second-resolution name lets the later run overwrite the
    earlier run's only undo.
    Oracle: two consecutive runs and the two paths they report.
    """
    op_list = [{'action': 'rewrite', 'id': 'a', 'content': ['The cap is three.']}]

    first = repair_rows.apply(
        'default', op_list, sqlite_store,
        dry_run=False, backup_dir=str(tmp_path / 'bk'))
    second = repair_rows.apply(
        'default', op_list, sqlite_store,
        dry_run=False, backup_dir=str(tmp_path / 'bk'))

    assert first['backup'] != second['backup']
    assert Path(first['backup']).is_file()
    assert Path(second['backup']).is_file()


def test_apply_refuses_a_bundle_without_the_store(
        sqlite_store, tmp_path, monkeypatch):
    """Verify a bundle whose snapshot of the store failed blocks the run.

    Mutation: trusting `build_bundle`, which records a failed store
    snapshot in its manifest instead of raising.
    Oracle: a manifest entry with status 'failed'.
    """
    def failed_bundle(data_dir, target):
        return {
            'bundle': f'{target}/b.tar.gz',
            'stores': [{'name': 'default', 'status': 'failed'}],
            }

    monkeypatch.setattr(repair_rows, 'build_bundle', failed_bundle)

    with pytest.raises(repair_rows.Refused, match='no snapshot'):
        repair_rows.apply(
            'default', [{'action': 'delete', 'id': 'a'}], sqlite_store,
            dry_run=False, backup_dir=str(tmp_path / 'bk'))

    assert _stored_ids(sqlite_store) == ['a']


def test_audit_reports_state_integrity_and_refusal(sqlite_store, capsys):
    """Verify each audit line names the row's state, chain and refusal.

    Mutation: calling a forgotten replaced row replaced, missing the
    dangling population, or skipping the remember refusal.
    Oracle: hand-built rows, one per state, and the label refusal.
    """
    with factory.open_backend('default', sqlite_store) as backend:
        _insert(backend, 'b', content='Note: the cap is three')
        _insert(backend, 'c')
        _insert(backend, 'd')
        assert backend.nodes.mark_replaced('c', 'b')
        backend.nodes.soft_delete('c')
        repair_rows.execute(
            backend, "update {t}insights set replaced_by = 'gone' where id = 'd'")

    repair_rows.audit('default', sqlite_store)

    line_by_id = {
        line['id']: line for line in map(
            json.loads, capsys.readouterr().out.splitlines())}
    assert {row_id: line['state'] for row_id, line in line_by_id.items()} == {
        'a': 'current', 'b': 'current', 'c': 'forgotten', 'd': 'replaced'}
    assert line_by_id['d']['integrity'] == ['dangling']
    assert line_by_id['a']['refusal'] is None
    assert 'label' in line_by_id['b']['refusal']


def test_audit_refuses_a_branch_parent(sqlite_store):
    """Verify audit takes the same branch guard as a write.

    Mutation: auditing a branch, whose overlay nodes lack the internals
    the SQL adapter reads.
    Oracle: the token key `memman store branch` writes into the parent.
    """
    with factory.open_backend('default', sqlite_store) as backend:
        backend.meta.set('branch_token:feature', 'tok')

    with pytest.raises(repair_rows.Refused, match='branch'):
        repair_rows.audit('default', sqlite_store)


@pytest.mark.parametrize(('argv', 'code', 'key'), [
    (['delete', 'default', 'nope', '--dry-run'], 2, 'refused'),
    (['delete', 'missing', 'a', '--dry-run'], 1, 'error'),
    ])
def test_main_reports_failures_as_json(
        sqlite_store, monkeypatch, capsys, argv, code, key):
    """Verify a calling process gets an exit code and one JSON line.

    Mutation: letting a non-refusal error escape as a traceback, or
    exiting 0 on a refusal.
    Oracle: an absent id, and a store no config names.
    """
    monkeypatch.setenv('MEMMAN_DATA_DIR', sqlite_store)
    monkeypatch.setattr(sys, 'argv', ['repair_rows.py', *argv])

    assert repair_rows.main() == code
    assert key in json.loads(capsys.readouterr().err)


def test_apply_enriches_every_rewritten_and_split_row(sqlite_store, tmp_path):
    """Verify a real run leaves its rewritten and new rows enriched.

    Mutation: leaving the rows pending for the drain, which enriches a
    store only after a queued write to it, or enriching only the row
    rewritten in place.
    Oracle: the mocked LLM and embedder, and the rows' own stamps.
    """
    report = repair_rows.apply(
        'default',
        [{'action': 'rewrite', 'id': 'a',
          'content': ['The cap is three.', 'The queue drains hourly.']}],
        sqlite_store, dry_run=False, backup_dir=str(tmp_path / 'bk'))

    row_ids = ['a', *report['inserted']]
    assert sorted(report['enriched']) == sorted(row_ids)
    with factory.open_backend('default', sqlite_store, read_only=True) as backend:
        for row_id in row_ids:
            raw = backend.nodes.get_raw(row_id)
            assert raw.enriched_at is not None
            assert raw.summary_model
            assert raw.embedding is not None


def test_apply_refuses_a_rewrite_with_no_embedder(
        sqlite_store, tmp_path, monkeypatch):
    """Verify a rewrite that cannot be enriched refuses before writing.

    Mutation: checking only that the clients build, so a blank API key
    commits the rewrite and strands every row it touched.
    Oracle: an embedder whose `available()` is False, as the registry
    returns for a client with no key.
    """
    class Unavailable:
        def available(self):
            return False

    monkeypatch.setattr(
        repair_rows.fingerprint, 'bound_embedder', lambda backend: Unavailable())

    with pytest.raises(repair_rows.Refused, match='embedder'):
        repair_rows.apply(
            'default',
            [{'action': 'rewrite', 'id': 'a', 'content': ['The cap is three.']}],
            sqlite_store, dry_run=False, backup_dir=str(tmp_path / 'bk'))

    with factory.open_backend('default', sqlite_store, read_only=True) as backend:
        assert backend.nodes.get('a').content == 'claim a holds'


def test_apply_reports_an_enrich_failure_after_the_commit(
        sqlite_store, tmp_path, monkeypatch):
    """Verify a failed enrichment keeps the committed run's report.

    Mutation: letting the error escape, which drops the report and the
    backup path of a repair that did apply.
    Oracle: an enrich step that raises after the rewrite committed.
    """
    def broken_enrich(backend, **kwargs):
        raise RuntimeError('database is locked')

    monkeypatch.setattr(repair_rows, 'enrich_pending', broken_enrich)

    report = repair_rows.apply(
        'default',
        [{'action': 'rewrite', 'id': 'a', 'content': ['The cap is three.']}],
        sqlite_store, dry_run=False, backup_dir=str(tmp_path / 'bk'))

    assert 'database is locked' in report['enrich_error']
    assert Path(report['backup']).is_file()
    with factory.open_backend('default', sqlite_store, read_only=True) as backend:
        assert backend.nodes.get('a').content == 'The cap is three.'


def test_meta_sets_pairs_in_place_without_reenriching(backend):
    """Verify a meta op swaps the pairs and leaves text and enrichment alone.

    Mutation: merging into the old pairs instead of setting them, as
    `--meta` on replace sets them; clearing the enrich stamps; or
    skipping the keyword index, which still finds the old value.
    Oracle: an enriched row whose meta holds a word its text lacks.
    """
    _chain(backend, 'p', 'a')
    repair_rows.execute(
        backend, "update {t}insights set meta = ? where id = 'a'",
        ('{"ticket": "quokka", "ref": "x"}',))
    backend.nodes.update_embedding('a', [0.1] * EMBEDDING_DIM, 'e')
    backend.nodes.stamp_enrich_attempted('a')
    backend.nodes.stamp_enriched('a')
    if not isinstance(backend, SqliteBackend):
        repair_rows.execute(
            backend,
            "update {t}insights set kw_tokens = ? where id = 'a'",
            (['claim', 'a', 'holds', 'ticket', 'quokka', 'ref', 'x'],))
    before = backend.nodes.get_raw('a')

    report = _run(backend, [{
        'action': 'meta', 'id': 'a', 'meta': {'hq': 'wombat', 'ref': ''}}])

    raw = backend.nodes.get_raw('a')
    assert raw.meta == {'hq': 'wombat'}
    assert (raw.content, raw.created_at, raw.enriched_at, raw.embedding_model) == (
        before.content, before.created_at, before.enriched_at, 'e')
    assert _rows(backend) == {'p': ('a', False), 'a': (None, False)}
    assert _keyword_hits(backend, 'wombat') == ['a']
    assert _keyword_hits(backend, 'quokka') == []
    assert report['meta_set'] == ['a']


def test_meta_with_only_empty_values_clears_the_pairs(backend):
    """Verify an all-empty meta op leaves the row with no meta.

    Mutation: storing `{}` as a JSON object where memman stores no meta.
    Oracle: the Insight contract, which stores None and {} as no meta.
    """
    _insert(backend, 'a', meta={'ticket': 'T1'})

    _run(backend, [{'action': 'meta', 'id': 'a', 'meta': {'ticket': ''}}])

    assert repair_rows.fetch(
        backend, "select meta from {t}insights where id = 'a'") == [(None,)]


@pytest.mark.parametrize(('meta', 'message'), [
    ({'Bad-Key': 'x'}, 'key'),
    ({'a=b': 'c'}, 'key'),
    ({'hq': 3}, 'text'),
    ('hq=x', 'meta'),
    ({'hq': 'a\nb'}, 'line break'),
    ])
def test_check_ops_refuses_bad_meta(meta, message):
    """Verify meta pairs face the same checks as `--meta`.

    Mutation: skipping the `--meta` checks, so a key memman's CLI
    refuses lands in the store.
    Oracle: one bad pair per case, against the `--meta` refusals.
    """
    with pytest.raises(repair_rows.Refused, match=message):
        repair_rows.check_ops([{'action': 'meta', 'id': 'a', 'meta': meta}])


@pytest.mark.postgres
@pytest.mark.parametrize('backend_kind', ['postgres'])
def test_meta_on_a_forgotten_row_keeps_its_token_set_empty(backend):
    """Verify a meta op leaves a forgotten Postgres row out of the index.

    Mutation: recomputing `kw_tokens` for every row, which puts a
    forgotten row's tokens back in the keyword index.
    Oracle: `soft_delete`, which empties the token set.
    """
    _insert(backend, 'a')
    backend.nodes.soft_delete('a')

    _run(backend, [{'action': 'meta', 'id': 'a', 'meta': {'hq': 'wombat'}}])

    assert repair_rows.fetch(
        backend, "select kw_tokens from {t}insights where id = 'a'") == [([],)]
