"""D1: one field per job — source, session_id, queue_uuid.

`source` is provenance stored verbatim; `session_id` is the temporal
chain key; idempotency rides on a uuid4 minted at enqueue. These
tests pin the decomposition end to end through the real queue drain.
"""

import json
import sqlite3

from memman import config
from memman.store.db import store_dir
from tests.conftest import force_drain, invoke, parse_remember


def _queue_row(data_dir, queue_id):
    """Return (session_id, queue_uuid) for a queue row."""
    from memman.queue import queue_db
    with queue_db(data_dir) as conn:
        return conn.execute(
            'select session_id, queue_uuid from queue where id = ?',
            (queue_id,)).fetchone()


def _stored(data_dir, store, where, params):
    """Rows of (id, source, session_id, queue_uuid) from the store."""
    db_path = f'{store_dir(data_dir, store)}/memman.db'
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            'select id, source, session_id, queue_uuid from insights'
            f' where {where} and deleted_at is null', params).fetchall()


def _requeue(data_dir, queue_id):
    """Flip a drained queue row back to pending (simulated replay)."""
    from memman.queue import queue_db
    with queue_db(data_dir) as conn:
        conn.execute(
            "update queue set status = 'pending', attempts = 0,"
            ' claimed_at = null, worker_pid = null,'
            ' processed_at = null where id = ?', (queue_id,))
        conn.commit()


def test_plan_fact_propagates_session_and_queue_uuid(mm_runner):
    """The STORED insight carries the queue row's session and uuid.

    Mutation: dropping either field where `_plan_fact` mints fresh
        Insights (`pipeline/remember.py`) — every other part of D1
        would still be correct and the feature a silent no-op.
    Oracle: the stored row's `session_id`/`queue_uuid` equal the
        queue row's, through the real pipeline.
    """
    result = invoke(mm_runner, [
        'remember', 'session propagation end to end',
        '--session', 'sess-prop', '--no-reconcile'])
    assert result.exit_code == 0, result.output
    raw = json.loads(result.output)
    _, data_dir = mm_runner
    session_id, queue_uuid = _queue_row(data_dir, raw['queue_id'])
    assert session_id == 'sess-prop'
    rows = _stored(data_dir, raw['store'],
                   'queue_uuid = ?', (queue_uuid,))
    assert len(rows) == 1
    assert rows[0][2] == 'sess-prop'
    assert rows[0][3] == queue_uuid


def test_multi_fact_row_shares_one_queue_uuid(mm_runner):
    """Several facts from one remember call share the row's uuid.

    Mutation: putting `unique` on the insights `queue_uuid` column —
        the second fact's insert would fail. Only the QUEUE table's
        column is unique.
    Oracle: two extracted facts, both stored, identical uuids.
    """
    from unittest.mock import patch

    def _two_facts(llm_client, content):
        return [
            {'text': 'Switched from Flask to FastAPI',
             'category': 'decision', 'importance': 4,
             'entities': ['FastAPI']},
            {'text': 'Redis cache configured with 4GB max memory',
             'category': 'fact', 'importance': 3,
             'entities': ['Redis']},
            ]

    with patch('memman.llm.extract.extract_facts', _two_facts):
        result = invoke(mm_runner, [
            'remember', 'Switched to FastAPI and configured Redis'])
    assert result.exit_code == 0, result.output
    raw = json.loads(result.output)
    _, data_dir = mm_runner
    _sess, queue_uuid = _queue_row(data_dir, raw['queue_id'])
    rows = _stored(data_dir, raw['store'],
                   'queue_uuid = ?', (queue_uuid,))
    assert len(rows) == 2


def test_source_round_trips_verbatim(mm_runner):
    """The default `user` source is stored as `'user'`, not `queue:N`.

    Mutation: restoring the `!= 'user'` mapping at the CLI enqueue.
    Oracle: default write stores `'user'`; `--source agent` stores
        `'agent'`.
    """
    _, data_dir = mm_runner
    r1 = invoke(mm_runner, [
        'remember', 'a default sourced note', '--no-reconcile'])
    raw1 = json.loads(r1.output)
    r2 = invoke(mm_runner, [
        'remember', 'an agent sourced note', '--no-reconcile',
        '--source', 'agent'])
    raw2 = json.loads(r2.output)
    _s1, u1 = _queue_row(data_dir, raw1['queue_id'])
    _s2, u2 = _queue_row(data_dir, raw2['queue_id'])
    assert _stored(
        data_dir, raw1['store'], 'queue_uuid = ?', (u1,))[0][1] == 'user'
    assert _stored(
        data_dir, raw2['store'], 'queue_uuid = ?', (u2,))[0][1] == 'agent'


def test_source_defaults_to_user_for_programmatic_enqueue(mm_runner):
    """A bare `enqueue()` with no hint yields `source = 'user'`.

    Mutation: dropping the `or 'user'` at the drain — a programmatic
        enqueue (hint_source None) would write NULL into a column the
        recall filter compares with `=`.
    Oracle: direct enqueue, drained, stores `'user'`.
    """
    from memman.queue import enqueue, queue_db
    _, data_dir = mm_runner
    with queue_db(data_dir) as conn:
        row_id = enqueue(conn, store='default',
                         content='programmatic enqueue note')
    force_drain(data_dir)
    _sess, queue_uuid = _queue_row(data_dir, row_id)
    rows = _stored(data_dir, 'default', 'queue_uuid = ?', (queue_uuid,))
    assert len(rows) == 1
    assert rows[0][1] == 'user'


def test_replace_inherits_source(mm_runner):
    """`replace` without `--source` keeps the old insight's source.

    Mutation: dropping the replace-side fix — its own
        `source_explicit` guard (independent of the remember-side
        mapping) discarded the inherited source as a None hint, and
        the drain then fell back to the default.
    Oracle: the replacement row carries `'agent'` from the original.
    """
    _, data_dir = mm_runner
    r1 = invoke(mm_runner, [
        'remember', 'original agent note', '--no-reconcile',
        '--source', 'agent'])
    old = parse_remember(r1, mm_runner)
    r2 = invoke(mm_runner, [
        'replace', old['id'], 'updated agent note'])
    assert r2.exit_code == 0, r2.output
    raw2 = json.loads(r2.output)
    _sess, queue_uuid = _queue_row(data_dir, raw2['queue_id'])
    rows = _stored(data_dir, raw2['store'],
                   'queue_uuid = ?', (queue_uuid,))
    assert len(rows) == 1
    assert rows[0][1] == 'agent'


def test_session_flag_and_env_default(mm_runner, monkeypatch):
    """`--session` beats `$MEMMAN_SESSION_ID`; the env fills the gap.

    Mutation: dropping the env fallback, or letting env override an
        explicit flag.
    Oracle: no flag stores the env value; an explicit flag stores its
        own value with the env still set.
    """
    _, data_dir = mm_runner
    monkeypatch.setenv('MEMMAN_SESSION_ID', 'env-sess')
    r1 = invoke(mm_runner, [
        'remember', 'env session note', '--no-reconcile'])
    raw1 = json.loads(r1.output)
    r2 = invoke(mm_runner, [
        'remember', 'flag session note', '--no-reconcile',
        '--session', 'cli-sess'])
    raw2 = json.loads(r2.output)
    _s1, u1 = _queue_row(data_dir, raw1['queue_id'])
    _s2, u2 = _queue_row(data_dir, raw2['queue_id'])
    assert _stored(
        data_dir, raw1['store'],
        'queue_uuid = ?', (u1,))[0][2] == 'env-sess'
    assert _stored(
        data_dir, raw2['store'],
        'queue_uuid = ?', (u2,))[0][2] == 'cli-sess'


def test_session_env_precedence_ladder(mm_runner, monkeypatch):
    """`--session` beats `$MEMMAN_SESSION_ID` beats `$CLAUDE_CODE_SESSION_ID`.

    Mutation: dropping `CLAUDE_CODE_SESSION_ID` from the envvar list
        (a subagent, which is never told the id, reverts to a NULL
        session), or listing it ahead of `MEMMAN_SESSION_ID` (the
        memman variable would stop being the operator override).
    Oracle: three writes under a fixed env, storing the Claude id,
        then the memman id, then the flag value.
    """
    _, data_dir = mm_runner
    monkeypatch.setenv('CLAUDE_CODE_SESSION_ID', 'claude-sess')
    claude_only = invoke(mm_runner, [
        'remember', 'claude env session note', '--no-reconcile'])
    monkeypatch.setenv('MEMMAN_SESSION_ID', 'memman-sess')
    memman_over_claude = invoke(mm_runner, [
        'remember', 'memman env session note', '--no-reconcile'])
    flag_over_both = invoke(mm_runner, [
        'remember', 'flag session note', '--no-reconcile',
        '--session', 'flag-sess'])
    stored = []
    for result in (claude_only, memman_over_claude, flag_over_both):
        assert result.exit_code == 0, result.output
        raw = json.loads(result.output)
        _sess, queue_uuid = _queue_row(data_dir, raw['queue_id'])
        stored.append(_stored(
            data_dir, raw['store'], 'queue_uuid = ?', (queue_uuid,))[0][2])
    assert stored == ['claude-sess', 'memman-sess', 'flag-sess']


def test_replace_reads_claude_code_session_env(mm_runner, monkeypatch):
    """`replace` honors `$CLAUDE_CODE_SESSION_ID` like `remember` does.

    Mutation: extending the envvar list on the `remember` option
        alone - a subagent's `replace` would keep storing NULL, or
        inherit the original row's session and join the wrong chain.
    Oracle: the replacement carries the exported Claude Code id with
        no flag passed, while the row it replaces carries a different
        session entirely.
    """
    _, data_dir = mm_runner
    r1 = invoke(mm_runner, [
        'remember', 'note awaiting replacement', '--no-reconcile',
        '--session', 'sess-original'])
    old = parse_remember(r1, mm_runner)
    monkeypatch.setenv('CLAUDE_CODE_SESSION_ID', 'claude-replace')
    r2 = invoke(mm_runner, ['replace', old['id'], 'replacement note'])
    assert r2.exit_code == 0, r2.output
    raw2 = json.loads(r2.output)
    _sess, queue_uuid = _queue_row(data_dir, raw2['queue_id'])
    rows = _stored(data_dir, raw2['store'], 'queue_uuid = ?', (queue_uuid,))
    assert len(rows) == 1
    assert rows[0][2] == 'claude-replace'


def test_claude_session_env_is_never_reported_or_persisted(
        mm_runner, monkeypatch):
    """A foreign session id stays out of both config surfaces.

    Mutation: adding `CLAUDE_SESSION_ID` to `INSTALLABLE_KEYS` or
        `_PROCESS_CONTROL_VARS`, which would leak a live session id
        into `memman config show` and open a path to persisting one
        - a stale persisted id fuses every later write into one
        false backbone chain.
    Oracle: `enumerate_effective_config`, the reporting path behind
        `memman config show`, omits the key while it is exported,
        and the env file never gains it.
    """
    _, data_dir = mm_runner
    monkeypatch.setenv('CLAUDE_CODE_SESSION_ID', 'claude-persist')
    assert 'CLAUDE_CODE_SESSION_ID' not in config.enumerate_effective_config()
    assert 'CLAUDE_CODE_SESSION_ID' not in config.env_file_path(
        data_dir).read_text()


def test_idempotency_keyed_on_queue_uuid(mm_runner):
    """A replay skips; a second write in the same session does not.

    Mutation: keying the drain replay check on `source` (the first
        `user` write would suppress every later default write) or on
        the integer row id.
    Oracle: two same-session writes both store; re-queueing the first
        row and re-draining adds nothing.
    """
    _, data_dir = mm_runner
    r1 = invoke(mm_runner, [
        'remember', 'first same-session note', '--no-reconcile',
        '--session', 'sess-i'])
    raw1 = json.loads(r1.output)
    r2 = invoke(mm_runner, [
        'remember', 'second same-session note', '--no-reconcile',
        '--session', 'sess-i'])
    raw2 = json.loads(r2.output)
    assert raw1['store'] == raw2['store']
    rows = _stored(data_dir, raw1['store'],
                   'session_id = ?', ('sess-i',))
    assert len(rows) == 2

    _requeue(data_dir, raw1['queue_id'])
    force_drain(data_dir)
    rows_after = _stored(data_dir, raw1['store'],
                         'session_id = ?', ('sess-i',))
    assert len(rows_after) == 2


def test_idempotency_check_runs_for_explicit_source(mm_runner):
    """The replay check fires even when a source hint is present.

    Mutation: restoring the old `hint_source is None` precondition —
        a replayed row with an explicit source would store twice.
    Oracle: re-queueing an `--source agent` row and re-draining
        leaves exactly one stored insight for its uuid.
    """
    _, data_dir = mm_runner
    r1 = invoke(mm_runner, [
        'remember', 'explicit source replay note', '--no-reconcile',
        '--source', 'agent'])
    raw = json.loads(r1.output)
    _sess, queue_uuid = _queue_row(data_dir, raw['queue_id'])
    _requeue(data_dir, raw['queue_id'])
    force_drain(data_dir)
    rows = _stored(data_dir, raw['store'],
                   'queue_uuid = ?', (queue_uuid,))
    assert len(rows) == 1


def test_queue_uuid_survives_counter_rewind(mm_runner):
    """A rebuilt queue.db that reuses row id 1 must not skip the write.

    `backup.restore` replaces queue.db wholesale and rewinds the
    AUTOINCREMENT counter; with the integer id as the key, a fresh
    enqueue drawing a used id is silently dropped.

    Mutation: keying idempotency on the queue row id.
    Oracle: after deleting queue.db, a second write that draws the
        same row id still stores (two insights total).
    """
    import os
    _, data_dir = mm_runner
    r1 = invoke(mm_runner, [
        'remember', 'note before rewind', '--no-reconcile'])
    raw1 = json.loads(r1.output)
    for suffix in ('', '-wal', '-shm'):
        try:
            os.remove(f'{data_dir}/queue.db{suffix}')
        except FileNotFoundError:
            pass
    r2 = invoke(mm_runner, [
        'remember', 'note after rewind', '--no-reconcile'])
    raw2 = json.loads(r2.output)
    assert raw2['queue_id'] == raw1['queue_id'], (
        'fixture failed to rewind the AUTOINCREMENT counter')
    rows = _stored(data_dir, raw1['store'], '1 = 1', ())
    assert len(rows) == 2


def test_latest_by_session_tiebreak_matches_across_backends(backend):
    """Equal timestamps tiebreak on `id desc` (both backends' clause).

    `psycopg` is absent in this environment, so the Postgres half is
    pinned textually: the source must carry the exact `order by`
    clause SQLite implements behaviorally here.

    Mutation: dropping `id desc` from one backend — equal-timestamp
        rows would then chain nondeterministically on SQLite (rowid)
        and Postgres (heap order).
    Oracle: SQLite returns the higher id of two equal-created_at
        rows; postgres.py's `get_latest_by_session` contains the
        literal clause.
    """
    import inspect
    from datetime import datetime, timezone
    from pathlib import Path

    from memman.store import node as node_mod
    from tests.conftest import make_insight, set_created_at
    ts = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)
    for rid in ('a-1', 'z-9'):
        backend.nodes.insert(make_insight(
            id=rid, content=f'row {rid}', session_id='sess-t'))
        set_created_at(backend, rid, ts)
    latest = backend.nodes.get_latest_by_session(
        session_id='sess-t', exclude_id='other')
    assert latest is not None
    assert latest.id == 'z-9'

    pg_source = (
        Path(inspect.getsourcefile(node_mod)).parent / 'postgres.py'
        ).read_text()
    _, _, after = pg_source.partition('def get_latest_by_session')
    assert 'order by created_at desc, id desc' in after[:800]


def test_prime_substitutes_session_id_into_guide(mm_runner):
    """`memman prime` bakes the real session id into the guide template.

    `prime` is the only caller that passes an id, so this is what
    makes D1's chain adoption real rather than
    documented-but-unexercised.

    Mutation: dropping the substitution in `_emit_guide` — the
        template would keep the literal `$SESSION_ID` placeholder.
    Oracle: prime's output contains `--session <real id>` and no
        placeholder.
    """
    from memman.cli import cli
    r, _data_dir = mm_runner
    result = r.invoke(
        cli, ['prime'], input='{"session_id": "sess-guide-1"}')
    assert result.exit_code == 0, result.output
    assert '--session sess-guide-1' in result.output
    assert '$SESSION_ID' not in result.output


def test_insight_column_lists_are_identical_across_backends():
    """`_INSIGHT_COLUMNS` and `_INSIGHT_COLS` are byte-identical.

    A transposition of the two new columns between backends is
    invisible to the type checker and to every single-backend test —
    each backend would round-trip its own transposed order happily.

    Mutation: transposing `session_id`/`queue_uuid` (or any pair) in
        one constant.
    Oracle: pure string compare, no server needed; the Postgres
        constant is read from source text since psycopg may be
        absent.
    """
    import ast
    import inspect
    from pathlib import Path

    from memman.store import node as node_mod
    from memman.store.node import _INSIGHT_COLUMNS
    pg_path = (
        Path(inspect.getsourcefile(node_mod)).parent / 'postgres.py')
    tree = ast.parse(pg_path.read_text())
    pg_value = None
    for stmt in ast.walk(tree):
        if (isinstance(stmt, ast.Assign)
                and any(getattr(t, 'id', '') == '_INSIGHT_COLS'
                        for t in stmt.targets)):
            pg_value = ast.literal_eval(stmt.value)
    assert pg_value == _INSIGHT_COLUMNS


def test_expected_insight_columns_covers_new_fields(backend):
    """`doctor.EXPECTED_INSIGHT_COLUMNS` matches the live schema.

    Mutation: adding the columns to the schema but not to doctor —
        `check_schema_columns` would then pass on a store doctor
        cannot actually vouch for.
    Oracle: every expected column exists on a freshly created store.
    """
    from memman.doctor import EXPECTED_INSIGHT_COLUMNS
    present = backend.introspect_columns('insights')
    assert EXPECTED_INSIGHT_COLUMNS <= present
    assert {'session_id', 'queue_uuid'} <= present
