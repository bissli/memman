"""`author` field on every stored insight.

`MEMMAN_AUTHOR` is a process-control var resolved at `remember`/`replace`
time from the agent's shell. The drain reads it from the queue row, not
from `os.environ`, so the scheduler subprocess never needs the directory's
environment. Recall rows and `insights show` carry the field. The
author-prefix refusal in `remember`/`replace` checks only an explicitly
set `MEMMAN_AUTHOR`, never the `getpass.getuser()` fallback.
"""

import json
import shutil
from datetime import datetime, timezone

import pytest
from memman.queue import queue_db
from memman.store.db import open_db, open_read_only, read_active, set_meta
from memman.store.db import store_dir
from memman.store.model import Insight
from memman.store.node import insert_insight
from memman.store.postgres import PostgresMigrator, _store_schema
from memman.store.sqlite import SqliteMigrator
from tests.conftest import force_drain, invoke, make_insight, parse_remember

# --- Helpers ---


def _queue_author(data_dir: str, queue_id: int) -> str | None:
    """Return the `author` from the queue row, or None when absent."""
    with queue_db(data_dir) as conn:
        row = conn.execute(
            'select author from queue where id = ?',
            (queue_id,)).fetchone()
    return row[0] if row else None


def _insight_author(data_dir: str, queue_uuid: str) -> str | None:
    """Return `author` from the SQLite insight matching `queue_uuid`."""
    db = open_read_only(store_dir(data_dir, read_active(data_dir)))
    try:
        row = db._query(
            'select author from insights where queue_uuid = ?',
            (queue_uuid,)).fetchone()
    finally:
        db.close()
    return row[0] if row else None


def _queue_uuid_for(data_dir: str, queue_id: int) -> str | None:
    """Return the `queue_uuid` for a queue row."""
    with queue_db(data_dir) as conn:
        row = conn.execute(
            'select queue_uuid from queue where id = ?',
            (queue_id,)).fetchone()
    return row[0] if row else None


# --- Author set at enqueue time, carried through drain ---

@pytest.mark.no_auto_drain
def test_author_carried_from_queue_to_insight(mm_runner, monkeypatch):
    """Verify author stamped at enqueue survives to the stored insight.

    Mutation: resolving author at drain time (where MEMMAN_AUTHOR is absent),
        or dropping the author column from the queue table.
    Oracle: direct DB reads of queue.author and insights.author after drain.
    """
    _, data_dir = mm_runner
    monkeypatch.setenv('MEMMAN_AUTHOR', 'alice')

    result = invoke(mm_runner, ['remember', 'something happened'])
    assert result.exit_code == 0, result.output
    raw = json.loads(result.output)
    queue_id = raw['queue_id']

    assert _queue_author(data_dir, queue_id) == 'alice'

    # The scheduler drain runs without the directory's environment.
    monkeypatch.delenv('MEMMAN_AUTHOR', raising=False)
    force_drain(data_dir)

    q_uuid = _queue_uuid_for(data_dir, queue_id)
    assert _insight_author(data_dir, q_uuid) == 'alice'


@pytest.mark.no_auto_drain
def test_author_unset_falls_back_to_getpass(mm_runner, monkeypatch):
    """Verify MEMMAN_AUTHOR unset falls back to getpass.getuser().

    Mutation: missing fallback to getpass, so unset yields None or raises.
    Oracle: the stored author matches the mocked getpass return value.
    """
    _, data_dir = mm_runner
    monkeypatch.delenv('MEMMAN_AUTHOR', raising=False)
    monkeypatch.setattr('getpass.getuser', lambda: 'bob')

    result = invoke(mm_runner, ['remember', 'another fact'])
    assert result.exit_code == 0, result.output
    raw = json.loads(result.output)
    queue_id = raw['queue_id']

    assert _queue_author(data_dir, queue_id) == 'bob'

    force_drain(data_dir)
    q_uuid = _queue_uuid_for(data_dir, queue_id)
    assert _insight_author(data_dir, q_uuid) == 'bob'


# --- Author round-trips through the store, sqlite and postgres ---

def test_insight_author_round_trips_through_store(backend):
    """Verify an inserted author is unchanged by a public store read.

    Mutation: dropping author in postgres.py's `_row_to_insight` row
        mapper (the sqlite parametrization is unaffected by that
        mutation, so it pins the postgres-only mapper).
    Oracle: `backend.nodes.get` returns the same author that was
        passed to `backend.nodes.insert`.
    """
    backend.nodes.insert(make_insight(id='author-rt-1', author='alice'))
    stored = backend.nodes.get('author-rt-1')
    assert stored is not None
    assert stored.author == 'alice'


@pytest.mark.postgres
def test_author_survives_migrate_round_trip_postgres(tmp_path, pg_dsn):
    """Verify author survives a sqlite -> postgres -> sqlite migrate round trip.

    Mutation: dropping author from the `MigrateInsight` built by
        `PostgresMigrator.gather`.
    Oracle: the insight gathered back from postgres, and the final
        sqlite row after re-apply, both carry the original author.
    """
    # psycopg is the optional postgres extra, so a sqlite-only install
    # still collects this file.
    import psycopg

    store = 'author_migrate_rt'
    sdir = store_dir(str(tmp_path), store)
    db = open_db(sdir)
    try:
        insert_insight(db, Insight(
            id='author-migrate-1', content='author migrate test',
            category='fact', importance=3, entities=[],
            source='test', access_count=0,
            updated_at=datetime.now(timezone.utc),
            deleted_at=None, last_accessed_at=None,
            author='alice'))
        set_meta(
            db, 'embed_fingerprint',
            '{"provider":"voyage","model":"voyage-3-lite","dim":512}')
    finally:
        db.close()

    schema = _store_schema(store)

    def _drop_schema() -> None:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'drop schema if exists {schema} cascade')

    _drop_schema()
    try:
        src_mig = SqliteMigrator(str(tmp_path))
        src_mig.preflight_source(store)
        payload = src_mig.gather(store)
        tgt_mig = PostgresMigrator(str(tmp_path), dsn=pg_dsn)
        tgt_mig.preflight_target(store)
        tgt_mig.apply(store, payload)
        shutil.rmtree(sdir)

        rev_src = PostgresMigrator(str(tmp_path), dsn=pg_dsn)
        rev_src.preflight_source(store)
        rev_payload = rev_src.gather(store)
        assert len(rev_payload.insights) == 1
        assert rev_payload.insights[0].author == 'alice'

        target = store_dir(str(tmp_path), store)
        rev_tgt = SqliteMigrator(str(tmp_path))
        rev_tgt.preflight_target(store)
        rev_tgt.apply(store, rev_payload)

        db2 = open_db(target)
        try:
            row = db2.conn.execute(
                'select author from insights where id = ?',
                ('author-migrate-1',)).fetchone()
        finally:
            db2.close()
        assert row
        assert row[0] == 'alice'
    finally:
        _drop_schema()


# --- Author in recall --brief and insights show ---

def test_recall_brief_carries_author(mm_runner, monkeypatch):
    """Verify recall --brief rows include the author field.

    Mutation: omitting author from insight_to_brief_dict's projection.
    Oracle: the parsed JSON row has an 'author' key equal to the set env var.
    """
    _, data_dir = mm_runner
    monkeypatch.setenv('MEMMAN_AUTHOR', 'alice')

    invoke(mm_runner, ['remember', 'recall brief test'])

    result = invoke(mm_runner, ['recall', '--brief', 'recall brief test'])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed['results'], 'expected at least one result'
    row = parsed['results'][0]
    insight = row.get('insight', row)
    assert 'author' in insight, f'author absent from brief row: {row}'
    assert insight['author'] == 'alice'


def test_insights_show_carries_author(mm_runner, monkeypatch):
    """Verify `insights show <id>` carries the author field.

    Mutation: omitting author from insight_to_full_dict's projection.
    Oracle: the JSON output has an 'author' key equal to the set env var.
    """
    _, data_dir = mm_runner
    monkeypatch.setenv('MEMMAN_AUTHOR', 'alice')

    first = invoke(mm_runner, ['remember', 'show test fact'])
    fact = parse_remember(first, mm_runner)

    result = invoke(mm_runner, ['insights', 'show', fact['id']])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed.get('author') == 'alice', f'author wrong: {parsed}'


# --- Author-prefix refusal (remember) ---

_REFUSE_CASES = [
    pytest.param(['  Alice: x'], id='leading-whitespace-colon'),
    pytest.param(['"alice decided"'], id='quoted'),
    pytest.param(['--', '- alice decided'], id='leading-dash'),
    pytest.param(['(alice) x'], id='parenthesized'),
    pytest.param(['**Alice**: x'], id='markdown-bold'),
    ]


@pytest.mark.parametrize('content_args', _REFUSE_CASES)
def test_remember_refuses_author_prefix_variants(
        mm_runner, monkeypatch, content_args):
    r"""Verify remember refuses an opening author behind punctuation.

    Mutation: stripping only leading whitespace (`lstrip`) instead of
        `\\W*`, which would accept every case here except the plain
        leading-whitespace one.
    Oracle: exit code != 0 on each variant.
    """
    monkeypatch.setenv('MEMMAN_AUTHOR', 'alice')
    result = invoke(mm_runner, ['remember'] + content_args)
    assert result.exit_code != 0, result.output


def test_remember_refusal_message_names_author_and_instruction(
        mm_runner, monkeypatch):
    """Verify the refusal message names the author and the fix.

    Mutation: dropping the author name or the 'start with the
        subject' instruction from the message text.
    Oracle: both substrings present in the CLI's error output.
    """
    monkeypatch.setenv('MEMMAN_AUTHOR', 'alice')
    result = invoke(mm_runner, ['remember', 'alice decided to do this'])
    assert result.exit_code != 0
    assert 'alice' in result.output
    assert 'start with the subject' in result.output


@pytest.mark.parametrize('content', [
    'alicex decided to do this',
    'The cap alice set was wrong',
    ], ids=['author-as-prefix-of-word', 'author-mid-sentence'])
def test_remember_accepts_author_substring_and_mid_sentence(
        mm_runner, monkeypatch, content):
    """Verify remember accepts text where the author is not the first word.

    Mutation: matching `author` as a bare substring rather than
        anchored with a trailing word boundary, which would refuse
        both cases here.
    Oracle: exit code == 0.
    """
    monkeypatch.setenv('MEMMAN_AUTHOR', 'alice')
    result = invoke(mm_runner, ['remember', content])
    assert result.exit_code == 0, result.output


def test_remember_author_with_regex_metacharacter_is_escaped(
        mm_runner, monkeypatch):
    """Verify a `.` in the author name matches only itself, not any char.

    Mutation: building the refusal pattern from the raw `author`
        string instead of `re.escape(author)`, so `.` acts as a
        regex wildcard and 'axb' wrongly matches 'a.b'.
    Oracle: exit code == 0 for content starting with 'axb'.
    """
    monkeypatch.setenv('MEMMAN_AUTHOR', 'a.b')
    result = invoke(mm_runner, ['remember', 'axb decided'])
    assert result.exit_code == 0, result.output


def test_remember_accepts_punctuation_leading_content_when_author_unset(
        mm_runner, monkeypatch):
    r"""Verify an unset MEMMAN_AUTHOR disables the refusal entirely.

    Mutation: dropping the `if not author: return None` guard, which
        leaves `re.escape('')` in the pattern and lets `\\W*` match a
        leading punctuation character on its own, refusing content
        that never touched an author name.
    Oracle: exit code == 0 for content opening with a quote mark.
    """
    monkeypatch.delenv('MEMMAN_AUTHOR', raising=False)
    result = invoke(mm_runner, ['remember', '"quoted content" here'])
    assert result.exit_code == 0, result.output


def test_remember_accepts_when_author_unset_and_getpass_matches_content(
        mm_runner, monkeypatch):
    """Verify the refusal checks only an explicitly set MEMMAN_AUTHOR.

    Mutation: refusing on `config.resolve_author()`'s getpass fallback
        instead of the explicit `MEMMAN_AUTHOR` env var, which would
        refuse this content since it opens with the mocked OS user.
    Oracle: exit code == 0, and the stored author is the getpass name.
    """
    _, data_dir = mm_runner
    monkeypatch.delenv('MEMMAN_AUTHOR', raising=False)
    monkeypatch.setattr('getpass.getuser', lambda: 'carol')

    result = invoke(mm_runner, ['remember', 'carol decided x'])
    assert result.exit_code == 0, result.output
    raw = json.loads(result.output)
    queue_id = raw['queue_id']
    assert _queue_author(data_dir, queue_id) == 'carol'


# --- Author-prefix refusal (replace) ---

def test_replace_refuses_text_starting_with_author(mm_runner, monkeypatch):
    """Verify replace refuses content whose first word is the author.

    Mutation: adding the refusal to remember but not replace, or a
        replace message that drops the fix instruction.
    Oracle: exit code != 0, and the message names the author and the fix.
    """
    monkeypatch.setenv('MEMMAN_AUTHOR', 'alice')

    first = invoke(mm_runner, ['remember', 'original fact to be replaced'])
    fact = parse_remember(first, mm_runner)

    result = invoke(
        mm_runner, ['replace', fact['id'], 'alice thinks it changed'])
    assert result.exit_code != 0
    assert 'alice' in result.output
    assert 'start with the subject' in result.output


def test_replace_accepts_text_not_starting_with_author(
        mm_runner, monkeypatch):
    """Verify replace accepts content whose first word is not the author.

    Mutation: refusing every replace regardless of content (e.g. an
        unconditional refusal check).
    Oracle: exit code == 0.
    """
    monkeypatch.setenv('MEMMAN_AUTHOR', 'alice')

    first = invoke(mm_runner, ['remember', 'original fact to be replaced'])
    fact = parse_remember(first, mm_runner)

    result = invoke(
        mm_runner, ['replace', fact['id'], 'the plan changed entirely'])
    assert result.exit_code == 0, result.output


# --- Column round-trip through migrate (SQLite side) ---

def test_author_survives_migrate_round_trip_sqlite(
        mm_runner, monkeypatch, tmp_path):
    """Verify author column is exported and re-imported by SqliteMigrator.

    Mutation: missing author from the export or import column list in
        sqlite.py's MigrateInsight path.
    Oracle: the destination store's insight carries the same author after
        a gather+apply round-trip.
    """
    _, data_dir = mm_runner
    monkeypatch.setenv('MEMMAN_AUTHOR', 'alice')

    first = invoke(mm_runner, ['remember', 'migrate test content'])
    fact = parse_remember(first, mm_runner)
    src_id = fact['id']

    store = read_active(data_dir)
    payload = SqliteMigrator(data_dir).gather(store)

    exported_authors = [
        ins.author for ins in payload.insights if ins.id == src_id]
    assert exported_authors == ['alice'], (
        f'exported author wrong: {exported_authors}')

    dest_dir = tmp_path / 'dest'
    dest_dir.mkdir()
    SqliteMigrator(str(dest_dir)).apply(store, payload)

    db2 = open_read_only(store_dir(str(dest_dir), store))
    try:
        row = db2._query(
            'select author from insights where id = ?',
            (src_id,)).fetchone()
    finally:
        db2.close()
    assert row and row[0] == 'alice', f'post-migrate author wrong: {row}'


# --- Author on the queue row shown by the CLI ---

@pytest.mark.no_auto_drain
def test_scheduler_queue_show_carries_author(mm_runner, monkeypatch):
    """Verify `scheduler queue show` reports the queue row's author.

    Mutation: dropping author from `queue.get_row`'s select or its dict.
    Oracle: the author set at enqueue, read back through the CLI.
    """
    monkeypatch.setenv('MEMMAN_AUTHOR', 'alice')
    queued = invoke(mm_runner, ['remember', 'queue show test'])
    queue_id = json.loads(queued.output)['queue_id']

    result = invoke(
        mm_runner, ['scheduler', 'queue', 'show', str(queue_id)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['author'] == 'alice'
