"""The recall page: one plain-text line per row, and nothing else.

`memman recall` prints `<id8> <score> <created_at> <author> <category> |
<text>` per row in rank order, and `--basic` prints the same line
without a score. The intent router, the meta block, the per-row signals
and the deleted flags leave the page, and recall no longer counts
accesses.
"""

import json
import os
import pathlib
import re
import subprocess
from datetime import datetime, timezone
from importlib.resources import files as pkg_files

import pytest
from click.testing import CliRunner
from memman.cli import cli
from memman.search.recall import intent_aware_recall
from memman.store.db import open_db
from memman.store.model import format_timestamp
from memman.store.node import insert_insight
from memman.store.sqlite import SqliteBackend
from tests.conftest import invoke, make_insight

_SCORED_LINE = re.compile(
    r'^(?P<id>\S{8}) (?P<score>-?\d+\.\d\d)'
    r' (?P<created>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)'
    r' (?P<author>\S+) (?P<category>\S+) \| (?P<text>.*)$')
_BASIC_LINE = re.compile(
    r'^(?P<id>\S{8})'
    r' (?P<created>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)'
    r' (?P<author>\S+) (?P<category>\S+) \| (?P<text>.*)$')

_LONG_CONTENT = 'zulu first line\nsecond line ' + 'y' * 300


def _seed(data_dir: str, rows: list[tuple]) -> None:
    """Insert `(insight, summary)` pairs into the default SQLite store."""
    db = open_db(str(pathlib.Path(data_dir) / 'data' / 'default'))
    try:
        for ins, summary in rows:
            insert_insight(db, ins)
            if summary:
                db._conn.execute(
                    'update insights set summary = ? where id = ?',
                    (summary, ins.id))
    finally:
        db.close()


@pytest.fixture
def page_rows(mm_runner):
    """Four rows that each exercise one text rule of the page line."""
    _, data_dir = mm_runner
    _seed(data_dir, [
        (make_insight(
            id='aaaaaaaa-0001', content='zulu decision on the retry cap',
            category='decision', author='alice'),
         'The retry cap stays at three.'),
        (make_insight(
            id='bbbbbbbb-0002', content=_LONG_CONTENT,
            category='fact', author='bob'),
         ''),
        (make_insight(
            id='cccccccc-0003', content='zulu short note',
            category='fact', author=None),
         ''),
        (make_insight(
            id='dddddddd-0004', content='zulu row with a two-line summary',
            category='fact', author='dave'),
         'first half\nsecond half'),
        ])
    return mm_runner


def _parse(stdout: str, pattern: re.Pattern) -> dict[str, dict]:
    """Map each page line's id8 to its parsed fields, failing on a stray line."""
    rows: dict[str, dict] = {}
    for line in stdout.splitlines():
        match = pattern.match(line)
        assert match, f'line off the page format: {line!r}'
        rows[match['id']] = match.groupdict()
    return rows


def test_scored_page_is_one_line_per_row_in_rank_order(page_rows):
    """Verify recall prints exactly one formatted line per row, best first.

    Mutation: printing the JSON envelope, adding any header or meta
        line, or printing rows out of score order.
    Oracle: the four seeded ids, each on one line matching the page
        regex, with scores read back in non-increasing order.
    """
    result = invoke(page_rows, ['recall', 'zulu'])

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    rows = _parse(result.stdout, _SCORED_LINE)
    assert set(rows) == {'aaaaaaaa', 'bbbbbbbb', 'cccccccc', 'dddddddd'}
    assert len(lines) == 4
    scores = [float(_SCORED_LINE.match(line)['score']) for line in lines]
    assert scores == sorted(scores, reverse=True)


def test_page_text_prefers_the_summary(page_rows):
    """Verify a row with a summary shows the summary, not its content.

    Mutation: always printing the content prefix.
    Oracle: the seeded summary string, which the content does not hold.
    """
    result = invoke(page_rows, ['recall', 'zulu'])

    rows = _parse(result.stdout, _SCORED_LINE)
    assert rows['aaaaaaaa']['text'] == 'The retry cap stays at three.'


def test_page_text_folds_line_breaks_and_marks_the_cut(page_rows):
    """Verify line breaks fold to spaces and a cut prefix ends in `...`.

    Mutation: printing content[:200] raw, so a line break splits the
        row across two lines; dropping the `...` marker; or folding
        the content fallback but not the summary.
    Oracle: hand-computed fold of the seeded content cut at 200
        characters plus `...`, and the seeded two-line summary joined
        by one space.
    """
    result = invoke(page_rows, ['recall', 'zulu'])

    rows = _parse(result.stdout, _SCORED_LINE)
    folded = ' '.join(_LONG_CONTENT.split())
    assert rows['bbbbbbbb']['text'] == folded[:200] + '...'
    assert rows['dddddddd']['text'] == 'first half second half'


def test_page_prints_a_dash_for_an_unset_author(page_rows):
    """Verify a row with no author shows `-` in the author field.

    Mutation: printing `None` or an empty field, which shifts every
        later field of the line.
    Oracle: the one seeded row with no author, beside a row with one.
    """
    result = invoke(page_rows, ['recall', 'zulu'])

    rows = _parse(result.stdout, _SCORED_LINE)
    assert rows['cccccccc']['author'] == '-'
    assert rows['aaaaaaaa']['author'] == 'alice'


def test_page_joins_whitespace_inside_an_author(mm_runner):
    r"""Verify an author holding whitespace stays one field on one line.

    Mutation: printing the author raw, so `MEMMAN_AUTHOR='Jane Doe'`
        reads as author `Jane` and category `Doe`, and a line break in
        it splits the row across two lines.
    Oracle: hand-computed `Jane_Doe_Smith` from the seeded
        `'Jane Doe\\nSmith'`, and the seeded category `fact`.
    """
    _, data_dir = mm_runner
    _seed(data_dir, [(make_insight(
        id='eeeeeeee-0005', content='zulu row by a spaced author',
        category='fact', author='Jane Doe\nSmith'), '')])

    result = invoke(mm_runner, ['recall', 'zulu'])

    rows = _parse(result.stdout, _SCORED_LINE)
    assert rows['eeeeeeee']['author'] == 'Jane_Doe_Smith'
    assert rows['eeeeeeee']['category'] == 'fact'


def test_basic_page_prints_the_line_without_a_score(page_rows):
    """Verify recall --basic prints the page line with no score field.

    Mutation: keeping the `{results, meta: {basic: true}}` envelope on
        the --basic branch, or printing a placeholder score.
    Oracle: the four seeded ids, each on a line matching the scoreless
        regex, and the category of the one seeded decision row.
    """
    result = invoke(page_rows, ['recall', '--basic', 'zulu'])

    assert result.exit_code == 0, result.output
    rows = _parse(result.stdout, _BASIC_LINE)
    assert set(rows) == {'aaaaaaaa', 'bbbbbbbb', 'cccccccc', 'dddddddd'}
    assert rows['aaaaaaaa']['category'] == 'decision'


def test_empty_page_prints_nothing(mm_runner):
    """Verify a recall over an empty store prints nothing and exits 0.

    Mutation: printing an empty envelope, a header, or a notice line.
    Oracle: the empty string on stdout from a store holding no row.
    """
    result = invoke(mm_runner, ['recall', 'zulu'])

    assert result.exit_code == 0, result.output
    assert result.stdout == ''


def test_default_limit_is_twenty(mm_runner):
    """Verify recall returns 20 rows when no --limit is given.

    Mutation: leaving the --limit default at 10.
    Oracle: 25 seeded matching rows, counted by their distinct
        8-character id prefixes on stdout.
    """
    _, data_dir = mm_runner
    _seed(data_dir, [
        (make_insight(id=f'{i:02d}limitrow', content=f'zulu row {i}'), '')
        for i in range(25)])

    result = invoke(mm_runner, ['recall', '--basic', 'zulu'])

    shown = [i for i in range(25) if f'{i:02d}limitr' in result.stdout]
    assert len(shown) == 20


@pytest.mark.parametrize('flag', [
    ['--brief'],
    ['--intent', 'WHY'],
    ['--expand'],
    ['--min-score', '0.5'],
    ], ids=['brief', 'intent', 'expand', 'min-score'])
def test_deleted_recall_flags_are_refused(page_rows, flag):
    """Verify each deleted recall flag is an unknown option.

    Mutation: keeping any of the four options on the command.
    Oracle: click's usage-error exit code 2 and its `No such option`
        message.
    """
    result = invoke(page_rows, ['recall', 'zulu', *flag])

    assert result.exit_code == 2, result.output
    assert 'No such option' in result.output


def test_recall_detail_row_keeps_only_q_limit_hits(page_rows):
    """Verify the recall-detail oplog row carries q, limit and hits.

    Mutation: leaving `intent` or `session` on the row or `via`, `kw`,
        `sim`, `gr` on each hit; or a leftover access increment after
        the column drop, whose OperationalError the bookkeeping guard
        swallows so the row silently never lands.
    Oracle: the exact key sets read back from the store's oplog after
        one scored recall.
    """
    _, data_dir = page_rows
    invoke(page_rows, ['recall', 'zulu'])

    db = open_db(str(pathlib.Path(data_dir) / 'data' / 'default'))
    try:
        row = db._conn.execute(
            "select detail from oplog where operation = 'recall-detail'"
            ' order by id desc limit 1').fetchone()
    finally:
        db.close()
    assert row, 'no recall-detail row written'
    detail = json.loads(row[0])
    assert set(detail) == {'q', 'limit', 'hits'}
    assert detail['hits']
    assert all(set(hit) == {'id', 'score'} for hit in detail['hits'])


def test_recall_scores_by_the_general_weights(backend):
    """Verify a WHY-shaped query scores by the GENERAL weight row.

    Mutation: leaving the intent router in place, so `why sqlite`
        routes to WHY and its keyword weight.
    Oracle: hand-computed score of the only row, which holds every
        query token, has no vector and no edge, so its score is the
        GENERAL keyword weight 0.25 / 0.85; WHY would give 0.15 / 0.90.
    """
    backend.nodes.insert(make_insight(
        id='why-row', content='why sqlite stays'))

    resp = intent_aware_recall(backend, 'why sqlite', None, 5)

    assert resp['results'][0]['score'] == pytest.approx(0.25 / 0.85)


def test_recall_return_carries_no_router_keys(backend):
    """Verify the recall return drops per-row intent and the router meta.

    Mutation: leaving `intent` on each row, `intent`, `intent_source`
        or `hint` in meta, or the `traversed` count, which only ever
        differed from `anchor_count` by the rows traversal added.
    Oracle: the exact meta key set, and the absence of `intent` on the
        one returned row.
    """
    backend.nodes.insert(make_insight(
        id='router-row', content='router keys row'))

    resp = intent_aware_recall(backend, 'router keys', None, 5)

    assert 'intent' not in resp['results'][0]
    assert set(resp['meta']) == {'anchor_count', 'reranked'}


def test_insights_show_has_no_access_count(page_rows):
    """Verify insights show no longer reports an access count.

    Mutation: leaving `access_count` in `insight_to_full_dict`.
    Oracle: the key set of one shown row.
    """
    result = invoke(page_rows, ['insights', 'show', 'aaaaaaaa-0001'])

    assert result.exit_code == 0, result.output
    assert 'access_count' not in json.loads(result.output)


def test_log_stats_has_no_never_accessed(page_rows):
    """Verify log list --stats no longer reports never-accessed rows.

    Mutation: leaving `never_accessed` in the stats printer or in
        `OpLogStats`.
    Oracle: the key set of the stats JSON.
    """
    result = invoke(page_rows, ['log', 'list', '--stats'])

    assert result.exit_code == 0, result.output
    assert 'never_accessed' not in json.loads(result.output)


def _recall_lines(text: str) -> list[str]:
    """Return the lines of `text` that name `memman recall`."""
    return [line for line in text.splitlines() if 'memman recall' in line]


def _run_asset(name: str, payload: str, tmp_path: pathlib.Path) -> str:
    """Run one shipped hook script and return its stdout."""
    script = str(pkg_files('memman.setup.assets').joinpath(f'claude/{name}'))
    done = subprocess.run(
        ['bash', script], check=True, input=payload,
        capture_output=True, text=True,
        env={**os.environ, 'HOME': str(tmp_path)})
    return done.stdout


def test_recall_prose_names_the_bare_command(tmp_path):
    """Verify every injected recall line names `memman recall "..."` alone.

    Mutation: leaving `--brief`, `--limit` or `--session` on the recall
        line of guide.md, user_prompt.sh, task_recall.sh or prime's
        compact hint.
    Oracle: each surface's recall lines, read from the shipped asset or
        the hook's own stdout, searched for the three flags.
    """
    guide = (pkg_files('memman.setup.assets')
             .joinpath('claude/guide.md').read_text())
    compact = CliRunner().invoke(
        cli, ['prime'], input='{"source": "compact", "session_id": "s"}',
        env={'MEMMAN_DATA_DIR': str(tmp_path)}).output
    surfaces = {
        'guide.md': guide,
        'user_prompt.sh': _run_asset(
            'user_prompt.sh', '{"session_id": "sess-6"}', tmp_path),
        'task_recall.sh': _run_asset('task_recall.sh', '{}', tmp_path),
        'prime compact': compact.split('\n# memman')[0],
        }

    for name, text in surfaces.items():
        lines = _recall_lines(text)
        assert lines, f'{name} names no recall command'
        flagged = [
            line for line in lines
            if any(flag in line for flag in ('--brief', '--limit', '--session'))]
        assert not flagged, f'{name}: {flagged}'


def test_shipped_prose_names_no_deleted_recall_flag():
    """Verify guide.md and SKILL.md name none of the deleted recall flags.

    Mutation: leaving a `--brief`, `--expand`, `--min-score` or
        `--intent` example in the guide or the skill.
    Oracle: the four flag strings searched for in both shipped files.
    """
    for name in ('guide.md', 'SKILL.md'):
        text = (pkg_files('memman.setup.assets')
                .joinpath(f'claude/{name}').read_text())
        found = [
            flag for flag in ('--brief', '--expand', '--min-score', '--intent')
            if flag in text]
        assert not found, f'{name} names {found}'


def _stamp(backend, insight_id: str, columns: dict) -> None:
    """Test-only: set raw column values on one stored row."""
    if isinstance(backend, SqliteBackend):
        assignments = ', '.join(f'{name} = ?' for name in columns)
        values = [
            format_timestamp(v) if isinstance(v, datetime) else v
            for v in columns.values()]
        backend._db._exec(
            f'update insights set {assignments} where id = ?',
            (*values, insight_id))
        return
    assignments = ', '.join(f'{name} = %s' for name in columns)
    with backend._conn.cursor() as cur:
        cur.execute(
            f'update {backend._schema}.insights set {assignments}'
            ' where id = %s',
            (*columns.values(), insight_id))
    backend._conn.commit()


def test_live_mappers_read_every_trailing_field(backend):
    """Verify the row mapper reads each field from its own column.

    Mutation: a stale positional index in `node._scan_insight` or
        `postgres._row_to_insight` after a column drop, which reads a
        neighbor's value into a field (summary into deleted_at,
        queue_uuid into superseded_by) with no error.
    Oracle: a distinct hand-set value per column, read back through
        `get_include_deleted` on both backends.
    """
    backend.nodes.insert(make_insight(
        id='fidelity-row', content='fidelity row', category='decision',
        queue_uuid='queue-f', author='carol'))
    stamps = {
        'summary': 'fidelity summary',
        'linked_at': datetime(2026, 1, 2, 3, 4, 1, tzinfo=timezone.utc),
        'enriched_at': datetime(2026, 1, 2, 3, 4, 2, tzinfo=timezone.utc),
        'deleted_at': datetime(2026, 1, 2, 3, 4, 3, tzinfo=timezone.utc),
        'superseded_by': 'successor-f',
        }
    _stamp(backend, 'fidelity-row', stamps)

    got = backend.nodes.get_include_deleted('fidelity-row')

    assert got.category == 'decision'
    assert got.summary == 'fidelity summary'
    assert got.linked_at == stamps['linked_at']
    assert got.enriched_at == stamps['enriched_at']
    assert got.deleted_at == stamps['deleted_at']
    assert got.queue_uuid == 'queue-f'
    assert (got.superseded_by, got.author) == ('successor-f', 'carol')
