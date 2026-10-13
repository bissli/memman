"""`--meta key=value` pairs: parsing, storage, display, keyword index, replace.
"""

import json
from datetime import datetime, timezone

import pytest
from memman.migrate import MigrateInsight
from memman.store.model import BRIEF_CONTENT_CHARS, insight_to_recall_line
from tests.conftest import invoke, make_insight, parse_remember
from tests.conftest import queued_contents


def _show(mm_runner: tuple, insight_id: str) -> dict:
    """`insights show` JSON for one id.
    """
    result = invoke(mm_runner, ['insights', 'show', insight_id])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


# --- Recall page line ---

def test_recall_line_keeps_meta_past_the_cut():
    """Verify meta prints whole after a content prefix that was cut.

    Mutation: appending meta before the BRIEF_CONTENT_CHARS cut, or
        leaving it off the line.
    Oracle: the hand-written `[ref=...]` suffix after the `...` mark.
    """
    ins = make_insight(
        id='meta-line-1', content='x' * (BRIEF_CONTENT_CHARS + 50),
        meta={'ref': '.handoff/auth-refresh/'})
    line = insight_to_recall_line(ins, 0.5)
    assert line.endswith('... [ref=.handoff/auth-refresh/]'), line


def test_recall_line_quotes_a_value_holding_a_space():
    """Verify a value with whitespace prints quoted, keys sorted.

    Mutation: printing the value bare, which splits one value into
        what reads as two tokens, or printing keys in insert order.
    Oracle: hand-written expected suffix.
    """
    ins = make_insight(
        id='meta-line-2', content='short',
        meta={'topic': 'token refresh', 'pr': '431'})
    line = insight_to_recall_line(ins, None)
    assert line.endswith('| short [pr=431 topic="token refresh"]'), line


@pytest.mark.parametrize('meta', [None, {}], ids=['unset', 'empty'])
def test_recall_line_without_meta_has_no_brackets(meta):
    """Verify a row with no meta ends at its text, with no `[]`.

    Mutation: testing `ins.meta is not None` in place of truthiness,
        which prints `[]` for an empty dict.
    Oracle: the line ends with the content itself.
    """
    line = insight_to_recall_line(
        make_insight(id='meta-line-3', content='plain row', meta=meta), None)
    assert line.endswith('| plain row'), line


# --- CLI write path, end to end ---

def test_remember_meta_reaches_show_and_recall(mm_runner):
    """Verify `remember --meta` lands on the row, the page and show.

    Mutation: dropping meta from the queue row, the worker's Insight,
        the insert, or insight_to_full_dict.
    Oracle: the pairs passed on the command line.
    """
    result = invoke(mm_runner, [
        'remember', '--meta', 'ref=.handoff/zebra-slug/',
        '--meta', 'ticket=ABC-12', 'the zebra rollout waits on review'])
    fact = parse_remember(result, mm_runner)
    shown = _show(mm_runner, fact['id'])
    assert shown['meta'] == {
        'ref': '.handoff/zebra-slug/', 'ticket': 'ABC-12'}
    page = invoke(mm_runner, ['recall', '--basic', 'zebra'])
    assert page.output.strip().endswith(
        '[ref=.handoff/zebra-slug/ ticket=ABC-12]'), page.output


@pytest.mark.parametrize('pairs', [
    pytest.param(['ref'], id='no-equals'),
    pytest.param(['Ref=x'], id='upper-case-key'),
    pytest.param(['=x'], id='empty-key'),
    pytest.param(['ref=a', 'ref=b'], id='duplicate-key'),
    pytest.param(['ref=a\nb'], id='line-break-in-value'),
    pytest.param(['ref=a\tb'], id='tab-in-value'),
    pytest.param(['ref=a\x9bb'], id='c1-control-in-value'),
    pytest.param(['ref=a\u202eb'], id='bidi-override-in-value'),
    pytest.param(['ref=' + 'x' * 200], id='over-the-byte-cap'),
    ])
def test_remember_refuses_bad_meta_and_queues_nothing(mm_runner, pairs):
    """Verify a malformed or oversized --meta is refused before queuing.

    Mutation: dropping one of the key, duplicate, control-character
        or byte-cap checks in the meta parser.
    Oracle: nonzero exit and an empty queue.
    """
    args = ['remember']
    for pair in pairs:
        args += ['--meta', pair]
    result = invoke(mm_runner, args + ['a claim about meta'])
    assert result.exit_code != 0, result.output
    assert queued_contents(mm_runner[1]) == []


def test_remember_accepts_meta_at_the_byte_cap(mm_runner):
    """Verify meta whose rendered form is exactly 200 bytes is stored.

    Mutation: `>=` in place of `>` in the byte-cap check.
    Oracle: `[ref=` + 194 chars + `]` is 200 bytes by hand count.
    """
    value = 'x' * 194
    result = invoke(mm_runner, [
        'remember', '--meta', f'ref={value}', 'a claim at the cap'])
    assert result.exit_code == 0, result.output
    fact = parse_remember(result, mm_runner)
    assert _show(mm_runner, fact['id'])['meta'] == {'ref': value}


# --- Replace ---

def _remember_with_ref(mm_runner: tuple) -> str:
    """Store one row carrying `ref=.handoff/a/` and return its id.
    """
    result = invoke(mm_runner, [
        'remember', '--meta', 'ref=.handoff/a/', 'the first claim'])
    return parse_remember(result, mm_runner)['id']


def test_replace_without_meta_keeps_the_target_pairs(mm_runner):
    """Verify a replace that passes no --meta inherits the target's.

    Mutation: the worker storing the successor with no meta when the
        queue row carries none.
    Oracle: the target's own pairs.
    """
    old_id = _remember_with_ref(mm_runner)
    result = invoke(mm_runner, ['replace', old_id, 'the corrected claim'])
    fact = parse_remember(result, mm_runner)
    assert _show(mm_runner, fact['id'])['meta'] == {'ref': '.handoff/a/'}


def test_replace_with_meta_stores_exactly_the_pairs_given(mm_runner):
    """Verify a replace with --meta replaces the whole set.

    Mutation: merging the given pairs over the target's.
    Oracle: only the pair passed to replace.
    """
    old_id = _remember_with_ref(mm_runner)
    result = invoke(mm_runner, [
        'replace', old_id, '--meta', 'ticket=B-2', 'the corrected claim'])
    fact = parse_remember(result, mm_runner)
    assert _show(mm_runner, fact['id'])['meta'] == {'ticket': 'B-2'}


def test_replace_with_an_empty_value_clears_the_pairs(mm_runner):
    """Verify `--meta ref=` on replace stores a row with no meta.

    Mutation: treating an all-empty --meta as no --meta, so the
        target's pairs are inherited.
    Oracle: `insights show` carries no `meta` key.
    """
    old_id = _remember_with_ref(mm_runner)
    result = invoke(mm_runner, [
        'replace', old_id, '--meta', 'ref=', 'the corrected claim'])
    fact = parse_remember(result, mm_runner)
    assert 'meta' not in _show(mm_runner, fact['id'])


# --- Store round trips, both backends ---

def test_meta_round_trips_through_the_store(backend):
    """Verify inserted meta comes back from a public store read.

    Mutation: dropping meta from either row mapper, or from the
        insert statement of either backend.
    Oracle: the dict passed to insert.
    """
    backend.nodes.insert(make_insight(
        id='meta-rt-1', meta={'ref': '.handoff/x/'}))
    assert backend.nodes.get('meta-rt-1').meta == {'ref': '.handoff/x/'}


def test_meta_round_trips_through_the_raw_path(backend):
    """Verify insert_raw and get_raw carry meta, as migrate and branches use.

    Mutation: leaving meta out of either backend's raw column list.
    Oracle: the dict on the MigrateInsight passed to insert_raw.
    """
    now = datetime.now(timezone.utc)
    backend.nodes.insert_raw(MigrateInsight(
        id='meta-raw-1', content='raw row', summary=None, embedding=None,
        enrich_attempted_at=None, enriched_at=None,
        created_at=now, updated_at=now, deleted_at=None,
        embedding_model=None, queue_uuid=None, replaced_by=None,
        author=None, summary_model=None, meta={'ref': '.handoff/y/'}))
    assert backend.nodes.get_raw('meta-raw-1').meta == {'ref': '.handoff/y/'}


@pytest.mark.parametrize(('meta', 'word'), [
    pytest.param({'ref': '.handoff/quokka-plan/'}, 'quokka', id='value'),
    pytest.param({'quokka_ref': 'x'}, 'quokka', id='key'),
    pytest.param({'ref': 'r\u00e9sum\u00e9-plan'}, 'r\u00e9sum\u00e9', id='non-ascii'),
    ])
def test_keyword_index_matches_a_meta_word(backend, meta, word):
    """Verify a word found only in a meta key or value is a keyword hit.

    Mutation: indexing content alone, dropping keys from insight_tokens,
        or json.dumps without ensure_ascii=False on the SQLite insert.
    Oracle: the word appears in meta and nowhere in content.
    """
    backend.nodes.insert(make_insight(
        id='meta-kw-1', content='a claim with no slug', meta=meta))
    with backend.recall_session() as session:
        assert session.keyword_counts({word}) == {'meta-kw-1': 1}


@pytest.mark.parametrize('word', [
    pytest.param('.handoff/quokka-plan/', id='path'),
    pytest.param('QUOKKA', id='case-folded'),
    ])
def test_basic_query_matches_a_meta_value(backend, word):
    """Verify `recall --basic`'s query finds a word held only in meta.

    Mutation: matching `content` alone in either backend's basic query.
    Oracle: the word appears in meta and nowhere in content.
    """
    backend.nodes.insert(make_insight(
        id='meta-basic-1', content='a claim with no slug',
        meta={'ref': '.handoff/quokka-plan/'}))
    assert [i.id for i in backend.nodes.query(keyword=word)] == [
        'meta-basic-1']


def test_scored_recall_page_prints_meta(mm_runner):
    """Verify the default scored recall page carries the meta suffix.

    Mutation: printing meta on the `--basic` path only.
    Oracle: the pair passed to remember.
    """
    invoke(mm_runner, [
        'remember', '--meta', 'ref=.handoff/kiwi/', 'the kiwi rollout plan'])
    page = invoke(mm_runner, ['recall', 'kiwi rollout plan'])
    assert '| the kiwi rollout plan [ref=.handoff/kiwi/]' in page.output
