"""`--meta key=value` pairs: parsing, storage, display, keyword index, replace.
"""

import json
from datetime import datetime, timezone

import pytest
from memman.migrate import MigrateInsight
from memman.queue import enqueue, queue_db
from memman.store import factory
from memman.store.model import BRIEF_CONTENT_CHARS, insight_to_recall_line
from tests.conftest import force_drain, invoke, make_insight, parse_remember
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
        meta={'ref': 'docs/auth-refresh/'})
    line = insight_to_recall_line(ins, 0.5)
    assert line.endswith('... [ref=docs/auth-refresh/]'), line


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
        'remember', '--meta', 'ref=docs/zebra-slug/',
        '--meta', 'ticket=ABC-12', 'the zebra cache evicts entries after an hour'])
    fact = parse_remember(result, mm_runner)
    shown = _show(mm_runner, fact['id'])
    assert shown['meta'] == {
        'ref': 'docs/zebra-slug/', 'ticket': 'ABC-12'}
    page = invoke(mm_runner, ['recall', '--basic', 'zebra'])
    assert page.output.strip().endswith(
        '[ref=docs/zebra-slug/ ticket=ABC-12]'), page.output


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
    """Store one row carrying `ref=docs/a/` and return its id.
    """
    result = invoke(mm_runner, [
        'remember', '--meta', 'ref=docs/a/', 'the first claim'])
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
    assert _show(mm_runner, fact['id'])['meta'] == {'ref': 'docs/a/'}


def test_replace_with_meta_sets_its_key_and_keeps_the_others(mm_runner):
    """Verify a replace with --meta sets its key over the target's pairs.

    Mutation: storing only the given pairs, as a whole-set replace does.
    Oracle: the target's ref plus the ticket passed to replace.
    """
    old_id = _remember_with_ref(mm_runner)
    result = invoke(mm_runner, [
        'replace', old_id, '--meta', 'ticket=B-2', 'the corrected claim'])
    fact = parse_remember(result, mm_runner)
    assert _show(mm_runner, fact['id'])['meta'] == {
        'ref': 'docs/a/', 'ticket': 'B-2'}


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


def _stored(mm_runner: tuple, *pairs: str) -> str:
    """Remember one claim with pairs, drain, and return its id.
    """
    args = ['remember']
    for pair in pairs:
        args += ['--meta', pair]
    result = invoke(mm_runner, args + ['The retry cap for batch jobs is three.'])
    assert result.exit_code == 0, result.output
    force_drain(mm_runner[1])
    return json.loads(result.output)['id']


def _queued_replace(
        mm_runner: tuple, target: str, *pairs: str,
        text: str = 'The retry cap for batch jobs is five.') -> str:
    """Queue a replace of target with pairs and return its id, undrained.
    """
    args = ['replace', target]
    for pair in pairs:
        args += ['--meta', pair]
    result = invoke(mm_runner, args + [text])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)['id']


@pytest.mark.no_auto_drain
def test_replace_meta_sets_its_key_and_keeps_the_other_keys(mm_runner):
    """Verify replace --meta changes the named key and keeps the rest.

    Mutation: storing only the given pairs (whole-set replace).
    Oracle: hand-merged {ref: docs/a/, ticket: B-2}.
    """
    target = _stored(mm_runner, 'ref=docs/a/', 'ticket=A-1')
    successor = _queued_replace(mm_runner, target, 'ticket=B-2')

    force_drain(mm_runner[1])

    assert _show(mm_runner, successor)['meta'] == {
        'ref': 'docs/a/', 'ticket': 'B-2'}


@pytest.mark.no_auto_drain
def test_replace_empty_value_removes_only_its_key(mm_runner):
    """Verify `--meta ticket=` on replace drops ticket and keeps ref.

    Mutation: an empty value clearing every pair, or being ignored.
    Oracle: hand-merged {ref: docs/a/}.
    """
    target = _stored(mm_runner, 'ref=docs/a/', 'ticket=A-1')
    successor = _queued_replace(mm_runner, target, 'ticket=')

    force_drain(mm_runner[1])

    assert _show(mm_runner, successor)['meta'] == {'ref': 'docs/a/'}


@pytest.mark.no_auto_drain
def test_redirected_replace_merges_onto_the_head_it_retires(mm_runner):
    """Verify a replace the drain redirects merges onto the chain head.

    Mutation: merging onto the row the queue names, or whole-set
        replace, so the first replace's ticket is lost.
    Oracle: hand-merged {ref: a, ticket: T, doc: d}. The second replace
        goes straight into queue.db, as two racing CLIs leave it.
    """
    _, data_dir = mm_runner
    target = _stored(mm_runner, 'ref=a')
    _queued_replace(mm_runner, target, 'ticket=T')
    with queue_db(data_dir) as conn:
        _, second = enqueue(
            conn, store='default', content='The retry cap is six.',
            replaced_id=target, meta={'doc': 'd'})

    force_drain(data_dir)

    assert _show(mm_runner, second)['meta'] == {
        'ref': 'a', 'ticket': 'T', 'doc': 'd'}


@pytest.mark.no_auto_drain
def test_degraded_replace_stores_only_its_own_pairs(mm_runner):
    """Verify a replace whose target was forgotten keeps only its pairs.

    Mutation: the base taken from the target when the link fails.
    Oracle: the replace's own {doc: d}.
    """
    target = _stored(mm_runner, 'hq=A')
    successor = _queued_replace(mm_runner, target, 'doc=d')
    forget = invoke(mm_runner, ['forget', target])
    assert forget.exit_code == 0, forget.output

    force_drain(mm_runner[1])

    assert _show(mm_runner, successor)['meta'] == {'doc': 'd'}


@pytest.mark.no_auto_drain
def test_replace_refuses_pairs_whose_merge_passes_the_cap(mm_runner):
    """Verify the cap applies to the merged pairs and nothing queues.

    Mutation: the cap checked on the given pairs alone.
    Oracle: `[ref=` + 100 + ` ticket=` + 100 + `]` is 214 bytes by hand.
    """
    target = _stored(mm_runner, 'ref=' + 'x' * 100)

    result = invoke(mm_runner, [
        'replace', target, '--meta', 'ticket=' + 'y' * 100,
        'The retry cap for batch jobs is five.'])

    assert result.exit_code == 1
    assert '214 bytes' in result.output
    assert len(queued_contents(mm_runner[1])) == 1


@pytest.mark.no_auto_drain
def test_replace_of_a_queued_target_refuses_a_merge_past_the_cap(mm_runner):
    """Verify a queued target's own pairs count toward the cap.

    Mutation: the cap checked on the given pairs alone for a queued
        target, so the drain stores 214 bytes.
    Oracle: `[ref=` + 100 + ` ticket=` + 100 + `]` is 214 bytes by hand.
    """
    queued = invoke(mm_runner, [
        'remember', '--meta', 'ref=' + 'x' * 100,
        'The retry cap for batch jobs is three.'])
    assert queued.exit_code == 0, queued.output

    result = invoke(mm_runner, [
        'replace', json.loads(queued.output)['id'],
        '--meta', 'ticket=' + 'y' * 100, 'The retry cap for batch jobs is five.'])

    assert result.exit_code == 1
    assert '214 bytes' in result.output
    assert len(queued_contents(mm_runner[1])) == 1


@pytest.mark.no_auto_drain
def test_drain_reads_the_base_after_the_link(mm_runner, monkeypatch):
    """Verify the successor's base is the target's pairs as of the link.

    Mutation: the target read before mark_replaced, which misses an
        edit committed between the read and the link.
    Oracle: a stub that edits the target's meta just before linking.
    """
    target = _stored(mm_runner, 'doc=old')
    successor = _queued_replace(mm_runner, target)
    with factory.open_backend('default', mm_runner[1]) as backend:
        node_class = type(backend.nodes)
    real_mark = node_class.mark_replaced

    def mark_replaced(self, predecessor_id, successor_id):
        self.update_meta(predecessor_id, {'doc': 'edited'})
        return real_mark(self, predecessor_id, successor_id)

    monkeypatch.setattr(node_class, 'mark_replaced', mark_replaced)

    force_drain(mm_runner[1])

    assert _show(mm_runner, successor)['meta'] == {'doc': 'edited'}


def test_remember_empty_value_sets_no_pair(mm_runner):
    """Verify remember stores no pair for an empty value.

    Mutation: the drain storing a queued empty value as a pair.
    Oracle: only doc=d, by hand.
    """
    result = invoke(mm_runner, [
        'remember', '--meta', 'hq=', '--meta', 'doc=d',
        'The retry cap for batch jobs is three.'])

    fact = parse_remember(result, mm_runner)

    assert _show(mm_runner, fact['id'])['meta'] == {'doc': 'd'}


# --- Store round trips, both backends ---

def test_meta_round_trips_through_the_store(backend):
    """Verify inserted meta comes back from a public store read.

    Mutation: dropping meta from either row mapper, or from the
        insert statement of either backend.
    Oracle: the dict passed to insert.
    """
    backend.nodes.insert(make_insight(
        id='meta-rt-1', meta={'ref': 'docs/x/'}))
    assert backend.nodes.get('meta-rt-1').meta == {'ref': 'docs/x/'}


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
        author=None, summary_model=None, meta={'ref': 'docs/y/'}))
    assert backend.nodes.get_raw('meta-raw-1').meta == {'ref': 'docs/y/'}


@pytest.mark.parametrize(('meta', 'word'), [
    pytest.param({'ref': 'docs/quokka-plan/'}, 'quokka', id='value'),
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
    pytest.param('docs/quokka-plan/', id='path'),
    pytest.param('QUOKKA', id='case-folded'),
    ])
def test_basic_query_matches_a_meta_value(backend, word):
    """Verify `recall --basic`'s query finds a word held only in meta.

    Mutation: matching `content` alone in either backend's basic query.
    Oracle: the word appears in meta and nowhere in content.
    """
    backend.nodes.insert(make_insight(
        id='meta-basic-1', content='a claim with no slug',
        meta={'ref': 'docs/quokka-plan/'}))
    assert [i.id for i in backend.nodes.query(keyword=word)] == [
        'meta-basic-1']


def test_scored_recall_page_prints_meta(mm_runner):
    """Verify the default scored recall page carries the meta suffix.

    Mutation: printing meta on the `--basic` path only.
    Oracle: the pair passed to remember.
    """
    invoke(mm_runner, [
        'remember', '--meta', 'ref=docs/kiwi/', 'the kiwi rollout plan'])
    page = invoke(mm_runner, ['recall', 'kiwi rollout plan'])
    assert '| the kiwi rollout plan [ref=docs/kiwi/]' in page.output
