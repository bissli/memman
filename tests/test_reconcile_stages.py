"""The three staged reconcile contracts: screen, verdict, merge.

Each stage's reader is pinned against the response shapes the probe
measured: the vocabulary, the id map onto the one shown row, the
fail-open and fail-closed readings, the body each stage sends and the
stage and ceiling each names on the client.
"""

import hashlib
import json

import pytest
from memman.llm import usage as llm_usage
from memman.llm.extract import MERGE_MAX_TOKENS, MERGE_SYSTEM
from memman.llm.extract import PAIRWISE_SCREEN_SYSTEM, RECONCILIATION_SYSTEM
from memman.llm.extract import SCREEN_MAX_TOKENS, SCREEN_RELATIONS, UNJUDGED
from memman.llm.extract import VERDICT_DISPOSITION, VERDICT_MAX_TOKENS
from memman.llm.extract import judge_memory, merge_successor, screen_memory


class _Client:
    """Record every `complete` call; answer one canned response or raise."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def complete(self, system, user, *, stage, max_tokens=None):
        self.calls.append({'system': system, 'user': user,
                           'stage': stage, 'max_tokens': max_tokens})
        if self.error is not None:
            raise self.error
        return self.response


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


RESTARTED_OBJECT = (
    '{"relation": "CONTRADICTS", "contradicted_clauses": ["the broker is '
    'kombu", wait, the memory also says {"relation": "REFINES", '
    '"contradicted_clauses": [')


# --- stage 1: the screen ---


def test_screen_reads_the_relation_and_its_quoted_clauses():
    """Verify the screen returns the relation with the clauses it quoted.

    Mutation: reading `relation` alone, so every supersede target
        reaches stage 3 under `(none)` and the merge writer is never
        told which clause to drop.
    Oracle: the canned response's relation and clause list.
    """
    client = _Client(json.dumps({
        'relation': 'CONTRADICTS',
        'contradicted_clauses': ['the broker is kombu'],
        'reason': 'the fact names redis'}))
    relation, clauses = screen_memory(
        client, 'the broker is redis', ('m-1', 'the broker is kombu'))
    assert (relation, clauses) == ('CONTRADICTS', ['the broker is kombu'])


def test_screen_keeps_only_string_clauses():
    """Verify a clause list is read as its non-empty strings and nothing else.

    Mutation: coercing every truthy element with `str()`, so a nested
        object reaches the merge body as its Python repr and the writer
        is told to drop a clause the row never held.
    Oracle: one string survives from a list mixing an object, a number,
        an empty string and the string.
    """
    client = _Client(json.dumps({
        'relation': 'CONTRADICTS',
        'contradicted_clauses': [{'clause': 'the broker is kombu'}, 7, '', 'ok']}))
    assert screen_memory(client, 'f', ('m-1', 'c')) == ('CONTRADICTS', ['ok'])


def test_screen_drops_clauses_outside_a_contradiction():
    """Verify clauses ride only on CONTRADICTS, as the screen text demands.

    Mutation: passing the list through on any relation, so a RESTATES
        row stage 2 later supersedes carries clauses the fact never
        overturned into the merge body.
    Oracle: the screen text's last line and the empty list.
    """
    client = _Client(json.dumps({
        'relation': 'RESTATES', 'contradicted_clauses': ['stray clause']}))
    assert screen_memory(client, 'f', ('m-1', 'c')) == ('RESTATES', [])


@pytest.mark.parametrize('raw', [
    json.dumps({'relation': 'MAYBE', 'contradicted_clauses': []}),
    json.dumps({'contradicted_clauses': []}),
    json.dumps({'relation': ['CONTRADICTS'], 'contradicted_clauses': []}),
    RESTARTED_OBJECT,
    '{"actions": ' + '[' * 1200,
    'no json at all',
    ])
def test_screen_reads_an_unreadable_verdict_as_unjudged(raw):
    """Verify an unknown relation or an unparsed body reads as UNJUDGED.

    Mutation: passing an unknown token through (the keep rule then
        drops the row, or the disposition raises), raising on a
        relation that is not a string or on a body nested past the
        recursion limit, or reading the restarted object the probe met
        as UNRELATED, so a row the screen could not judge never reaches
        stage 2.
    Oracle: the fail-open token, for a stray token, a missing key, a
        list where the token belongs, the reasoning-inside-the-array
        shape, a runaway nesting and plain prose.
    """
    assert screen_memory(_Client(raw), 'f', ('m-1', 'c')) == (UNJUDGED, [])


def test_screen_reads_a_clause_string_as_no_clauses():
    """Verify a clause field that is one string, not a list, yields no clauses.

    Mutation: iterating the value whatever its type, so the merge body
        lists one clause per character.
    Oracle: the relation with an empty clause list.
    """
    client = _Client(json.dumps({
        'relation': 'CONTRADICTS', 'contradicted_clauses': 'the broker is kombu'}))
    assert screen_memory(client, 'f', ('m-1', 'c')) == ('CONTRADICTS', [])


def test_screen_error_is_unjudged():
    """Verify a client error leaves the row UNJUDGED rather than raising.

    Mutation: letting the exception out, so one screen failure among
        twenty drops the whole write to the retry path.
    Oracle: the fail-open token on a raising client.
    """
    client = _Client(error=RuntimeError('boom'))
    assert screen_memory(client, 'f', ('m-1', 'c')) == (UNJUDGED, [])


def test_screen_sends_the_pair_under_its_stage_and_ceiling():
    """Verify the screen call's body, stage and token ceiling.

    Mutation: booking the call under `reconciliation` (the ledger then
        cannot separate twenty screen calls from the verdict calls), or
        leaving the role ceiling in place (one measured response hit
        1024 on a reasoning preamble).
    Oracle: the measured body `EXISTING MEMORY / NEW FACT`, the stage
        `screen`, and the constant, which is 2048.
    """
    client = _Client(json.dumps(
        {'relation': 'UNRELATED', 'contradicted_clauses': []}))
    screen_memory(client, 'the fact', ('m-1', 'the memory'))
    call = client.calls[0]
    assert call['system'] == PAIRWISE_SCREEN_SYSTEM
    assert call['user'] == 'EXISTING MEMORY:\nthe memory\n\nNEW FACT:\nthe fact'
    assert call['stage'] == llm_usage.STAGE_SCREEN == 'screen'
    assert call['max_tokens'] == SCREEN_MAX_TOKENS == 2048


def test_screen_text_is_the_measured_pairwise_v1():
    """Verify the screen text ships as measured, byte for byte.

    Mutation: any edit to the text (a rewrap, a softened clause), which
        moves the screen off the 51/64 and 8/86 it was measured at.
    Oracle: the sha256 prefix of `prompt_pairwise-v1.txt` as run in the
        probe, and the vocabulary line the disposition is keyed on.
    """
    assert _sha(PAIRWISE_SCREEN_SYSTEM) == '416e9b2b4ecfa357'
    enum_line = next(
        ln for ln in PAIRWISE_SCREEN_SYSTEM.splitlines()
        if ln.startswith('{"relation":'))
    named = set(enum_line.split('"')[3].split('|'))
    assert named == set(SCREEN_RELATIONS) == {
        'CONTRADICTS', 'REFINES', 'RESTATES', 'UNRELATED'}


# --- stage 2: the verdict ---


def test_verdict_lands_an_unshown_id_on_the_shown_row():
    """Verify a verdict on an id the call never showed lands on the row.

    Mutation: the id-map lookup dropping the entry, so the row scores
        keep: the 25 no-write rows of the probe, where the model
        numbered sections of the one memory and judged section 3.
    Oracle: `supersede` for a SUPERSEDE on target 3 when `[0]` alone
        was shown.
    """
    client = _Client(json.dumps({'actions': [
        {'action': 'SUPERSEDE', 'target_id': 3, 'reason': 'stale'}]}))
    assert judge_memory(client, 'f', ('m-1', 'c')) == 'supersede'


def _actions(*pairs):
    return [{'action': action, 'target_id': target} for action, target in pairs]


@pytest.mark.parametrize(('actions', 'expected'), [
    (_actions(('ADD', 0), ('SUPERSEDE', 3)), 'keep'),
    (_actions(('SUPERSEDE', 3), ('UPDATE', 0)), 'supersede'),
    (_actions(('ADD', None), ('SUPERSEDE', 0)), 'supersede'),
    (_actions(('NONE', 0), ('UPDATE', 1)), 'none'),
    (_actions(('bogus', 0), ('SUPERSEDE', 0)), 'supersede'),
    (_actions(('UPDATE', None), ('SUPERSEDE', 0)), 'supersede'),
    ])
def test_verdict_takes_the_first_action_that_names_the_row(actions, expected):
    """Verify the `first` rule: the first known entry naming any id decides.

    Mutation: the `supwins` variant (SUPERSEDE beats UPDATE beats
        NONE); a null-id entry blocking the entry after it; an unknown
        token blocking instead of being skipped. `first` and `supwins`
        were both measured and `first` shipped for its lower collateral.
    Oracle: the six response orders hand-disposed under the measured
        rule: skip a null id or an unknown token, then the first entry
        decides, ADD included.
    """
    client = _Client(json.dumps({'actions': actions}))
    assert judge_memory(client, 'f', ('m-1', 'c')) == expected


@pytest.mark.parametrize('raw', [
    json.dumps({'actions': [{'action': 'ADD', 'target_id': None}]}),
    json.dumps({'actions': [{'action': 'UPDATE', 'target_id': None}]}),
    json.dumps({'actions': [{'action': 'bogus', 'target_id': 0}]}),
    json.dumps({'actions': ['SUPERSEDE 0']}),
    json.dumps({'actions': []}),
    json.dumps({'actions': 'SUPERSEDE'}),
    json.dumps({'merged_text': 'x'}),
    RESTARTED_OBJECT,
    '{"actions": ' + '[' * 1200,
    'no json',
    ])
def test_verdict_reads_a_non_verdict_as_keep(raw):
    """Verify every non-verdict shape reads as keep.

    Mutation: writing the fact against the row on an unknown token;
        raising on an entry that is not a dict, on `actions` that is not
        a list, on a body nested past the recursion limit, or on the
        restarted object the probe met, instead of the fail-closed
        reading.
    Oracle: `keep`, the no-write disposition, on an ADD, an id-less row
        verdict, an unknown token, a string entry, an empty list, a
        string where the list belongs, a wrong key, the restarted
        object, a runaway nesting and prose.
    """
    assert judge_memory(_Client(raw), 'f', ('m-1', 'c')) == 'keep'


def test_verdict_error_is_keep():
    """Verify a client error is a no-write, never a raise.

    Mutation: letting the exception out of stage 2, so one failed row
        drops the whole write.
    Oracle: `keep` on a raising client.
    """
    client = _Client(error=RuntimeError('boom'))
    assert judge_memory(client, 'f', ('m-1', 'c')) == 'keep'


def test_verdict_reads_delete_as_supersede():
    """Verify a stray DELETE lands as supersede.

    Mutation: dropping the alias, so DELETE falls to keep and the
        contradicted row stays current beside the fact.
    Oracle: `supersede` on a DELETE naming the row.
    """
    client = _Client(json.dumps(
        {'actions': [{'action': 'DELETE', 'target_id': 0}]}))
    assert judge_memory(client, 'f', ('m-1', 'c')) == 'supersede'


def test_verdict_sends_one_row_under_its_stage_and_ceiling():
    """Verify the verdict call shows one row under `[0]` at its ceiling.

    Mutation: the pooled body (every kept row in one call), or the
        8192 ceiling of the merged-text contract.
    Oracle: the cell's body from `pcommon.user_body`, the stage
        `reconciliation`, and the constant, which is 2048.
    """
    client = _Client(json.dumps(
        {'actions': [{'action': 'ADD', 'target_id': None}]}))
    judge_memory(client, 'the fact', ('m-1', 'the memory'))
    call = client.calls[0]
    assert call['system'] == RECONCILIATION_SYSTEM
    assert call['user'] == (
        'EXISTING MEMORIES:\n[0] the memory\n\nNEW FACT:\nthe fact')
    assert call['stage'] == llm_usage.STAGE_RECONCILIATION
    assert call['max_tokens'] == VERDICT_MAX_TOKENS == 2048


def test_reconciliation_text_is_verdict_only_with_no_tie_break():
    """Verify the v3 verdict text ships with change 11 and nothing else.

    Mutation: the merged-text paragraph restored (the call writes the
        long tail again), or the several-memories tie-break left in,
        which folded 2 of 11 synthetic restatements as UPDATE at pool
        size one.
    Oracle: the sha256 prefix of `prompt_candidate-v3-verdict.txt` with
        the tie-break sentence removed, and the lines that carry both
        amendments.
    """
    assert _sha(RECONCILIATION_SYSTEM) == '0a21483f5cb8e4db'
    assert 'merged_text' not in RECONCILIATION_SYSTEM
    assert 'most complete' not in RECONCILIATION_SYSTEM
    lines = RECONCILIATION_SYSTEM.splitlines()
    update_line = next(ln for ln in lines if ln.startswith('- UPDATE'))
    assert update_line == (
        '- UPDATE <id>: the fact refines memory <id>. At most one.')
    none_line = next(ln for ln in lines if ln.startswith('- NONE'))
    assert '<id>' in none_line


def test_every_verdict_token_has_a_disposition():
    """Verify the disposition table covers exactly the tokens the text names.

    Mutation: a token added to the text without its disposition (the
        parser then skips every entry carrying it, so a row the model
        judged is never written against), or a disposition kept for a
        token the text no longer offers.
    Oracle: the `takes only` line of the shipped text, parsed.
    """
    line = next(
        ln for ln in RECONCILIATION_SYSTEM.splitlines()
        if ln.startswith('The action field takes only'))
    named = set(
        line.removeprefix('The action field takes only ').rstrip('.')
        .replace(' or ', ' ').replace(',', ' ').split())
    assert named == set(VERDICT_DISPOSITION)
    assert set(VERDICT_DISPOSITION.values()) <= {
        'supersede', 'update', 'none', 'keep'}
    with pytest.raises(KeyError):
        VERDICT_DISPOSITION['KEEP']


# --- stage 3: the merge ---


def test_merge_lists_the_target_its_clauses_then_the_fact():
    """Verify the merge body is the measured row-alone shape.

    Mutation: the whole-body shape (every target in one call), which
        dropped a true clause in 13 of 24 against 7 of 24, or the
        clause block omitted.
    Oracle: `c13_merge.alone_body`: the target under `[0]`, its clauses
        as `- ` lines, then the fact.
    """
    client = _Client(json.dumps(
        {'merged_text': 'the broker is redis; the queue is durable'}))
    text = merge_successor(
        client, 'the broker is redis',
        ('m-1', 'the broker is kombu and the queue is durable',
         ['the broker is kombu']))
    assert text == 'the broker is redis; the queue is durable'
    call = client.calls[0]
    assert call['system'] == MERGE_SYSTEM
    assert call['user'] == (
        'EXISTING MEMORIES:\n[0] the broker is kombu and the queue is durable\n'
        'CONTRADICTED CLAUSES of [0]:\n- the broker is kombu\n\n'
        'NEW FACT:\nthe broker is redis')
    assert call['stage'] == llm_usage.STAGE_MERGE == 'merge'
    assert call['max_tokens'] == MERGE_MAX_TOKENS == 8192


def test_merge_suppresses_clauses_copied_from_the_fact(monkeypatch):
    """Verify a clause copied from the fact is not rendered in the body.

    Mutation: rendering a clause the screen copied from the fact, so
        the merge deletes the fact's own correction from the successor;
        or comparing raw text, which lets a fact sentence through when
        the screen changed its case or wrapping.
    Oracle: three hand-built clauses - one verbatim in the target
        content (rendered), one in the fact and absent from the content,
        differing from the fact by case and a double space (not
        rendered), one present in both (rendered) - asserted on the
        exact clause block; the trace carries the suppressed count.
    """
    from memman import trace
    events = []
    monkeypatch.setattr(trace, 'event',
                        lambda name, **kw: events.append((name, kw)))
    client = _Client(json.dumps({'merged_text': 'ok'}))
    merge_successor(
        client,
        'the queue is redis and the cache is warm',
        ('m-1',
         'the queue is durable and the cache is warm and the logger is active',
         ['the queue is durable', 'The queue  is REDIS', 'the cache is warm']))
    body = client.calls[0]['user']
    clause_block = body.split('CONTRADICTED CLAUSES')[1].split('NEW FACT')[0]
    assert clause_block == ' of [0]:\n- the queue is durable\n- the cache is warm\n\n'
    merged = [kw for name, kw in events if name == 'reconcile_merge']
    assert merged[-1]['clauses'] == 2
    assert merged[-1]['suppressed'] == 1


def test_merge_marks_an_update_target_with_no_clauses():
    """Verify a target with no quoted clauses is listed under `(none)`.

    Mutation: an empty clause block, which the measured text did not
        see; the cell listed `(none)` for every update target.
    Oracle: the `(none)` line of the c13 body.
    """
    client = _Client(json.dumps({'merged_text': 't'}))
    merge_successor(client, 'f', ('m-1', 'c', []))
    body = client.calls[0]['user']
    assert 'CONTRADICTED CLAUSES of [0]:\n(none)\n\nNEW FACT:' in body


@pytest.mark.parametrize('client', [
    _Client(json.dumps({'merged_text': ''})),
    _Client(json.dumps({'merged_text': '  \n '})),
    _Client(json.dumps({'merged_text': None})),
    _Client('{"merged_text": ' + '[' * 1200),
    _Client(json.dumps({'text': 'wrong key'})),
    _Client('no json'),
    _Client(error=RuntimeError('boom')),
    ])
def test_merge_returns_none_when_no_text_came_back(client):
    """Verify every merge failure is None, so the caller stores the fact.

    Mutation: returning the empty or whitespace-only string (the
        successor stores nothing), or raising on a runaway nesting or
        an error (the write fails on a merge the fact never needed).
    Oracle: None on an empty text, a whitespace text, a null, a runaway
        nesting, a wrong key, prose and an error.
    """
    assert merge_successor(client, 'f', ('m-1', 'c', [])) is None


def test_merge_strips_the_returned_text():
    """Verify surrounding whitespace never reaches the stored row.

    Mutation: storing the text as returned, so an `(unmerged)` read on
        `content == fact_text` misses a fact the model echoed with a
        trailing newline.
    Oracle: the stripped text.
    """
    client = _Client(json.dumps({'merged_text': '  the text \n'}))
    assert merge_successor(client, 'f', ('m-1', 'c', [])) == 'the text'


def test_merge_text_is_the_measured_v1():
    """Verify the merge text ships as measured, byte for byte.

    Mutation: any edit to the text, which moves stage 3 off the 7/24
        dropped-clause line it was measured at.
    Oracle: the sha256 prefix of `prompt_merge-v1.txt` as run.
    """
    assert _sha(MERGE_SYSTEM) == '5f67a35d44a231b7'


def test_screen_and_merge_stages_are_valid():
    """Verify the two new stages are in the client's closed set.

    Mutation: a stage constant dropped from `VALID_STAGES` while the
        stage function still passes it, so the first live call raises
        in `record`.
    Oracle: `record` accepting both without raising, beside the stage
        tests above that pin what each function passes.
    """
    usage = {'prompt_tokens': 1, 'completion_tokens': 1}
    llm_usage.record(llm_usage.STAGE_SCREEN, usage)
    llm_usage.record(llm_usage.STAGE_MERGE, usage)
    assert {llm_usage.STAGE_SCREEN, llm_usage.STAGE_MERGE} <= llm_usage.VALID_STAGES
