"""memman holds lasting findings and no session state.

`remember` and `replace` refuse every leading label with one message.
It says to state the lasting finding as a sentence, or to store nothing
for session state. A `--meta` pointer to any work folder is accepted. The quality
check warns on wording that marks a transient state, and the write
still queues.
"""

import json

import pytest
from memman.search.quality import check_content_quality
from tests.conftest import invoke, queued_contents

STORE_NOTHING_PHRASE = 'store nothing when the text is session state'


def test_remember_accepts_a_meta_pointer_to_a_dotted_work_folder(mm_runner):
    """Verify `--meta hq=.work/x/` on a lasting finding queues the row.

    Mutation: a guard that refuses a dot-prefixed folder, or any
        folder path, in a meta value, which blocks the pointer a
        lasting finding carries to its work folder.
    Oracle: exit code 0 and the finding queued byte for byte.
    """
    _, data_dir = mm_runner
    text = 'the retry cap is five'

    result = invoke(mm_runner, ['remember', '--meta', 'hq=.work/x/', text])

    assert result.exit_code == 0, result.output
    assert queued_contents(data_dir) == [text]


@pytest.mark.parametrize('label', [
    'TODO:', 'Plan:', 'Open question:', 'WIP:', 'Fix:', '**TODO:**',
    ])
def test_a_leading_label_is_refused_with_the_session_state_cue(
        mm_runner, label):
    """Verify every leading label refuses, quoted, with the store-nothing cue.

    Mutation: a label set routed to a second message, so a label
        outside the set loses the cue to store nothing for session
        state; markdown emphasis around the label defeating the match;
        or the message not quoting the label.
    Oracle: exit code, an empty queue, the label in the output, and
        the store-nothing phrase in the output.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, [
        'remember', f'{label} wire the retry cap into the poller'])

    assert result.exit_code != 0
    assert label in result.output
    assert STORE_NOTHING_PHRASE in result.output
    assert queued_contents(data_dir) == []


def test_a_colon_after_the_three_word_slot_is_not_a_label(mm_runner):
    """Verify a colon after four words is no label and enqueues.

    Mutation: `search` in place of `match` with no `^` anchor, so the
        pattern finds `is simple:` mid-line, or a word slot widened
        past three words.
    Oracle: a four-word clause before a colon. It queues byte for byte.
    """
    _, data_dir = mm_runner
    text = 'The rule is simple: keep three retries'

    result = invoke(mm_runner, ['remember', text])

    assert result.exit_code == 0, result.output
    assert queued_contents(data_dir) == [text]


@pytest.mark.parametrize(('text', 'label'), [
    ('the next step is to rebuild the index', 'next step'),
    ('the next steps need a second pass', 'next steps'),
    ('step 3 finished before the lock cleared', 'step finished'),
    ('phase 2 is done and the index is built', 'phase done'),
    ('stage 4 completed without errors', 'stage completed'),
    ('cycle 7 ran the same checks again', 'cycle 7'),
    ('the default should be changed to five', 'should be changed to'),
    ])
def test_transient_wording_warns_without_blocking(mm_runner, text, label):
    """Verify session-state wording draws a warning and still queues.

    Mutation: a pattern dropped or narrowed, or a warning promoted to a
        refusal.
    Oracle: hand-labeled rows. Each exits 0, queues byte for byte, and
        returns at least one quality warning.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, ['remember', text])

    assert result.exit_code == 0, f'{label}: {result.output}'
    assert queued_contents(data_dir) == [text]
    assert json.loads(result.output)['quality_warnings'], label


@pytest.mark.parametrize('text', [
    'The retry cap is five and the backoff doubles each time',
    'the step size of the optimizer is 0.01',
    'the phase of the signal lags by 90 degrees',
    'a cycle of length 3 makes the graph cyclic',
    'the worker waits on the drain lock',
    'a row not yet embedded stays out of vector recall',
    'the scheduler can resume a failed drain',
    'the build step is done by the CI runner',
    ])
def test_lasting_findings_return_no_warning(text):
    """Verify lasting findings, even with near-miss words, draw none.

    Mutation: a pattern with no boundary (`step` alone, `cycle` without
        the digit), a finished-step pattern that takes a step with no
        number, or a `waits on`, `not yet` or `resume` pattern. Each
        flags ordinary technical prose and invites a `forget` of a true
        claim.
    Oracle: hand-labeled lasting claims.
    """
    assert check_content_quality(text) == []
