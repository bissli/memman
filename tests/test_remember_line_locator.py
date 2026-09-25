"""`remember` and `replace` must refuse text that names a line number.

A line number is a snapshot of a file at write time and goes stale on
the next edit. The CLI refuses it before enqueue and never rewrites
the text, so the agent names the file and the symbol in its own words.
"""

import pytest
from memman.cli import _line_locator_refusal_message
from tests.conftest import invoke, parse_remember, queued_contents


@pytest.mark.parametrize(('text', 'locator'), [
    ('See cli.py:182 for the fix', 'cli.py:182'),
    ('src/pkg/module/file.py:590 has the bug',
     'src/pkg/module/file.py:590'),
    ('Error on line 42 of the module', 'line 42'),
    ('See app/page.html:106 for the diff', 'app/page.html:106'),
    ('Edit config.yaml:12 then restart', 'config.yaml:12'),
    ('Bind to localhost:8080', None),
    ('The pool connects to db.example.com:5432 over TLS', None),
    ('See https://api.example.com:8443/v1 for the API', None),
    ('Reach 192.0.2.1:8000 over VPN', None),
    ('Worker fires daily at 14:18', None),
    ('Returns code:404 on miss', None),
    ('Use python:3.11 base image', None),
    ('redis:6379 is the cache', None),
    ('The deploy pipeline 2 stage retries', None),
])
def test_refusal_names_only_a_file_line_locator(text, locator):
    """Verify the refusal quotes a line locator and passes hosts and tags.

    Mutation: matching any dot-letter run before the colon, which reads
        `db.example.com:5432` as a file; or broadening the pattern until
        it also catches a `host:port`, a `HH:MM` clock, an image tag, a
        `code:404`, or `line` inside a longer word.
    Oracle: hand-paired rows; each refusing row names the exact locator
        the message must quote, each passing row names none.
    """
    message = _line_locator_refusal_message(text)

    if locator is None:
        assert message is None
    else:
        assert f'({locator!r})' in message


@pytest.mark.parametrize(('text', 'locator'), [
    ('The accesses at lines 427-431 route through from_dict',
     'lines 427'),
    ('The guard at app.py-1233 refuses the stamp', 'app.py-1233'),
    ('Production recall reranks at cli.py ~1190', 'cli.py ~1190'),
    ('The deny branch sits at :1774 in the gate', ':1774'),
    ('OnRefreshPressed (:243) calls CalculateFull', ':243'),
    ('The trim runs in libtc/tc/emsx.py L419', 'L419'),
    ('preprocess_excel_file (~L287) returns early', '~L287'),
    ('The oplog keeps keyword_str[:80] of the query', None),
    ('Export DISPLAY=:99 before drawio runs', None),
    ('The rewrite left 1175 lines in documents.py', None),
    ('The CPU L1 cache misses on the scan', None),
])
def test_refusal_names_every_line_number_form(text, locator):
    """Verify the refusal catches each stored line-number form.

    Mutation: refusing only `path.ext:N` and `line N`, which passes
        `lines N`, `path.ext-N`, `path.ext ~N`, a bare `:N` and an
        `L123`; or a bare-colon pattern that also reads a slice
        `[:80]` or `DISPLAY=:99`, a line count `1175 lines`, or an
        `L1` cache as a line number.
    Oracle: hand-paired rows drawn from forms stored rows carry; each
        refusing row names the exact locator the message must quote,
        each passing row names none.
    """
    message = _line_locator_refusal_message(text)

    if locator is None:
        assert message is None
    else:
        assert f'({locator!r})' in message


def test_remember_refuses_a_file_line_locator(mm_runner):
    """Verify `path.ext:N` fails the write and enqueues nothing.

    Mutation: the check dropped, or a strip that stores `auth.py` and
        reports success.
    Oracle: the CLI exit code, the error quoting the locator, and a
        direct read of the queue, which must stay empty.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, [
        'remember', 'The 401 branch sits at scripts/auth.py:88'])

    assert result.exit_code != 0
    assert "'scripts/auth.py:88'" in result.output
    assert 'line number' in result.output
    assert queued_contents(data_dir) == []


def test_remember_refuses_the_word_line_and_a_number(mm_runner):
    """Verify `line N` fails the write as `path.ext:N` does.

    Mutation: refusing only the `path.ext:N` form.
    Oracle: the CLI exit code and the empty queue.
    """
    _, data_dir = mm_runner

    result = invoke(mm_runner, [
        'remember', 'The 401 branch is at line 88 of the poller'])

    assert result.exit_code != 0
    assert "'line 88'" in result.output
    assert queued_contents(data_dir) == []


def test_remember_accepts_ports_versions_and_times(mm_runner):
    """Verify a colon and digits with no file extension enqueue whole.

    Mutation: a pattern matching any `name:digits`, which refuses a
        host port, a version pin, or a clock time.
    Oracle: the enqueued row compared byte for byte with the input,
        which holds one of each shape the locator must not match.
    """
    _, data_dir = mm_runner
    text = (
        'Redis listens on localhost:6379 and Postgres on'
        ' db.example.com:5432; the image pins python:3.11 and the'
        ' backup runs at 03:00 through pipeline 2')

    result = invoke(mm_runner, ['remember', text])

    assert result.exit_code == 0, result.output
    assert queued_contents(data_dir) == [text]


def test_replace_refuses_a_line_locator(mm_runner):
    """Verify `replace` shares the refusal, not just `remember`.

    Mutation: the check wired into `remember` alone.
    Oracle: the CLI exit code, plus the queue holding only the row the
        initial `remember` wrote.
    """
    _, data_dir = mm_runner
    first = invoke(mm_runner, [
        'remember', 'a note that will be replaced'])
    old = parse_remember(first, mm_runner)
    before = queued_contents(data_dir)

    result = invoke(mm_runner, [
        'replace', old['id'], 'The poller retries in poll.py:120'])

    assert result.exit_code != 0
    assert 'line number' in result.output
    assert queued_contents(data_dir) == before
