"""Tests for the agent-verb call log at `<data_dir>/logs/calls.log`.
"""

import json
import re
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import click
import pytest
from memman.cli import claude_callable, cli
from tests.conftest import invoke

_LINE_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)'
    r'\|(?P<verb>[a-z -]+)\|(?P<store>[\w-]+|\?)'
    r'\|(?P<exit>\d+)\|(?P<ms>\d+)$')

_AGENT_VERB_ARGS = [
    ('doctor', []),
    ('forget', ['no-such-id']),
    ('insights review', []),
    ('insights show', ['no-such-id']),
    ('recall', ['retry host']),
    ('remember', ['The retry host is db-a, since db-b is read-only.']),
    ('replace', ['no-such-id', 'The retry host is db-c.']),
    ('status', []),
    ]


def _call_lines(data_dir: str) -> list[dict[str, str]]:
    """Parsed lines of calls.log; empty when the file does not exist.
    """
    path = Path(data_dir) / 'logs' / 'calls.log'
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        match = _LINE_RE.match(line)
        assert match, f'line off the call-log format: {line!r}'
        rows.append(match.groupdict())
    return rows


@pytest.fixture
def exit_probe() -> Iterator[None]:
    """Register an agent-callable `exit-probe` verb for the test's length.

    `--mode` picks how the body ends: `exit3` calls `ctx.exit(3)`,
    `usage` raises `UsageError` (exit 2), `crash` raises `RuntimeError`,
    and `sleep` returns after 50 ms.
    """
    @claude_callable
    @cli.command('exit-probe')
    @click.option('--mode', required=True)
    @click.pass_context
    def probe(ctx: click.Context, mode: str) -> None:
        if mode == 'exit3':
            ctx.exit(3)
        if mode == 'usage':
            raise click.UsageError('probe')
        if mode == 'sleep':
            time.sleep(0.05)
            return
        raise RuntimeError('probe')

    yield
    cli.commands.pop('exit-probe')


def test_agent_verb_appends_one_line_per_call(mm_runner):
    """Verify each agent-callable call appends one line in call order.

    Mutation: no line written, or a line per nested context instead of
        per call.
    Oracle: two `status` calls on the default store, hand-counted.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['status'])
    invoke(mm_runner, ['status'])
    rows = _call_lines(data_dir)
    assert [(r['verb'], r['store'], r['exit']) for r in rows] == [
        ('status', 'default', '0'),
        ('status', 'default', '0'),
        ]


@pytest.mark.parametrize(('verb', 'args'), _AGENT_VERB_ARGS)
def test_each_agent_verb_logs_its_calls(mm_runner, verb, args):
    """Verify every verb an agent runs writes a line naming its full path.

    Mutation: dropping `@claude_callable` from one of these verbs, or
        logging only the leaf name (`show` for `insights show`).
    Oracle: the hand-kept list of the eight agent verbs.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, [*verb.split(), *args])
    assert [r['verb'] for r in _call_lines(data_dir)] == [verb]


def test_store_field_names_the_resolved_store(mm_runner, monkeypatch):
    """Verify the store field follows `MEMMAN_STORE`, then the flag.

    Mutation: logging the raw `--store` flag, which is empty here, a
        hard-coded `default`, or a name check stricter than
        `valid_store_name` that turns `my_store-2` into `?`.
    Oracle: `MEMMAN_STORE=work`, then `--store my_store-2`, set by hand.
    """
    _, data_dir = mm_runner
    monkeypatch.setenv('MEMMAN_STORE', 'work')
    invoke(mm_runner, ['status'])
    invoke(mm_runner, ['--store', 'my_store-2', 'status'])
    stores = [r['store'] for r in _call_lines(data_dir)]
    assert stores == ['work', 'my_store-2']


@pytest.mark.parametrize('store', [
    'a|b\nFAKE|remember|x|0|0',
    'work\n',
    ])
def test_forged_store_name_cannot_add_a_line(mm_runner, store):
    """Verify a store name carrying `|` or a newline yields one line.

    Mutation: writing the store name unchecked, or a name check that
        lets a trailing newline through, so one call forges a second
        line.
    Oracle: crafted names, and the `?` placeholder for a name
        `valid_store_name` rejects.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['--store', store, 'status'])
    rows = _call_lines(data_dir)
    assert [(r['verb'], r['store']) for r in rows] == [('status', '?')]


@pytest.mark.parametrize(('mode', 'expected_exit'), [
    ('exit3', 3),
    ('usage', 2),
    ('crash', 1),
    ])
def test_line_records_the_calls_exit_code(
        mm_runner, exit_probe, mode, expected_exit):
    """Verify the exit field matches how the command body ended.

    Mutation: dropping the `Exit` or `ClickException` arm, or a
        starting exit code of 0 in place of 1.
    Oracle: the CliRunner exit code, and the hand values 3, 2 and 1.
    """
    _, data_dir = mm_runner
    result = invoke(mm_runner, ['exit-probe', '--mode', mode])
    assert result.exit_code == expected_exit
    assert [r['exit'] for r in _call_lines(data_dir)] == [str(expected_exit)]


def test_ms_field_times_the_command_body(mm_runner, exit_probe):
    """Verify the ms field counts milliseconds of the body's run time.

    Mutation: storing whole seconds (the `* 1000` dropped), which logs 0.
    Oracle: a probe body that sleeps 50 ms.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['exit-probe', '--mode', 'sleep'])
    assert int(_call_lines(data_dir)[0]['ms']) >= 50


def test_timestamp_is_utc_under_a_non_utc_zone(mm_runner, monkeypatch):
    """Verify the start time is UTC whatever the local zone is.

    Mutation: a naive `datetime.now()`, which writes local time with a
        `Z` suffix.
    Oracle: `datetime.now(timezone.utc)` read around the call, with
        `TZ=Asia/Tokyo` putting local time nine hours off UTC.
    """
    _, data_dir = mm_runner
    monkeypatch.setenv('TZ', 'Asia/Tokyo')
    time.tzset()
    try:
        before = datetime.now(timezone.utc).replace(microsecond=0)
        invoke(mm_runner, ['status'])
        after = datetime.now(timezone.utc)
    finally:
        monkeypatch.undo()
        time.tzset()
    logged = datetime.strptime(
        _call_lines(data_dir)[0]['ts'], '%Y-%m-%dT%H:%M:%SZ').replace(
        tzinfo=timezone.utc)
    assert before <= logged <= after


def test_line_carries_no_argument_text(mm_runner):
    """Verify a `remember` line holds the verb and never the content.

    Mutation: logging argv or the content argument into the line.
    Oracle: a marker word present only in the remembered text.
    """
    _, data_dir = mm_runner
    invoke(mm_runner, ['remember', 'Zanzibarquux is the retry host name'])
    log_text = (Path(data_dir) / 'logs' / 'calls.log').read_text()
    assert [r['verb'] for r in _call_lines(data_dir)] == ['remember']
    assert 'Zanzibarquux' not in log_text


def test_non_agent_verb_writes_no_line(mm_runner):
    """Verify a command the agent never runs leaves the log untouched.

    Mutation: wrapping every command at the root group instead of only
        the agent-callable ones.
    Oracle: `log list`, which carries no agent-callable marker.
    """
    _, data_dir = mm_runner
    result = invoke(mm_runner, ['log', 'list'])
    assert result.exit_code == 0
    assert _call_lines(data_dir) == []


def test_failed_append_leaves_the_call_outcome_unchanged(mm_runner, caplog):
    """Verify an unwritable log dir warns and the call still succeeds.

    Mutation: letting the append's OSError escape, which fails the call.
    Oracle: `logs` pre-created as a plain file, so the mkdir must fail.
    """
    _, data_dir = mm_runner
    (Path(data_dir) / 'logs').write_text('')
    result = invoke(mm_runner, ['status'])
    assert result.exit_code == 0
    assert 'call log append failed' in caplog.text


def _write_call_log(data_dir: str, lines: list[str]) -> None:
    """Write `lines` as calls.log under `data_dir`.
    """
    log_dir = Path(data_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / 'calls.log').write_text(''.join(f'{x}\n' for x in lines))


def test_log_calls_counts_by_date_and_verb(mm_runner):
    """Verify `log calls` counts lines per UTC date and verb, newest first.

    Mutation: grouping by verb alone, grouping on the full timestamp,
        sorting dates oldest first, or ranking a date's verbs by name
        before count.
    Oracle: five hand-written lines over two dates, counted by hand.
        On 2026-09-29 the busier verb sorts last by name.
    """
    _, data_dir = mm_runner
    _write_call_log(data_dir, [
        '2026-09-28T09:00:00Z|recall|default|0|120',
        '2026-09-29T10:00:00Z|recall|default|0|110',
        '2026-09-29T10:05:00Z|remember|work|0|40',
        '2026-09-29T11:00:00Z|remember|work|1|90',
        '2026-09-28T12:00:00Z|insights show|default|0|30',
        ])
    result = invoke(mm_runner, ['log', 'calls'])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        'counts': [
            {'date': '2026-09-29', 'verb': 'remember', 'calls': 2},
            {'date': '2026-09-29', 'verb': 'recall', 'calls': 1},
            {'date': '2026-09-28', 'verb': 'insights show', 'calls': 1},
            {'date': '2026-09-28', 'verb': 'recall', 'calls': 1},
            ],
        'meta': {'total': 5, 'malformed': 0},
        }


def test_log_calls_skips_and_counts_damaged_lines(mm_runner):
    """Verify a damaged line is counted as malformed, never as a call.

    Mutation: splitting on `|` with no format check, which crashes on a
        line with no `|` and counts a cut-off `...|rec` as verb `rec`.
    Oracle: one good line among six hand-damaged ones: cut-off writes
        glued to the next line, one cut inside the ms field, and a byte
        that is not UTF-8.
    """
    _, data_dir = mm_runner
    log_dir = Path(data_dir) / 'logs'
    log_dir.mkdir(parents=True)
    (log_dir / 'calls.log').write_bytes(
        b'2026-09-29T10:00:00Z|recall|default|0|110\n'
        b'\n'
        b'garbage\n'
        b'2026-09-29T10:01:00Z|rec'
        b'2026-09-29T10:02:00Z|status|default|0|5\n'
        b'2026-09-29T10:03:00Z|st\xffatus|default|0|5\n'
        b'\x00\x00\x00\n'
        b'2026-09-29T10:04:00Z|recall|default|0|1'
        b'2026-09-29T10:05:00Z|recall|default|0|7\n')
    result = invoke(mm_runner, ['log', 'calls'])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        'counts': [{'date': '2026-09-29', 'verb': 'recall', 'calls': 1}],
        'meta': {'total': 1, 'malformed': 6},
        }


def test_log_calls_since_drops_older_lines(mm_runner):
    """Verify `log calls --since` keeps only lines at or after the cutoff.

    Mutation: ignoring `--since`, or comparing the cutoff the wrong way.
    Oracle: one line from 2000 and one from 2999 against `--since 1d`.
    """
    _, data_dir = mm_runner
    _write_call_log(data_dir, [
        '2000-01-01T00:00:00Z|recall|default|0|1',
        '2999-01-01T00:00:00Z|status|default|0|1',
        ])
    result = invoke(mm_runner, ['log', 'calls', '--since', '1d'])
    assert json.loads(result.output)['counts'] == [
        {'date': '2999-01-01', 'verb': 'status', 'calls': 1}]


def test_log_calls_without_a_log_reports_no_calls(mm_runner):
    """Verify `log calls` on a fresh data dir reports zero, not an error.

    Mutation: opening calls.log unguarded, so a fresh install exits 1.
    Oracle: a data dir where no agent verb has run yet.
    """
    result = invoke(mm_runner, ['log', 'calls'])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        'counts': [], 'meta': {'total': 0, 'malformed': 0}}
