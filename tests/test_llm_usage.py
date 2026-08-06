"""Per-stage LLM token accounting (F1).

Drives the real `MemmanLLMClient.complete` through a fake transport
(same pattern as test_llm_client_retry.py) and the `usage` ledger
directly, so attribution, retry accumulation, and the lock are all
exercised where they live.
"""

import sys
import threading

import httpx
import pytest
from memman import _http
from memman.llm import client as llm_client_mod
from memman.llm import usage
from memman.llm.client import MemmanLLMClient


def _install_fake_post(monkeypatch, responses):
    calls = []

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        spec = responses[min(len(calls) - 1, len(responses) - 1)]
        # A dict is an HTTP-200 JSON body; a (status, body) tuple
        # picks the status, with a str body sent as raw text.
        if isinstance(spec, tuple):
            status, body = spec
            if isinstance(body, str):
                return httpx.Response(
                    status, request=httpx.Request('POST', url),
                    text=body)
            return httpx.Response(
                status, request=httpx.Request('POST', url), json=body)
        return httpx.Response(
            200, request=httpx.Request('POST', url), json=spec)

    monkeypatch.setitem(
        _http._SESSIONS, llm_client_mod.__name__,
        type('FakeClient', (), {'post': staticmethod(_fake_post)})())
    return calls


def _client():
    return MemmanLLMClient(
        endpoint='http://localhost:11434/v1', api_key='',
        model='test-model')


def _valid(content='ok', prompt=7, completion=3):
    return {
        'choices': [{'message': {'content': content}}],
        'usage': {'prompt_tokens': prompt, 'completion_tokens': completion,
                  'total_tokens': prompt + completion},
        }


@pytest.mark.no_mock_llm
def test_usage_attributed_to_originating_stage(monkeypatch):
    """Every billed attempt lands in its caller's stage bucket.

    An empty-retrying call bills each attempt; a success-only
    recorder books zero for the empties, and a stage mixup charges
    enrichment's spend to causal.

    Mutation: swapping two stage labels, or reading `usage` only at
        the success site (dropping retry accumulation).
    Oracle: an [empty-with-usage, valid] sequence on 'enrichment'
        records 2 calls / 15 prompt tokens there; a single valid call
        on 'causal' records 1 call / 7 -- exact per-stage sums.
    """
    monkeypatch.setattr(llm_client_mod.time, 'sleep', lambda s: None)
    before = usage.snapshot()
    empty = {
        'choices': [],
        'usage': {'prompt_tokens': 8, 'completion_tokens': 0,
                  'total_tokens': 8},
        }
    _install_fake_post(monkeypatch, [empty, _valid()])
    assert _client().complete(
        'sys', 'user', stage=usage.STAGE_ENRICHMENT) == 'ok'
    _install_fake_post(monkeypatch, [_valid()])
    assert _client().complete(
        'sys', 'user', stage=usage.STAGE_CAUSAL) == 'ok'
    d = usage.delta(before, usage.snapshot())
    assert d[usage.STAGE_ENRICHMENT]['calls'] == 2
    assert d[usage.STAGE_ENRICHMENT]['prompt_tokens'] == 15
    assert d[usage.STAGE_ENRICHMENT]['completion_tokens'] == 3
    assert d[usage.STAGE_CAUSAL]['calls'] == 1
    assert d[usage.STAGE_CAUSAL]['prompt_tokens'] == 7


@pytest.mark.no_mock_llm
def test_exhausted_retries_still_charge_every_attempt(monkeypatch):
    """All-empty attempts are charged even though `complete` raises.

    Mutation: recording usage only on the success return path -- a
        raising call books zero despite MAX_RETRIES billed attempts.
    Oracle: exactly `MAX_RETRIES` calls and the summed prompt tokens
        of every empty body appear in the ledger after the raise.
    """
    from memman._http import MAX_RETRIES
    monkeypatch.setattr(llm_client_mod.time, 'sleep', lambda s: None)
    before = usage.snapshot()
    empty = {
        'choices': [],
        'usage': {'prompt_tokens': 5, 'completion_tokens': 0,
                  'total_tokens': 5},
        }
    _install_fake_post(monkeypatch, [empty])
    with pytest.raises(RuntimeError):
        _client().complete('sys', 'user', stage=usage.STAGE_EXTRACTION)
    d = usage.delta(before, usage.snapshot())
    assert d[usage.STAGE_EXTRACTION]['calls'] == MAX_RETRIES
    assert d[usage.STAGE_EXTRACTION]['prompt_tokens'] == 5 * MAX_RETRIES


@pytest.mark.no_mock_llm
def test_http_errors_split_from_billed_calls(monkeypatch):
    """Unbilled non-2xx retries land in http_errors, not calls.

    A 429/5xx rejection bills nothing; booking it as a call lets a
    rate-limit storm inflate the drain's billed-call signal 3x.

    Mutation: recording error attempts into `calls` (the 0.19.0
        form), or dropping the error-path record entirely.
    Oracle: a [500, 500, valid] sequence books exactly calls == 1,
        http_errors == 2, missing_usage == 0, and only the valid
        attempt's 7 prompt tokens.
    """
    monkeypatch.setattr(llm_client_mod.time, 'sleep', lambda s: None)
    before = usage.snapshot()
    _install_fake_post(monkeypatch, [
        (500, {'error': 'boom'}), (500, {'error': 'boom'}), _valid()])
    assert _client().complete(
        'sys', 'user', stage=usage.STAGE_ENRICHMENT) == 'ok'
    d = usage.delta(before, usage.snapshot())
    assert d[usage.STAGE_ENRICHMENT]['calls'] == 1
    assert d[usage.STAGE_ENRICHMENT]['http_errors'] == 2
    assert d[usage.STAGE_ENRICHMENT].get('missing_usage', 0) == 0
    assert d[usage.STAGE_ENRICHMENT]['prompt_tokens'] == 7


@pytest.mark.no_mock_llm
def test_unparseable_200_body_is_booked_and_retried(monkeypatch):
    """A billed 200 with a non-JSON body is booked, then retried.

    A truncating proxy returns 200 with an HTML or partial body;
    the provider billed the completion either way.

    Mutation: parsing `resp.json()` before the accounting (the
        0.19.0 form) -- the JSONDecodeError skips the ledger and
        aborts the call with no retry.
    Oracle: [html junk, valid] returns 'ok' with calls == 2 and
        missing_usage == 1 in the stage bucket.
    """
    monkeypatch.setattr(llm_client_mod.time, 'sleep', lambda s: None)
    before = usage.snapshot()
    _install_fake_post(monkeypatch, [
        (200, '<html>bad gateway page</html>'), _valid()])
    assert _client().complete(
        'sys', 'user', stage=usage.STAGE_CAUSAL) == 'ok'
    d = usage.delta(before, usage.snapshot())
    assert d[usage.STAGE_CAUSAL]['calls'] == 2
    assert d[usage.STAGE_CAUSAL]['missing_usage'] == 1
    assert d[usage.STAGE_CAUSAL]['prompt_tokens'] == 7


@pytest.mark.no_mock_llm
def test_missing_usage_block_counts_call_not_tokens(monkeypatch):
    """A body without `usage` bumps `missing_usage`, not the tokens.

    Mutation: conflating "provider reported zero" with "reported
        nothing" (e.g. treating a missing block as all-zero usage
        without counting it).
    Oracle: a usage-less body yields calls=1, missing_usage=1,
        zero tokens; an explicit all-zero block yields calls=1,
        missing_usage=0.
    """
    before = usage.snapshot()
    _install_fake_post(
        monkeypatch, [{'choices': [{'message': {'content': 'ok'}}]}])
    _client().complete('sys', 'user', stage=usage.STAGE_PROBE)
    d = usage.delta(before, usage.snapshot())
    assert d[usage.STAGE_PROBE]['calls'] == 1
    assert d[usage.STAGE_PROBE]['missing_usage'] == 1
    assert d[usage.STAGE_PROBE]['total_tokens'] == 0

    mid = usage.snapshot()
    _install_fake_post(monkeypatch, [_valid(prompt=0, completion=0)])
    _client().complete('sys', 'user', stage=usage.STAGE_PROBE)
    d2 = usage.delta(mid, usage.snapshot())
    assert d2[usage.STAGE_PROBE]['calls'] == 1
    assert d2[usage.STAGE_PROBE].get('missing_usage', 0) == 0


def test_concurrent_stages_do_not_interleave_usage():
    """Two threads recording into one stage sum exactly.

    Enrichment and causal run concurrently on a two-worker executor,
    so `record` races are the production case, not a theoretical one.

    Mutation: removing the `Lock` around the ledger update -- the
        read-modify-write interleaves and updates are lost.
    Oracle: 2 threads x 20000 records sum to exactly 40000 calls and
        40000 * 7 prompt tokens.
    """
    n = 20000
    before = usage.snapshot()
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        def _hammer():
            for _ in range(n):
                usage.record(
                    usage.STAGE_RECONCILIATION, {'prompt_tokens': 7})

        threads = [threading.Thread(target=_hammer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old_interval)
    d = usage.delta(before, usage.snapshot())
    assert d[usage.STAGE_RECONCILIATION]['calls'] == 2 * n
    assert d[usage.STAGE_RECONCILIATION]['prompt_tokens'] == 2 * n * 7


def test_all_call_sites_use_closed_set_stages():
    """Every production `complete()` call names a closed-set stage.

    Mutation: a typo'd stage string at a call site creating a
        phantom bucket that never appears in any report.
    Oracle: `record` raises on an unknown stage, and an ast scan of
        src/memman finds a `stage=usage.STAGE_*` keyword on every
        `.complete(` call.
    """
    import ast
    from pathlib import Path

    import memman

    with pytest.raises(ValueError, match='unknown LLM stage'):
        usage.record('extractoin', None)

    src_root = Path(memman.__file__).parent
    sites = []
    for py in src_root.rglob('*.py'):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'complete'):
                continue
            sites.append((py.name, node))
    assert len(sites) >= 6, 'expected the six documented call sites'
    for name, node in sites:
        stage_kw = [k for k in node.keywords if k.arg == 'stage']
        assert stage_kw, f'{name}: complete() call missing stage='
        val = stage_kw[0].value
        assert isinstance(val, ast.Attribute), (
            f'{name}: stage must reference a usage.STAGE_* constant')
        assert val.attr.startswith('STAGE_'), (
            f'{name}: stage constant {val.attr!r} not a STAGE_* name')
        assert getattr(usage, val.attr) in usage.VALID_STAGES


@pytest.mark.no_auto_drain
def test_drain_json_carries_llm_usage_delta(mm_runner, monkeypatch):
    """The drain's JSON output carries the drain-level usage delta.

    The ledger tests above never exercise the drain wiring; a
    snapshot taken after the row loop (or a dropped key) would ship
    green while reporting an empty summary forever.

    Mutation: taking `drain_usage_snap` after the loop, or dropping
        the `llm_usage` key from `_json_out`.
    Oracle: a stub row-processor records one extraction attempt of
        11 prompt tokens; the drain JSON must carry exactly that
        per-stage delta.
    """
    import json as _json

    from memman.cli import cli

    def _stub_row(row, ctx, executor):
        usage.record(
            usage.STAGE_EXTRACTION,
            {'prompt_tokens': 11, 'completion_tokens': 2,
             'total_tokens': 13})

    monkeypatch.setattr('memman.cli._process_queue_row', _stub_row)
    r, data_dir = mm_runner
    res = r.invoke(cli, [
        '--data-dir', data_dir, 'remember', 'drain usage probe row'])
    assert res.exit_code == 0, res.output
    res = r.invoke(cli, [
        '--data-dir', data_dir, 'scheduler', 'drain',
        '--limit', '5', '--timeout', '10'])
    assert res.exit_code == 0, res.output
    data = _json.loads(res.output)
    assert data['processed'] == 1
    stage = data['llm_usage'][usage.STAGE_EXTRACTION]
    assert stage['calls'] == 1
    assert stage['prompt_tokens'] == 11


def test_expand_usage_summary_uses_the_drain_event_key(
        mm_runner, monkeypatch):
    """`recall --expand` emits llm_usage_summary under the `usage` key.

    The drain's emitter names the payload `usage`; a second emitter
    for the same event name under a different key is invisible to
    every consumer keyed on the shipped shape -- the exact
    observability hole the emission exists to close.

    Mutation: emitting the delta under `llm_usage=`, or dropping the
        emission from the recall command.
    Oracle: a trace.event spy captures exactly one llm_usage_summary
        whose `usage` payload carries the query_expansion stage.
    """
    from memman import trace
    from memman.cli import cli

    events = []
    monkeypatch.setattr(
        trace, 'event',
        lambda name, **kw: events.append((name, kw)))
    monkeypatch.setattr(
        'memman.cli._get_llm_client_or_fail', lambda role: object())

    def _fake_expand(client, q):
        usage.record(
            usage.STAGE_QUERY_EXPANSION, {'prompt_tokens': 5})
        return {'expanded_query': q + ' broadened'}

    monkeypatch.setattr(
        'memman.llm.extract.expand_query', _fake_expand)
    r, data_dir = mm_runner
    res = r.invoke(cli, [
        '--data-dir', data_dir, 'remember', 'expansion seed row'])
    assert res.exit_code == 0, res.output
    res = r.invoke(cli, [
        '--data-dir', data_dir, 'recall', 'expansion', 'seed',
        '--expand'])
    assert res.exit_code == 0, res.output
    # The auto-drain after `remember` emits its own summary; the
    # recall's is the one carrying the expansion stage under the
    # shipped `usage` key.
    summaries = [
        kw for name, kw in events if name == 'llm_usage_summary']
    expansion = [
        kw for kw in summaries
        if usage.STAGE_QUERY_EXPANSION in kw.get('usage', {})]
    assert len(expansion) == 1
