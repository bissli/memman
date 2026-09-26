"""Tests for the OpenRouter candidate lister the operator picks from."""

import datetime
import os

import pytest
from memman import config
from memman.cli import cli
from memman.llm import openrouter_models as om

PINNED_VENDORS = frozenset({'amazon-bedrock', 'azure', 'google-vertex'})
CURRENT_MODEL = 'qwen/qwen3-235b-a22b-2507'


def _model(model_id, released=datetime.date(2025, 7, 21), reasoning=None):
    """Build a `/models` entry released at noon UTC on `released`.
    """
    noon = datetime.datetime.combine(
        released, datetime.time(12), datetime.UTC)
    entry = {'id': model_id, 'created': int(noon.timestamp())}
    if reasoning is not None:
        entry['reasoning'] = reasoning
    return entry


def _zdr(model_id, tag='google-vertex/us-south1',
         prompt='0.00000022', completion='0.00000088'):
    """Build a `/endpoints/zdr` row; prices are dollars per token.
    """
    return {
        'model_id': model_id,
        'tag': tag,
        'pricing': {'prompt': prompt, 'completion': completion},
        }


def _list(models, zdr_rows, **overrides):
    """Call the lister with the qwen family, the seeded ceilings and the pin.
    """
    kwargs = {
        'family': 'qwen/',
        'max_input_per_m': 0.25,
        'max_output_per_m': 1.00,
        'vendors': PINNED_VENDORS,
        }
    kwargs.update(overrides)
    return om.list_candidates(models, zdr_rows, **kwargs)


def _ids(candidates):
    """Model ids of `candidates`, in list order.
    """
    return [c.model_id for c in candidates]


def test_lists_only_the_family():
    """A model outside the configured family is never a candidate.

    Mutation: the family filter dropped, so an anthropic model priced
    inside the qwen ceilings is offered as a qwen replacement.
    Oracle: two models on the same vendor at the same price, one per
    family.
    """
    models = [_model(CURRENT_MODEL), _model('anthropic/claude-haiku-4.5')]
    zdr_rows = [_zdr(CURRENT_MODEL), _zdr('anthropic/claude-haiku-4.5')]
    assert _ids(_list(models, zdr_rows)) == [CURRENT_MODEL]


def test_drops_vision_coder_free_and_batch_variants():
    """Only general-purpose models are listed.

    Mutation: one of the `-vl-`, `coder`, `:free` or `:batch` rules
    dropped, so a specialist or rate-limited variant reaches the list.
    Oracle: every excluded id is a real catalog id, priced and hosted
    the same as the one general model.
    """
    excluded = [
        'qwen/qwen3-vl-235b-a22b-instruct',
        'qwen/qwen3-coder',
        'qwen/qwen3.8-27b:free',
        'qwen/qwen3-235b-a22b-2507:batch',
        ]
    models = [_model(i) for i in [*excluded, CURRENT_MODEL]]
    zdr_rows = [_zdr(i) for i in [*excluded, CURRENT_MODEL]]
    assert _ids(_list(models, zdr_rows)) == [CURRENT_MODEL]


def test_lists_only_models_present_in_both_catalogs():
    """A model needs a ZDR row and a `/models` entry to be listed.

    Mutation: the lister walks `/models` without requiring a ZDR row,
    or walks the ZDR rows without the `/models` join, which admits
    embedding and ASR models the chat endpoint cannot serve.
    Oracle: one model with no ZDR row, one ZDR row with no `/models`
    entry, and the current model with both.
    """
    models = [_model('qwen/qwen3.5-397b-a17b'), _model(CURRENT_MODEL)]
    zdr_rows = [_zdr('qwen/qwen3-embedding-8b'), _zdr(CURRENT_MODEL)]
    assert _ids(_list(models, zdr_rows)) == [CURRENT_MODEL]


def test_a_model_hosted_only_off_the_vendor_list_is_not_listed():
    """A ZDR endpoint counts only when its vendor is on the list.

    Mutation: the vendor filter dropped, or matched against the whole
    tag, so a model only a non-listed host serves is offered.
    Oracle: qwen3.6-35b-a3b's real ZDR hosts, none of them listed.
    """
    models = [_model('qwen/qwen3.6-35b-a3b'), _model(CURRENT_MODEL)]
    zdr_rows = [
        _zdr('qwen/qwen3.6-35b-a3b', tag='deepinfra/fp8'),
        _zdr('qwen/qwen3.6-35b-a3b', tag='siliconflow/fp8'),
        _zdr(CURRENT_MODEL),
        ]
    assert _ids(_list(models, zdr_rows)) == [CURRENT_MODEL]


def test_shown_price_and_vendor_come_from_a_listed_vendor():
    """A cheaper row on a non-listed host never sets the shown price.

    Mutation: the price is taken from the model's cheapest ZDR row
    whatever its vendor, so the operator sees a price the pin never
    routes to.
    Oracle: the current model's real DeepInfra and Vertex ZDR prices,
    0.09/0.55 against 0.25/1.00 per million tokens.
    """
    zdr_rows = [
        _zdr(CURRENT_MODEL, tag='deepinfra/fp8',
             prompt='0.00000009', completion='0.00000055'),
        _zdr(CURRENT_MODEL, prompt='0.00000025', completion='0.000001'),
        ]
    [row] = _list([_model(CURRENT_MODEL)], zdr_rows)
    assert row.vendor == 'google-vertex'
    assert (row.input_per_m, row.output_per_m) == pytest.approx((0.25, 1.00))


def test_a_model_priced_at_the_ceiling_is_listed():
    """Both ceilings are inclusive.

    Mutation: a strict `<` in either ceiling test, which drops the
    current model from a list whose ceilings its own price seeded.
    Oracle: the current model's dearest pinned price, 0.25/1.00 per
    million tokens, equal to the seeded ceilings.
    """
    zdr_rows = [_zdr(CURRENT_MODEL, prompt='0.00000025', completion='0.000001')]
    assert _ids(_list([_model(CURRENT_MODEL)], zdr_rows)) == [CURRENT_MODEL]


def test_a_model_over_either_ceiling_is_not_listed():
    """A model must sit inside the input AND the output ceiling.

    Mutation: only one ceiling checked, or the two prices summed
    against one limit.
    Oracle: qwen3-next-80b-a3b's real 0.15/1.20 fails on output alone;
    a model at 0.26/0.50 fails on input alone.
    """
    models = [
        _model('qwen/qwen3-next-80b-a3b-instruct'),
        _model('qwen/qwen3-over-input'),
        _model(CURRENT_MODEL),
        ]
    zdr_rows = [
        _zdr('qwen/qwen3-next-80b-a3b-instruct', tag='google-vertex/global',
             prompt='0.00000015', completion='0.0000012'),
        _zdr('qwen/qwen3-over-input',
             prompt='0.00000026', completion='0.0000005'),
        _zdr(CURRENT_MODEL),
        ]
    assert _ids(_list(models, zdr_rows)) == [CURRENT_MODEL]


def test_shows_at_most_three_dearest_first():
    """Five qualifying models yield the three dearest, dearest first.

    Mutation: the cap dropped, or the sort ascending, which offers the
    cheapest models and hides the one the ceiling was set for.
    Oracle: hand-ordered prices, input and output rising together.
    """
    prices = [
        ('qwen/m1', '0.00000005', '0.0000002'),
        ('qwen/m2', '0.0000001', '0.0000004'),
        ('qwen/m3', '0.00000015', '0.0000006'),
        ('qwen/m4', '0.0000002', '0.0000008'),
        ('qwen/m5', '0.00000025', '0.000001'),
        ]
    models = [_model(i) for i, _, _ in prices]
    zdr_rows = [_zdr(i, prompt=p, completion=c) for i, p, c in prices]
    assert _ids(_list(models, zdr_rows)) == ['qwen/m5', 'qwen/m4', 'qwen/m3']


def test_newest_breaks_a_price_tie():
    """Equal prices order by release date, newest first.

    Mutation: ties broken by id in either direction, or oldest first.
    Oracle: three equal-price models whose id order matches none of
    the wrong orders: b newest, a middle, c oldest.
    """
    models = [
        _model('qwen/a', released=datetime.date(2026, 3, 1)),
        _model('qwen/b', released=datetime.date(2026, 9, 1)),
        _model('qwen/c', released=datetime.date(2025, 7, 1)),
        ]
    zdr_rows = [_zdr('qwen/a'), _zdr('qwen/b'), _zdr('qwen/c')]
    assert _ids(_list(models, zdr_rows)) == ['qwen/b', 'qwen/a', 'qwen/c']


def test_a_row_carries_price_vendor_release_date_and_thinking():
    """Each row shows per-million prices, vendor slug, date and thinking.

    Mutation: prices left per token, the vendor shown as the whole tag
    with its region, or the date taken in local time rather than UTC.
    Oracle: the current model's real Vertex row and release date.
    """
    [row] = _list([_model(CURRENT_MODEL)], [_zdr(CURRENT_MODEL)])
    assert row.model_id == CURRENT_MODEL
    assert row.vendor == 'google-vertex'
    assert (row.input_per_m, row.output_per_m) == pytest.approx((0.22, 0.88))
    assert row.released == datetime.date(2025, 7, 21)
    assert row.thinks_by_default is False


@pytest.mark.parametrize(('reasoning', 'thinks'), [
    ({'mandatory': True}, True),
    ({'mandatory': False, 'default_enabled': True}, True),
    ({'mandatory': False, 'default_enabled': False}, False),
    ({'mandatory': False}, False),
    (None, False),
    ])
def test_thinks_by_default_reads_the_reasoning_flags(reasoning, thinks):
    """Thinking is on by default when reasoning is mandatory or enabled.

    Mutation: the mere presence of a `reasoning` object read as
    thinking, which marks claude-haiku-4.5 as a thinking model.
    Oracle: the real flags of qwen3-235b-a22b-thinking-2507
    (mandatory), qwen3.6-35b-a3b (default_enabled) and
    claude-haiku-4.5 (neither).
    """
    models = [_model(CURRENT_MODEL, reasoning=reasoning)]
    [row] = _list(models, [_zdr(CURRENT_MODEL)])
    assert row.thinks_by_default is thinks


def test_fetch_reads_only_the_two_public_catalogs(monkeypatch):
    """Fetching makes two keyless GETs and no chat call.

    Mutation: a live probe added to rank or verify candidates, or an
    Authorization header sent to the public endpoints.
    Oracle: the URLs and headers a stubbed HTTP client records, and a
    spy on the chat call.
    """
    endpoint = 'https://openrouter.ai/api/v1'
    payloads = {
        f'{endpoint}/endpoints/zdr': {'data': [_zdr(CURRENT_MODEL)]},
        f'{endpoint}/models': {'data': [_model(CURRENT_MODEL)]},
        }
    requests = []
    chat_calls = []

    class _Resp:
        status_code = 200

        def __init__(self, url):
            self.url = url

        def raise_for_status(self):
            return None

        def json(self):
            return payloads[self.url]

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, **kwargs):
            requests.append((url, kwargs.get('headers') or {}))
            return _Resp(url)

    monkeypatch.setattr('memman.llm.openrouter_models.httpx.Client', _Client)
    monkeypatch.setattr(
        'memman.llm.client.MemmanLLMClient.complete',
        lambda *a, **k: chat_calls.append(a))
    candidates = om.fetch_candidates(
        endpoint, family='qwen/', max_input_per_m=0.25,
        max_output_per_m=1.00, vendors=PINNED_VENDORS)
    assert sorted(url for url, _ in requests) == sorted(payloads)
    assert all('Authorization' not in headers for _, headers in requests)
    assert chat_calls == []
    assert _ids(candidates) == [CURRENT_MODEL]


def _two_candidates():
    """A dearer, newer candidate, then a cheaper thinking one.
    """
    return [
        om.Candidate(
            model_id='qwen/first', vendor='google-vertex',
            input_per_m=0.25, output_per_m=1.00,
            released=datetime.date(2026, 9, 1), thinks_by_default=False),
        om.Candidate(
            model_id='qwen/second', vendor='amazon-bedrock',
            input_per_m=0.20, output_per_m=0.80,
            released=datetime.date(2026, 3, 1), thinks_by_default=True),
        ]


def _env_model(data_dir):
    """MEMMAN_LLM_MODEL as the env file under `data_dir` holds it.
    """
    return config.parse_env_file(config.env_file_path(data_dir)).get(
        config.LLM_MODEL)


def test_config_models_writes_the_pick_in_a_tty(monkeypatch):
    """In a TTY, `config models` writes the picked candidate.

    Mutation: the pick index off by one, or the choice printed and
    never written to the env file.
    Oracle: two stubbed candidates and a prompt answering 2.
    """
    data_dir = os.environ[config.DATA_DIR]
    monkeypatch.setattr(
        'memman.llm.openrouter_models.fetch_candidates',
        lambda endpoint, **kwargs: _two_candidates())
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)
    monkeypatch.setattr('memman.cli.click.prompt', lambda *a, **k: 2)
    cli.main(
        ['--data-dir', data_dir, 'config', 'models'], standalone_mode=False)
    assert _env_model(data_dir) == 'qwen/second'


def test_config_models_only_lists_outside_a_tty(monkeypatch, capsys):
    """Outside a TTY, `config models` prints every row and writes nothing.

    Mutation: the command prompts or writes with no TTY, which hangs a
    script or rewrites the model unasked; or a row drops its vendor or
    release date.
    Oracle: two stubbed candidates, a prompt that fails the test if
    called, and the seeded model read back unchanged.
    """
    data_dir = os.environ[config.DATA_DIR]

    def _no_prompt(*a, **k):
        raise AssertionError('prompted outside a TTY')

    monkeypatch.setattr(
        'memman.llm.openrouter_models.fetch_candidates',
        lambda endpoint, **kwargs: _two_candidates())
    monkeypatch.setattr('sys.stdin.isatty', lambda: False)
    monkeypatch.setattr('memman.cli.click.prompt', _no_prompt)
    cli.main(
        ['--data-dir', data_dir, 'config', 'models'], standalone_mode=False)
    printed = capsys.readouterr().out
    for piece in ('qwen/first', 'google-vertex', '2026-09-01',
                  'qwen/second', 'amazon-bedrock', '2026-03-01'):
        assert piece in printed
    assert _env_model(data_dir) == config.INSTALL_DEFAULTS[config.LLM_MODEL]
