"""Retry behaviour of MemmanLLMClient.complete on empty LLM responses.

A transient empty body from a flaky local endpoint (Ollama, llama.cpp)
must be retried inside `complete()` rather than surfaced, because every
pipeline call site swallows the exception and permanently degrades the
row (passthrough fact, missing enrichment, no causal edges).
"""

import httpx
import pytest
from memman import _http
from memman._http import MAX_RETRIES
from memman.llm import client as llm_client_mod
from memman.llm import usage
from memman.llm.client import MemmanLLMClient

VALID = {'choices': [{'message': {'content': 'ok'}}]}


def _install_fake_post(monkeypatch, responses):
    calls = []

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        payload = responses[min(len(calls) - 1, len(responses) - 1)]
        return httpx.Response(
            200, request=httpx.Request('POST', url), json=payload)

    monkeypatch.setitem(
        _http._SESSIONS, llm_client_mod.__name__,
        type('FakeClient', (), {'post': staticmethod(_fake_post)})())
    return calls


def _client():
    return MemmanLLMClient(
        endpoint='http://localhost:11434/v1', api_key='', model='test-model')


@pytest.mark.no_mock_llm
def test_empty_choices_retries_then_succeeds(monkeypatch):
    """Two empty-choices bodies then a valid one yields the content.

    Mutation: raising on the first empty `choices` instead of retrying
        (deleting the retry `continue`).
    Oracle: spy transport counts exactly 3 POSTs and the valid third
        body's content is returned.
    """
    calls = _install_fake_post(
        monkeypatch, [{'choices': []}, {'choices': []}, VALID])
    monkeypatch.setattr(llm_client_mod.time, 'sleep', lambda s: None)
    assert _client().complete('sys', 'user', stage=usage.STAGE_PROBE) == 'ok'
    assert len(calls) == 3


@pytest.mark.no_mock_llm
def test_empty_choices_raises_after_last_attempt(monkeypatch):
    """A permanently empty endpoint raises after exactly MAX_RETRIES POSTs.

    Mutation: flipping the last-attempt guard (retrying forever, or
        raising one attempt early).
    Oracle: spy transport counts exactly MAX_RETRIES POSTs before the
        RuntimeError surfaces.
    """
    calls = _install_fake_post(monkeypatch, [{'choices': []}])
    monkeypatch.setattr(llm_client_mod.time, 'sleep', lambda s: None)
    with pytest.raises(RuntimeError):
        _client().complete('sys', 'user', stage=usage.STAGE_PROBE)
    assert len(calls) == MAX_RETRIES


@pytest.mark.no_mock_llm
@pytest.mark.parametrize('empty_content', ['', '   \n\t', None])
def test_empty_content_string_retries(monkeypatch, empty_content):
    """Empty, whitespace-only, and null content all retry like no choices.

    Mutation: treating only missing `choices` as empty — an empty or
        null `content` string would then be returned to the caller
        (None with no exception at all, the latent bug).
    Oracle: spy transport counts 2 POSTs and the valid second body's
        content is returned.
    """
    calls = _install_fake_post(
        monkeypatch,
        [{'choices': [{'message': {'content': empty_content}}]}, VALID])
    monkeypatch.setattr(llm_client_mod.time, 'sleep', lambda s: None)
    assert _client().complete('sys', 'user', stage=usage.STAGE_PROBE) == 'ok'
    assert len(calls) == 2


@pytest.mark.no_mock_llm
def test_empty_retry_does_not_sleep_backoff(monkeypatch):
    """Empty-body retries never wait a RETRY_BACKOFF-scale delay.

    An empty body is not rate limiting; `(1.0, 2.0, 4.0)` of sleep is
    charged against the 60 s drain timeout whose maintenance pass needs
    ~30 s.

    Mutation: reusing `RETRY_BACKOFF` on the empty-body path.
    Oracle: a sleep spy asserts no recorded delay reaches 1.0 (the
        smallest backoff step).
    """
    slept = []
    calls = _install_fake_post(
        monkeypatch, [{'choices': []}, {'choices': []}, VALID])
    monkeypatch.setattr(llm_client_mod.time, 'sleep', slept.append)
    assert _client().complete('sys', 'user', stage=usage.STAGE_PROBE) == 'ok'
    assert len(calls) == 3
    assert all(s < 1.0 for s in slept)


@pytest.mark.no_mock_llm
def test_structurally_malformed_response_is_not_retried(monkeypatch):
    """A response missing `message.content` structure raises immediately.

    A structurally malformed response does not self-heal, so retrying
    it only burns attempts.

    Mutation: widening the empty-body retry to the `KeyError`/
        `TypeError` branch.
    Oracle: spy transport counts exactly 1 POST before the RuntimeError
        surfaces.
    """
    calls = _install_fake_post(
        monkeypatch, [{'choices': [{'no_message': {}}]}])
    monkeypatch.setattr(llm_client_mod.time, 'sleep', lambda s: None)
    with pytest.raises(RuntimeError):
        _client().complete('sys', 'user', stage=usage.STAGE_PROBE)
    assert len(calls) == 1
