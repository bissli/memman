"""One request path each for the LLM, embeddings, and rerank.

All three read the shared `MEMMAN_ENDPOINT` and `MEMMAN_API_KEY`.
"""

import httpx
import pytest
from memman import config
from memman.embed import get_client
from memman.embed.fingerprint import Fingerprint, bound_embedder
from memman.embed.fingerprint import write_fingerprint
from memman.llm.client import get_llm_client
from memman.search.recall import run_recall
from tests.conftest import make_insight

ENDPOINT = 'https://openrouter.ai/api/v1'


@pytest.fixture
def shared_env(env_file):
    """Point all three paths at one OpenRouter endpoint and key.
    """
    env_file('MEMMAN_ENDPOINT', ENDPOINT)
    env_file('MEMMAN_API_KEY', 'sk-or-shared')
    env_file('MEMMAN_EMBED_MODEL', 'voyageai/voyage-4-lite')
    env_file('MEMMAN_RERANK_MODEL', 'voyageai/rerank-3-lite')


@pytest.fixture
def posts(monkeypatch):
    """Record every httpx POST and answer it with a canned body.

    Returns the list of `(url, headers, json)` tuples. An embeddings
    request gets one 4-dim vector per input, a chat request one fixed
    completion, and a rerank request its documents ranked last-first
    under `results`.
    """
    sent = []

    def fake_post(self, url, *, headers=None, json=None, timeout=None):
        sent.append((url, headers or {}, json or {}))
        if url.endswith('/embeddings'):
            body = {'data': [
                {'embedding': [0.5, 0.5, 0.5, 0.5]} for _ in json['input']]}
        elif url.endswith('/chat/completions'):
            body = {'choices': [{'message': {'content': '{"ok": true}'}}]}
        else:
            n = len(json['documents'])
            body = {'results': [
                {'index': i, 'relevance_score': 1.0 - (n - 1 - i) / n}
                for i in reversed(range(n))]}
        return httpx.Response(
            200, json=body, request=httpx.Request('POST', url))

    monkeypatch.setattr(httpx.Client, 'post', fake_post)
    return sent


@pytest.mark.no_mock_embed
def test_embed_posts_to_shared_endpoint_with_no_provider_block(
        shared_env, posts):
    """Verify embeddings go to the shared endpoint with no `provider` block.

    Mutation: the embed client reading a provider-specific endpoint or
        key, or sending a `provider` block whose pin can make
        OpenRouter refuse a voyageai model the account allows.
    Oracle: the captured request against the literal endpoint, key, and
        model the env file names.
    """
    vectors = get_client().embed_batch(['alpha', 'beta'])
    url, headers, body = posts[-1]
    assert len(vectors) == 2
    assert url == f'{ENDPOINT}/embeddings'
    assert headers['Authorization'] == 'Bearer sk-or-shared'
    assert body['model'] == 'voyageai/voyage-4-lite'
    assert 'provider' not in body


@pytest.mark.no_mock_rerank
def test_recall_rerank_posts_top_n_and_reads_results(
        shared_env, posts, tmp_backend):
    """Verify recall reranks through `/rerank` and applies `results`.

    Mutation: posting `top_k` or to the Voyage host, reading the
        ranking from `data` (OpenRouter answers under `results`, so
        recall silently keeps its baseline order), or sending a
        `provider` block.
    Oracle: a stub that ranks the last shortlisted document first; that
        document must lead the recall results.
    """
    for i in range(3):
        tmp_backend.nodes.insert(make_insight(
            id=f'rr-{i}', content=f'alpha shared topic row {i}'))
    resp = run_recall(
        tmp_backend, 'alpha shared topic', None, 5, rerank=True)
    url, headers, body = posts[-1]
    assert url == f'{ENDPOINT}/rerank'
    assert headers['Authorization'] == 'Bearer sk-or-shared'
    assert body['model'] == 'voyageai/rerank-3-lite'
    assert body['top_n'] == 3
    assert 'top_k' not in body
    assert 'provider' not in body
    assert resp['meta']['reranked'] is True
    assert resp['results'][0]['insight'].content == body['documents'][-1]


@pytest.mark.no_mock_llm
def test_llm_posts_to_shared_endpoint_with_no_provider_block(
        shared_env, posts):
    """Verify a completion goes to the shared pair with no `provider` block.

    Mutation: the LLM reading a separate endpoint or key, or sending a
        vendor pin that refuses a host the account allows.
    Oracle: the captured request against the literal endpoint and key
        the env file names.
    """
    get_llm_client().complete('sys', 'user', stage='enrichment')
    url, headers, body = posts[-1]
    assert url == f'{ENDPOINT}/chat/completions'
    assert headers['Authorization'] == 'Bearer sk-or-shared'
    assert 'provider' not in body


def test_fingerprint_names_model_and_dim_only():
    """Verify a fingerprint round-trips as `(model, dim)` alone.

    Mutation: the fingerprint still requiring a provider field, so a
        store written by the single embed path fails to open.
    Oracle: a hand-written JSON value with only the two keys.
    """
    stored = '{"dim": 1024, "model": "voyageai/voyage-4-lite"}'
    assert Fingerprint.from_json(stored).to_json() == stored


@pytest.mark.no_default_env
def test_install_seeds_shared_key_and_latest_models(monkeypatch, tmp_path):
    """Verify install seeds one key and the default models.

    Mutation: install still demanding a provider-specific key such as
        `MEMMAN_VOYAGE_API_KEY`, or persisting a removed provider or
        routing knob such as `MEMMAN_ZDR`.
    Oracle: the vendor-native `OPENROUTER_API_KEY` export and the
        model ids OpenRouter serves.
    """
    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-or-native')
    knobs = config.collect_install_knobs(str(tmp_path))
    assert knobs['MEMMAN_API_KEY'] == 'sk-or-native'
    assert knobs['MEMMAN_EMBED_MODEL'] == 'voyageai/voyage-4-lite'
    assert knobs['MEMMAN_RERANK_MODEL'] == 'voyageai/rerank-3-lite'
    assert not [
        key for key in knobs
        if 'VOYAGE' in key or 'OPENROUTER' in key or 'PROVIDER' in key
        or key in {'MEMMAN_ZDR', 'MEMMAN_DATA_COLLECTION'}]


@pytest.mark.no_mock_embed
def test_opening_a_fingerprinted_store_sends_no_embed(
        shared_env, posts, tmp_backend):
    """Verify binding a store's embedder reuses the stored dim, unbilled.

    Mutation: the registry probing the endpoint for a model whose dim
        the store's fingerprint already records, so every store open
        bills one embed.
    Oracle: an HTTP recorder that must see no request.
    """
    write_fingerprint(
        tmp_backend, Fingerprint(model='voyageai/voyage-4', dim=4))
    posts.clear()
    ec = bound_embedder(tmp_backend)
    assert ec.dim == 4
    assert posts == []
