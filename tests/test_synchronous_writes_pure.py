"""Audit: synchronous (non-queued) write commands must be LLM/embed-free.

`forget`, `graph link`, and `insights protect` mutate the store DB
synchronously. They are not queued because the queue exists to defer
LLM/embed/network work, and these commands are O(1) SQL with none of
that. This audit pins the contract: if a future change adds an LLM or
embed call to any of these paths, the corresponding test fails loudly.
"""

import json

import pytest
from click.testing import CliRunner
from memman.cli import cli
from memman.embed.fingerprint import active_fingerprint
from memman.embed.vector import serialize_vector
from memman.store.node import insert_insight, update_embedding
from tests.conftest import make_insight


@pytest.fixture
def runner_with_seed(tmp_path):
    """CliRunner over an isolated data dir with two seeded insights.

    Direct DB seeding (instead of going through `remember`) avoids
    invoking the LLM for setup, so the assertion that the test
    target makes no LLM calls is meaningful.
    """
    from memman.store.db import open_db, store_dir, write_active

    data_dir = str(tmp_path)
    name = 'default'
    write_active(data_dir, name)
    sdir = store_dir(data_dir, name)
    db = open_db(sdir)
    fp = active_fingerprint()
    from memman.embed.fingerprint import write_fingerprint
    from memman.store.sqlite import SqliteBackend
    write_fingerprint(SqliteBackend(db), fp)

    a = make_insight(id='aud-a', content='alpha', importance=3)
    b = make_insight(id='aud-b', content='beta', importance=3)
    insert_insight(db, a)
    insert_insight(db, b)
    update_embedding(db, 'aud-a',
                     serialize_vector([0.1] * fp.dim), fp.model)
    db.close()

    return CliRunner(), data_dir


def _make_failing_complete(*_args, **_kwargs):
    """Stand-in for OpenRouterClient.complete that fails the test loudly."""
    raise AssertionError(
        'synchronous write must not invoke the LLM')


def _make_failing_embed(*_args, **_kwargs):
    """Stand-in for embed clients that fails the test loudly."""
    raise AssertionError(
        'synchronous write must not invoke the embed client')


def test_forget_makes_no_llm_or_embed_calls(runner_with_seed, monkeypatch):
    """`forget` is pure SQL: no LLM, no embed."""
    monkeypatch.setattr(
        'memman.llm.openrouter_client.OpenRouterClient.complete',
        _make_failing_complete)
    monkeypatch.setattr(
        'memman.embed.voyage.Client.embed', _make_failing_embed)

    r, data_dir = runner_with_seed
    out = r.invoke(cli, ['--data-dir', data_dir, 'forget', 'aud-a'])
    assert out.exit_code == 0, out.output


def test_graph_link_makes_no_llm_or_embed_calls(
        runner_with_seed, monkeypatch):
    """`graph link` is pure SQL."""
    monkeypatch.setattr(
        'memman.llm.openrouter_client.OpenRouterClient.complete',
        _make_failing_complete)
    monkeypatch.setattr(
        'memman.embed.voyage.Client.embed', _make_failing_embed)

    r, data_dir = runner_with_seed
    out = r.invoke(cli, ['--data-dir', data_dir,
                         'graph', 'link', 'aud-a', 'aud-b'])
    assert out.exit_code == 0, out.output


def test_insights_protect_makes_no_llm_or_embed_calls(
        runner_with_seed, monkeypatch):
    """`insights protect` is pure SQL."""
    monkeypatch.setattr(
        'memman.llm.openrouter_client.OpenRouterClient.complete',
        _make_failing_complete)
    monkeypatch.setattr(
        'memman.embed.voyage.Client.embed', _make_failing_embed)

    r, data_dir = runner_with_seed
    out = r.invoke(cli, ['--data-dir', data_dir,
                         'insights', 'protect', 'aud-a'])
    assert out.exit_code == 0, out.output


def test_remember_returns_queued_only(runner_with_seed, monkeypatch):
    """`remember` returns immediately with action=queued; no LLM on hot path.

    The autouse `_scheduler_started` fixture forces inline drain mode
    in this test file's parents. Override it here so the LLM mock
    actually fails if the hot path tries to enrich.
    """
    monkeypatch.setattr(
        'memman.llm.openrouter_client.OpenRouterClient.complete',
        _make_failing_complete)
    monkeypatch.setattr(
        'memman.embed.voyage.Client.embed', _make_failing_embed)

    r, data_dir = runner_with_seed
    out = r.invoke(
        cli, ['--data-dir', data_dir, 'remember', 'fresh content'])
    assert out.exit_code == 0, out.output
    payload = json.loads(out.output)
    assert payload['action'] == 'queued'
    assert 'queue_id' in payload
