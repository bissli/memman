"""Contracts the FTS5 keyword channel has to hold.

`RecallSession.keyword_counts` replaced a per-recall tokenization of
every active row. It fills `kw_score`'s numerator, so a count that
disagrees with `keyword.insight_tokens` moves `signals.keyword`,
`--min-score`, the rerank blend and `meta.sparse` together. These pin
the count, the query-language safety rule, and the index's sync with
the rows it indexes.
"""

import sqlite3
from pathlib import Path
from unittest import mock

import memman.store.db as db_module
import pytest
from memman.search.keyword import insight_tokens, keyword_search, tokenize
from memman.search.recall import intent_aware_recall
from memman.store.db import open_db, open_read_only
from memman.store.errors import BackendError
from memman.store.sqlite import SqliteBackend
from tests.conftest import make_insight

# Every one of these raises `OperationalError` when handed straight to
# `match`: `?` `-` `/` `.` `[` `)` are operators or syntax errors and
# a bare `NOT` is a keyword.
HOSTILE_QUERIES = [
    'what is _ALLOWED_BOOL_FLAGS?',
    'how does embed swap work - cutover',
    'store/sqlite.py adjacency',
    'NOT reindex_auto_edges',
    'meta.sparse contract',
    'kw_score [0,1] range',
    'FTS5 bm25() ranking',
    'edges.all() vs adjacency()',
    ]

CORPUS = [
    ('kw-a', 'the quick brown fox jumps over cutover',
     ['vulpes', 'reindex_auto_edges']),
    ('kw-b', 'a slow brown bear sleeps near the adjacency map',
     ['ursus', 'sqlite']),
    ('kw-c', 'quantum entanglement of photons and bm25 ranking',
     ['physics', '_ALLOWED_BOOL_FLAGS']),
    ('kw-d', 'meta sparse contract for the kw_score range',
     ['contract']),
]


def _seed(backend):
    """Insert the shared corpus and return its insights by id."""
    out = {}
    for iid, content, ents in CORPUS:
        ins = make_insight(id=iid, content=content, entities=ents)
        backend.nodes.insert(ins)
        out[iid] = ins
    return out


def _python_counts(insights, query_tokens):
    """Match counts the pre-index route produced, as the oracle."""
    return {
        iid: n for iid, n in (
            (iid, sum(1 for t in query_tokens if t in insight_tokens(ins)))
            for iid, ins in insights.items())
        if n
        }


def test_counts_match_python_tokenization(backend):
    """Verify the index reproduces `insight_tokens` overlap exactly.

    Mutation: indexing `content` but not `entities`, or swapping the
        tokenizer for a stemming one - both keep recall working while
        silently moving every `kw_score` that depends on an entity or
        on an inflected word.
    Oracle: the counts recomputed in Python from `insight_tokens`,
        which is the route the drain still uses.

    Notes
    -----
    - The last query is inflected on purpose and matches nothing
      under either route. It is what pins `unicode61`: a `porter`
      tokenizer would stem `jumping`/`rankings` onto the stored
      `jumps`/`ranking` and return hits Python never returns.
    """
    insights = _seed(backend)
    for query in ('brown fox', 'vulpes physics', 'adjacency sqlite map',
                  'bm25 ranking photons', 'contract sparse',
                  'jumping photons rankings'):
        query_tokens = tokenize(query)
        with backend.recall_session() as session:
            got = session.keyword_counts(query_tokens)
        assert got == _python_counts(insights, query_tokens), query


def test_punctuation_and_case_count_the_same_as_python(backend):
    """Verify messy real queries count correctly, not merely run.

    Mutation: collapsing the per-token probes into one `match`
        expression - space-joined it means AND, so a row matching
        part of the query scores 0 instead of a fraction; or dropping
        `lower()` from the Postgres split, which loses every
        mixed-case token.
    Oracle: the Python token-overlap counts for the same strings,
        which no FTS5 syntax and no SQL folding can reach.

    Notes
    -----
    - These strings are the ones that raise `OperationalError` when
      handed to `match` raw. That they cannot reach `match` is
      structural, not something this test defends: `keyword_counts`
      takes an already-tokenized `set[str]`. What it does defend is
      that the counts are right for them, `_ALLOWED_BOOL_FLAGS`
      included, the corpus's only mixed-case token.
    """
    insights = _seed(backend)
    for query in HOSTILE_QUERIES:
        query_tokens = tokenize(query)
        assert query_tokens, query
        with backend.recall_session() as session:
            got = session.keyword_counts(query_tokens)
        assert got == _python_counts(insights, query_tokens), query
    # Non-vacuous: at least one hostile query has to match something,
    # or the loop above would pass on an empty dict throughout.
    with backend.recall_session() as session:
        assert session.keyword_counts(tokenize(HOSTILE_QUERIES[3]))


def test_soft_deleted_rows_never_surface(backend):
    """Verify a deleted row is filtered out of the count.

    Mutation: dropping the join to `insights` or its `deleted_at is
        null` predicate. Every row stays indexed by design, so the
        read-side filter is the only thing keeping ghosts out.
    Oracle: the same probe before and after the delete.
    """
    _seed(backend)
    with backend.recall_session() as session:
        assert session.keyword_counts({'brown'}) == {'kw-a': 1, 'kw-b': 1}

    backend.nodes.soft_delete('kw-b')

    with backend.recall_session() as session:
        assert session.keyword_counts({'brown'}) == {'kw-a': 1}


def test_edits_reindex_and_unrelated_writes_do_not(backend):
    """Verify the index tracks content and entity edits, and only those.

    Mutation: omitting the paired `'delete'` in the update trigger,
        so the old terms linger and the row keeps matching a word it
        no longer holds.
    Oracle: probes for the removed and the added entity, by value.

    Notes
    -----
    - The access bump asserts the index survives an unrelated write.
      It does NOT pin the trigger's `of content, entities` scoping:
      a bare `after update` rewrites the row with identical values,
      so it costs writes and changes no output. Catching that needs
      a write-count spy, not this assertion.
    """
    _seed(backend)
    backend.nodes.update_entities('kw-a', ['canis'])

    with backend.recall_session() as session:
        assert session.keyword_counts({'vulpes'}) == {}
        assert session.keyword_counts({'canis'}) == {'kw-a': 1}

    backend.nodes.increment_access_count('kw-a')
    assert backend.integrity_check()['ok']
    with backend.recall_session() as session:
        assert session.keyword_counts({'canis'}) == {'kw-a': 1}


def test_recall_stops_tokenizing_every_row(backend, monkeypatch):
    """Verify no active row is tokenized to answer one recall.

    Mutation: reinstating the per-row `insight_tokens` scan - passing
        `None` for the counts, re-importing `insight_tokens` into
        `recall.py` for the scoring loop, or weakening
        `keyword_search`'s `counts is None` test to `not counts` so
        an empty index falls back to the scan.
    Oracle: a spy on `insight_tokens`, bound in BOTH modules that
        can hold a reference, against a store with rows to tokenize.

    Notes
    -----
    - The second query matches nothing, so `keyword_counts` returns
      an empty dict. That is the input that separates `is None` from
      a falsiness test, and it is why the assertion runs twice.
    """
    _seed(backend)
    import memman.search.keyword as keyword_module
    import memman.search.recall as recall_module

    calls: list[str] = []
    real = keyword_module.insight_tokens

    def spy(ins):
        calls.append(ins.id)
        return real(ins)

    monkeypatch.setattr(keyword_module, 'insight_tokens', spy)
    monkeypatch.setattr(
        recall_module, 'insight_tokens', spy, raising=False)

    resp = intent_aware_recall(backend, 'brown adjacency', None, 10)
    assert resp['results']
    assert calls == []

    resp = intent_aware_recall(
        backend, 'zzznomatch qqqnomatch', None, 10)
    assert resp['results'], 'recency anchors should still return rows'
    assert calls == []


def test_keyword_search_reads_the_counts_it_is_given(backend):
    """Verify the counts branch ranks by the supplied numbers alone.

    Mutation: defaulting a missing id to 1 instead of 0 in
        `counts.get`, or falling back to tokenizing the insight when
        an id is absent. Either silently promotes every unmatched row
        into the anchor pool, which changes the RRF seeds without
        touching `signals.keyword`, so no scoring assertion sees it.
    Oracle: hand-computed. Two query tokens, so a supplied count of 2
        is 1.0 and 1 is 0.5; the two ids absent from the dict must
        not appear at all.
    """
    insights = list(_seed(backend).values())
    counts = {'kw-a': 2, 'kw-b': 1}

    ranked = keyword_search(insights, 'brown fox', 10, counts)

    assert [(i.id, score) for i, score in ranked] == [
        ('kw-a', 1.0), ('kw-b', 0.5)]


def test_keyword_signal_is_the_overlap_fraction(backend):
    """Verify `signals.keyword` stays matched tokens over query tokens.

    Mutation: any rescale of the score - `bm25()`, an IDF weight, or
        a denominator of matched-rather-than-query tokens. All three
        keep ordering plausible while moving `--min-score`, the
        rerank blend and `meta.sparse` off their documented range.
    Oracle: hand-computed. 'brown fox jumps quantum' has four
        tokens; kw-a holds three of them and kw-b one.

    Notes
    -----
    - No row matches all four on purpose. With a query every row
        could match in full, `max(matched)` equals the query length
        and a denominator swapped for it is invisible.
    """
    _seed(backend)
    resp = intent_aware_recall(
        backend, 'brown fox jumps quantum', None, 10)
    signals = {r['insight'].id: r['signals']['keyword']
               for r in resp['results']}
    assert signals['kw-a'] == pytest.approx(3 / 4)
    assert signals['kw-b'] == pytest.approx(1 / 4)


def test_integrity_check_catches_a_drifted_index(tmp_path):
    """Verify drift is detected, which needs the rank-1 probe.

    Mutation: reporting `pragma integrity_check` alone, or FTS5's
        default `'integrity-check'`. Measured: both pass on an index
        whose terms no longer match the rows, so either one turns
        this check into a tautology.
    Oracle: the same store before and after the base row is edited
        behind the index's back.

    Notes
    -----
    - SQLite-only: the FTS5 index is the SQLite keyword channel, and
      Postgres counts against the rows themselves with nothing to
      drift.
    """
    store = tmp_path / 'drift'
    db = open_db(str(store))
    backend = SqliteBackend(db)
    _seed(backend)
    assert backend.integrity_check()['ok']
    db.close()

    raw = sqlite3.connect(Path(store) / 'memman.db')
    raw.execute('drop trigger insights_fts_update')
    raw.execute("update insights set content = 'rewritten behind the index'"
                " where id = 'kw-a'")
    raw.commit()
    raw.close()

    db = open_db(str(store))
    result = SqliteBackend(db).integrity_check()
    db.close()
    assert not result['ok']
    assert 'insights_fts' in result['detail']


def test_creating_the_index_populates_it(tmp_path):
    """Verify a store predating the index opens with it already filled.

    Mutation: creating the virtual table without the backfill. The
        triggers only carry rows written afterwards, so every
        existing row would be unfindable by keyword and recall would
        degrade silently rather than fail.
    Oracle: the probe result before the table is dropped, re-asserted
        after it is dropped and the store reopened.

    Notes
    -----
    - Dropping the table AND its triggers is how a store that
      predates the index reaches `_migrate`, and also how one
      restored from an older backup does. Dropping only the table
      leaves a state `_migrate` cannot produce and every write
      rejects, since the triggers would reference a missing table.
    """
    store = tmp_path / 'backfill'
    db = open_db(str(store))
    backend = SqliteBackend(db)
    _seed(backend)
    with backend.recall_session() as session:
        before = session.keyword_counts({'brown'})
    assert before
    db.close()

    raw = sqlite3.connect(Path(store) / 'memman.db')
    for name in ('insert', 'delete', 'update'):
        raw.execute(f'drop trigger insights_fts_{name}')
    raw.execute('drop table insights_fts')
    raw.commit()
    raw.close()

    db = open_db(str(store))
    backend = SqliteBackend(db)
    with backend.recall_session() as session:
        assert session.keyword_counts({'brown'}) == before
    assert backend.integrity_check()['ok']
    db.close()


def test_the_index_is_created_and_filled_atomically(tmp_path):
    """Verify a failed creation leaves nothing, so the next open retries.

    Mutation: creating the table from `_BASELINE_SCHEMA` instead of
        inside the explicit transaction. `executescript` commits, and
        the connection is autocommit, so the table would survive a
        failed backfill - and the absence check would then read as
        "already migrated" forever, leaving the keyword channel dead
        with no error. That is the snapshot's failure mode exactly.
    Oracle: `sqlite_master` after the failure (nothing left), then a
        clean reopen that indexes every row.

    Notes
    -----
    - SQLite-only: the transaction is SQLite DDL. Postgres has no
      index here to create.
    """
    store = tmp_path / 'atomic'
    db = open_db(str(store))
    backend = SqliteBackend(db)
    _seed(backend)
    db.close()

    raw = sqlite3.connect(Path(store) / 'memman.db')
    for name in ('insert', 'delete', 'update'):
        raw.execute(f'drop trigger insights_fts_{name}')
    raw.execute('drop table insights_fts')
    raw.commit()
    raw.close()

    broken = db_module._FTS_STATEMENTS + ('select not_a_real_column',)
    with mock.patch.object(db_module, '_FTS_STATEMENTS', broken):
        with pytest.raises(BackendError):
            open_db(str(store))

    raw = sqlite3.connect(Path(store) / 'memman.db')
    left = raw.execute(
        "select count(*) from sqlite_master"
        " where name like 'insights_fts%'").fetchone()[0]
    raw.close()
    assert left == 0, 'a failed creation must leave no half-built index'

    db = open_db(str(store))
    backend = SqliteBackend(db)
    with backend.recall_session() as session:
        assert session.keyword_counts({'brown'}) == {'kw-a': 1, 'kw-b': 1}
    assert backend.integrity_check()['ok']
    db.close()


def test_integrity_check_does_not_cry_drift_on_a_read_only_handle(
        tmp_path):
    """Verify a handle that cannot write reports "not checked", not drift.

    Mutation: catching `sqlite3.DatabaseError` around the rank-1
        probe without separating `OperationalError` first. The probe
        needs a write transaction, so a read-only handle or a busy
        writer then reports a healthy store as corrupt and `memman
        doctor` exits 1 on it.
    Oracle: the same store read through `open_read_only`, which must
        agree with the read-write handle that the store is healthy.

    Notes
    -----
    - SQLite-only, and `open_read_only` is how the benchmark and
      ablation harnesses open a live store.
    """
    store = tmp_path / 'readonly'
    db = open_db(str(store))
    backend = SqliteBackend(db)
    _seed(backend)
    assert backend.integrity_check()['ok']
    db.close()

    result = SqliteBackend(open_read_only(str(store))).integrity_check()

    assert result['ok'] is True
    assert 'not checked' in result['detail']


def test_non_ascii_divergence_stays_where_it_is(backend, backend_kind):
    """Pin the known SQLite/Python tokenizer gap so it cannot widen.

    Mutation: any change that moves the boundary - `remove_diacritics
        1` or `2`, a different tokenizer, or a Postgres split that
        stops matching `_WORD_RE`. All three would silently reshape
        `signals.keyword` for accented content, in opposite
        directions on the two backends.
    Oracle: hand-computed from the two tokenizers' rules. `_WORD_RE`
        is `[a-zA-Z0-9]+`, so `naive` with an i-diaeresis is `na` +
        `ve`; `unicode61` keeps it whole and matches neither.

    Notes
    -----
    - This asserts a DIVERGENCE, on purpose. SQLite cannot reproduce
      `_WORD_RE` without changing `_WORD_RE` itself, which would move
      the drain's reconciliation and causal-edge inference and needs
      its own sweep. The gap is documented on
      `RecallSession.keyword_counts` with its measured cost; this
      test is what stops it growing unnoticed.
    - Postgres is the faithful side and is asserted as such, so a
      regression there fails even though the value differs by
      backend.
    """
    backend.nodes.insert(make_insight(
        id='kw-nonascii', content='a naïve fallback', entities=[]))

    with backend.recall_session() as session:
        got = session.keyword_counts({'na', 've', 'fallback'})

    assert got.get('kw-nonascii') == (1 if backend_kind == 'sqlite' else 3), (
        'sqlite indexes the whole word and matches only "fallback";'
        ' postgres splits exactly as _WORD_RE does')
