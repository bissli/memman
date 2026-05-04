"""Phase 3 -- cross-backend `backend` fixture sanity checks.

Confirms the new `backend` fixture parametrizes over available
backends, yields an isolated Backend per test, and routes
nodes/meta verbs cleanly. SQLite-only `make test` runs are
unaffected; the postgres parametrization slot emits only when
`psycopg` + `testcontainers` are importable, and the postgres
slot carries `pytest.mark.postgres` so `-m "not postgres"` skips it.
"""

import pytest
from memman.store.backend import Backend
from tests.conftest import make_insight


def test_backend_fixture_yields_a_runtime_backend(backend, backend_kind):
    """Fixture yields an object satisfying the runtime Backend Protocol.
    """
    assert isinstance(backend, Backend)
    assert backend_kind in {'sqlite', 'postgres'}


def test_backend_fixture_starts_empty(backend):
    """Each parametrized invocation sees a fresh empty store.
    """
    assert backend.nodes.count_active() == 0
    assert backend.nodes.count_total() == 0


def test_backend_fixture_supports_insert_and_count(backend):
    """Round-trip: insert two nodes, count reflects both.
    """
    backend.nodes.insert(make_insight(id='cb-1'))
    backend.nodes.insert(make_insight(id='cb-2'))
    assert backend.nodes.count_active() == 2


def test_backend_fixture_meta_round_trip(backend):
    """`meta.set` / `meta.get` round-trips on both backends.
    """
    backend.meta.set('phase3-probe', 'value-x')
    assert backend.meta.get('phase3-probe') == 'value-x'


def test_backend_fixture_isolation_between_tests(backend):
    """Independent test invocations do not share state.

    Inserting here must not be visible to other tests; the
    matching assertion lives in `test_backend_fixture_starts_empty`
    because pytest collects tests in module order and parametrize
    repeats both tests per kind, so a leak would surface as a
    non-zero count there.
    """
    backend.nodes.insert(make_insight(id='iso-only'))
    assert backend.nodes.count_active() == 1
