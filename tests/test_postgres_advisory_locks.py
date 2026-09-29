"""Tests for `_lock_id` deterministic advisory-lock id derivation.

Python's built-in `hash()` is randomized per-process via PYTHONHASHSEED,
so two memman processes touching the same store would compute different
lock ids and fail to serialize. `_lock_id` uses blake2b for cross-process
determinism.
"""

import subprocess
import sys

from memman.store.postgres import _advisory_lock_key, _lock_id


def test_lock_id_deterministic_within_process():
    """Verify the same lock name yields the same id within one interpreter.

    Mutation: mixing a per-call random or counter value into the id.
    Oracle: two calls with one literal input.
    """
    assert _lock_id('store_main:drain') == _lock_id('store_main:drain')


def test_lock_id_deterministic_across_processes():
    """Verify two interpreters compute identical lock ids.

    Mutation: deriving the id from built-in `hash()`, which PYTHONHASHSEED
        randomizes per process, so two memman processes take different locks.
    Oracle: the printed ids from two subprocess runs.
    """
    code = (
        "from memman.store.postgres import _lock_id; "
        "print(_lock_id('store_main:drain'))"
    )
    out1 = subprocess.check_output([sys.executable, '-c', code]).strip()
    out2 = subprocess.check_output([sys.executable, '-c', code]).strip()
    assert out1 == out2
    assert out1 != b''


def test_lock_id_distinct_per_schema():
    """Verify two stores do not collide on the same lock name.

    Mutation: dropping the schema part of the name before hashing.
    Oracle: ids for 'store_main:drain' and 'store_shared:drain' differ.
    """
    a = _lock_id('store_main:drain')
    b = _lock_id('store_shared:drain')
    assert a != b


def test_lock_id_distinct_per_lock_name():
    """Verify two lock names within one schema do not collide.

    Mutation: hashing only the schema part of the name.
    Oracle: ids for 'store_main:drain' and 'store_main:reembed' differ.
    """
    a = _lock_id('store_main:drain')
    b = _lock_id('store_main:reembed')
    assert a != b


def test_lock_id_fits_signed_int8():
    """Verify the id fits the signed int8 that pg_advisory_lock takes.

    Mutation: returning an unsigned 64-bit digest, which Postgres rejects
        above 2**63 - 1.
    Oracle: the int8 bounds -(2**63) and 2**63 - 1.
    """
    value = _lock_id('store_main:drain')
    assert -(2 ** 63) <= value <= (2 ** 63) - 1


def test_advisory_lock_key_routes_through_lock_id():
    """Verify _advisory_lock_key(schema, name) equals _lock_id('schema:name').

    Mutation: joining schema and name with a different separator, so the
        key differs from the id other code derives.
    Oracle: _lock_id on the hand-joined string.
    """
    assert _advisory_lock_key('store_main', 'drain') == _lock_id(
        'store_main:drain')
