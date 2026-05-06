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
    """Same input yields same output within one interpreter.
    """
    assert _lock_id('store_main:drain') == _lock_id('store_main:drain')


def test_lock_id_deterministic_across_processes():
    """Two subprocess invocations compute identical lock ids.
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
    """Two stores never collide on the same lock name.
    """
    a = _lock_id('store_main:drain')
    b = _lock_id('store_shared:drain')
    assert a != b


def test_lock_id_distinct_per_lock_name():
    """Two lock names within one schema do not collide.
    """
    a = _lock_id('store_main:drain')
    b = _lock_id('store_main:reembed')
    assert a != b


def test_lock_id_fits_signed_int8():
    """Postgres pg_advisory_*lock takes signed int8; output must fit.
    """
    value = _lock_id('store_main:drain')
    assert -(2 ** 63) <= value <= (2 ** 63) - 1


def test_advisory_lock_key_routes_through_lock_id():
    """`_advisory_lock_key(schema, name)` matches `_lock_id(f'{schema}:{name}')`.
    """
    assert _advisory_lock_key('store_main', 'drain') == _lock_id(
        'store_main:drain')
