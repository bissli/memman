"""Store names must not overflow or collide in a Postgres schema.

Postgres truncates an identifier to NAMEDATALEN-1 = 63 bytes without
complaint. memman prefixes `store_`, so a 58-character store name
already produces a 64-character schema. Two names differing only past
that point map to one schema and share its rows.
"""

import pytest
from memman.store.errors import ConfigError
from memman.store.postgres import _store_schema

PG_NAME_MAX = 63
PREFIX_LEN = len('store_')


def test_schema_name_within_the_postgres_limit_is_accepted():
    """The longest name that still fits is not rejected.

    Mutation: an off-by-one in the length guard (`>=` for `>`, or
    checking the bare name rather than the prefixed one), which would
    refuse the longest legal name.
    Oracle: a name sized so `store_<name>` is exactly 63 characters.
    """
    name = 'a' * (PG_NAME_MAX - PREFIX_LEN)
    schema = _store_schema(name)

    assert len(schema) == PG_NAME_MAX


def test_schema_name_over_the_limit_is_refused():
    """One character too long raises rather than silently truncating.

    Mutation: dropping the length guard, restoring the silent
    collision -- Postgres truncates and two stores share a schema.
    Oracle: a name one longer than the accepted case above, which
    yields a 64-character schema.
    """
    name = 'a' * (PG_NAME_MAX - PREFIX_LEN + 1)

    with pytest.raises(ConfigError) as caught:
        _store_schema(name)

    assert '63' in str(caught.value)


def test_two_long_names_cannot_silently_share_a_schema():
    """Names that Postgres would fold together are both refused.

    This is the defect itself: `('store_' || repeat('a',57) || 'XX')
    ::name` equals the same expression ending `YY` on a live server,
    so both stores would read and write one another's rows.

    Mutation: length-checking the bare store name instead of the
    prefixed schema, which passes both of these at 59 characters.
    Oracle: the two names differ, yet truncation to 63 would erase
    the difference, so neither may be accepted.
    """
    stem = 'a' * (PG_NAME_MAX - PREFIX_LEN)
    first, second = f'{stem}XX', f'{stem}YY'

    assert first != second
    assert f'store_{first}'[:PG_NAME_MAX] == f'store_{second}'[:PG_NAME_MAX]

    for name in (first, second):
        with pytest.raises(ConfigError):
            _store_schema(name)
