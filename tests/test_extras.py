"""Tests for `memman.extras` runtime detection."""

import sys
import tomllib
from pathlib import Path

from memman import extras


def test_postgres_unavailable_when_psycopg_missing(monkeypatch):
    """is_available returns False when any probe module is missing."""
    monkeypatch.setitem(sys.modules, 'psycopg', None)

    def _missing(name: str):
        if name == 'psycopg':
            return None
        from importlib.util import find_spec
        return find_spec(name)

    monkeypatch.setattr('memman.extras.find_spec', _missing)
    assert extras.is_available('postgres') is False


def test_extras_keys_match_pyproject():
    """Backend extras must match `pyproject.toml::[tool.poetry.extras]`.

    The set of backend extras is derived from the static
    `BACKENDS` registry; this test pins it against the poetry
    declaration so adding a backend without a matching extras
    block in pyproject is caught at test time.
    """
    pyproject = Path(__file__).resolve().parent.parent / 'pyproject.toml'
    with Path(pyproject).open('rb') as fh:
        data = tomllib.load(fh)
    declared = set(data['tool']['poetry'].get('extras', {}).keys())
    detected = set(extras._extras_map().keys())
    assert declared == detected, (
        f'pyproject.toml extras {declared} drifted from'
        f' extras._extras_map() {detected}')
