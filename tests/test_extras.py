"""Tests for `memman.extras` runtime detection.
"""

import sys
import tomllib
from importlib.util import find_spec
from pathlib import Path

from memman import extras


def test_postgres_unavailable_when_psycopg_missing(monkeypatch):
    """Verify is_available('postgres') is False when psycopg cannot be found.

    Mutation: is_available ignoring the probe result and reporting True.
    Oracle: a find_spec stub that returns None for psycopg only.
    """
    monkeypatch.setitem(sys.modules, 'psycopg', None)

    def _missing(name: str):
        if name == 'psycopg':
            return None
        return find_spec(name)

    monkeypatch.setattr('memman.extras.find_spec', _missing)
    assert extras.is_available('postgres') is False


def test_extras_keys_match_pyproject():
    """Verify backend extras match the poetry extras declaration.

    Mutation: adding a backend to the registry without a matching extras
        block in pyproject.toml.
    Oracle: the key set parsed from pyproject.toml.
    """
    pyproject = Path(__file__).resolve().parent.parent / 'pyproject.toml'
    with Path(pyproject).open('rb') as fh:
        data = tomllib.load(fh)
    declared = set(data['tool']['poetry'].get('extras', {}).keys())
    detected = set(extras._extras_map().keys())
    assert declared == detected, (
        f'pyproject.toml extras {declared} drifted from'
        f' extras._extras_map() {detected}')
