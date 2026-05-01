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


def test_postgres_available_when_all_probes_resolve(monkeypatch):
    """is_available returns True when every probe spec is non-None."""
    monkeypatch.setattr(
        'memman.extras.find_spec',
        lambda name: object())
    assert extras.is_available('postgres') is True


def test_detect_active_extras_returns_list(monkeypatch):
    """detect_active_extras returns the names of importable extras."""
    monkeypatch.setattr(
        'memman.extras.find_spec',
        lambda name: object())
    assert extras.detect_active_extras() == ['postgres']

    monkeypatch.setattr(
        'memman.extras.find_spec',
        lambda name: None)
    assert extras.detect_active_extras() == []


def test_extras_keys_match_pyproject():
    """`_EXTRAS` keys must match `pyproject.toml::[tool.poetry.extras]`."""
    pyproject = Path(__file__).resolve().parent.parent / 'pyproject.toml'
    with Path(pyproject).open('rb') as fh:
        data = tomllib.load(fh)
    declared = set(data['tool']['poetry'].get('extras', {}).keys())
    detected = set(extras._EXTRAS.keys())
    assert declared == detected, (
        f'pyproject.toml extras {declared} drifted from'
        f' extras._EXTRAS {detected}')
