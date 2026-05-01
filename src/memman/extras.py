"""Runtime detection of memman optional install extras.

Each entry in `_EXTRAS` maps an extra name (matching a key in
`pyproject.toml::[tool.poetry.extras]`) to a tuple of probe modules.
An extra is "active" when every probe module is importable via
`importlib.util.find_spec`. Probes never execute the modules, so this
is cheap and side-effect-free; the install wizard, doctor, and tests
all use it to decide whether postgres-only paths are reachable.
"""

from importlib.util import find_spec

_EXTRAS = {
    'postgres': ('psycopg', 'psycopg_pool', 'pgvector'),
    }


def is_available(extra: str) -> bool:
    """Return True when every probe module for `extra` is importable.
    """
    return all(find_spec(m) is not None for m in _EXTRAS[extra])


def detect_active_extras() -> list[str]:
    """Return the names of extras that resolve at runtime."""
    return [name for name in _EXTRAS if is_available(name)]
