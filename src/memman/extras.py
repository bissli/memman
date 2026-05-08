"""Runtime detection of memman optional install extras.

Probe-module lists derive from the static backend registry in
`memman.store.factory.BACKENDS`; each backend descriptor declares
its `extras_packages` tuple, and a backend's extra is "active"
when every package is importable via `importlib.util.find_spec`.
Probes never execute the modules, so this is cheap and side-effect-
free; the install wizard, doctor, and tests use this to decide
whether a backend's paths are reachable. Adding a new RDBMS backend
to the registry surfaces it here automatically.
"""

from importlib.util import find_spec


def _extras_map() -> dict[str, tuple[str, ...]]:
    from memman.store.factory import all_descriptors
    return {
        d.name: d.extras_packages
        for d in all_descriptors()
        if d.extras_packages}


def is_available(extra: str) -> bool:
    """Return True when every probe module for `extra` is importable.
    """
    probes = _extras_map()[extra]
    return all(find_spec(m.replace('-', '_')) is not None
               for m in probes)


def detect_active_extras() -> list[str]:
    """Return the names of extras that resolve at runtime."""
    return [name for name in _extras_map() if is_available(name)]
