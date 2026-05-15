"""Backend-level error hierarchy.

These errors are raised by `Backend` Protocol implementations and
caught by pipeline / CLI code. Defined in their own module so callers
can `from memman.store.errors import BackendError` without pulling in
the full Protocol typing surface.
"""

from memman.exceptions import ConfigError as _RuntimeConfigError


class BackendError(Exception):
    """Base class for all backend errors."""


class IntegrityError(BackendError):
    """Raised on constraint violations (FK, unique, check)."""


class ConfigError(BackendError, _RuntimeConfigError):
    """Raised when the backend selection or DSN is misconfigured.

    Inherits from both `BackendError` (so backend code that catches
    `BackendError` keeps working) and the runtime `ConfigError`
    (so a single `except memman.exceptions.ConfigError` at the CLI
    seam catches both runtime config failures and backend dispatch
    failures uniformly).
    """
