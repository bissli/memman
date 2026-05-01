"""Backend-level error hierarchy.

These errors are raised by `Backend` Protocol implementations and
caught by pipeline / CLI code. Defined in their own module so callers
can `from memman.store.errors import NotFound` without pulling in the
full Protocol typing surface.
"""


class BackendError(Exception):
    """Base class for all backend errors."""


class NotFound(BackendError):
    """Raised when a lookup misses (insight, edge, meta key)."""


class IntegrityError(BackendError):
    """Raised on constraint violations (FK, unique, check)."""


class DrainBusy(BackendError):
    """Raised when an exclusive write op cannot acquire the drain lock."""


class SchemaVersionMismatch(BackendError):
    """Raised when the application version does not match the DB schema.

    Postgres-only signal: if a binary running an older application
    version connects to a database whose schema has been migrated
    forward, every call should refuse rather than corrupt data.
    SQLite is a single-file local store and does not raise this.
    """


class ConfigError(BackendError):
    """Raised when the backend selection or DSN is misconfigured."""
