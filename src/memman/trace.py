"""Structured JSONL trace mode for memman debug sessions.

Default off. Two ways to enable:

1. Export `MEMMAN_DEBUG=1` in the current shell. `MEMMAN_DEBUG` is a
   process-control variable -- the env file is never consulted for it.
2. Run `memman scheduler debug on` to flip the persistent toggle in
   `~/.memman/debug.state`. Affects future scheduler-fired drains and
   any CLI invocation in a shell that has not exported MEMMAN_DEBUG.
   `memman scheduler debug off` clears it.

When enabled and `setup()` has been called, every `event(name, **fields)`
call writes one JSON line to `~/.memman/logs/debug.log`, which is
size-rotated by `RotatingFileHandler` to cap total disk use.

The handler attaches to `logging.getLogger('memman')` at DEBUG level, so
every `logger.debug()` call across the codebase is captured for free in
the same file.

Header redaction replaces values for `Authorization`, `x-api-key`, and
`Api-Key` (case-insensitive) with `'***REDACTED***'`. Bodies are logged
verbatim -- this is deliberate per the feature's explicit design.
"""

import json
import logging
import logging.handlers
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memman import config
from memman.setup.scheduler import get_debug

TRACE_FILENAME = 'debug.log'
LOG_DIR_NAME = 'logs'
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3
REDACT_HEADER_NAMES = {'authorization', 'x-api-key', 'api-key'}
REDACT_VALUE = '***REDACTED***'

_DSN_PASSWORD_RE = re.compile(
    r'(?P<scheme>[a-z][a-z0-9+.-]*://[^:@/\s]*):[^/\s]+@', re.IGNORECASE)
_DSN_QUERY_PASSWORD_RE = re.compile(
    r'(?P<key>[?&](?:ssl)?password=)[^&#\s]*', re.IGNORECASE)
_DSN_KEYWORD_PASSWORD_RE = re.compile(
    r"(?P<key>(?<!\S)(?:ssl)?password\s*=\s*)"
    r"(?:'(?:[^'\\]|\\.)*(?:'|$)|(?:\\.|[^\s\\])+)",
    re.IGNORECASE)


def redact_dsn(value: str) -> str:
    """Mask the password in a URI or libpq key=value DSN.

    Masks the userinfo password of `scheme://user:pass@host`, a
    `password=` or `sslpassword=` URI query parameter, and the value of
    a `password` or `sslpassword` keyword, quoted or bare. A quote left
    open masks to the end of the string. A passwordless DSN, or a
    string that does not match, returns unchanged. Call at any log site
    that may carry a postgres connection string.
    """
    masked = _DSN_PASSWORD_RE.sub(r'\g<scheme>:***@', value)
    masked = _DSN_QUERY_PASSWORD_RE.sub(r'\g<key>***', masked)
    return _DSN_KEYWORD_PASSWORD_RE.sub(r'\g<key>***', masked)


def is_enabled() -> bool:
    """Return True when trace mode is on.

    `MEMMAN_DEBUG` is read from `os.environ` alone; the env file is
    ignored. A truthy value enables trace; any other non-empty value
    disables it. When the variable is unset or empty, the answer is
    `~/.memman/debug.state`, written by `memman scheduler debug on`.
    """
    raw = os.environ.get(config.DEBUG)
    if raw is not None and raw != '':
        return raw.strip().lower() in config.TRUTHY
    return get_debug()


def _trace_path() -> Path:
    """Path of `~/.memman/logs/debug.log` under the current HOME.
    """
    return Path.home() / '.memman' / LOG_DIR_NAME / TRACE_FILENAME


class JsonlFormatter(logging.Formatter):
    """Format each log record as a single-line JSON object.

    Record payload carried via `extra={'trace_fields': {...}}`. Other
    records (plain logger.debug calls) are captured as:
    {"ts": ..., "level": ..., "event": "<msg>"}.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render the record as compact JSON.
        """
        ts = datetime.fromtimestamp(
            record.created, tz=timezone.utc).isoformat()
        payload: dict[str, Any] = {
            'ts': ts,
            'level': record.levelname,
            'event': record.getMessage(),
            }
        fields = getattr(record, 'trace_fields', None)
        if isinstance(fields, dict):
            for k, v in fields.items():
                if k in payload:
                    continue
                payload[k] = v
        return json.dumps(payload, default=_json_default, sort_keys=False)


def _json_default(obj: Any) -> str:
    """Fallback serializer for non-JSON-native objects in trace payloads.
    """
    try:
        return repr(obj)
    except Exception:
        return f'<unserializable {type(obj).__name__}>'


def setup() -> None:
    """Attach a rotating JSONL file handler to the 'memman' logger.

    A no-op when tracing is disabled. Repeated calls attach no
    duplicate handler. The log file is chmod 600 right after creation,
    so raw memory content never sits at world-readable permissions.
    """
    if not is_enabled():
        return

    logger = logging.getLogger('memman')
    for h in logger.handlers:
        if getattr(h, '_memman_trace', False):
            return

    log_path = _trace_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding='utf-8',
        delay=False,
        )
    handler.setFormatter(JsonlFormatter())
    handler.setLevel(logging.DEBUG)
    handler._memman_trace = True
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > logging.DEBUG:
        logger.setLevel(logging.DEBUG)

    try:
        Path(log_path).chmod(0o600)
    except OSError:
        pass


def event(name: str, **fields: Any) -> None:
    """Emit a structured trace event. No-op when tracing is disabled.
    """
    if not is_enabled():
        return
    logger = logging.getLogger('memman')
    logger.debug(name, extra={'trace_fields': fields})


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of headers with secret values masked.

    Case-insensitive match on header names in REDACT_HEADER_NAMES.
    """
    out = {}
    for k, v in headers.items():
        if k.lower() in REDACT_HEADER_NAMES:
            out[k] = REDACT_VALUE
        else:
            out[k] = v
    return out
