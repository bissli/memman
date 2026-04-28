"""Atomic secure-write helper for setup files holding sensitive data.

Used by scheduler state, env, and openclaw config writers. Creates the
.tmp at mode 0o600 from the start (no race window where another
process could observe a more permissive mode), enforces the mode
against the open descriptor with fchmod, then atomically renames into
place.
"""

import os
from pathlib import Path


def atomic_write_secure(path: Path, contents: str) -> None:
    """Atomically write contents to path at mode 0o600."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write(contents)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    tmp.replace(path)
