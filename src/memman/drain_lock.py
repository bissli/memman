"""Process-level exclusion for `_drain_queue`.

Acquires an advisory flock on `<data_dir>/drain.lock`. Released
automatically by the kernel when the holding process exits (clean
or crash), so no stale-lock recovery is needed.

Caveats:
- SIGSTOP / `docker pause` / cgroup freezer hold the lock until the
  process actually dies. This is the kernel's behavior, not ours.
- Assumes a local filesystem. The lock lives at `<data_dir>/`, which
  is the base data dir (typically `~/.memman/`), not the per-store
  directory under `data/`. In nanoclaw containers this sits on the
  container-local writable layer, not on the bind-mounted store volume.
"""
import fcntl
import os
from pathlib import Path


class DrainLockBusy(Exception):
    """Another drain is already running on this data_dir."""


def acquire(data_dir: str) -> int:
    """Acquire the drain lock; return the file descriptor.

    Raises DrainLockBusy if another process holds it. Caller MUST
    call `release(fd)` when done.
    """
    path = Path(data_dir) / 'drain.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise DrainLockBusy(str(path))
    return fd


def release(fd: int) -> None:
    """Release the lock and close the fd.
    """
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
