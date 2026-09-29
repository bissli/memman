"""Symlink-based asset deployment for memman integrations.

Each shipped asset under `src/memman/setup/assets/` is exposed via
`importlib.resources`. `symlink_asset` replaces any existing file or
symlink at `dest` with a fresh symlink pointing at the resolved
package-relative path. This keeps deployed assets in lock-step with
the installed package - wheel installs resolve into site-packages,
editable installs resolve into the source tree.
"""

from importlib.resources import files as pkg_files
from pathlib import Path
from tempfile import TemporaryDirectory


def is_asset_link(rel_path: str, dest: Path) -> bool:
    """Recognize a packaged link, including one to a removed environment.

    Read the link rather than following it so cyclic foreign links cannot
    crash discovery. Deployment always writes resolved package paths.
    """
    try:
        target = dest.readlink()
    except OSError:
        return False
    suffix = ('memman', 'setup', 'assets', *Path(rel_path).parts)
    return target.parts[-len(suffix):] == suffix


def symlink_asset(rel_path: str, dest: Path) -> None:
    """Atomically replace dest with a symlink pointing at a shipped asset.

    Parameters
    ----------
    rel_path : str
        Asset path relative to `memman/setup/assets/`.
    dest : Path
        Symlink location. Missing parent directories are created, and
        any existing file or symlink there is replaced.
    """
    target = Path(str(pkg_files('memman.setup.assets')
                      .joinpath(rel_path))).resolve()
    dest.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    # Stage on the same filesystem so a failed link creation or rename
    # leaves the previous asset installed. The temporary directory also
    # avoids collisions between simultaneous install processes.
    with TemporaryDirectory(prefix='.memman-', dir=dest.parent) as staging:
        staged = Path(staging) / dest.name
        staged.symlink_to(target)
        staged.replace(dest)
