"""Symlink-based asset deployment for memman integrations.

Each shipped asset under `src/memman/setup/assets/` is exposed via
`importlib.resources`. `symlink_asset` replaces any existing file or
symlink at `dest` with a fresh symlink pointing at the resolved
package-relative path. This keeps deployed assets in lock-step with
the installed package — wheel installs resolve into site-packages,
editable installs resolve into the source tree.
"""

from importlib.resources import files as pkg_files
from pathlib import Path


def symlink_asset(rel_path: str, dest: Path) -> None:
    """Create or replace a symlink at dest pointing at a shipped asset.
    """
    target = Path(str(pkg_files('memman.setup.assets')
                      .joinpath(rel_path))).resolve()
    dest.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    try:
        dest.unlink()
    except FileNotFoundError:
        pass
    dest.symlink_to(target)
