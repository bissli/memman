"""Distribution and filesystem boundaries for the Codex skill lifecycle."""

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from importlib.resources import files
from pathlib import Path

from memman.setup import deploy, detect
from memman.setup.codex import install_codex, uninstall_codex


def test_upgrade_from_removed_environment_uses_new_package(monkeypatch, tmp_path):
    """Verify a skill linked into a deleted environment is found and relinked.

    Mutation: detect_codex or is_asset_link tests link.exists(), so a dangling
    old link is missed, or install_codex keeps the stale target.
    Oracle: Two fake asset trees whose SKILL.md holds the environment name; the
    linked text is read after each install, with no .memman-* litter.
    """
    monkeypatch.setattr(detect.shutil, 'which', lambda name: None)
    env = detect.detect_codex()
    link = Path(env['skills_dir']) / 'memman'
    for version in ('old environment', 'new environment'):
        assets = tmp_path / version / 'memman/setup/assets'
        (assets / 'codex').mkdir(parents=True)
        (assets / 'codex/SKILL.md').write_text(version)
        monkeypatch.setattr(deploy, 'pkg_files', lambda package, p=assets: p)
        install_codex(env)
        assert (link / 'SKILL.md').read_text() == version
        if version == 'old environment':
            shutil.rmtree(tmp_path / version)
            assert not link.exists()
            assert detect.detect_codex()['detected']

    assert not list(link.parent.glob('.memman-*'))
    uninstall_codex(env)
    assert (assets / 'codex/SKILL.md').read_text() == 'new environment'
    assert not detect.detect_codex()['detected']


def test_lifecycle_through_symlinked_skills_root_preserves_siblings(tmp_path):
    """Verify a symlinked skills root survives install and uninstall intact.

    Mutation: symlink_asset or uninstall_codex swaps the skills-dir symlink for
    a real directory, or removes sibling skills during cleanup.
    Oracle: A hand-built shared root modeling a dotfile manager, with a sibling
    skill; readlink, sibling text, and the root listing are checked.
    """
    root = tmp_path / 'shared skills'
    other = root / 'other/SKILL.md'
    other.parent.mkdir(parents=True)
    other.write_text('unrelated skill')
    skills = Path.home() / '.agents/skills'
    skills.parent.mkdir(parents=True)
    skills.symlink_to(root, target_is_directory=True)
    env = {'skills_dir': str(skills),
           'config_dir': str(Path.home() / '.codex')}

    install_codex(env)
    install_codex(env)
    assert (root / 'memman/SKILL.md').is_file()
    uninstall_codex(env)

    assert skills.is_symlink()
    assert skills.readlink() == root
    assert other.read_text() == 'unrelated skill'
    assert sorted(p.name for p in root.iterdir()) == ['other']


def test_distributions_ship_a_loadable_codex_skill(tmp_path):
    """Verify wheel and sdist ship the skill and a wheel install loads it.

    Mutation: The packaging config drops assets/codex/SKILL.md from the wheel
    or sdist, or codex.py imports a module the wheel omits.
    Oracle: Source-tree SKILL.md bytes compared to each archive member; a fresh
    interpreter installs from the extracted wheel and checks the link.
    """
    poetry = shutil.which('poetry')
    project = Path(__file__).resolve().parents[1]
    dist = tmp_path / 'dist'
    result = subprocess.run(
        [poetry, 'build', '--output', str(dist)], cwd=project,
        env=dict(os.environ, POETRY_VIRTUALENVS_CREATE='false'),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr

    shipped = files('memman.setup.assets').joinpath('codex/SKILL.md').read_bytes()
    asset_path = 'memman/setup/assets/codex/SKILL.md'
    installed = tmp_path / 'installed wheel'
    with zipfile.ZipFile(next(dist.glob('*.whl'))) as archive:
        assert archive.read(asset_path) == shipped
        archive.extractall(installed)
    with tarfile.open(next(dist.glob('*.tar.gz'))) as archive:
        candidates = [m for m in archive.getmembers()
                      if m.name.endswith('/src/' + asset_path)]
        assert len(candidates) == 1
        assert archive.extractfile(candidates[0]).read() == shipped

    # A fresh interpreter prevents an editable import already in sys.modules
    # from hiding missing modules or resources in the release archive.
    script = """
import sys
from pathlib import Path
import memman
from memman.setup.codex import install_codex, uninstall_codex
from memman.setup.deploy import is_asset_link

root = Path(sys.argv[1])
assert Path(memman.__file__).is_relative_to(root)
env = {'skills_dir': str(Path(sys.argv[2])),
       'config_dir': str(Path(sys.argv[2]) / 'codex')}
install_codex(env)
link = Path(env['skills_dir']) / 'memman'
assert is_asset_link('codex', link)
assert link.resolve().is_relative_to(root)
assert (link / 'SKILL.md').read_text().startswith('---\\nname: memman\\n')
uninstall_codex(env)
assert not link.is_symlink()
assert (root / 'memman/setup/assets/codex/SKILL.md').is_file()
"""
    result = subprocess.run(
        [sys.executable, '-c', script, str(installed), str(tmp_path / 'skills')],
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(installed)),
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
