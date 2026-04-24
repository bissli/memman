"""NanoClaw integration: deliver the /add-memman skill to Claude Code.

NanoClaw is a containerised agent platform. Unlike Claude Code or
OpenClaw, its memman integration is driven by an agentic skill
(`/add-memman`) that the user invokes inside Claude Code while editing
their NanoClaw project — the skill patches the Dockerfile, adds a
container skill, and sets up volume mounts. Our job here is to deliver
the skill file into the Claude Code skills directory so `/add-memman`
is invokable.
"""

import shutil
from pathlib import Path

from memman.setup.deploy import symlink_asset
from memman.setup.prompt import status_ok
from memman.setup.settings import remove_if_empty

SKILL_NAME = 'add-memman'


def _claude_skill_dir() -> Path:
    """Resolve ~/.claude/skills/add-memman/."""
    return Path.home() / '.claude' / 'skills' / SKILL_NAME


def install_nanoclaw(env: dict, data_dir: str) -> None:
    """Install the /add-memman agentic skill into Claude Code's skills dir."""
    skill_dir = _claude_skill_dir()
    skill_path = skill_dir / 'SKILL.md'
    container_path = skill_dir / 'container-skill.md'

    symlink_asset('nanoclaw/SKILL.md', skill_path)
    symlink_asset('nanoclaw/container-skill.md', container_path)

    print(f'\nSetting up NanoClaw integration ({skill_dir})...')
    status_ok('Skill', str(skill_path))
    status_ok('Container', str(container_path))

    print()
    print('Setup complete!')
    print(f'  Skill: {skill_path}')
    print()
    print('Next: open a Claude Code session inside your NanoClaw project'
          ' and run /add-memman to patch the Dockerfile + mount memman'
          ' into the container.')


def uninstall_nanoclaw(config_dir: str | None = None) -> list[Exception]:
    """Remove the /add-memman skill from Claude Code's skills dir."""
    errs: list[Exception] = []
    skill_dir = _claude_skill_dir()
    print(f'\nRemoving NanoClaw integration ({skill_dir})...')
    try:
        if skill_dir.exists():
            shutil.rmtree(skill_dir, ignore_errors=True)
            status_ok('Skill', f'{skill_dir} removed')
    except Exception as e:
        errs.append(e)
    remove_if_empty(Path(str(skill_dir)).parent)
    return errs
