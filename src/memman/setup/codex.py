"""Codex integration through a user-scoped memory skill and command rules."""

import contextlib
import json
import sys
from pathlib import Path

import click
from memman.cli import list_agent_commands
from memman.setup.deploy import is_asset_link, symlink_asset
from memman.setup.prompt import status_ok


def _skill_path(skills_dir: str) -> Path:
    return Path(skills_dir) / 'memman'


def _rules_path(config_dir: str) -> Path:
    return Path(config_dir) / 'rules' / 'memman.rules'


def _check_owned(link: Path) -> None:
    """Refuse to overwrite or remove another skill named memman."""
    # The environment path can change during a Python/package upgrade.
    # Recognize even a dangling link to an earlier packaged asset.
    if is_asset_link('codex', link):
        return
    if link.is_symlink() or link.exists():
        raise click.ClickException(
            f'{link} already exists and is not the packaged memman skill;'
            ' move it aside before installing or uninstalling')


def check_codex_skill(env: dict) -> None:
    """Check for a conflict before changing any integration or settings."""
    _check_owned(_skill_path(env['skills_dir']))


def install_codex(env: dict, no_wizard: bool = False) -> None:
    """Link the packaged skill and allow the agent verbs in Codex rules.

    Parameters
    ----------
    env : dict
        `detect_codex` output.
    no_wizard : bool, default False
        Write the rules without asking, even on a TTY.

    Notes
    -----
    - An allowed verb runs outside the Codex sandbox with no approval
      prompt. Without the rule, the sandbox blocks the writes to the
      data dir and the provider calls, and Codex stops to ask.
    """
    link = _skill_path(env['skills_dir'])
    _check_owned(link)
    print(f'\nSetting up Codex ({env["skills_dir"]})...')
    symlink_asset('codex', link)
    status_ok('Skill', str(link))

    rules_path = _rules_path(env['config_dir'])
    rules = [
        f'prefix_rule(pattern={json.dumps(["memman", *path])},'
        ' decision="allow")'
        for path in list_agent_commands()]
    approved = True
    if sys.stdin.isatty() and not no_wizard:
        print(f'\nmemman would like to write these rules to {rules_path}:')
        for rule in rules:
            print(f'  {rule}')
        approved = click.confirm('Add them?', default=True)
    if approved:
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        rules_path.write_text('\n'.join(rules) + '\n')
        status_ok('Permissions', f'{len(rules)} memman verbs allowed')
    else:
        status_ok('Permissions',
                  'skipped (approve each call in Codex, or re-run'
                  ' memman install --codex)')
    print('Use $memman in Codex to recall or save memories.')
    print('Start a new Codex session if the skill is not visible.')


def uninstall_codex(env: dict) -> None:
    """Remove the packaged skill link and the memman rules file.
    """
    link = _skill_path(env['skills_dir'])
    _check_owned(link)
    print(f'\nRemoving Codex integration ({env["skills_dir"]})...')
    link.unlink(missing_ok=True)
    status_ok('Skill', str(link) + ' removed')
    rules_path = _rules_path(env['config_dir'])
    rules_path.unlink(missing_ok=True)
    status_ok('Permissions', str(rules_path) + ' removed')
    # An empty config dir left behind would still detect Codex.
    for directory in (rules_path.parent, rules_path.parent.parent):
        with contextlib.suppress(OSError):
            directory.rmdir()
