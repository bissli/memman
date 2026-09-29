"""JSON configuration management with JSON5 support.
"""

import json
import os
from pathlib import Path
from typing import Any


def strip_json5(s: str) -> str:
    r"""Strip JSON5 affordances down to plain JSON.

    Handles: `//` line comments, `/* */` block comments, double-quoted
    and single-quoted strings (both pass through unchanged, with escape
    handling so `\\'` or `\\"` inside a string does not end it), and
    trailing commas before `]`, `}`, `)`.

    Parameters
    ----------
    s : str
        JSON5 text.

    Returns
    -------
    str
        Plain JSON text.
    """
    result = []
    in_string: str | None = None
    escaped = False
    i = 0
    while i < len(s):
        ch = s[i]
        if escaped:
            result.append(ch)
            escaped = False
            i += 1
            continue
        if in_string is not None:
            if ch == '\\':
                escaped = True
            elif ch == in_string:
                in_string = None
            result.append(ch)
            i += 1
            continue
        if ch in {'"', "'"}:
            in_string = ch
            result.append(ch)
            i += 1
            continue
        if ch == '/' and i + 1 < len(s) and s[i + 1] == '/':
            while i < len(s) and s[i] != '\n':
                i += 1
            continue
        if ch == '/' and i + 1 < len(s) and s[i + 1] == '*':
            i += 2
            while i + 1 < len(s) and not (s[i] == '*' and s[i + 1] == '/'):
                i += 1
            i += 2
            continue
        if ch == ',':
            j = i + 1
            while j < len(s) and s[j] in ' \t\n\r':
                j += 1
            if j < len(s) and s[j] in ']})':
                i += 1
                continue
        result.append(ch)
        i += 1
    return ''.join(result)


def read_json_file(path: str) -> dict:
    """Read a JSON file into a dict; a missing file gives an empty dict.
    """
    try:
        data = Path(path).read_text()
    except OSError:
        return {}
    if not data:
        return {}
    cleaned = strip_json5(data)
    return json.loads(cleaned)


def write_json_file(path: str, data: dict) -> None:
    """Write a dict to a JSON file atomically via .tmp + rename.
    """
    content = json.dumps(data, indent=2) + '\n'
    Path(path).parent.mkdir(mode=0o755, exist_ok=True, parents=True)
    tmp = path + '.tmp'
    Path(tmp).write_text(content)
    Path(tmp).replace(path)


def write_or_remove_json_file(path: str, data: dict) -> None:
    """Write the settings, or remove the file if the dict is empty.
    """
    if not data:
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass
        return
    write_json_file(path, data)


def _contains_memman(v: Any) -> bool:
    """True when any string inside v mentions memman.
    """
    if isinstance(v, str):
        return 'memman' in v
    if isinstance(v, dict):
        return any(_contains_memman(val) for val in v.values())
    if isinstance(v, list):
        return any(_contains_memman(item) for item in v)
    return False


def remove_claude_hooks(data: dict) -> None:
    """Remove all memman-related entries from Claude Code hooks.
    """
    hooks = data.get('hooks')
    if not isinstance(hooks, dict):
        return
    for key in ('UserPromptSubmit', 'SessionStart',
                'PreCompact', 'PreToolUse'):
        arr = hooks.get(key)
        if not isinstance(arr, list):
            continue
        filtered = [entry for entry in arr if not _contains_memman(entry)]
        if not filtered:
            hooks.pop(key, None)
        else:
            hooks[key] = filtered
    if not hooks:
        data.pop('hooks', None)


def add_claude_hooks_selective(
        data: dict, hooks_dir: str,
        remind: bool = False,
        compact: bool = False,
        task_recall: bool = False,
        exit_plan: bool = False) -> None:
    """Idempotently set memman hooks in Claude Code settings.

    Parameters
    ----------
    data : dict
        Parsed settings, mutated in place. Existing memman hooks are
        removed first.
    hooks_dir : str
        Directory holding the hook scripts.
    remind, compact, task_recall, exit_plan : bool
        Also register the user-prompt, pre-compact, task-recall, and
        exit-plan hooks. The session-start prime hook is always set.
    """
    remove_claude_hooks(data)
    hooks = data.setdefault('hooks', {})
    home = str(Path.home())

    registrations = [
        (True, 'SessionStart', 'prime.sh', None),
        (remind, 'UserPromptSubmit', 'user_prompt.sh', None),
        (compact, 'PreCompact', 'compact.sh', None),
        (task_recall, 'PreToolUse', 'task_recall.sh', 'Agent|Task'),
        (exit_plan, 'PreToolUse', 'exit_plan.sh', 'ExitPlanMode'),
        ]
    for enabled, event, script, matcher in registrations:
        if not enabled:
            continue
        command = os.path.join(hooks_dir, script)
        if command.startswith(home):
            command = '~' + command[len(home):]
        entry = {
            'hooks': [
                {
                    'type': 'command',
                    'command': command,
                    },
                ],
            }
        if matcher:
            entry['matcher'] = matcher
        arr = hooks.get(event, [])
        if not isinstance(arr, list):
            arr = []
        arr.append(entry)
        hooks[event] = arr


_PERMISSION_SECTIONS = ('allow', 'deny', 'ask')


def add_memman_permission(data: dict, entries: list[str]) -> None:
    """Append entries to permissions.allow, skipping any already there.

    Parameters
    ----------
    data : dict
        Parsed settings, mutated in place.
    entries : list[str]
        Allow strings, as `memman.cli.list_claude_permissions` returns.
    """
    perms = data.setdefault('permissions', {})
    allow = perms.setdefault('allow', [])
    for entry in entries:
        if entry not in allow:
            allow.append(entry)


def remove_memman_permission(data: dict) -> None:
    """Drop every string mentioning memman from permissions allow/deny/ask.

    Parameters
    ----------
    data : dict
        Parsed settings, mutated in place. An emptied list, and an
        emptied `permissions`, are removed.
    """
    perms = data.get('permissions')
    if not isinstance(perms, dict):
        return
    for key in _PERMISSION_SECTIONS:
        arr = perms.get(key)
        if not isinstance(arr, list):
            continue
        filtered = [
            item for item in arr
            if not (isinstance(item, str) and 'memman' in item)
            ]
        if not filtered:
            perms.pop(key, None)
        else:
            perms[key] = filtered
    if not perms:
        data.pop('permissions', None)


_REMOVE_IF_EMPTY_ROOTS = frozenset({
    '.claude',
    })
_REMOVE_IF_EMPTY_LEAVES = frozenset({
    'hooks', 'skills',
    })


def remove_if_empty(dir_path: str) -> None:
    """Remove a directory only if it exists and is empty.

    Parameters
    ----------
    dir_path : str
        Directory whose basename is an agent-config root (`.claude`)
        or a known leaf inside one (`hooks`, `skills`).

    Raises
    ------
    ValueError
        The basename is outside the allowlist, which guards against a
        caller passing a path like `/tmp/x` or `/`.
    """
    p = Path(dir_path)
    basename = p.name
    if basename not in _REMOVE_IF_EMPTY_ROOTS \
            and basename not in _REMOVE_IF_EMPTY_LEAVES:
        raise ValueError(
            f'remove_if_empty refused path {dir_path!r}: basename'
            f' must be one of {sorted(_REMOVE_IF_EMPTY_ROOTS | _REMOVE_IF_EMPTY_LEAVES)}')
    try:
        entries = os.listdir(dir_path)
        if not entries:
            p.rmdir()
    except OSError:
        pass
