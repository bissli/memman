"""Smoke test: no stale CLI command references in shipped artifacts.

Walks src/memman, docs, README.md, CONTRIBUTING.md, .claude (project),
and tests/ for known-removed command strings. Fails fast with file:line
citations. Drives the rename commits by example: this test goes red on
master and stays red until every legacy reference is gone.
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BANNED_PATTERNS: list[tuple[str, str]] = [
    (r'memman\s+search\b', 'top-level `memman search` (use recall --basic)'),
    (r'memman\s+viz\b', 'memman viz (removed entirely)'),
    (r'memman\s+gc\b', 'top-level `memman gc` (use insights candidates/review/protect)'),
    (r'memman\s+enrich\b', 'memman enrich (use scheduler drain, hidden)'),
    (r'memman\s+link\s', 'top-level `memman link` (use graph link)'),
    (r'memman\s+related\s', 'top-level `memman related` (use graph related)'),
    (r'memman\s+queue\s+(list|cat|retry|purge|list-failed|failed|show)\b',
     'top-level `memman queue` (use scheduler queue ...)'),
    (r'memman\s+scheduler\s+off\b', '`scheduler off` (use scheduler disable)'),
    (r'memman\s+scheduler\s+resume\b', '`scheduler resume` (use scheduler enable)'),
    (r'memman\s+scheduler\s+logs\b', '`scheduler logs` (use log worker)'),
    (r'memman\s+scheduler\s+debug\s+tail\b', '`scheduler debug tail` (use log trace)'),
    (r'memman\s+store\s+set\b', '`store set` (use store use)'),
    (r'memman\s+keys\s+test\b', '`keys test` (subsumed by doctor)'),
    (r'queue\s+list-failed\b', '`queue list-failed` (use queue failed)'),
    (r'queue\s+cat\b', '`queue cat` (use queue show)'),
]

ROOTS_TO_SCAN = [
    'src/memman/setup/assets',
    'docs',
    'README.md',
    'CONTRIBUTING.md',
    'CHANGELOG.md',
    '.claude/CLAUDE.md',
    '.claude/reference',
    '.claude/rules',
    'tests',
]

ALLOWED_FILES: set[str] = {
    'tests/test_no_stale_references.py',
}

EXTENSIONS = {'.md', '.py', '.sh', '.json', '.toml', '.yaml', '.yml', '.txt'}


def _iter_files() -> list[Path]:
    files: list[Path] = []
    for root in ROOTS_TO_SCAN:
        path = PROJECT_ROOT / root
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        for f in path.rglob('*'):
            if not f.is_file():
                continue
            if f.suffix not in EXTENSIONS and f.name != 'Makefile':
                continue
            files.append(f)
    return files


def test_no_stale_command_references() -> None:
    """Fail with file:line:pattern citations for every legacy command found."""
    violations: list[str] = []
    for file_path in _iter_files():
        rel = file_path.relative_to(PROJECT_ROOT).as_posix()
        if rel in ALLOWED_FILES:
            continue
        try:
            text = file_path.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern, description in BANNED_PATTERNS:
                if re.search(pattern, line):
                    violations.append(f'{rel}:{line_no}: {description}\n    > {line.strip()}')

    if violations:
        msg = ['Stale CLI command references found:']
        msg.extend(violations)
        msg.append(f'\nTotal: {len(violations)} violations across {len({v.split(":")[0] for v in violations})} files.')
        raise AssertionError('\n'.join(msg))


def test_e2e_script_no_stale_commands() -> None:
    """The bash e2e script is scanned separately because it has no .ext-based suffix."""
    e2e = PROJECT_ROOT / 'scripts' / 'e2e_test.sh'
    if not e2e.exists():
        return
    text = e2e.read_text(encoding='utf-8')
    violations: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern, description in BANNED_PATTERNS:
            if re.search(pattern, line):
                violations.append(f'scripts/e2e_test.sh:{line_no}: {description}\n    > {line.strip()}')
    if violations:
        raise AssertionError('Stale CLI references in e2e script:\n' + '\n'.join(violations))
