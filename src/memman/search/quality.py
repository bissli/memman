"""Content quality signals for the remember pipeline."""

import re

TRANSIENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'i-[0-9a-f]{17}'), 'AWS instance ID'),
    (re.compile(r'\d+ resources? total'), 'resource count'),
    (re.compile(
        r'(?:all|every)\b.{0,30}\bverified', re.IGNORECASE),
        'verification receipt'),
    (re.compile(r'state (?:is )?clean', re.IGNORECASE),
        'state observation'),
    (re.compile(
        r'(?:deployed|completed|applied) via', re.IGNORECASE),
        'deployment receipt'),
    (re.compile(r'(?<!\w)line \d+\b', re.IGNORECASE), 'line number reference'),
    (re.compile(r'\b\d{2,} lines\b'), 'line count'),
    (re.compile(
        r'\b(?!(?:localhost|port|python|node|ruby|go|java|php'
        r'|redis|mysql|postgres|npm|yarn|docker|alpine|ubuntu'
        r'|debian|v?\d)\b)\w+:\d{2,}\b'),
        'function/symbol line reference'),
    (re.compile(r'\d+→\d+'), 'line number correction'),
    (re.compile(r'\bmemor(?:y|ies)\s*\[\s*\d+\s*\]', re.IGNORECASE),
        'back-reference'),
    (re.compile(r'\b[A-Z][A-Z _-]{4,}:\s+'),
        'uppercase section header'),
    (re.compile(r'\bcurrently\b', re.IGNORECASE),
        'transient time marker'),
    (re.compile(r'\bas of \d{4}-\d{2}-\d{2}\b', re.IGNORECASE),
        'dated observation'),
    ]


def check_content_quality(content: str) -> list[str]:
    """Scan content for transient patterns and return warnings."""
    warnings = []
    for pattern, label in TRANSIENT_PATTERNS:
        if pattern.search(content):
            warnings.append(label)
    return warnings
