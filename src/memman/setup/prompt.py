"""Status output helpers for setup (ANSI-aware, non-interactive)."""

import sys

from memman.setup.detect import home_dir

COLOR_GREEN = '\033[32m'
COLOR_DIM = '\033[2m'
COLOR_RED = '\033[31m'
COLOR_BOLD = '\033[1m'
COLOR_RESET = '\033[0m'
SYM_OK = '✓'
SYM_FAIL = '✗'
SYM_DOT = '·'

_colors_inited = False


def _init_colors() -> None:
    """Clear ANSI codes when stdout is not a TTY."""
    global COLOR_GREEN, COLOR_DIM, COLOR_RED, COLOR_BOLD, COLOR_RESET, _colors_inited
    if _colors_inited:
        return
    _colors_inited = True
    if not sys.stdout.isatty():
        COLOR_GREEN = ''
        COLOR_DIM = ''
        COLOR_RED = ''
        COLOR_BOLD = ''
        COLOR_RESET = ''


def status_ok(label: str, detail: str) -> None:
    """Print a green checkmark status line."""
    _init_colors()
    print(f'  {COLOR_GREEN}{SYM_OK}{COLOR_RESET}'
          f' {label:<12s} {COLOR_DIM}{detail}{COLOR_RESET}')


def status_updated(label: str, detail: str) -> None:
    """Print a green checkmark with 'updated' note."""
    _init_colors()
    print(f'  {COLOR_GREEN}{SYM_OK}{COLOR_RESET}'
          f' {label:<12s} {COLOR_DIM}{detail}{COLOR_RESET}'
          f'  {COLOR_GREEN}updated{COLOR_RESET}')


def status_error(label: str, err: object) -> None:
    """Print a red cross status line."""
    _init_colors()
    print(f'  {COLOR_RED}{SYM_FAIL}{COLOR_RESET}'
          f' {label:<12s} {COLOR_RED}{err}{COLOR_RESET}')


def detection_line(detected: bool, display: str,
                   version: str, path: str) -> None:
    """Print a detection result line."""
    _init_colors()
    display_path = path.replace(home_dir(), '~', 1)
    if detected:
        print(f'  {COLOR_GREEN}{SYM_OK}{COLOR_RESET}'
              f' {display:<14s} {COLOR_DIM}{version:<12s}'
              f' {display_path}{COLOR_RESET}')
    else:
        print(f'  {COLOR_DIM}{SYM_DOT} {display:<14s}'
              f' (not found){COLOR_RESET}')
