#!/bin/bash
# SessionStart hook: prints the status line, any model notice, and
# the guide.

if ! command -v memman >/dev/null 2>&1; then
  echo "[memman] Warning: memman is not on PATH, so its hooks are inactive."
  exit 0
fi

memman prime
exit 0
