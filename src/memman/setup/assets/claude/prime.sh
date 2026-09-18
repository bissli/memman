#!/bin/bash
# memman SessionStart hook - thin shim that delegates to `memman prime`.
# The `memman prime` command reads SessionStart JSON on stdin, emits
# the status line, compact-recall hint (when applicable), and the
# shipped behavioral guide.

if [ -t 0 ]; then
  INPUT='{}'
else
  INPUT=$(cat)
fi

if ! command -v memman >/dev/null 2>&1; then
  echo "[memman] Warning: memman not on PATH; hooks inactive."
  exit 0
fi

echo "$INPUT" | memman prime
exit 0
