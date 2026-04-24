#!/bin/bash
# memman SessionStart hook — thin shim that delegates to `memman prime`.
# The `memman prime` command reads SessionStart JSON on stdin, emits
# the status line, compact-recall hint (when applicable), and the
# behavioral guide (shipped + ~/.memman/prompt/guide.local.md overrides).

if [ -t 0 ]; then
  INPUT='{}'
else
  INPUT=$(cat)
fi

find "$HOME/.memman/stop_fired" -mindepth 1 -maxdepth 1 -type d -mmin +120 \
    -exec rmdir {} \; 2>/dev/null

if ! command -v memman >/dev/null 2>&1; then
  echo "[memman] Warning: memman not on PATH; hooks inactive."
  exit 0
fi

echo "$INPUT" | memman prime
exit 0
