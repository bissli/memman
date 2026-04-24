#!/bin/bash
# memman SessionStart hook - load memory context.
# Reads SessionStart JSON input to detect post-compact restarts.

PROMPT_DIR="${HOME}/.memman/prompt"

if [ -t 0 ]; then
  INPUT='{}'
else
  INPUT=$(cat)
fi
SOURCE=$(echo "$INPUT" | sed -n 's/.*"source": *"\([^"]*\)".*/\1/p' | head -1)

# Clean stale stop-fired flags (older than 2 hours)
find "$HOME/.memman/stop_fired" -mindepth 1 -maxdepth 1 -type d -mmin +120 \
    -exec rmdir {} \; 2>/dev/null

if ! command -v memman >/dev/null 2>&1; then
  echo "[memman] Warning: memman not found in PATH."
  [ -f "${PROMPT_DIR}/guide.md" ] && cat "${PROMPT_DIR}/guide.md"
  exit 0
fi

STATS=$(memman status 2>/dev/null)
if [ -n "$STATS" ]; then
  INSIGHTS=$(echo "$STATS" | sed -n 's/.*"total_insights": *\([0-9]*\).*/\1/p' | head -1)
  EDGES=$(echo "$STATS" | sed -n 's/.*"edge_count": *\([0-9]*\).*/\1/p' | head -1)
  echo "[memman] Memory active (${INSIGHTS:-0} insights, ${EDGES:-0} edges)."
else
  echo "[memman] Memory active."
fi

if [ "$SOURCE" = "compact" ]; then
  SESSION_ID=$(echo "$INPUT" | sed -n 's/.*"session_id": *"\([^"]*\)".*/\1/p' | head -1)
  FLAG="${HOME}/.memman/compact/${SESSION_ID}.json"
  TRIGGER=""
  if [ -f "$FLAG" ]; then
    TRIGGER=$(sed -n 's/.*"trigger":"\([^"]*\)".*/\1/p' "$FLAG" | head -1)
  fi
  echo "[memman] Context was just compacted (${TRIGGER:-auto}). Recall critical context now: memman recall \"<topic>\" --limit 5"
fi

[ -f "${PROMPT_DIR}/guide.md" ] && cat "${PROMPT_DIR}/guide.md"

exit 0
