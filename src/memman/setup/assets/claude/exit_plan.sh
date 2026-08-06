#!/bin/bash
# memman PreToolUse(ExitPlanMode) hook - remind agent to store memories.
if [ -t 0 ]; then
  INPUT='{}'
else
  INPUT=$(cat)
fi
SESSION_ID=$(echo "$INPUT" | sed -n 's/.*"session_id": *"\([^"]*\)".*/\1/p' | head -1)
SESSION_HINT=''
[ -n "$SESSION_ID" ] && SESSION_HINT=" --session $SESSION_ID"
echo "[memman] Plan-to-execute transition: store any conclusions, decisions, or preferences from this planning session via Bash (memman remember ...$SESSION_HINT) before proceeding."
exit 0
