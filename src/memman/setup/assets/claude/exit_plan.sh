#!/bin/bash
# Notes:
# - Claude Code discards a hook's plain stdout on every event but
#   SessionStart and UserPromptSubmit, so the reminder rides
#   hookSpecificOutput.additionalContext rather than an echo.
# - printf keeps the JSON escapes literal while interpolating the
#   session hint; an unquoted heredoc would expand them.
if [ -t 0 ]; then
  INPUT='{}'
else
  INPUT=$(cat)
fi
SESSION_ID=$(echo "$INPUT" | sed -n 's/.*"session_id": *"\([^"]*\)".*/\1/p' | head -1)
SESSION_HINT=''
[ -n "$SESSION_ID" ] && SESSION_HINT=" --session $SESSION_ID"
printf '{"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": "[memman] Plan-to-execute transition: store any conclusions, decisions, or preferences from this planning session via Bash (memman remember ...%s) before proceeding."}}\n' "$SESSION_HINT"
exit 0
