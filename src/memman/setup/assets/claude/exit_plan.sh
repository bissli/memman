#!/bin/bash
# Notes:
# - Claude Code discards a hook's plain stdout on every event but
#   SessionStart and UserPromptSubmit, so the reminder rides
#   hookSpecificOutput.additionalContext rather than an echo.
# - printf keeps the JSON escapes literal; an unquoted heredoc would
#   expand them.
if [ ! -t 0 ]; then
  cat > /dev/null
fi
printf '{"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": "[memman] Plan-to-execute transition: store any conclusions, decisions, or preferences from this planning session via Bash (memman remember ...) before proceeding."}}\n'
exit 0
