#!/bin/bash
# Notes:
# - Claude Code discards a hook's plain stdout on every event but
#   SessionStart and UserPromptSubmit, so the reminder rides
#   hookSpecificOutput.additionalContext rather than an echo.
# - PreToolUse fires once the brief is already written, so the text
#   aims at the next delegation rather than this one.
cat <<'EOF'
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": "[memman] Memory was not checked before this delegation. Run memman recall \"<focused query>\" --brief --limit 20 and carry anything relevant into the next one."}}
EOF
