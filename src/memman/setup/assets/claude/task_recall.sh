#!/bin/bash
# Notes:
# - Claude Code discards a hook's plain stdout on every event but
#   SessionStart and UserPromptSubmit, so the reminder goes in
#   hookSpecificOutput.additionalContext.
# - PreToolUse runs after the brief is written, so the text aims at
#   the next delegation.
cat <<'EOF'
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": "[memman] Before the next delegation, run memman recall \"<focused query>\" and carry anything relevant into its brief."}}
EOF
