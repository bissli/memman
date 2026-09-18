#!/bin/bash
# memman UserPromptSubmit hook - remind agent to recall.

INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | sed -n 's/.*"session_id": *"\([^"]*\)".*/\1/p' | head -1)

SESSION_HINT=''
[ -n "$SESSION_ID" ] && SESSION_HINT=" --session $SESSION_ID"
echo '[memman] Recall: memman recall "<focused query>" --brief --limit 20'"$SESSION_HINT"
