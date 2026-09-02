#!/bin/bash
# memman UserPromptSubmit hook - remind agent to recall/remember,
# and reset the stop-hook once-per-turn flag.

INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | sed -n 's/.*"session_id": *"\([^"]*\)".*/\1/p' | head -1)
[ -n "$SESSION_ID" ] && rmdir "$HOME/.memman/stop_fired/$SESSION_ID" 2>/dev/null

SESSION_HINT=''
[ -n "$SESSION_ID" ] && SESSION_HINT=" Pass --session $SESSION_ID on every recall/remember/replace."
echo '[memman] Recall: run memman recall "<focused query>" --brief --limit 20 unless topic is already in context. Rows are relevance-ordered - judge each against the query, not against a fixed score. After responding, evaluate: remember needed?'"$SESSION_HINT"
