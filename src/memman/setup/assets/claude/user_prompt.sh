#!/bin/bash
# memman UserPromptSubmit hook - remind agent to recall/remember,
# and reset the stop-hook once-per-turn flag.

INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | sed -n 's/.*"session_id": *"\([^"]*\)".*/\1/p' | head -1)
[ -n "$SESSION_ID" ] && rmdir "$HOME/.memman/stop_fired/$SESSION_ID" 2>/dev/null

echo '[memman] Recall: run memman recall "<focused query>" --limit 5 unless topic is already in context. After responding, evaluate: remember needed?'
