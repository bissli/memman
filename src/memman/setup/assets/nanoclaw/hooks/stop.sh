#!/bin/bash
# memman Stop hook — prompt agent to evaluate remembering.
# Returns JSON decision:block so the agent sees the reason and gets
# one more turn. Checks stop_hook_active to prevent infinite loops.

INPUT=$(cat)

if echo "$INPUT" | grep -q '"stop_hook_active"[[:space:]]*:[[:space:]]*true'; then
  exit 0
fi

cat <<'EOF'
{"decision": "block", "reason": "[memman] Memory check: does this exchange contain anything worth storing (user preferences, decisions, corrections, insights, architectural facts)? If yes, call `memman remember \"<self-contained text>\"` directly via Bash in your next turn (no sub-agent delegation). If nothing qualifies, stop without comment."}
EOF
