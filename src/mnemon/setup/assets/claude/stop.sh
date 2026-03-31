#!/bin/bash
# mnemon Stop hook — evaluate memory storage once per user turn.
# Uses atomic mkdir to fire only on first Stop after each user message.
# Reset via user_prompt.sh rmdir on new human input.

INPUT=$(cat)

# Prevent infinite loop on memory-eval re-entry
if echo "$INPUT" | grep -q '"stop_hook_active"[[:space:]]*:[[:space:]]*true'; then
  exit 0
fi

# Extract session_id (sed pattern matches existing compact.sh/prime.sh/exit_plan.sh)
SESSION_ID=$(echo "$INPUT" | sed -n 's/.*"session_id": *"\([^"]*\)".*/\1/p' | head -1)

if [ -z "$SESSION_ID" ]; then
  # Can't determine session — fall back to always blocking (safe default)
  cat <<'EOF'
{"decision": "block", "reason": "[mnemon] Memory check: did the user state a preference, make a decision, give a correction, or reach a conclusion? If yes, store via Agent(model=sonnet) sub-agent. Only skip if the exchange was purely open-ended questions with no resolution."}
EOF
  exit 0
fi

# Ensure parent dir exists (first run)
mkdir -p "$HOME/.mnemon/stop_fired" 2>/dev/null

# Atomic gate: only first mkdir succeeds per user turn
if mkdir "$HOME/.mnemon/stop_fired/$SESSION_ID" 2>/dev/null; then
  cat <<'EOF'
{"decision": "block", "reason": "[mnemon] Memory check: did the user state a preference, make a decision, give a correction, or reach a conclusion? If yes, store via Agent(model=sonnet) sub-agent. Only skip if the exchange was purely open-ended questions with no resolution."}
EOF
else
  # Flag already set — subsequent stop in same turn, skip silently
  exit 0
fi
