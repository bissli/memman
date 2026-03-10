#!/bin/bash
# mnemon PreToolUse(ExitPlanMode) hook — prompt memory storage before
# leaving plan mode. Exit 2 blocks the tool; stderr becomes agent feedback.
# Flag file ensures the second attempt (after agent stores memories) passes.

INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | sed -n 's/.*"session_id": *"\([^"]*\)".*/\1/p' | head -1)

FLAG_DIR="${HOME}/.mnemon/exit_plan"
FLAG="${FLAG_DIR}/${SESSION_ID}.flag"

if [ -f "$FLAG" ]; then
    rm -f "$FLAG"
    exit 0
fi

mkdir -p "$FLAG_DIR"
touch "$FLAG"

echo "[mnemon] Plan-to-execute transition: before exiting plan mode, evaluate this planning session for memories. Did the user state preferences, make decisions, reach conclusions, or discuss architectural choices? If yes, store now via Bash (mnemon remember ...) -- context will be cleared after this point. Only skip if the session was purely exploratory with no conclusions." >&2
exit 2
