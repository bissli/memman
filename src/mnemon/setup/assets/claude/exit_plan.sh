#!/bin/bash
# mnemon PreToolUse(ExitPlanMode) hook — remind agent to store memories.
cat > /dev/null
echo "[mnemon] Plan-to-execute transition: store any conclusions, decisions, or preferences from this planning session via Bash (mnemon remember ...) before proceeding."
exit 0
