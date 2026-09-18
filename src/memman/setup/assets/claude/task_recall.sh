#!/bin/bash
# memman PreToolUse hook - recall reminder before a sub-agent launch.
echo "[memman] Before delegating: recall relevant context first (memman recall \"<query>\" --brief --limit 20) unless already done for this topic."
