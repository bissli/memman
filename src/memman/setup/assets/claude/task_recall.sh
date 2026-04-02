#!/bin/bash
# memman PreToolUse hook - recall reminder before Task agent launches.
echo "[memman] Before delegating: recall relevant context first (memman recall \"<query>\" --limit 5) unless already done for this topic."
