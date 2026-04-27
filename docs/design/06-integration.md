# 6. LLM CLI Integration

[< Back to Design Overview](../DESIGN.md)

---

![Integration Architecture](../diagrams/08-three-layer-integration.drawio.png)

MemMan integrates with LLM CLIs through lifecycle hooks, a skill file, and a behavioral guide. Claude Code's [hook system](https://docs.anthropic.com/en/docs/claude-code/hooks) is the reference implementation — all components are deployed automatically via `memman install`.

## 6.1 Integration Architecture

Six hooks drive the memory lifecycle:

```
Session starts
    │
    ▼
  Prime (SessionStart) ─── prime.sh ──→ load guide.md (memory execution manual)
    │
    ▼
  User sends message
    │
    ▼
  Remind (UserPromptSubmit) ─── user_prompt.sh ──→ remind agent to recall & remember
    │
    ▼
  Skill (SKILL.md) ── command syntax reference (auto-discovered)
    │
    ▼
  LLM generates response (following guide.md behavioral rules)
    │
    ▼
  Nudge (Stop) ─── stop.sh ──→ remind agent to remember
    │
    ▼
  (when context compacts)
  Compact (PreCompact) ─── compact.sh ──→ flag file for post-compact recall
    │
    ▼
  (before invoking the Task tool)
  Recall (PreToolUse) ─── task_recall.sh ──→ remind agent to recall before delegation
    │
    ▼
  (before exiting plan mode)
  ExitPlan (PreToolUse) ─── exit_plan.sh ──→ prompt memory storage before transition
```

Three layers work together:

| Layer     | What                                                                        | Where                                       | Role                                                                                                                                                |
| --------- | --------------------------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hooks** | Shell scripts triggered by Claude Code lifecycle events                     | `.claude/hooks/memman/`                     | Prime (guide), Remind (recall & remember), Nudge (remember), Compact (pre-compact bridge), Recall (pre-delegation), ExitPlan (plan-mode transition) |
| **Skill** | `SKILL.md` — command reference in Claude Code skill format                  | `.claude/skills/memman/`                    | Teaches the LLM *how* to use memman commands                                                                                                        |
| **Guide** | `guide.md` — detailed execution manual for recall, remember, and delegation | Installed package (read via `memman guide`) | Teaches the LLM *when* to recall, *what* to remember, and *how* to delegate                                                                         |

## 6.2 Hook Details

Claude Code fires hooks at specific lifecycle events. MemMan registers up to six, each with a distinct role in the memory lifecycle:

**Prime (SessionStart) — `prime.sh`**

Runs once when a session starts. Delegates to `memman prime`, which emits the status line, a compact-recall hint when `SessionStart.source == 'compact'`, and the shipped behavioral guide via `importlib.resources`:

```bash
if ! command -v memman >/dev/null 2>&1; then
  echo "[memman] Warning: memman not on PATH; hooks inactive."
  exit 0
fi
echo "$INPUT" | memman prime
```

The hook's stdout is injected into the agent's context before the first user turn. The guide content (recall/remember policy, phase awareness, Tier A/B rules) sets behavior for the entire session.

**Remind (UserPromptSubmit) — `user_prompt.sh`**

Runs on every user message. A lightweight prompt that reminds the agent to evaluate whether recall and remember are needed before starting work:

```bash
echo '[memman] Recall: run memman recall "<focused query>" --limit 5 unless topic is already in context. After responding, evaluate: remember needed?'
```

The agent decides whether to act on this reminder based on the guide.md rules — it is a suggestion, not forced execution.

**Nudge (Stop) — `stop.sh`**

Runs after each LLM response. Returns a `decision: block` JSON so the agent gets one more turn to evaluate memory. Directive-aware: prompts the agent to store if a user preference, decision, or conclusion emerged. Fires once per user turn (gated by a `stop_fired/` directory lock) and stays silent when `stop_hook_active` is true (preventing infinite loops). Simplified excerpt:

```bash
INPUT=$(cat)
if echo "$INPUT" | grep -q '"stop_hook_active"[[:space:]]*:[[:space:]]*true'; then
  exit 0
fi
cat <<'EOF'
{"decision": "block", "reason": "[memman] Memory check: did the user state a preference, make a decision, give a correction, or reach a conclusion? If yes, call `memman remember \"<self-contained text>\"` directly via Bash in your next turn (no sub-agent, no delegation). Dereference anaphora before storing. Only skip if the exchange was purely open-ended questions with no resolution."}
EOF
```

**Compact (PreCompact + SessionStart) — `compact.sh` + `prime.sh` (optional)**

A two-part bridge that preserves memory context across context compaction. PreCompact cannot inject context into the agent's conversation (stdout is verbose-mode only), so the solution uses a flag file relay:

1. `compact.sh` fires at PreCompact — writes a flag file to `~/.memman/compact/<session_id>.json` with the trigger type and timestamp
2. After compaction, Claude Code fires SessionStart with `source=compact`
3. `prime.sh` detects `source=compact`, reads the flag file for enrichment, and injects a recall instruction the agent can see

The design is defensively layered — `prime.sh` detects compaction from the SessionStart `source` field regardless of whether `compact.sh` ran. The flag file enriches the message with trigger type but is not required.

```bash
# compact.sh (PreCompact) — writes flag file
cat > "${COMPACT_DIR}/${SESSION_ID}.json" <<EOF
{"trigger":"${TRIGGER:-auto}","ts":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF

# prime.sh (SessionStart) — detects compact source, injects recall
if [ "$SOURCE" = "compact" ]; then
  echo "[memman] Context was just compacted (${TRIGGER:-auto}). Recall critical context now."
fi
```

**Recall (PreToolUse) — `task_recall.sh` (optional)**

Fires before the agent delegates to a sub-agent. Reminds the agent to recall relevant context before delegation, ensuring sub-agents receive informed prompts:

```bash
echo "[memman] Before delegating: recall relevant context first (memman recall \"<query>\" --limit 5) unless already done for this topic."
```

**ExitPlan (PreToolUse) — `exit_plan.sh` (optional)**

Fires before the agent exits plan mode. Outputs an advisory reminder to store memories before the plan-to-execute transition. Non-blocking — the agent always proceeds:

```bash
echo "[memman] Plan-to-execute transition: store any conclusions, decisions, or preferences from this planning session via Bash (memman remember ...) before proceeding."
exit 0
```

Stale `stop_fired/` directories (older than 2 hours) are cleaned up by `prime.sh` at session start.

## 6.3 Automated Setup

`memman install` deploys everything via symlinks to the installed package, so `pipx upgrade memman` refreshes hook scripts and SKILL.md automatically — no re-run needed:

```
$ memman install

Detecting LLM CLI environments...
  ✓ Claude Code (v1.x)    ~/.claude/

Setting up Claude Code (~/.claude/)...

[1/2] Skill
  ✓ Skill     ~/.claude/skills/memman/SKILL.md

[2/2] Hooks
  ✓ Hook: prime     ~/.claude/hooks/memman/prime.sh
  ✓ Hook: remind    ~/.claude/hooks/memman/user_prompt.sh
  ✓ Hook: nudge     ~/.claude/hooks/memman/stop.sh
  ✓ Hook: compact   ~/.claude/hooks/memman/compact.sh
  ✓ Hook: recall    ~/.claude/hooks/memman/task_recall.sh
  ✓ Hook: exit_plan ~/.claude/hooks/memman/exit_plan.sh
  ✓ Settings         ~/.claude/settings.json (updated)
  ✓ Permission       Bash(memman:*) added to settings.json

Setup complete!
  Hooks   prime, remind, nudge, compact, recall, exit_plan

Start a new Claude Code session to activate.
```

Deployment model:

- `~/.claude/skills/memman/SKILL.md` → symlink into the installed package's `memman/setup/assets/claude/SKILL.md`.
- `~/.claude/hooks/memman/*.sh` → symlinks into the same package path. `prime.sh` is a thin shim that delegates to `memman prime` (status + compact hint + guide in one Python call).
- Shipped `guide.md` is never deployed to disk — `memman guide` reads it from the package via `importlib.resources` every time `prime.sh` fires.

`pipx upgrade memman` refreshes hook scripts and `SKILL.md` automatically through the symlinks; `guide.md` is read live from the new package version. Asset-only changes propagate without any re-install step.

Key install options:

| Command / Flag                        | Effect                           |
| ------------------------------------- | -------------------------------- |
| `memman install --target claude-code` | Install into `~/.claude/` only   |
| `memman install --target openclaw`    | Install into `~/.openclaw/` only |
| `memman install --target nanoclaw`    | Install into `~/.nanoclaw/` only |
| `memman uninstall`                    | Remove all memman integrations   |
| `memman uninstall --target <name>`    | Remove from a single environment |

`memman uninstall` never deletes anything under `~/.memman/` — memory store, API-key env file, caches, queue, and scheduler logs all survive. To fully remove the binary: `pipx uninstall memman`.

The Prime hook is always installed. Remind, Nudge, Compact, Recall, and ExitPlan hooks are optional (all enabled by default).

## 6.4 Direct-Bash Invocation (No Sub-Agent)

The host agent calls `memman remember` directly via Bash in the same turn —
no sub-agent, no Task delegation, no context isolation. This is intentional:

- **The binary is a fast queue-append** (~50 ms). The cost that would
  justify offloading to a sub-agent (LLM extraction, embedding, edge
  inference) doesn't run in-band — it runs in the scheduler worker
  out of band. The host turn pays only the queue-append latency.
- **The host LLM already holds the context** needed to choose the right
  `--cat`, `--imp`, `--entities`, and to dereference anaphora before
  storing. A sub-agent would pay tokens to re-read context the host
  already has.
- **One-way visibility** (writes are not recallable in the same turn)
  means there's no callback the sub-agent could provide that the host
  couldn't get itself. Recall remains a separate Bash call.

The shipped `guide.md` enforces this rule explicitly: *"call `memman
remember "<self-contained text>"` directly via Bash in your current
turn. No sub-agent delegation."*

## 6.5 Adapting to Other LLM CLIs

For CLIs with hook support, replicate the Claude Code pattern: register lifecycle hooks that call memman commands, deploy the skill file, and provide the behavioral guide.

For CLIs without hook support, merge the recall/remember guidance into the corresponding system prompt or rules file:

- OpenClaw -> `memman install --target openclaw` deploys skill + guide, but hooks require manual plugin configuration
- Others -> System prompt / rules file
