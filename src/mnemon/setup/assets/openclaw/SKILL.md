---
name: mnemon
description: "Persistent memory CLI for LLM agents. Store facts, recall past knowledge, link related memories, manage lifecycle."
metadata:
  openclaw:
    emoji: "🧠"
    requires:
      bins: ["mnemon"]
---

# mnemon

## Install & Configure

### 1. Install mnemon

**From source (Poetry)**:

```bash
git clone https://github.com/bissli/mnemon.git && cd mnemon
make install
```

### 2. Set up OpenClaw integration

```bash
mnemon setup --target openclaw --yes
```

This single command deploys all components:
- **Skill** → `~/.openclaw/skills/mnemon/SKILL.md`
- **Hook** → `~/.openclaw/hooks/mnemon-prime/` (agent:bootstrap — injects behavioral guide)
- **Plugin** → `~/.openclaw/extensions/mnemon/` (remind, nudge hooks)
- **Prompts** → `~/.mnemon/prompt/` (guide.md, skill.md)

Restart the OpenClaw gateway to activate.

### 3. Customize (optional)

Edit `~/.mnemon/prompt/guide.md` to tune recall/remember behavior.

Plugin hooks are configured in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "mnemon": {
        "enabled": true,
        "config": {
          "remind": true,
          "nudge": true
        }
      }
    }
  }
}
```

| Hook     | Default | Description                                             |
| -------- | ------- | ------------------------------------------------------- |
| `remind` | on      | Recall relevant memories + remind agent on each message |
| `nudge`  | on      | Suggest remember sub-agent after each reply             |

### 4. Uninstall

```bash
mnemon setup --eject --target openclaw --yes
```

## Workflow

1. **Remember**: `mnemon remember "<fact>" --cat <cat> --imp <1-5> --entities "e1,e2" --source agent`
   - Diff is built-in: duplicates skipped, conflicts auto-replaced.
   - Output includes `action` (added/updated/skipped/replaced), `enrichment` (keywords, summary, entities), and `edges_created` (temporal, entity, causal).
   - All edge creation, LLM enrichment, and causal inference run inline before `remember` returns.
   - **Replace**: `mnemon replace <id> "<new content>"` — deterministic replacement by ID. Inherits metadata from original unless overridden. Carries `access_count` forward.
2. **Link** (manual linking when you identify relationships):
   - Syntax: `mnemon link <id> <target> --type <causal|semantic> --weight <0-1> [--meta '<json>']`
   - For causal links, pass sub_type via `--meta`: `mnemon link <id> <target> --type causal --meta '{"sub_type": "causes"}'` (values: `causes`, `enables`, `prevents`)
3. **Recall**: `mnemon recall "<query>" --limit 10`

## Commands

```bash
mnemon remember "<fact>" --cat <cat> --imp <1-5> --entities "e1,e2" --source agent
mnemon link <id1> <id2> --type <type> --weight <0-1> [--meta '<json>']
mnemon recall "<query>" --limit 10
mnemon search "<query>" --limit 10
mnemon replace <id> "<new content>" [--cat] [--imp] [--tags] [--entities] [--source]
mnemon forget <id>
mnemon related <id> --edge causal
mnemon gc --threshold 0.4
mnemon gc --keep <id>
mnemon graph link
mnemon status
mnemon log [--since 7d] [--group-by operation] [--stats]
mnemon store list
mnemon store create <name>
mnemon store set <name>
mnemon store remove <name>
```

## Guardrails

- Use the `exec` tool to run mnemon commands.
- Do not store secrets, passwords, or tokens.
- Categories (`--cat`):
  - `preference` — user-stated likes, dislikes, style choices ("I prefer X over Y")
  - `decision` — architectural/design choices with rationale ("chose X because Y")
  - `fact` — discovered truths about systems, tools, domains
  - `insight` — non-trivial conclusions from multi-source reasoning
  - `context` — background about user, project, environment
- Edge types: `temporal` · `semantic` · `causal` · `entity`
- Max 8,000 chars per insight.
