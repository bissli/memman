---
name: memman
description: "Persistent memory CLI for LLM agents. Store facts, recall past knowledge, link related memories, manage lifecycle."
metadata:
  openclaw:
    emoji: "🧠"
    requires:
      bins: ["memman"]
---

# memman

## Install & Configure

### 1. Install memman

**From source (Poetry)**:

```bash
git clone https://github.com/bissli/memman.git && cd memman
make install
```

### 2. Set up OpenClaw integration

```bash
memman setup --target openclaw --yes
```

This single command deploys all components:
- **Skill** → `~/.openclaw/skills/memman/SKILL.md`
- **Hook** → `~/.openclaw/hooks/memman-prime/` (agent:bootstrap — injects behavioral guide)
- **Plugin** → `~/.openclaw/extensions/memman/` (remind, nudge hooks)
- **Prompts** → `~/.memman/prompt/` (guide.md, skill.md)

Restart the OpenClaw gateway to activate.

### 3. Customize (optional)

Edit `~/.memman/prompt/guide.md` to tune recall/remember behavior.

Plugin hooks are configured in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "memman": {
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
memman setup --eject --target openclaw --yes
```

## Workflow

1. **Remember**: `memman remember "<fact>" --cat <cat> --imp <1-5> --entities "e1,e2" --source agent`
   - Diff is built-in: duplicates skipped, conflicts auto-replaced.
   - Output includes `action` (added/updated/skipped/replaced), `enrichment` (keywords, summary, entities), and `edges_created` (temporal, entity, causal).
   - All edge creation, LLM enrichment, and causal inference run inline before `remember` returns.
   - **Replace**: `memman replace <id> "<new content>"` — deterministic replacement by ID. Inherits metadata from original unless overridden. Carries `access_count` forward.
2. **Link** (manual linking when you identify relationships):
   - Syntax: `memman link <id> <target> --type <causal|semantic> --weight <0-1> [--meta '<json>']`
   - For causal links, pass sub_type via `--meta`: `memman link <id> <target> --type causal --meta '{"sub_type": "causes"}'` (values: `causes`, `enables`, `prevents`)
3. **Recall**: `memman recall "<query>" --limit 10`

## Commands

```bash
memman remember "<fact>" --cat <cat> --imp <1-5> --entities "e1,e2" --source agent
memman link <id1> <id2> --type <type> --weight <0-1> [--meta '<json>']
memman recall "<query>" --limit 10
memman search "<query>" --limit 10
memman replace <id> "<new content>" [--cat] [--imp] [--tags] [--entities] [--source]
memman forget <id>
memman related <id> --edge causal
memman gc --threshold 0.4
memman gc --keep <id>
memman graph rebuild
memman graph reindex
memman status
memman doctor
memman log [--since 7d] [--group-by operation] [--stats]
memman store list
memman store create <name>
memman store set <name>
memman store remove <name>
```

## Guardrails

- Use the `exec` tool to run memman commands.
- Do not store secrets, passwords, or tokens.
- Categories (`--cat`):
  - `preference` — user-stated likes, dislikes, style choices ("I prefer X over Y")
  - `decision` — architectural/design choices with rationale ("chose X because Y")
  - `fact` — discovered truths about systems, tools, domains
  - `insight` — non-trivial conclusions from multi-source reasoning
  - `context` — background about user, project, environment
- Edge types: `temporal` · `semantic` · `causal` · `entity`
- Max 8,000 chars per insight.
