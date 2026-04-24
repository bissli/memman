---
name: add-memman
description: Add persistent graph-based memory to NanoClaw agents using memman. Agents recall context before responding and remember insights after. Each group gets isolated memory with optional global shared store.
---

# /add-memman

Add [memman](https://github.com/bissli/memman) persistent memory to your NanoClaw installation. After running this skill, every agent session will have access to a per-group memory graph that persists across conversations.

## Architecture

```
Host                              Container
~/.memman/data/{group}/ ──rw──→ /home/node/.memman/data/default/  (private)
~/.memman/data/global/  ──ro──→ /home/node/.memman/data/global/   (shared)
```

Each group gets its own isolated memman store. An optional global store provides shared read-only memory across all groups.

---

## Phase 1: Pre-flight

1. Verify memman is installed on the host:
   ```bash
   memman --version
   ```
   If not installed, follow the README at
   https://github.com/bissli/memman and stop. Do not attempt to
   install memman from this skill.

2. Verify the container image exists:
   ```bash
   docker image inspect nanoclaw-agent:latest >/dev/null 2>&1 && echo "OK"
   ```

3. Fetch the latest memman version for the Dockerfile:
   ```bash
   curl -s https://api.github.com/repos/bissli/memman/releases/latest | grep -o '"tag_name": "v[^"]*"' | cut -d'"' -f4 | sed 's/^v//'
   ```

---

## Phase 2: Apply Code Changes

### 2a. Install memman in the container image

**File**: `container/Dockerfile`

Add the following block **after** the `apt-get install` section and **before** the `npm install -g` line. Replace `<version>` with the version from Phase 1 step 3:

```dockerfile
# Install memman for persistent agent memory
RUN pip install --no-cache-dir git+https://github.com/bissli/memman.git
```

### 2b. Add the container skill

**File**: `container/skills/memman/SKILL.md`

Create this file with the memman container skill content. This skill teaches the agent inside the container when and how to use memman. It should include:

- **Memory stores section**: Explain that the default store is per-group (private, read-write) and the global store is shared (read-only, accessed via `--store global --readonly`).
- **Recall guide**: Default recall on every new user message. Use `memman recall "<query>" --limit 5`. Also check the global store: `memman recall "<query>" --store global --readonly --limit 5`. Craft focused keyword-rich queries.
- **Remember guide**: Decision tree — Step 1: Does this exchange contain a user directive, reasoning conclusion, or durable observed state? Step 2: Does a memory already exist (create/update/skip)? Step 3: Is it worth storing?
- **Workflow**: remember → link (evaluate semantic/causal candidates with judgment) → recall.
- **Commands**: Full memman command reference (remember, link, recall, search, forget, related, gc, status, log).
- **Guardrails**: Never store secrets. Never write to the global store. Categories: preference, decision, insight, fact, context. Max 8,000 chars per insight.

### 2c. Add volume mounts for memman data

**File**: `src/container-runner.ts`

In the function that builds volume mounts (where the existing group folder and Claude session mounts are defined), add two new mounts **after** the Claude sessions mount:

```typescript
// Per-group memman memory store (private, read-write)
const groupMemManDir = path.join(homedir(), '.memman', 'data', group.folder);
fs.mkdirSync(groupMemManDir, { recursive: true });
mounts.push({
  hostPath: groupMemManDir,
  containerPath: '/home/node/.memman/data/default',
  readonly: false,
});

// Global shared memman memory (read-only, optional)
const globalMemManDir = path.join(homedir(), '.memman', 'data', 'global');
if (fs.existsSync(globalMemManDir)) {
  mounts.push({
    hostPath: globalMemManDir,
    containerPath: '/home/node/.memman/data/global',
    readonly: true,
  });
}
```

Adapt the mount syntax to match the existing pattern in `container-runner.ts` (it may use string format like `hostPath:containerPath:ro` or an object format — match whichever the file uses).

**Important**: The `mkdirSync` call ensures the per-group memman directory exists on the host before the container starts, preventing mount failures.

### 2d. Add lifecycle hook scripts

Create `container/hooks/memman/` with four shell scripts. These run inside the container at Claude Code lifecycle events to actively drive memory operations.

**File**: `container/hooks/memman/prime.sh`

```bash
#!/bin/bash
# memman SessionStart hook — report memory stats on session init.
STATS=$(memman status 2>/dev/null)
if [ -n "$STATS" ]; then
  INSIGHTS=$(echo "$STATS" | sed -n 's/.*"total_insights": *\([0-9]*\).*/\1/p' | head -1)
  EDGES=$(echo "$STATS" | sed -n 's/.*"edge_count": *\([0-9]*\).*/\1/p' | head -1)
  echo "[memman] Memory active (${INSIGHTS:-0} insights, ${EDGES:-0} edges)."
else
  echo "[memman] Memory active."
fi
```

**File**: `container/hooks/memman/user_prompt.sh`

```bash
#!/bin/bash
# memman UserPromptSubmit hook — remind agent to evaluate recall/remember.
echo "[memman] Evaluate: recall needed? After responding, evaluate: remember needed?"
```

**File**: `container/hooks/memman/stop.sh`

```bash
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
```

Make all scripts executable: `chmod +x container/hooks/memman/*.sh`

### 2e. Copy hooks into container and register in settings.json

**File**: `container/Dockerfile`

Add after the memman binary install block:

```dockerfile
# Copy memman hook scripts
COPY hooks/memman/ /app/hooks/memman/
RUN chmod +x /app/hooks/memman/*.sh
```

**File**: `src/container-runner.ts`

In the block where `settings.json` is created for each group session (look for `writeFileSync` with `settings.json`), merge memman hooks into the settings object:

```typescript
// Register memman lifecycle hooks
const memmanHooks = {
  SessionStart: [{
    hooks: [{ type: 'command', command: '/app/hooks/memman/prime.sh' }]
  }],
  UserPromptSubmit: [{
    hooks: [{ type: 'command', command: '/app/hooks/memman/user_prompt.sh' }]
  }],
  Stop: [{
    hooks: [{ type: 'command', command: '/app/hooks/memman/stop.sh' }]
  }],
};

// Merge into existing settings.hooks (preserve any existing hooks)
const existingHooks = settings.hooks || {};
for (const [event, entries] of Object.entries(memmanHooks)) {
  existingHooks[event] = [...(existingHooks[event] || []), ...entries];
}
settings.hooks = existingHooks;
```

Adapt this to match the existing settings.json construction pattern in `container-runner.ts`.

---

## Phase 3: Setup

1. Initialize the global shared store on the host (optional — skip if you don't need cross-group shared memory):
   ```bash
   memman store create global
   ```

2. Rebuild the container image:
   ```bash
   ./container/build.sh
   ```

3. Restart the NanoClaw service:
   ```bash
   # macOS with launchd:
   launchctl kickstart -k "gui/$(id -u)/com.nanoclaw"
   # Or manually:
   npm run dev
   ```

---

## Phase 4: Verify

1. Verify memman works inside the container:
   ```bash
   docker run --rm --entrypoint memman nanoclaw-agent:latest --version
   ```

2. Verify memman status inside a running container:
   ```bash
   docker run --rm --entrypoint memman nanoclaw-agent:latest status
   ```

3. Send a test message to the WhatsApp bot and check that the agent mentions memory operations in its reasoning.

4. Verify data persistence on the host:
   ```bash
   ls ~/.memman/data/
   # Should show directories for each active group
   ```

---

## Removal

To remove memman from your NanoClaw installation:

1. Remove from Dockerfile: delete the `ARG MEMMAN_VERSION` + `RUN ... memman` block and the `COPY hooks/memman/` line
2. Remove container skill: `rm -rf container/skills/memman/`
3. Remove hook scripts: `rm -rf container/hooks/memman/`
4. Remove volume mounts from `src/container-runner.ts`: delete the memman mount blocks
5. Remove hooks registration from `src/container-runner.ts`: delete the memman hooks merge in settings.json
6. Rebuild: `./container/build.sh`
7. (Optional) Remove data: `rm -rf ~/.memman/data/`
