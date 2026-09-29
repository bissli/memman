---
name: memman
description: Recall and save persistent project knowledge with memman. Use for past decisions, user preferences, lessons from earlier sessions, or explicit requests to remember or correct a memory.
---

# Persistent memory

Run `memman` through the shell. Before work that depends on earlier decisions
or preferences, recall the relevant topic. Save durable conclusions from the
task when useful for future sessions.

```bash
memman recall "billing retry decisions"
memman remember "The billing retry cap stays at three because a fourth try only adds load." --cat decision
memman replace <id> "The billing retry cap is now four because the upstream timeout was reduced."
memman insights show <id> --history
```

Each write is one self-contained claim on one line, at most 1,000 UTF-8 bytes.
Categories are `fact` (default), `decision`, `preference`, `insight`, and
`context`. Store reasoning, resolutions, and user preferences; avoid transient
status, verification receipts, secrets, and facts easily recovered from code
or configuration. Behavioral instructions needing guaranteed loading belong
in the appropriate `AGENTS.md`, within the user's requested scope.

Recall returns memory IDs. Correct an existing claim with `replace`, using its
ID or unambiguous prefix; `remember` adds a row and leaves the old claim active.
A replacement must retain any still-true claims of the old row. If the ID is
unknown, recall the topic first. `forget <id>` removes a memory that should no
longer exist; it cannot remove a write that is still queued or a current
replacement whose predecessor has not been forgotten. Follow the command's
error guidance; use `replace` when the row needs a correction.

Writes return an ID immediately and become searchable after the background
worker drains the queue. A queued write can already be corrected with
`replace`. A `related` list in the write response contains potentially related
memories; only correct one when the new information makes it outdated.

Use `memman --store NAME ...` only when a specific store is needed, since any
global option before the verb needs approval. Otherwise the process's
`MEMMAN_STORE`, then the user's configured default, selects it.
Codex must inherit `MEMMAN_STORE` in its shell environment for that selection
to apply. Store names contain letters, digits, dashes, or underscores and
start with a letter or digit; they are not filesystem paths.
Use the same store for recall and writes. For a custom installation, supply
`--data-dir PATH` before the subcommand or use `MEMMAN_DATA_DIR`.

If a command fails, report the failure rather than claiming a memory was
saved. `memman scheduler status` checks the worker; a stopped worker rejects
writes. `memman recall "topic" --basic` searches text without semantic recall.
Use `memman <command> --help` for additional options. Codex runs `memman
doctor`, `forget`, `insights review`, `insights show`, `recall`, `remember`,
`replace`, and `status` without approval. Any other command, and any call
with a global option before the verb, needs approval. A memman call that runs
inside the sandbox fails.
