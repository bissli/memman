# memman

Memory persists across sessions. The agent reads it before responding
and writes to it after.

## Recall

Before a new user message, a new task, a topic switch, a design
decision, or a delegation, the agent runs

    memman recall "<focused query>"

The query is focused keywords, never the raw user prompt. The one
exception is a direct follow-up whose topic is already in context.
The page is one line per row, best first: id, score, date, author,
category, then the text. The agent judges each row against the
query, never against a fixed score, and opens one with
`memman insights show <id>`.

## Remember

After responding, the agent stores a user preference, decision, or
correction at once, and any conclusion that would stand if the
exchange stopped here. It defers pure deliberation that has reached
no conclusion.

    memman remember "<self-contained text>" --cat <category> --session $SESSION_ID

One thought per call, as one paragraph that opens on its subject and
keeps its reasoning with it. The text names its subject outright,
with no "this" or "it", and never opens with a label or with who
wrote it or when: `author` and `created_at` carry those.

    BAD   Decision (alice, 2026-09-24): retry cap stays at three.
    GOOD  The retry cap stays at three, since a fourth try only adds load.

memman refuses text over 1,000 bytes, text naming a line number, and
text off this shape, and says why. A behavioral rule ("always",
"never") goes to the project CLAUDE.md `## Directives` section
instead.

A literal `$SESSION_ID` means the host supplied no id: omit the flag.

The memman skill is the full manual: recall triggers, categories,
what never to store, corrections, the write pipeline, and scheduler
controls.
