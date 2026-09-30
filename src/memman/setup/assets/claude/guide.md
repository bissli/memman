# memman

Memory persists across sessions. The agent reads it before responding
and writes to it after.

## Recall

Before a new user message, a new task, a topic switch, a design
decision, or a delegation, the agent runs

    memman recall "<focused query>"

The query is focused keywords, never the raw user prompt. The
exception is a direct follow-up whose topic is already in context.
The agent judges each row against the query, never against a fixed
score, and opens one with `memman insights show <id>`.

## Remember

After responding, the agent stores a user preference or decision at
once, and any conclusion that would stand if the exchange stopped
here. It defers deliberation that has reached no conclusion.

    memman remember "<self-contained text>"

One thought per call, as one paragraph that opens on its subject and
keeps its reasoning with it. The text names its subject outright,
with no "this" or "it", and never opens with a label or with who
wrote it or when: `author` and `created_at` carry those.

    BAD   Decision (alice, 2026-09-24): retry cap stays at three.
    GOOD  The retry cap stays at three, since a fourth try only adds load.

A correction, stored at once, replaces the row it corrects. A write
saying something changed (a migration ran, a value moved, a step
finished) corrects the row that stated the old state. The new text
restates every claim of the old row still true.

    memman replace <id> "<corrected text>"

The id is on a recall page or is the `id` an earlier write printed,
even one still queued. Without either, recall the topic first.

`remember` replies with `related`, rows sharing the most words with
the new text. The agent acts only on a row with a sentence now false:
the new row is already queued, so `memman forget <id>` retires that
row when the new row holds all its still-true claims, else `replace`
it with only those claims. When every listed row is stale, the agent
recalls the topic: at most three are listed.

memman refuses text over 1,000 bytes and says why. A behavioral rule
("always", "never") goes to the project CLAUDE.md `## Directives`
section instead.

The memman skill is the full manual: what never to store,
corrections, `related`, pipeline, scheduler.
