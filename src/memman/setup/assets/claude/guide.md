# memman

Memory persists across sessions. The agent reads it before responding
and writes to it after.

## Recall

On a new user message, a new task, a topic switch, a design decision,
or a delegation, the agent first runs

    memman recall "<focused query>"

The query is focused keywords, never the raw user prompt. The
exception is a direct follow-up whose topic is already in context.
The agent judges each row against the query, never against a fixed
score, and opens one with `memman insights show <id>`.

## Remembering

A finding is worth storing when a later session without it would
redo work, repeat a mistake, or ask the user again. Anything else
stays out, however true.

Use cases of memories worth storing:

- a user preference, only when it changes future work and no file
  that loads every session already holds it
- a decision about the work, with its reason
- a lasting fact about a system, project, or tool
- a root cause and its fix, stated as the resulting fact
- an experiment's result and what it means
- an approach ruled out, with the reason

The agent stores a user correction at once, and any other finding
as soon as it is worth storing. Progress, next steps, plans, open
questions, working-tree state, and receipts never go in: the agent
stores the decision, fact, or result the work reached.

    memman remember "<self-contained text>"

Every write states the claim that holds, never the event that
changed it.

One thought per call, as one paragraph that opens on its subject and
keeps its reasoning with it. The text names its subject outright,
with no "this" or "it", and never opens with a label or with who
wrote it or when: `author` and `created_at` carry those.

    BAD   Decision (alice, 2026-09-24): retry cap stays at three.
    GOOD  The retry cap stays at three, since a fourth try only adds load.

A correction, stored at once, replaces the row it corrects. The new
text restates every claim of the old row still true.

    memman replace <id> "<corrected text>"

The id is from a recall page or an earlier write's reply, even a
write still queued. With neither, recall the topic first.

`remember` replies with `related`: at most three rows sharing the
most words with the new text. The agent acts only on a row with an
outdated sentence. The new row is already queued, so `memman forget
<id>` retires the outdated row when the new row holds all its
still-true claims. Otherwise `replace` it with only those claims.
When every listed row is outdated, the agent recalls the topic.

memman refuses text over 1,000 bytes and says why. A behavioral rule
("always", "never") goes to the project CLAUDE.md `## Directives`
section instead.

The memman skill is the full manual: what never to store,
corrections, `related`, pipeline, scheduler.
