# memman

Memory persists across sessions. The agent reads it before responding
and writes to it after.

## Recall

Before a new user message, a new task, a topic switch, a design
decision, or a delegation, the agent runs

    memman recall "<focused query>" --brief --limit 20 --session $SESSION_ID

The query is focused keywords, never the raw user prompt. The one
exception is a direct follow-up whose topic is already in context.
Rows come back in relevance order. The agent judges each row against
the query, never against a fixed score, and opens one with
`memman insights show <id>`.

## Remember

After responding, the agent stores a user preference, decision, or
correction at once, and any conclusion that would stand if the
exchange stopped here. It defers pure deliberation that has reached
no conclusion.

    memman remember "<self-contained text>" --cat <category> --session $SESSION_ID

The text names its subject outright, with no "this" or "it". One
claim per call. memman refuses text over 1,000 bytes. A behavioral
rule ("always", "never") goes to the project CLAUDE.md
`## Directives` section instead.

A literal `$SESSION_ID` means the host supplied no id: omit the flag.

The memman skill is the full manual: recall triggers and response
keys, categories, what never to store, corrections, the write
pipeline, and scheduler controls.
