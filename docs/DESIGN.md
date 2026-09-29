# Design and architecture

memman gives Claude Code persistent, searchable memory. The agent decides what to save, recall, replace, or forget. A background worker prepares queued memories for search, and the CLI retrieves them on demand.

This guide explains the implementation and the reasons behind it. The [usage guide](USAGE.md) covers installation and commands.

## Reading guide

| Chapter                                                    | What it explains                                                      |
| ---------------------------------------------------------- | --------------------------------------------------------------------- |
| [1. Background](design/01-background.md)                   | The problem, scope, and main design trade-offs.                       |
| [2. Core concepts and architecture](design/02-concepts.md) | Memory records, stores, schemas, modules, and data paths.             |
| [3. Read and write pipelines](design/03-pipelines.md)      | Queued writes, retries, enrichment, search ranking, and model checks. |
| [4. Lifecycle and embedding](design/04-lifecycle.md)       | Retention, embedding model bindings, recovery, and model changes.     |
| [5. Claude Code integration](design/05-integration.md)     | Hooks, agent instructions, installation, and upgrades.                |

The chapters build on each other in order, and each links to the command reference.

## Terms used here

| Term                  | Meaning                                                                         |
| --------------------- | ------------------------------------------------------------------------------- |
| Memory                | One saved claim. Called an **insight** in the database and inspection commands. |
| Store                 | A named collection of memories with its own database file or schema.            |
| Drain                 | One worker run that processes queued writes.                                    |
| Enrichment            | A model-generated summary; the original memory text stays unchanged.            |
| Embedding fingerprint | The provider, model, and vector dimension recorded for a store.                 |
| Current memory        | A memory that has been neither replaced nor forgotten.                          |
