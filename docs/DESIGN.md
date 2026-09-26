# memman - Design & Architecture

memman is persistent memory for Claude Code, supervised by the agent. Five chapters cover the reasons for memman's design, its data model, its write and recall pipelines, the memory lifecycle, and its Claude Code integration.

---

## Chapters

| #   | Chapter                        | File                                          | Topics                                                                                 |
| --- | ------------------------------ | --------------------------------------------- | -------------------------------------------------------------------------------------- |
| 1   | Background                     | [01-background.md](design/01-background.md)   | Context loss, scope, the LLM-supervised pattern, retrieval, design trade-offs, storage |
| 2   | Core Concepts and Architecture | [02-concepts.md](design/02-concepts.md)       | The memory record, schema, system architecture, data directory, store isolation        |
| 3   | Read & Write Pipelines         | [03-pipelines.md](design/03-pipelines.md)     | The turn and the worker, write pipeline, LLM calls, recall, handling model changes     |
| 4   | Lifecycle & Embedding          | [04-lifecycle.md](design/04-lifecycle.md)     | Retention, inspecting memories, embedding providers, embedding swap and re-embed       |
| 5   | Claude Code Integration        | [05-integration.md](design/05-integration.md) | Integration layers, hooks, automated setup, direct Bash calls                          |
