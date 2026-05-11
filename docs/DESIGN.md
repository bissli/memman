# memman — Design & Architecture

memman is an LLM-supervised persistent memory store for LLM agents. This document indexes the six chapters that cover its data model, retrieval, write pipeline, lifecycle, and integration surface.

---

## Chapters

| # | Chapter             | File                                          | Topics                                                                         |
| --- | ------------------- | --------------------------------------------- | ------------------------------------------------------------------------------ |
| 1 | Background          | [01-background.md](design/01-background.md)   | Amnesia, LLM-Supervised pattern, MAGMA + RRF, deviations, storage pluggability |
| 2 | Core Concepts       | [02-concepts.md](design/02-concepts.md)       | Insight/Edge model, schema, system layers, store isolation                     |
| 3 | Graph Model         | [03-graph-model.md](design/03-graph-model.md) | Four edge types, intent weights, thresholds                                    |
| 4 | Read & Write        | [04-pipelines.md](design/04-pipelines.md)     | Deferred queue, scheduler, recall, cross-encoder rerank                        |
| 5 | Lifecycle & Embed   | [05-lifecycle.md](design/05-lifecycle.md)     | EI decay, immunity, pruning, embedder sovereignty, swap                        |
| 6 | LLM CLI Integration | [06-integration.md](design/06-integration.md) | Lifecycle hooks, skill, guide, install, alt CLIs                               |
