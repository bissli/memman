# 1. Vision & Problem

[< Back to Design Overview](../DESIGN.md)

---

## 1.1 The Amnesia Problem

LLM agents suffer from three memory deficiencies:

- **Context compression loss**: After compaction or automatic compression, prior decisions and context are lost
- **Cross-session forgetting**: Each new session starts from scratch
- **Long-session decay**: Once the context window fills, early information is pushed out of attention range

Users must repeatedly restate preferences, re-explain project context, and re-derive conclusions already reached.

## 1.2 Mnemon's Goal

Make an LLM remember decisions, preferences, and project context across arbitrarily many sessions.

Mnemon is not a library embedded within an agent framework. It is a standalone memory engine — callable via the command line by Claude Code, Cursor, or any LLM CLI.
