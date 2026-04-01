# 1. Vision & Problem

[< Back to Design Overview](../DESIGN.md)

---

## 1.1 The "Amnesia" Problem of LLMs

LLM agents suffer from three critical memory deficiencies:

- **Context compression loss**: After `/compact` or automatic compression, all prior decisions, discoveries, and context are lost
- **Cross-session forgetting**: Each new session starts from scratch, with no knowledge of previous sessions
- **Long-session decay**: Once the context window fills up, critical early information is pushed out of the attention range

For a digital assistant that needs to "continuously learn the user's thinking and become an extension of the user," these three deficiencies mean users must repeatedly restate preferences, re-explain project context, and re-derive conclusions already reached.

## 1.2 Structural Bottlenecks of Traditional Approaches

Existing RAG/Memory solutions have fundamental design limitations:

1. **Memory is an afterthought** — its lifecycle is tied to the agent session, not an independent entity
2. **Writing is reactive** — summaries are extracted after conversation ends, losing structural information
3. **Retrieval is flat** — relying solely on vector similarity, unable to express temporal/causal/contradictory relationships
4. **No forgetting mechanism** — either remember everything or TTL-based blanket expiration, no intelligent decay
5. **Heavy dependencies** — requires external databases, complex network configurations

## 1.3 Mnemon's Mission

Mnemon's goal is: **to make an LLM remember your decisions, understand your preferences, and track project context like an experienced assistant — across arbitrarily many sessions.**

It is not a library or plugin embedded within an agent framework, but a standalone memory engine — callable via the command line by Claude Code, Cursor, or any LLM CLI.

