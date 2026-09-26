---
name: Bug Report
about: A bug in memman
title: ''
labels: bug
assignees: ''
---

**Bug description**
What goes wrong.

**To reproduce**
1. Run `memman ...`
2. See the error

**Expected behavior**
The behavior expected instead.

**Environment**
- OS: [e.g. macOS 15.3, Ubuntu 24.04]
- memman version: [output of `memman --version`]
- Claude Code version: [output of `claude --version`]
- LLM endpoint: [value of MEMMAN_LLM_ENDPOINT, e.g. https://openrouter.ai/api/v1]
- Embedding provider: [value of MEMMAN_EMBED_PROVIDER]
- Storage backend: [sqlite or postgres]
- API keys set: [names only, e.g. MEMMAN_LLM_API_KEY, MEMMAN_VOYAGE_API_KEY]

**Additional context**
Relevant output of `memman doctor --text`, `memman status`, `memman log list`, `memman log worker --errors`, or `memman log worker --stack`, with API keys and DSN passwords removed.
