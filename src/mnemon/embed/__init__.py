"""Embedding provider factory."""

import os
from typing import Any


def get_client() -> Any | None:
    """Return an embedding client based on MNEMON_EMBED_PROVIDER env var."""
    provider = os.environ.get('MNEMON_EMBED_PROVIDER', '').lower()
    if provider == 'openai':
        # Deferred: avoids importing openai client unless selected
        from mnemon.embed.openai import Client
        return Client()
    if provider == 'ollama':
        # Deferred: symmetric with openai branch
        from mnemon.embed.ollama import Client
        return Client()
    # Auto-detect: try Ollama if env var not set
    from mnemon.embed.ollama import Client  # noqa: deferred by design
    client = Client()
    if client.available():
        return client
    return None
