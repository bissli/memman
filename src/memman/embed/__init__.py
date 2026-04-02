"""Embedding provider — Voyage AI only."""

from memman.embed.voyage import Client


def get_client() -> Client:
    """Return Voyage embedding client."""
    return Client()
