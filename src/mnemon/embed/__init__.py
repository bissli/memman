"""Embedding provider — Voyage AI only."""

from mnemon.embed.voyage import Client


def get_client() -> Client:
    """Return Voyage embedding client."""
    return Client()
