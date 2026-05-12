"""JSON parsing helpers for LLM responses.

The LLM client class lives in `client.py`. HTTP retry/timeout policy
lives in `memman._http` (the single source of truth for HTTP policy
shared between LLM and embed paths).
"""

import json

import httpx


def strip_code_fences(raw: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = raw.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:])
        text = text.removesuffix('```').strip()
    return text


def parse_json_response(raw: str) -> dict | None:
    """Parse JSON dict from LLM response, handling code blocks."""
    for text in (raw, strip_code_fences(raw)):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def parse_json_list_response(raw: str) -> list | None:
    """Parse JSON list from LLM response, handling code blocks."""
    for text in (raw, strip_code_fences(raw)):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def safe_json(resp: httpx.Response) -> object:
    """Return parsed JSON or the raw text if decoding fails."""
    try:
        return resp.json()
    except Exception:
        return resp.text
