"""Provider-agnostic helpers used by every LLM client.

JSON parsing (tolerant of code fences), retry policy constants, and a
`_safe_json` helper for trace logging. Concrete provider classes
(OpenRouter, and future Gemini/Groq/OpenAI-compat entries) import
from here so they don't depend on each other.
"""

import json

import httpx

ENRICHMENT_TIMEOUT = 10.0
MAX_RETRIES = 3
RETRY_BACKOFF = (1.0, 2.0)
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 529)


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
