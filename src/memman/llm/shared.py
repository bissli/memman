"""JSON parsing helpers for LLM responses.

The LLM client class lives in `client.py`. HTTP retry/timeout policy
lives in `memman._http` (the single source of truth for HTTP policy
shared between LLM and embed paths).
"""

import json
import logging

import httpx

logger = logging.getLogger('memman')

# Guardrail bounding pathological LLM output (a sentence or paragraph
# emitted as one entity/keyword), NOT a retrieval tunable -- it needs
# no ablation-harness sweep. Measured: the longest legitimate strings
# fleet-wide (cloud ARNs, directory DNs, Windows paths) reach 137
# chars, so 200 passes every observed legitimate value.
MAX_ENRICH_STRING_CHARS = 200


def drop_overlong_strings(
        values: list[str], *, kind: str, owner: str) -> list[str]:
    """Drop strings over `MAX_ENRICH_STRING_CHARS`, logging each drop.

    Parameters
    ----------
    values : list[str]
        LLM-proposed entities or keywords. Never pass user-supplied
        values -- `--entities` is uncapped by design.
    kind : str
        'entity' or 'keyword', for the drop log line.
    owner : str
        Insight id (or producer label) named in the drop log line.

    Returns
    -------
    list[str]
        The surviving values, order preserved.

    Notes
    -----
    - Drop, never truncate: a truncated entity is still a valid
      exact-match edge key and still lands in the embedding,
      preserving the pathology under a new name.
    """
    kept = []
    for v in values:
        if len(v) > MAX_ENRICH_STRING_CHARS:
            logger.info(
                f'dropped over-long {kind} ({len(v)} chars) for'
                f' {owner}: {v[:40]!r}...')
            continue
        kept.append(v)
    return kept


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
