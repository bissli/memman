"""JSON parsing helpers for LLM responses.

The LLM client class lives in `client.py`. HTTP retry/timeout policy
lives in `memman._http` (the single source of truth for HTTP policy
shared between LLM and embed paths).
"""

import json
import logging
import re
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from memman.llm.client import MemmanLLMClient

logger = logging.getLogger('memman')


def strip_code_fences(raw: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = raw.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:])
        text = text.removesuffix('```').strip()
    return text


# A valid escape pair is matched first, so a lone backslash is one that
# no escape character follows (`NT AUTHORITY\SYSTEM`); only it doubles.
_ESCAPE_RUN_RE = re.compile(r'\\\\|\\(?!["\\/bfnrtu])')


def _top_level_json_values(raw: str, opener: str) -> list:
    """Every JSON value that decodes from an `opener` in `raw`, in order.

    Parameters
    ----------
    raw : str
        The response text as the model returned it.
    opener : str
        `'{'` for objects, `'['` for lists.

    Returns
    -------
    list
        The decoded values. The scan resumes after each decoded value;
        an opener whose value fails to decode (a truncated outer
        object) is skipped by one character, so the values inside it
        decode on their own. The text is scanned as sent and, when
        nothing decodes, with lone backslashes repaired; a valid escape
        pair is never touched.
    """
    # strict=False keeps a raw newline or tab inside a string value:
    # a model that copies a memory's paragraph breaks into a summary
    # emits them unescaped, and the strict default refuses the object.
    decoder = json.JSONDecoder(strict=False)
    repaired = _ESCAPE_RUN_RE.sub(
        lambda m: '\\\\' if m.group(0) == '\\' else m.group(0), raw)
    for text in (raw, repaired) if repaired != raw else (raw,):
        found: list = []
        pos = 0
        while True:
            start = text.find(opener, pos)
            if start == -1:
                break
            try:
                value, end = decoder.raw_decode(text, start)
            except (ValueError, RecursionError):
                pos = start + 1
                continue
            found.append(value)
            pos = end
        if found:
            return found
    return []


def parse_json_response(raw: str) -> dict | None:
    """The JSON object an LLM response carries, or None.

    Parameters
    ----------
    raw : str
        The response text as the model returned it.

    Returns
    -------
    dict | None
        The whole text decoded as an object when it is one (fences
        stripped if present); else the LAST object the scan of
        `_top_level_json_values` finds, so a response that reasons
        before its JSON, or emits a block, says "let me reconsider" and
        emits another, is read at its final answer; None when no object
        decodes.

    Notes
    -----
    - When an enclosing object fails to decode (a response cut at the
      token ceiling), the objects inside it are what the scan finds;
      none carries the top-level key a caller reads, so the caller's
      failure path runs as before.
    """
    for text in (raw, strip_code_fences(raw)):
        try:
            parsed = json.loads(text, strict=False)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError, RecursionError):
            pass
    objects = [v for v in _top_level_json_values(raw, '{') if isinstance(v, dict)]
    return objects[-1] if objects else None


def complete_parsed(
        llm_client: 'MemmanLLMClient', system: str, user: str, *,
        stage: str,
        max_tokens: int | None = None) -> tuple[dict | None, str]:
    """The object a completion carries, re-rolling once when none decodes.

    Parameters
    ----------
    llm_client : MemmanLLMClient
        The client for the stage; both attempts go through it.
    system : str
        System prompt.
    user : str
        User prompt.
    stage : str
        Pipeline stage both attempts are charged to.
    max_tokens : int | None, default None
        Output budget per attempt; None sends the role ceiling the
        client was built with.

    Returns
    -------
    tuple[dict | None, str]
        The decoded object and the body it came from. On a re-roll
        this is the SECOND attempt's pair, so a caller that traces a
        failure records the body it gave up on.

    Notes
    -----
    - The parse failure is the only signal that a response is
      unusable. A provider reports `finish_reason` `stop` on a body it
      cut mid-string, so no field of the response separates a complete
      answer from a cut one.
    - Exactly one re-roll, because the failures are sampling
      accidents: an unescaped quote inside a string value, a stream
      the provider cut. A second draw clears one or the shape is out
      of the model's reach.
    - An exception propagates on either attempt. The caller already
      separates a transport failure from an unusable body, and the
      two carry different oplog outcomes.
    """
    raw = llm_client.complete(
        system, user, stage=stage, max_tokens=max_tokens)
    parsed = parse_json_response(raw)
    if parsed is not None:
        return parsed, raw
    logger.debug(
        f'{stage} body of {len(raw)} chars did not decode; re-rolling')
    raw = llm_client.complete(
        system, user, stage=stage, max_tokens=max_tokens)
    return parse_json_response(raw), raw


def parse_json_list_response(raw: str) -> list | None:
    """The JSON list an LLM response carries, or None.

    Parameters
    ----------
    raw : str
        The response text as the model returned it.

    Returns
    -------
    list | None
        The whole text decoded as a list when it is one (fences stripped
        if present); else the last list of objects the scan finds, so a
        bracketed index the model wrote in its prose (`[0]`) never
        replaces the answer; else the last list of any shape; None when
        no list decodes.
    """
    for text in (raw, strip_code_fences(raw)):
        try:
            parsed = json.loads(text, strict=False)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError, RecursionError):
            pass
    lists = [v for v in _top_level_json_values(raw, '[') if isinstance(v, list)]
    of_objects = [v for v in lists if all(isinstance(x, dict) for x in v)]
    chosen = of_objects or lists
    return chosen[-1] if chosen else None


def safe_json(resp: httpx.Response) -> object:
    """Return parsed JSON or the raw text if decoding fails."""
    try:
        return resp.json()
    except Exception:
        return resp.text
