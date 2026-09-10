"""LLM-based fact extraction, memory reconciliation, and query expansion."""

import logging
import re

import cachetools
from memman import config, trace
from memman.llm import usage as llm_usage
from memman.llm.client import MemmanLLMClient
from memman.llm.shared import drop_overlong_strings, parse_json_response

logger = logging.getLogger('memman')

_LINE_WORD_RE = re.compile(r'\bline \d+\b', re.IGNORECASE)
# Notes:
# - Only a source, config or doc extension marks a locator: a bare
#   dot-letter run would also match a dotted host such as
#   `db.example.com:5432` and strip its port.
# - `localhost:8080`, `192.0.2.1:8000`, `14:18`, `python:3.11` and
#   `code:404` carry no such extension before the colon.
_FILE_LINE_RE = re.compile(
    r'(\b[\w./-]+\.(?:py|pyi|js|jsx|ts|tsx|md|rst|txt|html|htm|css|scss'
    r'|json|jsonl|yaml|yml|toml|ini|cfg|conf|sql|sh|bash|zsh|ps1|rs|go'
    r'|java|kt|c|h|cc|cpp|hpp|cs|rb|php|swift|lua|xml|csv|tsv|ipynb'
    r'|drawio|tf|vue|svelte|proto|mk|cmake)):\d{1,5}\b', re.IGNORECASE)
_WS_COLLAPSE_RE = re.compile(r'\s+')


def _strip_line_refs(text: str) -> str:
    """Drop `line N` and the `:N` of a `file.ext:N` locator, keeping the path.

    Parameters
    ----------
    text : str
        Fact text as the model returned it.

    Returns
    -------
    str
        The text with `line N` gone, the `:N` of every `file.ext:N`
        locator gone (a source, config or doc extension), the path kept,
        and whitespace collapsed to single spaces.

    Notes
    -----
    - The path locates the claim and stays; the line number is a
      snapshot of the file at write time and goes.
    """
    text = _LINE_WORD_RE.sub('', text)
    text = _FILE_LINE_RE.sub(r'\1', text)
    return _WS_COLLAPSE_RE.sub(' ', text).strip()


FACT_EXTRACTION_SYSTEM = """You are a personal memory system curator. The user is storing memories for future recall via an upstream fast LLM that emits raw, unpolished content. Your job has THREE steps applied in order.

## Step 1: Skip-judgment

Decide if the input is durable knowledge worth long-term storage. Return {"facts": [], "skip_reason": "..."} for non-durable shapes:

- status updates ("All tests passed", "deployed v1.4 to staging")
- task receipts ("All drives verified after the maintenance window")
- in-progress observations ("migration is 60% complete")
- ephemeral metrics ("queue depth is 4 right now")
- "I just did X" reports
- one-off action confirmations
- greetings, filler, or unintelligible text

Otherwise continue to Step 2.

## Step 2: Canonical-shape rewrite

Rewrite into clean prose, claim-for-claim. The rewrite is strictly 1-to-1: same facts, reshaped surface form. Apply:

- Strip all-caps section markers (REBUILD COST:, KEY FINDING:, ROOT CAUSE:, ARCHITECTURAL CONSTRAINT:)
- Strip back-references (memory [N], see [0], "as mentioned earlier")
- Replace anaphoric openers ("This means", "That implies") with explicit subjects
- Strip transient adverbs (currently, today, "as of YYYY-MM-DD")
- Drop preamble filler ("This is relevant for...", "It is important to note that...")
- Convert past-tense decision narratives ("Rejected X. Chose Y.") into present-tense durable rules
- When the input frames something as non-default state (preserved, enabled, set to, retained, kept), keep that framing - don't flatten to plain present-tense
- Preserve specific names, numbers, and technical terms verbatim. Do not generalize.

## Step 3: Return as a SINGLE atomic fact

Return EXACTLY ONE fact whose `text` is the cleaned content from Step 2. Do NOT split a coherent input into multiple facts. The user remembers in coherent chunks; recall synthesizes across chunks via the graph and similarity. Splitting at storage time loses context.

CRITICAL: even if the input contains 3-5 distinct claims about a single coherent topic (e.g., a multi-section blob describing one system's behavior), return ONE fact whose text covers all the claims in canonical paragraph form. Splitting is not the curator's job.

Output JSON:
{"facts": [{"text": "<cleaned content>", "category": "preference|decision|fact|insight|context", "entities": [...]}], "skip_reason": null}

## Category mapping

- preference: user likes/dislikes/prefers
- decision: explicit choice of X over Y with rationale, or "X uses Y rather than Z"
- fact: how something works (formulas, API behavior, data layout, code patterns)
- insight: lesson learned from experience
- context: project background or user role

A formula or behavior description is a fact, not a decision.

## CRITICAL DIRECTIVE: Preserve the user's domain vocabulary

The examples below illustrate STYLE transformations only. Never substitute indexing, pipeline, database, or other domain terminology if the user's input did not use it. If the input is about React components, return facts about React components. If about CloudFormation, return facts about CloudFormation. The lessons below are about SHAPE (skip / cleanup / single-fact), not subject matter.

---

## Examples - Category 1: Accepted, barely changed

Input: "Pre-fetch object keys into a set for 1M+ file transfers to avoid 1M HEAD requests."
Output: {"facts": [{"text": "Pre-fetch object keys into a set for 1M+ file transfers to avoid 1M HEAD requests.", "category": "fact", "entities": ["HEAD requests"]}], "skip_reason": null}

Input: "Avoid materializing all files upfront with list(fs.walk(recursive=True)) because it causes 30-60 minute enumeration delay and ~300MB memory spike."
Output: {"facts": [{"text": "Avoid materializing all files upfront with list(fs.walk(recursive=True)) because it causes 30-60 minute enumeration delay and ~300MB memory spike.", "category": "fact", "entities": ["fs.walk"]}], "skip_reason": null}

Input: "render_grouped() added a sort_key parameter to enable custom sorting when grouping by non-date partition keys."
Output: {"facts": [{"text": "render_grouped() added a sort_key parameter to enable custom sorting when grouping by non-date partition keys.", "category": "fact", "entities": ["render_grouped"]}], "skip_reason": null}

Input: "A timeout of 5.0 seconds and max_retries=1 were added to the LLM HTTP client to bound worst-case LLM latency."
Output: {"facts": [{"text": "A timeout of 5.0 seconds and max_retries=1 were added to the LLM HTTP client to bound worst-case LLM latency.", "category": "decision", "entities": ["LLM HTTP client"]}], "skip_reason": null}

Input: "CLI surface should be minimal. Every flag must earn its existence."
Output: {"facts": [{"text": "CLI surface should be minimal. Every flag must earn its existence.", "category": "preference", "entities": ["CLI"]}], "skip_reason": null}

## Examples - Category 2: Accepted, rewritten (style cleanup, claims preserved, ONE fact only)

Input: "Decision: rejected the _cleanup_helper() approach for orphan directory migration. Instead, manual rm -rf during rollout. Rationale: aligns with memory [3] preference for manual one-time cleanup over migration code in small-userbase projects."
Output: {"facts": [{"text": "Orphan directory migration uses manual rm -rf during rollout rather than a _cleanup_helper() function; in small-userbase projects, manual one-time cleanup is preferred over migration code.", "category": "decision", "entities": ["_cleanup_helper", "rm -rf"]}], "skip_reason": null}

Input: "INTENT_PARSER.PY DEPRECATION PATH: intent_parser.py (an LLM-based query parser) has been DELETED. Deletion was safe because all three deprecation conditions were met: (1) the search MCP tool docstring was updated to teach LLM callers to extract metadata filters; (2) callers internalized the new filter-extraction pattern; (3) the local LLM dependency was removed from the runtime."
Output: {"facts": [{"text": "intent_parser.py, an LLM-based query parser, was deleted after its three deprecation conditions were met: the search MCP tool docstring was updated to teach callers to extract metadata filters; callers internalized the new filter-extraction pattern; and the local LLM runtime dependency was removed.", "category": "fact", "entities": ["intent_parser.py", "MCP"]}], "skip_reason": null}

Input: "The pipeline currently uses Postgres as the system of record. This means the search index is fully recomputable from Postgres, which makes Postgres backups the only durable persistence layer."
Output: {"facts": [{"text": "The pipeline uses Postgres as the system of record; the search index is fully recomputable from Postgres, which makes Postgres backups the only durable persistence layer.", "category": "fact", "entities": ["Postgres"]}], "skip_reason": null}

(Multi-section synthesis blobs with multiple SHOUTY headers are also single-fact outputs: rewrite as one canonical paragraph, all claims preserved, no decomposition.)

## Examples - Category 3: Skip as non-durable

Input: "All tests passed in the latest CI run after the rebase."
Output: {"facts": [], "skip_reason": "test_run_receipt"}

Input: "Currently processing the backlog at about 12 documents per second."
Output: {"facts": [], "skip_reason": "ephemeral_throughput"}

Input: "Just deployed v1.4 to staging via the release script."
Output: {"facts": [], "skip_reason": "deployment_receipt"}

Input: "Hi there"
Output: {"facts": [], "skip_reason": "greeting"}

## Examples - Edge cases (surface phrasing matches but durability differs)

Input: "All EC2 application updates are deployed via the standard release script, never via direct SSH."
Output: {"facts": [{"text": "All EC2 application updates are deployed via the standard release script, never via direct SSH.", "category": "preference", "entities": ["EC2", "SSH"]}], "skip_reason": null}

Input: "Every login attempt is verified against the LDAP directory before the session token is issued."
Output: {"facts": [{"text": "Every login attempt is verified against the LDAP directory before the session token is issued.", "category": "fact", "entities": ["LDAP"]}], "skip_reason": null}

---

Now process the user's input. Return ONLY JSON, no commentary."""

# Notes:
# - The three stage texts ship byte for byte as the probe measured
#   them: the screen is `prompt_pairwise-v1.txt`; the
#   verdict is `prompt_candidate-v3-verdict.txt` without the
#   several-memories tie-break sentence, which has no referent when one
#   row is shown per call and folded 2 of 11 synthetic restatements as
#   UPDATE; the merge is `prompt_merge-v1.txt`.
#   `tests/test_reconcile_stages.py` pins each text by hash, so an edit
#   re-pins deliberately, after a measurement.
# - Ceilings: one of 1,998 screen responses reached 1024 on a
#   reasoning preamble and none 2048; the verdict cell ran at 2048 with
#   one truncation in 1,221, a runaway enumeration; one whole-body
#   merge over 37.6 K chars exhausted 8192, and the per-target body is
#   the size cure.
SCREEN_MAX_TOKENS = 2048
VERDICT_MAX_TOKENS = 2048
MERGE_MAX_TOKENS = 8192

SCREEN_RELATIONS = frozenset({'CONTRADICTS', 'REFINES', 'RESTATES', 'UNRELATED'})
UNJUDGED = 'UNJUDGED'

# Notes:
# - The write path's disposition of every verdict token the text
#   offers, keyed on the `takes only` line of RECONCILIATION_SYSTEM;
#   `tests/test_reconcile_stages.py` pins the keys to that line, so a
#   token added to the text without a row fails the suite.
# - The reader skips an entry whose token is outside the table (the
#   model's stray word), so it can neither write nor block, and a
#   lookup here never sees one.
VERDICT_DISPOSITION = {
    'SUPERSEDE': 'supersede',
    'UPDATE': 'update',
    'NONE': 'none',
    'ADD': 'keep',
    }

PAIRWISE_SCREEN_SYSTEM = """You are a memory manager. ONE EXISTING MEMORY and ONE NEW FACT arrive. The fact is the newer statement. Judge the memory's present-tense claims against the fact and name their relation.

Relations, judged on present-tense claims only:
- CONTRADICTS: the memory asserts something the fact says is no longer so or never was: a changed value or name, a mechanism that was removed or replaced, a decision that was reversed, or a question the memory left open that the fact settles the other way. A memory that reports a dated event, or what was true at a stated time, is not contradicted by a later state. A fact that says something is no longer true, or corrects an earlier statement, contradicts a memory that asserts the old state.
- REFINES: the fact adds compatible detail to the memory's subject and overturns nothing the memory asserts.
- RESTATES: the memory already carries every claim the fact makes.
- UNRELATED: a different subject, or compatible independent claims.

Return ONLY JSON, no commentary:
{"relation": "CONTRADICTS|REFINES|RESTATES|UNRELATED",
 "contradicted_clauses": ["<each clause of the memory the fact overturns, quoted from the memory>"],
 "reason": "brief explanation"}
contradicted_clauses is empty unless the relation is CONTRADICTS."""

RECONCILIATION_SYSTEM = """You are a memory manager. One NEW FACT arrives with a list of EXISTING MEMORIES, each under a numeric id. Judge EVERY memory against the fact and return one action per memory whose state the fact changes. A memory whose state the fact does not change gets no entry.

Relations, judged on present-tense claims only:
- CONTRADICTS: the memory asserts something the fact says is no longer so or never was: a changed value or name, a mechanism that was removed or replaced, a decision that was reversed, or a question the memory left open that the fact settles the other way. A memory that reports a dated event, or what was true at a stated time, is not contradicted by a later state. A fact that says something is no longer true, or corrects an earlier statement, contradicts every memory that asserts the old state.
- REFINES: the fact adds compatible detail to the memory's subject.
- RESTATES: the memory already carries every claim the fact makes.
- UNRELATED: a different subject, or compatible independent claims.

Actions:
- SUPERSEDE <id>: the fact contradicts memory <id>. Name EVERY contradicted memory, not only the closest one. A memory the fact merely repeats or extends is never superseded.
- UPDATE <id>: the fact refines memory <id>. At most one.
- NONE <id>: memory <id> restates the fact. Alone.
- ADD: no memory is contradicted, refined, or restating. Alone.
The action field takes only ADD, UPDATE, SUPERSEDE, or NONE.

Return JSON:
{"actions": [
  {"action": "ADD|UPDATE|SUPERSEDE|NONE",
   "target_id": null for ADD, else the numeric id,
   "reason": "brief explanation"}
 ]}

When one memory restates the fact and another contradicts it, the restating memory takes UPDATE (it is folded into the successor) and the contradicted one takes SUPERSEDE; NONE is for a fact that changes nothing.

Use the numeric IDs shown, not UUIDs. A contradicted memory gets SUPERSEDE, never ADD."""

MERGE_SYSTEM = """You are a memory manager. A NEW FACT arrives with the EXISTING MEMORIES it changes. Each memory is listed under a numeric id with the clauses the fact overturns (CONTRADICTED CLAUSES); a memory listed with no contradicted clauses is one the fact only adds detail to. Write the SUCCESSOR TEXT that replaces every listed memory.

The successor text:
- states the new fact;
- keeps EVERY clause of every listed memory that is not among its contradicted clauses, in that memory's own words where possible;
- drops each contradicted clause and never restates it as true;
- adds nothing that neither the fact nor a listed memory states;
- is one canonical paragraph of prose (no headers, no bullet lists, no back-references such as "memory 0").

Return ONLY JSON, no commentary:
{"merged_text": "<the successor text>"}"""

QUERY_EXPANSION_SYSTEM = (
    'Expand a search query for a personal memory system.\n\n'
    'Return JSON:\n'
    '{"expanded_query": "original plus synonyms and related terms",\n'
    ' "intent": "WHY|WHEN|ENTITY|GENERAL"}\n\n'
    'Keep expanded_query under 50 words.')


def extract_facts(
        llm_client: MemmanLLMClient,
        content: str) -> list[dict]:
    """Extract the one canonical fact of `content` via the LLM.

    Parameters
    ----------
    llm_client : MemmanLLMClient
        The slow canonical client; one `complete` call per invocation.
    content : str
        The text as the caller wrote it.

    Returns
    -------
    list[dict]
        Dicts with keys `text`, `category`, `entities`. Empty when the
        model skipped the input as non-durable. A single passthrough
        fact wrapping `content` on an LLM error, a parse error, or an
        empty `facts` list.

    Notes
    -----
    - Importance is not extracted: the caller's `--imp` is stored as
      passed (default 3) and the model has no say in it.
    """
    trace.event('extract_facts_start', content_len=len(content),
                content=content)
    try:
        raw = llm_client.complete(
            FACT_EXTRACTION_SYSTEM, content,
            stage=llm_usage.STAGE_EXTRACTION)
    except Exception as exc:
        logger.debug('LLM fact extraction failed, using passthrough')
        trace.event(
            'extract_facts_result',
            outcome='passthrough',
            error=f'{type(exc).__name__}: {exc}')
        return _passthrough_fact(content, 'fact')

    parsed = parse_json_response(raw)
    if parsed is None:
        logger.debug('LLM fact extraction parse error, using passthrough')
        trace.event(
            'extract_facts_result',
            outcome='parse_error',
            raw=raw)
        return _passthrough_fact(content, 'fact')

    skip_reason = parsed.get('skip_reason')
    if skip_reason:
        logger.debug(f'LLM skipped: {skip_reason}')
        trace.event(
            'extract_facts_result',
            outcome='skipped',
            skip_reason=skip_reason)
        return []

    raw_facts = parsed.get('facts', [])
    if not isinstance(raw_facts, list) or not raw_facts:
        trace.event('extract_facts_result', outcome='no_facts', raw=raw)
        return _passthrough_fact(content, 'fact')

    facts = []
    for f in raw_facts:
        if not isinstance(f, dict):
            continue
        text = _strip_line_refs(f.get('text', '').strip())
        if not text:
            continue
        category = f.get('category', 'fact')
        if category not in {'preference', 'decision', 'fact',
                            'insight', 'context'}:
            category = 'fact'
        entities = f.get('entities', [])
        if not isinstance(entities, list):
            entities = []
        entities = drop_overlong_strings(
            [str(e) for e in entities if e],
            kind='entity', owner=f'extracted fact {text[:32]!r}')
        facts.append({
            'text': text,
            'category': category,
            'entities': entities,
            })

    result = facts or _passthrough_fact(content, 'fact')
    trace.event(
        'extract_facts_result',
        outcome='ok',
        fact_count=len(result),
        skip_reason=skip_reason,
        facts=result)
    return result


def _passthrough_fact(content: str, category: str) -> list[dict]:
    """Wrap raw content as a single fact for fallback."""
    return [{
        'text': content,
        'category': category,
        'entities': [],
        }]


def screen_memory(
        llm_client: MemmanLLMClient,
        fact_text: str,
        memory: tuple[str, str]) -> tuple[str, list[str]]:
    """Stage 1: one row's relation to the fact and the clauses overturned.

    Parameters
    ----------
    llm_client : MemmanLLMClient
        The slow canonical client; one `complete` call per invocation.
    fact_text : str
        The new fact as extracted.
    memory : tuple[str, str]
        `(real_id, content)`, the row under judgment.

    Returns
    -------
    tuple[str, list[str]]
        The relation, one of `SCREEN_RELATIONS` or `UNJUDGED`, and the
        clauses of the row the fact overturns, quoted from the row;
        empty unless the relation is CONTRADICTS.

    Notes
    -----
    - Fail-open: an LLM error, an unparsed body, a missing or unknown
      relation is `UNJUDGED`, and the keep rule shows an UNJUDGED row
      to stage 2 rather than drop it.
    """
    real_id, content = memory
    body = f'EXISTING MEMORY:\n{content}\n\nNEW FACT:\n{fact_text}'
    try:
        raw = llm_client.complete(
            PAIRWISE_SCREEN_SYSTEM, body,
            stage=llm_usage.STAGE_SCREEN,
            max_tokens=SCREEN_MAX_TOKENS)
    except Exception as exc:
        trace.event(
            'screen_result', target_id=real_id, outcome='error',
            error=f'{type(exc).__name__}: {exc}')
        return UNJUDGED, []
    parsed = parse_json_response(raw)
    relation = (str(parsed.get('relation', '')).upper()
                if isinstance(parsed, dict) else '')
    if relation not in SCREEN_RELATIONS:
        trace.event(
            'screen_result', target_id=real_id, outcome='unjudged', raw=raw)
        return UNJUDGED, []
    clauses: list[str] = []
    if relation == 'CONTRADICTS':
        quoted = parsed.get('contradicted_clauses')
        if isinstance(quoted, list):
            clauses = [c for c in quoted if isinstance(c, str) and c.strip()]
    trace.event(
        'screen_result', target_id=real_id, outcome='ok',
        relation=relation, clauses=len(clauses))
    return relation, clauses


def judge_memory(
        llm_client: MemmanLLMClient,
        fact_text: str,
        memory: tuple[str, str]) -> str:
    """Stage 2: the write path's verdict on one screened row.

    Parameters
    ----------
    llm_client : MemmanLLMClient
        The slow canonical client; one `complete` call per invocation.
    fact_text : str
        The new fact as extracted.
    memory : tuple[str, str]
        `(real_id, content)`, the one row shown under `[0]`.

    Returns
    -------
    str
        `supersede`, `update`, `none`, or `keep` for no write against
        the row.

    Notes
    -----
    - The id map is the measured `first` variant: the first entry
      with a known token that names ANY non-null id decides, whatever
      id it names. The model numbers sections of the one row and judges
      a section (25 rows of 1,221 on the probe), and every such verdict
      is about the row shown.
    - An entry with a null id is skipped: an ADD there neither decides
      nor blocks, a row verdict there is dropped. DELETE reads as
      SUPERSEDE. An entry whose token is outside `VERDICT_DISPOSITION`
      is skipped the same way, so the next known entry decides.
    - Fail-closed: an LLM error or an unparsed body is `keep`.
    """
    real_id, content = memory
    body = f'EXISTING MEMORIES:\n[0] {content}\n\nNEW FACT:\n{fact_text}'
    trace.event('reconcile_start', target_id=real_id)
    try:
        raw = llm_client.complete(
            RECONCILIATION_SYSTEM, body,
            stage=llm_usage.STAGE_RECONCILIATION,
            max_tokens=VERDICT_MAX_TOKENS)
    except Exception as exc:
        trace.event(
            'reconcile_result', target_id=real_id, outcome='error',
            error=f'{type(exc).__name__}: {exc}')
        return 'keep'

    parsed = parse_json_response(raw)
    if parsed is None or not isinstance(parsed.get('actions'), list):
        trace.event(
            'reconcile_result', target_id=real_id, outcome='parse_error',
            raw=raw)
        return 'keep'

    verdict = 'keep'
    for entry in parsed['actions']:
        if not isinstance(entry, dict) or entry.get('target_id') is None:
            continue
        action = str(entry.get('action', '')).upper()
        if action == 'DELETE':
            action = 'SUPERSEDE'
        if action not in VERDICT_DISPOSITION:
            continue
        verdict = VERDICT_DISPOSITION[action]
        break
    trace.event(
        'reconcile_result', target_id=real_id, outcome='ok', verdict=verdict)
    return verdict


def merge_successor(
        llm_client: MemmanLLMClient,
        fact_text: str,
        target: tuple[str, str, list[str]]) -> str | None:
    """Stage 3: the successor text for one retiring target.

    Parameters
    ----------
    llm_client : MemmanLLMClient
        The slow canonical client; one `complete` call per invocation.
    fact_text : str
        The new fact as extracted.
    target : tuple[str, str, list[str]]
        `(real_id, content, clauses)`: the retiring row and the clauses
        the screen quoted as contradicted; empty for an update target
        or a row the screen did not call CONTRADICTS.

    Returns
    -------
    str | None
        The merged text, stripped; None on an LLM error, an unparsed
        body or an empty text, so the caller stores the fact and marks
        the oplog row `(unmerged)`.

    Notes
    -----
    - One target per call: on the probe the row-alone body kept the
      clauses the whole body dropped, b = 6 to 8 against c = 0 on
      every judged line.
    - Over the 0.35.0 gate's 534 SCREENED case-reps, 46 of 4,239
      clauses were sentences copied from the NEW FACT that do not occur
      in the target memory (31 case-reps); on those the merge deleted
      the fact's own correction from the successor. A clause whose
      normalized text appears in the fact but not in the target content
      is a fact-copied clause and is suppressed before rendering.
    """
    real_id, content, clauses = target
    norm_fact = ' '.join(fact_text.lower().split())
    norm_content = ' '.join(content.lower().split())
    # Notes:
    # - A clause found in the fact but not in the target is a fact
    #   sentence the screen copied, not a memory clause; handed to the
    #   merge it makes the merge delete the fact's own correction.
    # - Whitespace and case are collapsed because the screen returns
    #   clauses in the model's own wrapping and casing.
    filtered_clauses = [
        c for c in clauses
        if (nc := ' '.join(c.lower().split())) in norm_content
        or nc not in norm_fact
        ]
    suppressed = len(clauses) - len(filtered_clauses)
    clause_lines = [f'- {clause}' for clause in filtered_clauses] or ['(none)']
    body = (
        f'EXISTING MEMORIES:\n[0] {content}\n'
        'CONTRADICTED CLAUSES of [0]:\n' + '\n'.join(clause_lines)
        + f'\n\nNEW FACT:\n{fact_text}')
    try:
        raw = llm_client.complete(
            MERGE_SYSTEM, body,
            stage=llm_usage.STAGE_MERGE,
            max_tokens=MERGE_MAX_TOKENS)
    except Exception as exc:
        trace.event(
            'reconcile_merge', target_id=real_id,
            clauses=len(filtered_clauses), suppressed=suppressed,
            outcome='error', error=f'{type(exc).__name__}: {exc}')
        return None
    parsed = parse_json_response(raw)
    text = parsed.get('merged_text') if isinstance(parsed, dict) else None
    if not isinstance(text, str) or not text.strip():
        trace.event(
            'reconcile_merge', target_id=real_id,
            clauses=len(filtered_clauses), suppressed=suppressed,
            outcome='no_text', raw=raw)
        return None
    trace.event(
        'reconcile_merge', target_id=real_id,
        clauses=len(filtered_clauses), suppressed=suppressed,
        outcome='ok')
    return text.strip()


_EXPAND_CACHE_TTL = 300
_EXPAND_CACHE_MAX = 256


def _normalize_for_cache(query: str) -> str:
    """Lowercase + collapse whitespace; nothing else."""
    return ' '.join(query.lower().split())


def _expand_cache_key(query: str) -> str:
    """Salt with the configured fast-model id.

    The model id is resolved at install time and persisted to
    `~/.memman/env`, so `config.require` always returns a real value
    here. Reaching this with an unset key means install was never run,
    which is a `ConfigError` upstream callers handle.
    """
    import hashlib
    salt = config.require(config.LLM_MODEL_FAST)
    digest = hashlib.sha256(
        f'{_normalize_for_cache(query)}|{salt}'.encode())
    return digest.hexdigest()[:16]


_expand_cache: cachetools.TTLCache = cachetools.TTLCache(
    maxsize=_EXPAND_CACHE_MAX, ttl=_EXPAND_CACHE_TTL)


def reset_expand_cache() -> None:
    """Drop cached query expansions. Used by tests that swap env vars."""
    _expand_cache.clear()


def expand_query(
        llm_client: MemmanLLMClient,
        query: str) -> dict:
    """Expand recall query with synonyms and related terms.

    Returns dict with: expanded_query, intent.
    On failure: passthrough with original query. Repeated calls with
    the same query in the same process hit a `cachetools.TTLCache`
    keyed by sha256(normalized_query | $MEMMAN_LLM_MODEL_FAST). Cache
    lives only for the duration of one CLI invocation (memman is a
    one-shot CLI), so persistence across processes is left to the
    LLM provider's own response cache.
    """
    cache_key = _expand_cache_key(query)
    cached = _expand_cache.get(cache_key)
    if cached is not None:
        trace.event(
            'query_expand_result', outcome='cache_hit', **cached)
        return dict(cached)

    trace.event('query_expand_start', query=query)
    try:
        raw = llm_client.complete(
            QUERY_EXPANSION_SYSTEM, query,
            stage=llm_usage.STAGE_QUERY_EXPANSION)
    except Exception as exc:
        logger.debug('LLM query expansion failed, using passthrough')
        trace.event(
            'query_expand_result',
            outcome='error',
            error=f'{type(exc).__name__}: {exc}')
        return {'expanded_query': query, 'intent': None}

    parsed = parse_json_response(raw)
    if parsed is None:
        return {'expanded_query': query, 'intent': None}

    expanded = parsed.get('expanded_query', query)
    if not isinstance(expanded, str) or not expanded.strip():
        expanded = query

    intent = parsed.get('intent')
    if intent not in {'WHY', 'WHEN', 'ENTITY', 'GENERAL'}:
        intent = None

    result = {
        'expanded_query': expanded,
        'intent': intent,
        }
    _expand_cache[cache_key] = dict(result)
    trace.event('query_expand_result', outcome='ok', **result)
    return result
