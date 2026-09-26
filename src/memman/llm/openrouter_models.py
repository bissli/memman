"""OpenRouter model candidates for the operator to pick from.

`fetch_candidates` reads two public OpenRouter endpoints, with no API
key: `/endpoints/zdr` for every zero-data-retention endpoint and its
price, and `/models` for each model's release date and reasoning
defaults. `list_candidates` joins them and keeps the models the
operator may pick. memman never switches a model on its own: the
install wizard and `memman config models` show the list and write only
the operator's pick.
"""

import dataclasses
import datetime
import time
from decimal import Decimal

import httpx
from memman import trace

FETCH_TIMEOUT_SECONDS = 10.0
MAX_CANDIDATES = 3
SPECIALIST_MARKERS = ('-vl-', 'coder')
EXCLUDED_SUFFIXES = (':free', ':batch')


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One model the operator may pick.

    Attributes
    ----------
    model_id : str
        OpenRouter model id, e.g. `qwen/qwen3-235b-a22b-2507`.
    vendor : str
        Slug of the vendor whose endpoint sets the price, in the form
        `MEMMAN_LLM_PROVIDER_ONLY` uses, e.g. `google-vertex`.
    input_per_m : float
        Dollars per million input tokens on that endpoint.
    output_per_m : float
        Dollars per million output tokens on that endpoint.
    released : datetime.date
        UTC date of the model's catalog `created` stamp.
    thinks_by_default : bool
        True when the catalog marks reasoning mandatory or on by default.
    """

    model_id: str
    vendor: str
    input_per_m: float
    output_per_m: float
    released: datetime.date
    thinks_by_default: bool

    def label(self) -> str:
        """One display line: id, vendor, prices, release date, thinking.
        """
        thinking = 'yes' if self.thinks_by_default else 'no'
        return (
            f'{self.model_id}  {self.vendor}'
            f'  ${self.input_per_m:g} in / ${self.output_per_m:g} out per M'
            f'  released {self.released.isoformat()}'
            f'  thinks by default: {thinking}')


def list_candidates(
        models: list[dict],
        zdr_endpoints: list[dict],
        *,
        family: str,
        max_input_per_m: float,
        max_output_per_m: float,
        vendors: frozenset[str]) -> list[Candidate]:
    """Up to three general-purpose models in `family` the operator may pick.

    Parameters
    ----------
    models : list[dict]
        The `data` rows of `GET /models`.
    zdr_endpoints : list[dict]
        The `data` rows of `GET /endpoints/zdr`.
    family : str
        Model-id prefix including its slash, e.g. `qwen/`.
    max_input_per_m : float
        Input ceiling in dollars per million tokens, inclusive.
    max_output_per_m : float
        Output ceiling in dollars per million tokens, inclusive.
    vendors : frozenset[str]
        Vendor slugs allowed to serve: the operator's provider pin.

    Returns
    -------
    list[Candidate]
        At most `MAX_CANDIDATES`, dearest first by (output, input)
        price, newest first on a tie.

    Notes
    -----
    - A model qualifies through a ZDR endpoint whose vendor slug (the
      tag before its first `/`) is in `vendors` and whose prices sit
      inside both ceilings. With several, the dearest one sets the
      shown price and vendor.
    - General-purpose excludes the `-vl-` and `coder` lines and the
      `:free` and `:batch` variants. A ZDR row with no `/models` entry
      (embedding, ASR, reranker) drops out at the join.
    - Thinks-by-default is `reasoning.mandatory or
      reasoning.default_enabled`; no `reasoning` object means no.
    """
    models_by_id = {entry['id']: entry for entry in models}
    dearest_by_id: dict[str, tuple[float, float, str]] = {}
    for endpoint in zdr_endpoints:
        model_id = endpoint['model_id']
        vendor = endpoint['tag'].split('/', 1)[0]
        if (not model_id.startswith(family)
                or model_id not in models_by_id
                or vendor not in vendors
                or any(marker in model_id for marker in SPECIALIST_MARKERS)
                or model_id.endswith(EXCLUDED_SUFFIXES)):
            continue
        # Decimal keeps a catalog price equal to a ceiling set at that
        # price; float multiplication can land a hair above it.
        input_per_m = float(Decimal(endpoint['pricing']['prompt']) * 1_000_000)
        output_per_m = float(
            Decimal(endpoint['pricing']['completion']) * 1_000_000)
        if input_per_m > max_input_per_m or output_per_m > max_output_per_m:
            continue
        price = (output_per_m, input_per_m, vendor)
        if price > dearest_by_id.get(model_id, (-1.0, -1.0, '')):
            dearest_by_id[model_id] = price
    candidates = []
    for model_id, (output_per_m, input_per_m, vendor) in dearest_by_id.items():
        entry = models_by_id[model_id]
        reasoning = entry.get('reasoning') or {}
        candidates.append(Candidate(
            model_id=model_id,
            vendor=vendor,
            input_per_m=input_per_m,
            output_per_m=output_per_m,
            released=datetime.datetime.fromtimestamp(
                entry['created'], datetime.UTC).date(),
            thinks_by_default=bool(
                reasoning.get('mandatory') or reasoning.get('default_enabled'))))
    candidates.sort(
        key=lambda c: (c.output_per_m, c.input_per_m, c.released),
        reverse=True)
    return candidates[:MAX_CANDIDATES]


def _fetch_rows(client: httpx.Client, url: str) -> list[dict]:
    """GET a public OpenRouter catalog and return its `data` rows.

    Raises
    ------
    httpx.HTTPError
        On a transport failure or a non-2xx status.
    RuntimeError
        When the response carries no `data` list.
    """
    trace.event('openrouter_catalog_request', url=url)
    t0 = time.monotonic()
    resp = client.get(url, timeout=FETCH_TIMEOUT_SECONDS)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    resp.raise_for_status()
    rows = resp.json().get('data')
    if not isinstance(rows, list):
        raise RuntimeError(f'unexpected OpenRouter catalog shape at {url}')
    trace.event(
        'openrouter_catalog_response',
        url=url,
        status=resp.status_code,
        elapsed_ms=elapsed_ms,
        row_count=len(rows))
    return rows


def fetch_candidates(
        endpoint: str,
        *,
        family: str,
        max_input_per_m: float,
        max_output_per_m: float,
        vendors: frozenset[str]) -> list[Candidate]:
    """Read both public catalogs under `endpoint` and list the candidates.

    Parameters
    ----------
    endpoint : str
        OpenRouter API base, e.g. `https://openrouter.ai/api/v1`.
    family : str
        As `list_candidates`.
    max_input_per_m : float
        As `list_candidates`.
    max_output_per_m : float
        As `list_candidates`.
    vendors : frozenset[str]
        As `list_candidates`.

    Returns
    -------
    list[Candidate]
        As `list_candidates`.

    Raises
    ------
    httpx.HTTPError
        When either catalog cannot be fetched.
    RuntimeError
        When a catalog response carries no `data` list.
    """
    base = endpoint.rstrip('/')
    with httpx.Client() as client:
        zdr_endpoints = _fetch_rows(client, f'{base}/endpoints/zdr')
        models = _fetch_rows(client, f'{base}/models')
    return list_candidates(
        models, zdr_endpoints, family=family,
        max_input_per_m=max_input_per_m,
        max_output_per_m=max_output_per_m, vendors=vendors)
