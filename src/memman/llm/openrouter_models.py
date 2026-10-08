"""The check that the configured OpenRouter model still routes.

`fetch_model_notice` reads two public OpenRouter catalogs, with no API
key: `/endpoints/zdr` for every zero-data-retention endpoint and
`/models` for each model's retirement date. The scheduler drain runs
the check once a day and the install runs it at once; each records the
verdict in `<data_dir>/model.state`, and `memman prime` prints a
standing notice. memman never switches a model on its own.
"""

import json
import time
from pathlib import Path

import httpx
from memman import config, trace
from memman.setup._atomic import atomic_write_secure

FETCH_TIMEOUT_SECONDS = 10.0
CHECK_INTERVAL_SECONDS = 86_400
MODEL_STATE_FILENAME = 'model.state'
UNROUTABLE_NOTICE = (
    'LLM model {model} has no ZDR endpoint, so enrichment fails; the'
    ' user picks a replacement: memman config set MEMMAN_LLM_MODEL <id>')
RETIRING_NOTICE = (
    'LLM model {model} retires on {date}; the user picks a replacement:'
    ' memman config set MEMMAN_LLM_MODEL <id>')


def model_notice(
        models: list[dict],
        zdr_endpoints: list[dict],
        *,
        model: str) -> str:
    """The notice `model` earns against both catalogs, '' when it routes.

    Parameters
    ----------
    models : list[dict]
        The `data` rows of `GET /models`.
    zdr_endpoints : list[dict]
        The `data` rows of `GET /endpoints/zdr`.
    model : str
        The configured `MEMMAN_LLM_MODEL`, matched exactly. A `~` alias
        is checked for ZDR through its `/models` `alias_target`, and for
        retirement as the alias itself.

    Returns
    -------
    str
        `UNROUTABLE_NOTICE` when no ZDR endpoint serves `model`; else
        `RETIRING_NOTICE` when `/models` carries an `expiration_date`
        for it; else ''.
    """
    routed_model = next(
        (entry['alias_target']['slug'] for entry in models
         if entry['id'] == model and entry.get('alias_target')),
        model)
    if not any(endpoint['model_id'] == routed_model for endpoint in zdr_endpoints):
        return UNROUTABLE_NOTICE.format(model=model)
    for entry in models:
        if entry['id'] == model and entry.get('expiration_date'):
            return RETIRING_NOTICE.format(
                model=model, date=entry['expiration_date'])
    return ''


def _fetch_rows(client: httpx.Client, url: str) -> list[dict]:
    """GET a public OpenRouter catalog and return its `data` rows.

    Parameters
    ----------
    client : httpx.Client
        Open client; no API key is sent.
    url : str
        Full catalog URL, e.g. `https://openrouter.ai/api/v1/models`.

    Returns
    -------
    list[dict]
        The catalog's `data` rows, unfiltered.

    Raises
    ------
    httpx.HTTPError
        On a transport failure or a non-2xx status.
    RuntimeError
        When the body is not a JSON object carrying a `data` list.
    """
    trace.event('openrouter_catalog_request', url=url)
    t0 = time.monotonic()
    resp = client.get(url, timeout=FETCH_TIMEOUT_SECONDS)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    resp.raise_for_status()
    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(f'OpenRouter catalog at {url} is not JSON') from exc
    rows = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError(f'unexpected OpenRouter catalog shape at {url}')
    trace.event(
        'openrouter_catalog_response',
        url=url,
        status=resp.status_code,
        elapsed_ms=elapsed_ms,
        row_count=len(rows))
    return rows


def fetch_model_notice(endpoint: str, *, model: str) -> str:
    """Read both public catalogs under `endpoint` and check `model`.

    Parameters
    ----------
    endpoint : str
        OpenRouter API base, e.g. `https://openrouter.ai/api/v1`.
    model : str
        As `model_notice`.

    Returns
    -------
    str
        As `model_notice`.

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
    return model_notice(models, zdr_endpoints, model=model)


def _read_state(data_dir: str) -> dict:
    """The recorded check under `data_dir`; {} when absent or unreadable.
    """
    try:
        return json.loads((Path(data_dir) / MODEL_STATE_FILENAME).read_text())
    except (OSError, ValueError):
        return {}


def refresh_model_state(data_dir: str, *, force: bool) -> str | None:
    """Check the configured model when due and record the verdict.

    Parameters
    ----------
    data_dir : str
        Holds the env file that names the endpoint and model, and
        the `model.state` file this writes.
    force : bool
        Check even when the recorded check of this model is younger
        than `CHECK_INTERVAL_SECONDS`.

    Returns
    -------
    str or None
        The notice recorded, '' for a clean check. None when no check
        ran: the endpoint is not OpenRouter, no model is set, or
        (without `force`) this model's recorded check is still fresh.

    Raises
    ------
    httpx.HTTPError
        When a catalog cannot be fetched, raised after the state is
        written.
    RuntimeError
        When a catalog carries no `data` list, raised the same way.
    """
    endpoint = config.get_scoped(config.ENDPOINT, data_dir) or ''
    model = config.get_scoped(config.LLM_MODEL, data_dir)
    if not config.is_openrouter_endpoint(endpoint) or not model:
        return None
    state = _read_state(data_dir)
    same_model = state.get('model') == model
    now = int(time.time())
    if (not force and same_model
            and now - state.get('checked_at', 0) < CHECK_INTERVAL_SECONDS):
        return None
    # A failed fetch still restarts the clock, so an outage costs one
    # attempt per interval rather than one per drain. It keeps the
    # standing notice for this model and records '' for a new one.
    notice = state.get('notice', '') if same_model else ''
    try:
        notice = fetch_model_notice(endpoint, model=model)
    finally:
        atomic_write_secure(
            Path(data_dir) / MODEL_STATE_FILENAME,
            json.dumps({'model': model, 'checked_at': now, 'notice': notice}))
    return notice


def read_model_notice(data_dir: str) -> str:
    """The recorded notice while it names the configured model, else ''.

    Parameters
    ----------
    data_dir : str
        Holds the env file and the `model.state` file.

    Returns
    -------
    str
        '' as well when no check has run or the last one was clean.
    """
    state = _read_state(data_dir)
    if state.get('model') != config.get_scoped(config.LLM_MODEL, data_dir):
        return ''
    return state.get('notice', '')
