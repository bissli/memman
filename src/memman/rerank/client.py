"""Rerank client for the shared endpoint's `/rerank` route.

The route takes `{model, query, documents, top_n}` and returns ranked
rows under `results`.
"""

import time

from memman import config, trace
from memman._http import api_headers, get_session, post_with_retry


class Client:
    """HTTP client for `<MEMMAN_ENDPOINT>/rerank`.
    """

    def __init__(self) -> None:
        """Bind to the shared endpoint and key and `MEMMAN_RERANK_MODEL`.

        Raises
        ------
        ConfigError
            `MEMMAN_ENDPOINT` or `MEMMAN_RERANK_MODEL` is unset, or
            `MEMMAN_API_KEY` is blank on a non-loopback endpoint.
        """
        self.endpoint = config.require(config.ENDPOINT).rstrip('/')
        self._api_key = config.api_key_for(self.endpoint)
        self.model = config.require(config.RERANK_MODEL)

    def rerank(self, query: str, documents: list[str],
               top_n: int | None = None) -> list[tuple[int, float]]:
        """Score each (query, document) pair with the cross-encoder.

        Parameters
        ----------
        query : str
            Search text.
        documents : list[str]
            Candidates to score; empty returns [] with no HTTP call.
        top_n : int or None, default None
            Return only this many best pairs; None returns all.

        Returns
        -------
        list[tuple[int, float]]
            `(original_index, relevance_score)`, best first.

        Raises
        ------
        RuntimeError
            When the endpoint answers with a non-200 status.
        """
        if not documents:
            return []
        url = f'{self.endpoint}/rerank'
        headers = api_headers(self.endpoint, self._api_key)
        body: dict = {
            'model': self.model, 'query': query, 'documents': documents}
        if top_n is not None:
            body['top_n'] = top_n
        trace.event(
            'rerank_request',
            url=url,
            model=self.model,
            n_docs=len(documents),
            query_len=len(query),
            headers=trace.redact_headers(headers))
        t0 = time.monotonic()
        resp = post_with_retry(
            get_session(__name__), url,
            headers=headers, json=body, timeout=30.0)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code != 200:
            trace.event(
                'rerank_response',
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                error='http_status')
            raise RuntimeError(
                f'rerank request returned status {resp.status_code}')
        data = resp.json()
        items = data.get('results', [])
        trace.event(
            'rerank_response',
            status=resp.status_code,
            elapsed_ms=elapsed_ms,
            n_items=len(items),
            usage=data.get('usage'))
        return [
            (int(d['index']), float(d['relevance_score'])) for d in items]
