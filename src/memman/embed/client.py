"""Embedding client for the shared endpoint's `/embeddings` route.

The endpoint speaks OpenAI's embeddings schema. `dim` is learned from
the first successful embed.
"""

import logging
import time

from memman import config, trace
from memman._http import api_headers, get_session, post_with_retry

logger = logging.getLogger('memman')


class Client:
    """HTTP client for `<MEMMAN_ENDPOINT>/embeddings`.
    """

    def __init__(self, model: str) -> None:
        """Bind to the shared endpoint and key, and to `model`.

        Parameters
        ----------
        model : str
            Model id sent verbatim, e.g. `voyageai/voyage-4-lite`.

        Raises
        ------
        ConfigError
            `MEMMAN_ENDPOINT` is unset, or `MEMMAN_API_KEY` is blank on
            a non-loopback endpoint.
        """
        self.endpoint = config.require(config.ENDPOINT).rstrip('/')
        self._api_key = config.api_key_for(self.endpoint)
        self.model = model
        self.dim = 0
        self._availability_cache: bool | None = None

    def prepare(self) -> None:
        """Learn `dim` with a one-token embed when it is not yet known.

        A success also answers `available()`, so it does not probe again.
        A failure is logged and swallowed; the next `embed` call raises
        the real error.
        """
        if self.dim:
            return
        try:
            self.dim = len(self.embed('test'))
            self._availability_cache = True
        except Exception as exc:
            logger.debug(
                f'embed prepare probe failed for model={self.model!r}:'
                f' {type(exc).__name__}: {exc}')

    def available(self) -> bool:
        """True when a one-token embed succeeds; the result is memoized.
        """
        if self._availability_cache is not None:
            return self._availability_cache
        try:
            self.dim = len(self.embed('test'))
            result = True
        except Exception as exc:
            logger.debug(
                f'embed available probe failed for model={self.model!r}:'
                f' {type(exc).__name__}: {exc}')
            result = False
        self._availability_cache = result
        return result

    def embed(self, text: str) -> list[float]:
        """Embedding vector for one text.
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in input order, from one HTTP request.

        Raises
        ------
        RuntimeError
            A non-200 status, a vector count that differs from the
            input count, or a row with no embedding.
        """
        if not texts:
            return []
        url = f'{self.endpoint}/embeddings'
        headers = api_headers(self.endpoint, self._api_key)
        body: dict = {
            'model': self.model,
            'input': texts,
            'encoding_format': 'float',
            }
        trace.event(
            'embed_request',
            url=url,
            model=self.model,
            batch_size=len(texts),
            input_lens=[len(t) for t in texts],
            headers=trace.redact_headers(headers))
        t0 = time.monotonic()
        resp = post_with_retry(
            get_session(__name__), url,
            headers=headers, json=body, timeout=30.0)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code != 200:
            trace.event(
                'embed_response',
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                error='http_status')
            raise RuntimeError(
                f'embed request returned status {resp.status_code}')
        data = resp.json()
        items = data.get('data', [])
        if len(items) != len(texts):
            trace.event(
                'embed_response',
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
                error='length_mismatch',
                expected=len(texts),
                got=len(items))
            raise RuntimeError(
                f'embed request returned {len(items)} vectors for'
                f' {len(texts)} inputs')
        vectors = [item.get('embedding') for item in items]
        if any(v is None for v in vectors):
            raise RuntimeError('embed request returned a row with no embedding')
        # A component can arrive as a bare JSON int, and psycopg refuses
        # a list that mixes int and float.
        vectors = [[float(x) for x in vec] for vec in vectors]
        if self.dim == 0:
            self.dim = len(vectors[0])
        trace.event(
            'embed_response',
            status=resp.status_code,
            elapsed_ms=elapsed_ms,
            batch_size=len(vectors),
            dim=self.dim,
            usage=data.get('usage'))
        return vectors

    def unavailable_message(self) -> str:
        """Operator-facing reason the embed probe failed.
        """
        return (
            f'embed model {self.model!r} is not reachable at'
            f' {self.endpoint}; check {config.API_KEY} and that the'
            ' endpoint serves the model')
