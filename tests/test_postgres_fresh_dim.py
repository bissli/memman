"""A fresh Postgres schema takes its vector width from the embed model.
"""

import pytest

pytest.importorskip('psycopg')

from memman.embed.fingerprint import Fingerprint
from memman.exceptions import ConfigError
from memman.store import postgres as pg_mod
from memman.store.errors import BackendError


def _unreachable() -> Fingerprint:
    return Fingerprint(model='voyageai/voyage-4-lite', dim=0)


def _unconfigured() -> Fingerprint:
    raise ConfigError('MEMMAN_EMBED_MODEL is not set')


@pytest.mark.parametrize('active', [_unreachable, _unconfigured])
def test_fresh_schema_refuses_unknown_dim(monkeypatch, active):
    """Verify a fresh schema refuses to open when no embed dim is known.

    Mutation: falling back to a fixed vector width, which builds a column
        the configured model cannot fill, so every later write fails.
    Oracle: an unreachable model (dim 0) and an unset model, each of
        which leaves no width to build.
    """
    monkeypatch.setattr(pg_mod, 'seed_default_fingerprint', active)
    with pytest.raises(BackendError, match='vector column'):
        pg_mod._resolve_active_dim(expected_dim=None)
