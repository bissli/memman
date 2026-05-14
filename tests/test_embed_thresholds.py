"""Unit tests for per-(provider, model, surface) AUTO_SEMANTIC_THRESHOLD.

Covers:
- `thresholds.resolve` returns the calibrated float for known triples.
- `thresholds.resolve` returns None for uncalibrated triples.
- `thresholds.resolve` default `surface='code'` resolves the code row.
- `thresholds.is_calibrated` matches the table membership.
- `thresholds.resolve_with_fallback` returns ('calibrated', val) for
  known triples and ('surface_median', val) for uncalibrated triples.
- `thresholds.SURFACE_FALLBACK` is the median of calibrated thresholds
  for each surface (deterministic, derived from the shipped table).
- `create_semantic_edges(threshold=None)` short-circuits to zero edges.
- `create_semantic_edges` honors an explicit `threshold=` kwarg.
- `doctor.check_embed_threshold` passes / warns based on calibration.
"""

from memman.doctor import check_embed_threshold
from memman.embed import thresholds
from memman.embed.fingerprint import Fingerprint, write_fingerprint
from memman.graph.semantic import create_semantic_edges
from tests.conftest import _vec as _vec_512
from tests.conftest import make_insight


def test_calibrated_pairs_match_documented_values():
    """Shipped calibrated entries match the sweep-measured values.

    Hardcoded regression guard so an accidental regeneration that
    rewrites the table with garbage (zeros, Nones) trips here before
    landing. Default surface='code' keys both lookups.
    """
    assert thresholds.resolve('voyage', 'voyage-3-lite') == 0.645
    assert thresholds.resolve('openrouter', 'baai/bge-m3') == 0.662
    assert thresholds.resolve(
        'voyage', 'voyage-3-lite', 'claw') == 0.497


def test_resolve_default_surface_is_code():
    """`resolve(provider, model)` is the same as `resolve(..., 'code')`.

    Default surface='code' means stores without explicit
    `MEMMAN_SURFACE_<store>` get the code-surface row.
    """
    assert thresholds.resolve('voyage', 'voyage-3-lite') == \
        thresholds.resolve('voyage', 'voyage-3-lite', 'code')


def test_resolve_unknown_pair_returns_none():
    """An uncalibrated `(provider, model)` returns None.
    """
    assert thresholds.resolve('fake', 'fake') is None
    assert thresholds.resolve('fake', 'fake', 'code') is None
    assert thresholds.resolve('fake', 'fake', 'claw') is None


def test_resolve_known_pair_unknown_surface_returns_none():
    """A known (provider, model) on a surface outside the table returns None.

    Calibrated surfaces are the closed set `{code, claw}`; any other
    value must miss the dict rather than fall back to the code row
    (would silently mis-calibrate). Guards against caller bugs that
    pass an arbitrary string.
    """
    assert thresholds.resolve(
        'voyage', 'voyage-3-lite', 'unknown') is None


def test_is_calibrated_table_membership():
    """`is_calibrated` reflects table membership; unknown triples are False.
    """
    assert thresholds.is_calibrated(
        'voyage', 'voyage-3-lite') is True
    assert thresholds.is_calibrated(
        'openrouter', 'baai/bge-m3', 'code') is True
    assert thresholds.is_calibrated(
        'voyage', 'voyage-3-lite', 'claw') is True
    assert thresholds.is_calibrated('fake', 'fake') is False
    assert thresholds.is_calibrated(
        'voyage', 'voyage-3-lite', 'unknown') is False


def test_surface_fallback_constants_match_calibrated_medians():
    """`SURFACE_FALLBACK[surf]` is the median of all calibrated thresholds
    for that surface in the shipped `_THRESHOLDS` table.
    """
    import statistics

    from memman.embed._thresholds_generated import _THRESHOLDS
    for surf in ('code', 'claw'):
        vals = [v for (p, m, s), v in _THRESHOLDS.items() if s == surf]
        expected = statistics.median(vals)
        assert thresholds.SURFACE_FALLBACK[surf] == expected


def test_resolve_with_fallback_calibrated_triple():
    """A calibrated triple resolves to ('calibrated', shipped_value).
    """
    value, source = thresholds.resolve_with_fallback(
        'voyage', 'voyage-3-lite', 'code')
    assert source == 'calibrated'
    assert value == 0.645


def test_resolve_with_fallback_uncalibrated_code():
    """An uncalibrated triple on code returns ('surface_median', median_code).
    """
    value, source = thresholds.resolve_with_fallback(
        'fake', 'fake-model', 'code')
    assert source == 'surface_median'
    assert value == thresholds.SURFACE_FALLBACK['code']


def test_resolve_with_fallback_uncalibrated_claw():
    """An uncalibrated triple on claw returns ('surface_median', median_claw).
    """
    value, source = thresholds.resolve_with_fallback(
        'fake', 'fake-model', 'claw')
    assert source == 'surface_median'
    assert value == thresholds.SURFACE_FALLBACK['claw']


def test_resolve_with_fallback_default_surface_is_code():
    """`resolve_with_fallback(p, m)` is the same as `(p, m, 'code')`.
    """
    assert thresholds.resolve_with_fallback('fake', 'x') == \
        thresholds.resolve_with_fallback('fake', 'x', 'code')


def test_resolve_with_fallback_unknown_surface_raises():
    """An unknown surface raises KeyError; surface is a closed set.
    """
    import pytest as _pytest
    with _pytest.raises(KeyError):
        thresholds.resolve_with_fallback('voyage', 'voyage-3-lite', 'legal')


def test_create_semantic_edges_skips_when_threshold_is_none(backend):
    """`threshold=None` short-circuits to zero edges even at cosine ~1.
    """
    ins1 = make_insight(id='nt-1', content='a')
    ins2 = make_insight(id='nt-2', content='b')
    backend.nodes.insert(ins1)
    backend.nodes.insert(ins2)

    vec1 = _vec_512(1.0, 0.0)
    vec2 = _vec_512(0.99, 0.01)
    cache = {'nt-1': vec1, 'nt-2': vec2}

    count = create_semantic_edges(
        backend, ins1, embed_cache=cache, threshold=None)
    assert count == 0


def test_create_semantic_edges_respects_explicit_threshold_kwarg(backend):
    """Explicit `threshold=` overrides the module default for the call.
    """
    ins1 = make_insight(id='et-1', content='a')
    ins2 = make_insight(id='et-2', content='b')
    backend.nodes.insert(ins1)
    backend.nodes.insert(ins2)

    vec1 = _vec_512(1.0, 0.0)
    vec2 = _vec_512(0.65, 0.76)
    cache = {'et-1': vec1, 'et-2': vec2}

    count_below = create_semantic_edges(
        backend, ins1, embed_cache=cache, threshold=0.55)
    assert count_below >= 2

    backend.edges.delete_auto_for_node('et-1', 'semantic')
    backend.edges.delete_auto_for_node('et-2', 'semantic')

    count_above = create_semantic_edges(
        backend, ins1, embed_cache=cache, threshold=0.75)
    assert count_above == 0


def test_doctor_passes_on_calibrated_fingerprint(backend):
    """Default-seeded `(voyage, voyage-3-lite, code)` reports pass with
    source='calibrated' and the calibrated threshold in the detail.
    """
    result = check_embed_threshold(backend)
    assert result['name'] == 'embed_threshold'
    assert result['status'] == 'pass'
    assert result['detail']['provider'] == 'voyage'
    assert result['detail']['model'] == 'voyage-3-lite'
    assert result['detail']['surface'] == 'code'
    assert result['detail']['source'] == 'calibrated'
    assert result['detail']['threshold'] == 0.645


def test_doctor_passes_on_calibrated_claw_fingerprint(backend, env_file):
    """`(voyage, voyage-3-lite, claw)` resolves to the claw-row threshold.
    """
    from memman import config
    env_file(config.SURFACE_FOR('clawstore'), 'claw')

    result = check_embed_threshold(backend, store_name='clawstore')
    assert result['name'] == 'embed_threshold'
    assert result['status'] == 'pass'
    assert result['detail']['surface'] == 'claw'
    assert result['detail']['source'] == 'calibrated'
    assert result['detail']['threshold'] == 0.497


def test_doctor_warns_on_uncalibrated_fingerprint_with_fallback(backend):
    """An uncalibrated fingerprint warns with source='surface_median'
    and the fallback threshold in the detail payload.
    """
    write_fingerprint(
        backend, Fingerprint(provider='fake', model='fake', dim=512))

    result = check_embed_threshold(backend)
    assert result['name'] == 'embed_threshold'
    assert result['status'] == 'warn'
    assert result['detail']['source'] == 'surface_median'
    assert result['detail']['threshold'] == thresholds.SURFACE_FALLBACK['code']
    assert result['detail']['surface'] == 'code'
    assert 'fallback' in result['detail']['hint']


def test_doctor_override_skip_with_claw_surface_hints_surface_moot(
        backend, env_file):
    """`skip` override + non-default surface adds a hint that surface is moot.

    The surface setting has no observable effect once the override
    disables semantic edges; doctor's hint helps operators clean up.
    """
    from memman import config
    env_file(config.SURFACE_FOR('skipclaw'), 'claw')
    env_file(config.AUTO_THRESHOLD_FOR('skipclaw'), 'skip')

    result = check_embed_threshold(backend, store_name='skipclaw')
    assert result['status'] == 'pass'
    assert result['detail']['source'] == 'override_skip'
    assert 'has no effect' in result['detail']['hint']


def test_doctor_override_masks_calibrated_emits_hint(
        backend, env_file):
    """A numeric override that masks a calibrated row emits a hint."""
    from memman import config
    env_file(config.AUTO_THRESHOLD_FOR('masked'), '0.55')

    result = check_embed_threshold(backend, store_name='masked')
    assert result['status'] == 'pass'
    assert result['detail']['source'] == 'override'
    assert result['detail']['threshold'] == 0.55
    assert result['detail']['masked_calibrated'] == 0.645
    assert 'masking the calibrated value' in result['detail']['hint']


def test_doctor_override_matches_calibrated_no_mask_hint(
        backend, env_file):
    """An override matching the calibrated value adds no masking hint."""
    from memman import config
    env_file(config.AUTO_THRESHOLD_FOR('exactmatch'), '0.645')

    result = check_embed_threshold(backend, store_name='exactmatch')
    assert result['status'] == 'pass'
    assert result['detail']['source'] == 'override'
    assert result['detail']['threshold'] == 0.645
    assert 'masked_calibrated' not in result['detail']
    assert 'hint' not in result['detail']
