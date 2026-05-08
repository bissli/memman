"""Artifact dataclass shape tests.

Pins the wire-shape of `Artifact` so cross-backend orchestration
can rely on `kind` being a typed Literal and `location` being a
string-or-None. The CLI's outro line for the migrate command
formats `f'(artifact={artifact.kind}: {artifact.location})'`, so
silent shape drift would corrupt operator-visible output.
"""
from __future__ import annotations

from memman.migrate import Artifact


def test_artifact_kind_filesystem_carries_location():
    """`kind='filesystem'` artifacts have a string location."""
    a = Artifact(
        kind='filesystem',
        location='/tmp/archive/store/20260507_01',
        metadata={})
    assert a.kind == 'filesystem'
    assert isinstance(a.location, str)
    assert a.metadata == {}


def test_artifact_kind_none_has_null_location():
    """`kind='none'` is reserved for backends with no source-side dump."""
    a = Artifact(
        kind='none', location=None,
        metadata={'reason': 'apply is a full migration'})
    assert a.kind == 'none'
    assert a.location is None
    assert a.metadata.get('reason')


def test_artifact_kind_object_store_string_location():
    """`kind='object_store'` reserves a uri-shaped location."""
    a = Artifact(
        kind='object_store',
        location='s3://memman-archives/store/20260507',
        metadata={'bucket': 'memman-archives'})
    assert a.kind == 'object_store'
    assert a.location.startswith('s3://')


def test_artifact_metadata_default_factory():
    """`metadata` defaults to an empty dict, not a shared reference."""
    a1 = Artifact(kind='none', location=None)
    a2 = Artifact(kind='none', location=None)
    a1.metadata['k'] = 'v'
    assert 'k' not in a2.metadata, (
        'Artifact.metadata default must be a fresh dict per instance,'
        ' not a shared mutable default')
