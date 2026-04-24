"""Vector serialization, deserialization, and cosine similarity.

Cosine uses numpy for the hot path (recall ranks thousands of
embeddings per query) but keeps the list[float] public API so callers
don't have to think about ndarray types.
"""

import struct

import numpy as np


def cosine_similarity(
        a: list[float] | np.ndarray | None,
        b: list[float] | np.ndarray | None) -> float:
    """Compute cosine similarity between two vectors.

    None, empty, zero-norm, or mismatched-shape inputs return 0.0.
    Accepts Python lists or numpy arrays (avoids a list->ndarray
    round-trip when callers already have ndarray embeddings).
    """
    if a is None or b is None:
        return 0.0
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.size == 0 or bv.size == 0 or av.shape != bv.shape:
        return 0.0
    norm_a = float(np.linalg.norm(av))
    norm_b = float(np.linalg.norm(bv))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(av, bv)) / (norm_a * norm_b)


def serialize_vector(v: list[float]) -> bytes:
    """Encode float64 vector as little-endian binary blob."""
    if not v:
        return b''
    return struct.pack(f'<{len(v)}d', *v)


def deserialize_vector(b: bytes) -> list[float] | None:
    """Decode little-endian binary blob to float64 vector."""
    if not b:
        return None
    if len(b) % 8 != 0:
        return None
    count = len(b) // 8
    return list(struct.unpack(f'<{count}d', b))
