"""Vector serialization and deserialization.
"""

import struct


def serialize_vector(v: list[float]) -> bytes:
    """Encode float64 vector as little-endian binary blob.
    """
    if not v:
        return b''
    return struct.pack(f'<{len(v)}d', *v)


def deserialize_vector(b: bytes) -> list[float] | None:
    """Decode little-endian binary blob to float64 vector.
    """
    if not b:
        return None
    if len(b) % 8 != 0:
        return None
    count = len(b) // 8
    return list(struct.unpack(f'<{count}d', b))
