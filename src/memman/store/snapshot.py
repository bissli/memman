"""Recall snapshot file: read-side cache materialized by the worker.

The worker writes one binary snapshot per store after each drain that
processed rows. `recall` reads the snapshot instead of issuing
full-table scans (`get_all_active_insights`, `get_all_embeddings`,
per-node `get_edges_by_node`). Eliminates the synchronous DB I/O that
otherwise grows O(N) with the store size.

The file is written atomically (tmp + fsync + rename) so concurrent
recall readers either see the prior file or the new one in full,
never a partial write. Readers holding an open fd to the prior inode
keep reading the prior content (POSIX rename semantics).

Snapshots are stale by up to one drain interval; this is the same
visibility lag the queue already imposes on `remember -> recall` and
is acceptable.
"""

import json
import logging
import os
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from memman.embed.fingerprint import Fingerprint
from memman.embed.vector import deserialize_vector
from memman.model import Insight, parse_timestamp
from memman.store.edge import get_all_edges
from memman.store.node import get_all_active_insights, get_all_embeddings

logger = logging.getLogger('memman')

SNAPSHOT_FILENAME = 'recall_snapshot.v1.bin'
SNAPSHOT_TMP_SUFFIX = '.tmp'
SNAPSHOT_MAX_INSIGHTS = 1000
SNAPSHOT_MAGIC = b'MMS1'
SNAPSHOT_VERSION = 1


@dataclass
class Snapshot:
    """In-memory representation of a deserialized snapshot."""

    written_at: datetime
    embedding_dim: int
    embedding_model: str
    insights: list[Insight]
    embeddings: dict[str, list[float]]
    adjacency: dict[str, list[tuple[str, str, float]]]


def snapshot_path(store_dir: str) -> Path:
    """Return the canonical snapshot path for a store directory."""
    return Path(store_dir) / SNAPSHOT_FILENAME


def write_snapshot(db, store_dir: str, fingerprint: Fingerprint) -> bool:
    """Materialize the recall snapshot for a store.

    Skipped (returns False) when active-insight count exceeds
    `SNAPSHOT_MAX_INSIGHTS` — recall falls back to SQL in that regime
    rather than carrying a multi-megabyte file. Otherwise writes the
    snapshot atomically and returns True.
    """
    insights = get_all_active_insights(db)
    if len(insights) > SNAPSHOT_MAX_INSIGHTS:
        logger.info(
            f'snapshot skipped: {len(insights)} active insights exceeds'
            f' SNAPSHOT_MAX_INSIGHTS={SNAPSHOT_MAX_INSIGHTS}')
        return False

    embedding_pairs = []
    for eid, _content, blob in get_all_embeddings(db):
        vec = deserialize_vector(blob)
        if vec is None:
            continue
        if len(vec) != fingerprint.dim:
            continue
        embedding_pairs.append((eid, vec))

    insight_ids = [i.id for i in insights]
    embed_index = {eid: vec for eid, vec in embedding_pairs}
    ids_in_order = [iid for iid in insight_ids if iid in embed_index]

    embeddings_array = np.zeros(
        (len(ids_in_order), fingerprint.dim), dtype=np.float64)
    for row_idx, iid in enumerate(ids_in_order):
        embeddings_array[row_idx, :] = embed_index[iid]

    insight_meta = [
        {
            'id': i.id,
            'content': i.content,
            'category': i.category,
            'importance': i.importance,
            'entities': list(i.entities),
            'tags': list(i.tags),
            'source': i.source,
            'access_count': i.access_count,
            'created_at': i.created_at.astimezone(timezone.utc).isoformat(),
            }
        for i in insights
        ]

    adjacency: dict[str, list[list]] = {}
    for edge in get_all_edges(db):
        adjacency.setdefault(edge.source_id, []).append(
            [edge.target_id, edge.edge_type, edge.weight])

    header = {
        'version': SNAPSHOT_VERSION,
        'written_at': datetime.now(timezone.utc).isoformat(),
        'embedding_dim': fingerprint.dim,
        'embedding_model': (
            f'{fingerprint.provider}:{fingerprint.model}:{fingerprint.dim}'),
        'total_insights': len(insights),
        'embedded_ids': ids_in_order,
        }

    out_path = snapshot_path(store_dir)
    tmp_path = out_path.with_suffix(out_path.suffix + SNAPSHOT_TMP_SUFFIX)

    header_bytes = json.dumps(header, sort_keys=True).encode('utf-8')
    meta_bytes = json.dumps(insight_meta).encode('utf-8')
    adj_bytes = json.dumps(adjacency).encode('utf-8')
    embed_bytes = embeddings_array.tobytes(order='C')

    with open(tmp_path, 'wb') as f:
        f.write(SNAPSHOT_MAGIC)
        f.write(struct.pack('<I', len(header_bytes)))
        f.write(header_bytes)
        f.write(struct.pack('<Q', len(embed_bytes)))
        f.write(embed_bytes)
        f.write(struct.pack('<I', len(meta_bytes)))
        f.write(meta_bytes)
        f.write(struct.pack('<I', len(adj_bytes)))
        f.write(adj_bytes)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, out_path)
    logger.debug(
        f'snapshot wrote {out_path}'
        f' (insights={len(insights)} embedded={len(ids_in_order)}'
        f' edges={sum(len(v) for v in adjacency.values())})')
    return True


def read_snapshot(
        store_dir: str,
        active_fingerprint: Fingerprint) -> Snapshot | None:
    """Load and return the snapshot, or None when unusable.

    Returns None when:
    - the snapshot file is missing
    - the magic / version don't match
    - the embedding fingerprint doesn't match `active_fingerprint`
      (recall must fall back to SQL after a provider swap)
    - any structural read error occurs
    """
    path = snapshot_path(store_dir)
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != SNAPSHOT_MAGIC:
                logger.warning(f'snapshot {path} bad magic; ignoring')
                return None
            (header_len,) = struct.unpack('<I', f.read(4))
            header = json.loads(f.read(header_len).decode('utf-8'))
            if header.get('version') != SNAPSHOT_VERSION:
                logger.warning(
                    f'snapshot {path} version mismatch'
                    f' ({header.get("version")} != {SNAPSHOT_VERSION})')
                return None

            stored_model = header.get('embedding_model', '')
            active_model = (
                f'{active_fingerprint.provider}:'
                f'{active_fingerprint.model}:{active_fingerprint.dim}')
            if stored_model != active_model:
                logger.info(
                    f'snapshot fingerprint mismatch'
                    f' (stored={stored_model} active={active_model});'
                    ' falling back to SQL')
                return None

            (embed_len,) = struct.unpack('<Q', f.read(8))
            embed_buf = f.read(embed_len)
            (meta_len,) = struct.unpack('<I', f.read(4))
            meta_blob = f.read(meta_len).decode('utf-8')
            (adj_len,) = struct.unpack('<I', f.read(4))
            adj_blob = f.read(adj_len).decode('utf-8')
    except (OSError, ValueError, struct.error, json.JSONDecodeError) as exc:
        logger.warning(f'snapshot {path} unreadable: {exc}')
        return None

    embedded_ids = header.get('embedded_ids', [])
    dim = int(header.get('embedding_dim', 0))
    embeddings: dict[str, list[float]] = {}
    if embedded_ids and dim > 0:
        arr = np.frombuffer(embed_buf, dtype=np.float64)
        if arr.size == len(embedded_ids) * dim:
            arr = arr.reshape((len(embedded_ids), dim))
            for row_idx, iid in enumerate(embedded_ids):
                embeddings[iid] = arr[row_idx, :].tolist()
        else:
            logger.warning(
                f'snapshot {path} embedding shape mismatch'
                f' ({arr.size} != {len(embedded_ids)} * {dim})')

    meta_list = json.loads(meta_blob)
    insights: list[Insight] = []
    for entry in meta_list:
        insights.append(Insight(
            id=entry['id'],
            content=entry['content'],
            category=entry['category'],
            importance=int(entry['importance']),
            entities=list(entry.get('entities', [])),
            tags=list(entry.get('tags', [])),
            source=entry.get('source', 'user'),
            access_count=int(entry.get('access_count', 0)),
            created_at=parse_timestamp(entry['created_at']),
            ))

    raw_adj = json.loads(adj_blob)
    adjacency: dict[str, list[tuple[str, str, float]]] = {}
    for source_id, edges in raw_adj.items():
        adjacency[source_id] = [
            (e[0], e[1], float(e[2])) for e in edges]

    return Snapshot(
        written_at=parse_timestamp(header['written_at']),
        embedding_dim=dim,
        embedding_model=header.get('embedding_model', ''),
        insights=insights,
        embeddings=embeddings,
        adjacency=adjacency,
        )


def delete_snapshot(store_dir: str) -> None:
    """Remove the snapshot file if present (used by tests)."""
    path = snapshot_path(store_dir)
    if path.exists():
        path.unlink()
