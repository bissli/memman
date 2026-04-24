"""Click CLI for memman — all 16 commands."""

import json
import logging
import os
import pathlib
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone

import click
import memman
from memman.model import VALID_CATEGORIES, VALID_EDGE_TYPES, Edge, Insight
from memman.model import format_timestamp, is_immune
from memman.store.db import default_data_dir, list_stores, open_db
from memman.store.db import open_read_only, read_active, store_dir
from memman.store.db import store_exists, valid_store_name, write_active
from tqdm import tqdm

logger = logging.getLogger('memman')


def _json_out(obj: object) -> None:
    """Write JSON to stdout with 2-space indent, sorted keys."""
    click.echo(json.dumps(obj, indent=2, sort_keys=True))


def _resolve_store_name(data_dir: str, store_flag: str) -> str:
    """Resolve effective store name."""
    if store_flag:
        return store_flag
    env = os.environ.get('MEMMAN_STORE', '')
    if env:
        return env
    return read_active(data_dir)


def _open_db(ctx: click.Context) -> 'DB':
    """Open the database using context options."""
    data_dir = ctx.obj['data_dir']
    store_flag = ctx.obj['store']
    read_only = ctx.obj['readonly']

    name = _resolve_store_name(data_dir, store_flag)
    sdir = store_dir(data_dir, name)

    if read_only:
        return open_read_only(sdir)

    return open_db(sdir)


def _trunc_id(id: str) -> str:
    """Truncate an ID to 8 characters for display."""
    return id[:8] if len(id) > 8 else id


def _parse_since(since: str) -> str:
    """Parse a relative time string (e.g. '7d', '24h') to ISO timestamp."""
    m = re.match(r'^(\d+)([dhm])$', since)
    if not m:
        raise click.ClickException(
            f'Invalid --since format: {since} (use e.g. 7d, 24h, 30m)')
    val, unit = int(m.group(1)), m.group(2)
    delta = {'d': timedelta(days=val), 'h': timedelta(hours=val),
             'm': timedelta(minutes=val)}[unit]
    cutoff = datetime.now(timezone.utc) - delta
    return format_timestamp(cutoff)


def _insight_to_dict(i: Insight) -> dict:
    """Serialize an Insight for JSON output."""
    d = {
        'id': i.id,
        'content': i.content,
        'category': i.category,
        'importance': i.importance,
        'tags': i.tags,
        'entities': i.entities,
        'source': i.source,
        'access_count': i.access_count,
        'created_at': format_timestamp(i.created_at),
        'updated_at': format_timestamp(i.updated_at),
        }
    if i.deleted_at:
        d['deleted_at'] = format_timestamp(i.deleted_at)
    return d


def _parse_tags(tags: str) -> list[str]:
    """Parse and validate comma-separated tags."""
    tag_list: list[str] = []
    if not tags:
        return tag_list
    for t in tags.split(','):
        t = t.strip()
        if t:
            if len(t) > 100:
                raise click.ClickException(
                    f'tag too long ({len(t)} chars, max 100):'
                    f' {t[:50]}')
            tag_list.append(t)
    if len(tag_list) > 20:
        raise click.ClickException(
            f'too many tags ({len(tag_list)}, max 20)')
    return tag_list


def _parse_entities(entities: str) -> list[str]:
    """Parse and validate comma-separated entities."""
    entity_list: list[str] = []
    if not entities:
        return entity_list
    for e in entities.split(','):
        e = e.strip()
        if e:
            if len(e) > 200:
                raise click.ClickException(
                    f'entity too long ({len(e)} chars, max 200):'
                    f' {e[:50]}')
            entity_list.append(e)
    if len(entity_list) > 50:
        raise click.ClickException(
            f'too many entities ({len(entity_list)}, max 50)')
    return entity_list


@click.group()
@click.version_option(version=memman.__version__, prog_name='memman')
@click.option('--data-dir', default=None, help='Base data directory (env: MEMMAN_DATA_DIR)')
@click.option('--store', 'store_name', default='', help='Named memory store')
@click.option('--readonly', is_flag=True, default=False, help='Open database in read-only mode')
@click.pass_context
def cli(ctx: click.Context, data_dir: str | None, store_name: str, readonly: bool) -> None:
    """Memory daemon for LLM agents."""
    if data_dir is None:
        data_dir = os.environ.get('MEMMAN_DATA_DIR', default_data_dir())
    ctx.ensure_object(dict)
    ctx.obj['data_dir'] = data_dir
    ctx.obj['store'] = store_name
    ctx.obj['readonly'] = readonly


@cli.command()
@click.argument('content', nargs=-1, required=True)
@click.option('--cat', default='general', help='Category')
@click.option('--imp', default=3, type=int, help='Importance (1-5)')
@click.option('--tags', default='', help='Comma-separated tags')
@click.option('--source', default='user', help='Source')
@click.option('--entities', default='', help='Comma-separated entities')
@click.option('--no-reconcile', is_flag=True, default=False, help='Skip LLM reconciliation')
@click.option('--defer/--sync', 'defer', default=True,
              help='Defer to background worker (default) or run synchronously')
@click.option('--priority', default=0, type=int,
              help='Queue priority when --defer (higher drains first)')
@click.pass_context
def remember(ctx: click.Context, content: tuple[str, ...], cat: str,
             imp: int, tags: str, source: str, entities: str,
             no_reconcile: bool, defer: bool, priority: int) -> None:
    """Store a new insight."""
    content_str = ' '.join(content)
    content_bytes = len(content_str.encode('utf-8'))
    if content_bytes > 8000:
        raise click.ClickException(
            f'content too long ({content_bytes} bytes, max 8000);'
            ' consider chunking into multiple remember calls')

    if cat not in VALID_CATEGORIES:
        raise click.ClickException(
            f'invalid category {cat!r}; valid: preference, decision,'
            ' fact, insight, context, general')
    if imp < 1 or imp > 5:
        raise click.ClickException(
            f'importance must be 1-5, got {imp}')

    tag_list = _parse_tags(tags)
    entity_list = _parse_entities(entities)

    data_dir_val = ctx.obj['data_dir']
    store_flag = ctx.obj['store']
    name = _resolve_store_name(data_dir_val, store_flag)

    if defer:
        from memman.queue import enqueue, open_queue_db
        conn = open_queue_db(data_dir_val)
        try:
            cat_hint = cat if cat != 'general' else None
            imp_hint = imp if imp != 3 else None
            row_id = enqueue(
                conn, store=name, content=content_str,
                hint_cat=cat_hint, hint_imp=imp_hint,
                hint_tags=tags or None,
                hint_source=source if source != 'user' else None,
                hint_entities=entities or None,
                priority=priority)
        finally:
            conn.close()
        _json_out({
            'action': 'queued',
            'queue_id': row_id,
            'store': name,
            })
        return

    now = datetime.now(timezone.utc)
    insight = Insight(
        id=str(uuid.uuid4()), content=content_str,
        category=cat, importance=imp, tags=tag_list,
        entities=entity_list, source=source,
        created_at=now, updated_at=now)

    cat_explicit = (ctx.get_parameter_source('cat')
                    == click.core.ParameterSource.COMMANDLINE)
    imp_explicit = (ctx.get_parameter_source('imp')
                    == click.core.ParameterSource.COMMANDLINE)

    db = _open_db(ctx)
    try:
        _remember_impl(db, insight, content_str, no_reconcile,
                       data_dir=data_dir_val,
                       store_name=name,
                       cat_explicit=cat_explicit,
                       imp_explicit=imp_explicit)
    finally:
        db.close()


def _remember_impl(db: 'DB', insight: Insight, content: str,
                   no_reconcile: bool, replaced_id: str = '',
                   data_dir: str = '',
                   store_name: str = '',
                   cat_explicit: bool = False,
                   imp_explicit: bool = False) -> None:
    """Core remember implementation — single-tier synchronous write path.

    Sequential: quality check, embed, LLM fact extraction,
    LLM reconciliation, insert, fast edges, EI, prune.
    Parallel: LLM enrichment + LLM causal inference (ThreadPoolExecutor).
    Sequential: write enrichment results, stamp linked_at.
    """
    from concurrent.futures import ThreadPoolExecutor

    from memman.embed import get_client
    from memman.embed.vector import cosine_similarity, deserialize_vector
    from memman.embed.vector import serialize_vector
    from memman.graph.causal import infer_llm_causal_edges
    from memman.graph.engine import fast_edges
    from memman.graph.enrichment import build_enriched_text, enrich_with_llm
    from memman.graph.entity import create_entity_edges
    from memman.graph.semantic import create_semantic_edges
    from memman.llm.client import get_llm_client
    from memman.llm.extract import extract_facts, reconcile_memories
    from memman.search.keyword import keyword_search
    from memman.search.quality import check_content_quality
    from memman.store.edge import delete_auto_edges_for_node, insert_edge
    from memman.store.node import MAX_INSIGHTS, auto_prune
    from memman.store.node import get_all_active_insights, get_all_embeddings
    from memman.store.node import insert_insight, refresh_effective_importance
    from memman.store.node import soft_delete_insight, stamp_enriched
    from memman.store.node import stamp_linked, update_embedding
    from memman.store.node import update_enrichment, update_entities
    from memman.store.oplog import log_op

    quality_warnings = check_content_quality(content)
    if len(quality_warnings) >= 2:
        log_op(db, 'quality-reject', insight.id,
               f'{content[:200]}|warnings={quality_warnings}')
        _json_out({
            'id': insight.id,
            'content': content,
            'action': 'rejected',
            'quality_warnings': quality_warnings,
            })
        return

    llm_client = get_llm_client()
    ec = get_client()
    llm_calls = 0

    if no_reconcile:
        facts = [{'text': content, 'category': insight.category,
                  'importance': insight.importance,
                  'entities': []}]
    else:
        facts = extract_facts(llm_client, content)
        llm_calls += 1

        if not facts:
            _json_out({
                'id': insight.id,
                'content': content,
                'action': 'skipped',
                'skip_reason': 'trivial content',
                'quality_warnings': quality_warnings,
                'llm_calls': llm_calls,
                })
            return

    embed_cache: dict[str, list[float]] = {}
    db_embeds = get_all_embeddings(db)
    if db_embeds:
        for eid, _content, blob in db_embeds:
            v = deserialize_vector(blob)
            if v is not None:
                embed_cache[eid] = v

    fact_results = []
    deleted_ids: set[str] = set()

    for fact in facts:
        all_insights = get_all_active_insights(db)
        fact_text = fact['text']
        fact_category = (insight.category if cat_explicit
                         else fact.get('category', insight.category))
        fact_importance = (insight.importance if imp_explicit
                           else fact.get('importance', insight.importance))
        fact_entities = fact.get('entities', [])

        fact_vec = None
        fact_blob = None
        try:
            fact_vec = ec.embed(fact_text)
            fact_blob = serialize_vector(fact_vec)
        except Exception:
            pass

        action = 'ADD'
        target_id = None
        merged_text = None

        if replaced_id:
            action = 'REPLACE'
            target_id = replaced_id
            replaced_id = None
        elif not no_reconcile:
            keyword_hits = keyword_search(
                all_insights, fact_text, limit=5)
            similar: list[tuple[str, str]] = []
            seen_ids: set[str] = set()

            for hit_ins, _score in keyword_hits:
                if hit_ins.id not in seen_ids:
                    similar.append((hit_ins.id, hit_ins.content))
                    seen_ids.add(hit_ins.id)

            if fact_vec is not None:
                for eid, evec in embed_cache.items():
                    if eid in seen_ids or eid in deleted_ids:
                        continue
                    sim = cosine_similarity(fact_vec, evec)
                    if sim >= 0.5:
                        ins = next(
                            (i for i in all_insights if i.id == eid),
                            None)
                        if ins is not None:
                            similar.append((ins.id, ins.content))
                            seen_ids.add(eid)
                    if len(similar) >= 10:
                        break

            if similar:
                recon = reconcile_memories(
                    llm_client, [fact], similar)
                llm_calls += 1
                if recon:
                    r = recon[0]
                    action = r['action']
                    target_id = r.get('target_id')
                    merged_text = r.get('merged_text')

        if (action in {'UPDATE', 'REPLACE'}
                and target_id
                and target_id in deleted_ids):
            fact_results.append({
                'id': str(uuid.uuid4()),
                'content': merged_text or fact_text,
                'action': 'skipped',
                'reason': 'target already deleted',
                })
            continue

        fact_id = str(uuid.uuid4())
        effective_text = merged_text or fact_text

        fact_insight = Insight(
            id=fact_id,
            content=effective_text,
            category=fact_category,
            importance=fact_importance,
            tags=list(insight.tags),
            entities=fact_entities + list(insight.entities),
            source=insight.source,
            access_count=insight.access_count,
            created_at=insight.created_at,
            updated_at=insight.updated_at)

        embedded = False
        embed_vec = fact_vec
        embed_blob = fact_blob
        if merged_text:
            try:
                embed_vec = ec.embed(effective_text)
                embed_blob = serialize_vector(embed_vec)
            except Exception:
                pass

        if action == 'NONE':
            fact_results.append({
                'id': fact_id,
                'content': effective_text,
                'action': 'skipped',
                'reason': 'already captured',
                })
            continue

        if action == 'DELETE' and target_id:
            if target_id in deleted_ids:
                fact_results.append({
                    'id': fact_id,
                    'content': effective_text,
                    'action': 'skipped',
                    'reason': 'target already deleted',
                    })
                continue
            deleted_ids.add(target_id)

            def delete_tx(tid: str = target_id) -> None:
                soft_delete_insight(db, tid)
                log_op(db, 'reconcile-delete', tid,
                       f'contradicted by: {fact_text[:200]}')
            db.in_transaction(delete_tx)
            embed_cache.pop(target_id, None)
            fact_results.append({
                'id': fact_id,
                'content': effective_text,
                'action': 'deleted',
                'target_id': target_id,
                })
            continue

        edge_stats = {'temporal': 0, 'entity': 0, 'semantic': 0}
        ei = 0.0
        pruned = 0

        def insert_tx(
                fi: Insight = fact_insight,
                eb: bytes | None = embed_blob,
                act: str = action,
                tid: str | None = target_id) -> None:
            nonlocal embedded

            if act in {'UPDATE', 'REPLACE'} and tid:
                if tid not in deleted_ids:
                    deleted_ids.add(tid)
                    op_name = ('replace' if act == 'REPLACE'
                               else 'reconcile-update')
                    soft_delete_insight(db, tid)
                    log_op(db, op_name, tid,
                           f'replaced by {fi.id}')

            insert_insight(db, fi)

            if eb is not None:
                update_embedding(db, fi.id, eb)
                embedded = True

            if fi.entities:
                update_entities(db, fi.id, fi.entities)

            log_op(db, 'remember', fi.id, fi.content)

        db.in_transaction(insert_tx)

        if action in {'UPDATE', 'REPLACE'} and target_id:
            embed_cache.pop(target_id, None)

        if embed_vec is not None:
            embed_cache[fact_insight.id] = embed_vec

        def edge_tx(
                fi: Insight = fact_insight) -> None:
            nonlocal edge_stats, ei, pruned
            edge_stats = fast_edges(db, fi)
            edge_stats['semantic'] = create_semantic_edges(
                db, fi, embed_cache)
            try:
                ei = refresh_effective_importance(db, fi.id)
            except Exception:
                ei = 0.0
            try:
                pruned = auto_prune(db, MAX_INSIGHTS, [fi.id])
            except Exception:
                pruned = 0

        db.in_transaction(edge_tx)

        enrichment: dict = {}
        causal_edges: list = []
        data_dir_for_ro = pathlib.Path(db.path).parent

        def _do_enrich() -> dict:
            return enrich_with_llm(fact_insight, llm_client)

        def _do_causal() -> list:
            ro_db = open_read_only(data_dir_for_ro)
            try:
                return infer_llm_causal_edges(
                    ro_db, fact_insight, llm_client)
            finally:
                ro_db.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_e = pool.submit(_do_enrich)
            fut_c = pool.submit(_do_causal)
            try:
                enrichment = fut_e.result()
                llm_calls += 1
            except Exception:
                enrichment = {}
            try:
                causal_edges = fut_c.result()
                llm_calls += 1
            except Exception:
                causal_edges = []

        enriched_vec = None
        keywords = enrichment.get('keywords', [])
        if keywords and ec is not None and ec.available():
            enriched_text = build_enriched_text(
                fact_insight.content, keywords)
            try:
                enriched_vec = ec.embed(enriched_text)
                embed_cache[fact_insight.id] = enriched_vec
                embedded = True
            except Exception:
                pass

        def enrichment_tx(
                fi: Insight = fact_insight,
                evec: list[float] | None = enriched_vec) -> None:
            nonlocal edge_stats
            now = format_timestamp(datetime.now(timezone.utc))

            if enrichment:
                update_enrichment(
                    db, fi.id,
                    enrichment.get('keywords', []),
                    enrichment.get('summary', ''),
                    enrichment.get('semantic_facts', []))
                update_entities(
                    db, fi.id, enrichment.get('entities', []))
                fi.entities = enrichment.get('entities', [])

            if evec is not None:
                update_embedding(
                    db, fi.id, serialize_vector(evec))

            delete_auto_edges_for_node(db, fi.id, 'entity')
            edge_stats['entity'] = create_entity_edges(db, fi)

            delete_auto_edges_for_node(db, fi.id, 'semantic')
            edge_stats['semantic'] = create_semantic_edges(
                db, fi, embed_cache)

            delete_auto_edges_for_node(db, fi.id, 'causal')
            for edge in causal_edges:
                try:
                    insert_edge(db, edge)
                except Exception:
                    pass

            stamp_linked(db, fi.id, now)
            if enrichment:
                stamp_enriched(db, fi.id, now)

        db.in_transaction(enrichment_tx)

        fact_results.append({
            'id': fact_insight.id,
            'content': fact_insight.content,
            'category': fact_insight.category,
            'importance': fact_insight.importance,
            'tags': fact_insight.tags,
            'entities': fact_insight.entities,
            'action': action.lower(),
            'created_at': format_timestamp(fact_insight.created_at),
            'edges_created': {
                **edge_stats,
                'causal': len(causal_edges),
                },
            'enrichment': {
                'keywords': enrichment.get('keywords', []),
                'summary': enrichment.get('summary', ''),
                'entities': enrichment.get('entities', []),
                'semantic_facts': enrichment.get(
                    'semantic_facts', []),
                },
            'embedded': embedded,
            'effective_importance': ei,
            'auto_pruned': pruned,
            **({'replaced_id': target_id} if target_id else {}),
            })

    _json_out({
        'facts': fact_results,
        'quality_warnings': quality_warnings,
        'llm_calls': llm_calls,
        })


@cli.command()
@click.option('--pending', is_flag=True, default=False,
              help='Drain the deferred-write queue')
@click.option('--limit', default=100, type=int,
              help='Max blobs processed per invocation')
@click.option('--timeout', default=300, type=int,
              help='Max wall-clock seconds per invocation')
@click.option('--stores', default='',
              help='Comma-separated store names; default all')
@click.option('--verbose', is_flag=True, default=False,
              help='Echo per-blob progress')
@click.pass_context
def enrich(ctx: click.Context, pending: bool, limit: int,
           timeout: int, stores: str, verbose: bool) -> None:
    """Background enrichment worker (drains the queue)."""
    if not pending:
        raise click.ClickException(
            'only --pending mode is supported; pass --pending explicitly')
    _drain_queue(ctx, limit, timeout, stores, verbose)


def _drain_queue(ctx: click.Context, limit: int, timeout: int,
                 stores_filter: str, verbose: bool) -> None:
    """Claim and process queue rows until limit, timeout, or empty."""
    import time as _time

    from memman.queue import claim, mark_done, mark_failed, open_queue_db
    from memman.queue import stats

    data_dir_val = ctx.obj['data_dir']
    worker_pid = os.getpid()
    deadline = _time.monotonic() + timeout
    store_list = [s.strip() for s in stores_filter.split(',') if s.strip()]

    conn = open_queue_db(data_dir_val)
    processed = 0
    failed = 0

    try:
        while processed + failed < limit:
            if _time.monotonic() >= deadline:
                logger.info(f'enrich: timeout after {timeout}s')
                break
            row = claim(conn, worker_pid=worker_pid,
                        stores=store_list or None)
            if row is None:
                break

            sdir = store_dir(data_dir_val, row.store)

            try:
                _process_queue_row(row, sdir, data_dir_val)
                mark_done(conn, row.id)
                processed += 1
                if verbose:
                    click.echo(
                        f'[enrich] done id={row.id} store={row.store}',
                        err=True)
            except Exception as exc:
                mark_failed(conn, row.id, f'{type(exc).__name__}: {exc}')
                failed += 1
                if verbose:
                    click.echo(
                        f'[enrich] fail id={row.id} store={row.store}'
                        f' err={exc}', err=True)
                logger.exception(f'enrich row {row.id} failed')
    finally:
        s = stats(conn)
        conn.close()

    _json_out({
        'processed': processed,
        'failed': failed,
        'remaining': s,
        })


def _process_queue_row(row, store_data_dir: str,
                       base_data_dir: str) -> None:
    """Run the full remember pipeline on a claimed queue row."""
    from memman.store.db import open_db as _open_store_db

    tag_list = _parse_tags(row.hint_tags or '')
    entity_list = _parse_entities(row.hint_entities or '')
    category = row.hint_cat or 'general'
    importance = row.hint_imp if row.hint_imp is not None else 3
    source = row.hint_source or 'queue'

    if category not in VALID_CATEGORIES:
        category = 'general'
    if importance < 1 or importance > 5:
        importance = 3

    now = datetime.now(timezone.utc)
    insight = Insight(
        id=str(uuid.uuid4()), content=row.content,
        category=category, importance=importance,
        tags=tag_list, entities=entity_list, source=source,
        created_at=now, updated_at=now)

    db = _open_store_db(store_data_dir)
    try:
        _remember_impl(
            db, insight, row.content,
            no_reconcile=False,
            data_dir=base_data_dir,
            store_name=row.store,
            cat_explicit=row.hint_cat is not None,
            imp_explicit=row.hint_imp is not None)
    finally:
        db.close()


@cli.command()
@click.argument('keyword', nargs=-1, required=True)
@click.option('--cat', default='', help='Filter by category')
@click.option('--limit', default=10, type=int, help='Max results')
@click.option('--source', default='', help='Filter by source')
@click.option('--basic', is_flag=True, default=False, help='Simple SQL LIKE matching')
@click.option('--intent', default='', help='Override intent')
@click.pass_context
def recall(ctx: click.Context, keyword: tuple[str, ...], cat: str,
           limit: int, source: str, basic: bool,
           intent: str) -> None:
    """Retrieve insights by keyword."""
    from memman.embed import get_client
    from memman.llm.client import get_llm_client
    from memman.llm.extract import expand_query
    from memman.search.intent import intent_from_string
    from memman.search.recall import intent_aware_recall
    from memman.store.node import increment_access_count, query_insights
    from memman.store.oplog import log_op

    keyword_str = ' '.join(keyword)
    db = _open_db(ctx)
    try:
        if basic:
            results = query_insights(
                db, keyword=keyword_str, category=cat,
                source=source, limit=limit)
            for r in results:
                increment_access_count(db, r.id)
                r.access_count += 1
            log_op(db, 'recall:basic', '',
                   f'q={keyword_str} hits={len(results)}')
            _json_out([_insight_to_dict(r) for r in results])
            return

        llm_client = get_llm_client()
        expansion = expand_query(llm_client, keyword_str)
        keyword_str = expansion['expanded_query']

        intent_override = None
        if intent:
            try:
                intent_override = intent_from_string(intent)
            except ValueError as e:
                raise click.ClickException(str(e))
        elif expansion.get('intent'):
            try:
                intent_override = intent_from_string(
                    expansion['intent'])
            except ValueError:
                pass

        ec = get_client()
        query_vec = None
        try:
            query_vec = ec.embed(keyword_str)
        except Exception:
            pass

        query_entities = list(expansion.get('entities', []))

        fetch_limit = limit * 3 if (cat or source) else limit
        resp = intent_aware_recall(
            db, keyword_str, query_vec, query_entities,
            fetch_limit, intent_override)
        if cat:
            resp['results'] = [
                r for r in resp['results']
                if r['insight'].category == cat][:limit]
        if source:
            resp['results'] = [
                r for r in resp['results']
                if r['insight'].source == source][:limit]

        for r in resp['results']:
            increment_access_count(db, r['insight'].id)
            r['insight'].access_count += 1

        hits = [{'id': r['insight'].id[:8], 'via': r.get('via', ''),
                 'score': round(r['score'], 3),
                 'kw': round(r['signals']['keyword'], 3),
                 'sim': round(r['signals']['similarity'], 3),
                 'gr': round(r['signals']['graph'], 3),
                 'ent': round(r['signals']['entity'], 3)}
                for r in resp['results']]
        log_op(db, 'recall-detail', '',
               json.dumps({'intent': resp['meta']['intent'],
                           'q': keyword_str[:80], 'hits': hits}))

        out = {
            'results': [
                {
                    'insight': _insight_to_dict(r['insight']),
                    'score': r['score'],
                    'intent': r['intent'],
                    'signals': r['signals'],
                    **({'via': r['via']} if r.get('via') else {}),
                    }
                for r in resp['results']
                ],
            'meta': resp['meta'],
            }
        _json_out(out)
    finally:
        db.close()


@cli.command()
@click.argument('query', nargs=-1, required=True)
@click.option('--limit', default=10, type=int, help='Max results')
@click.pass_context
def search(ctx: click.Context, query: tuple[str, ...], limit: int) -> None:
    """Token-based keyword search."""
    from memman.search.keyword import keyword_search
    from memman.store.node import get_all_active_insights
    from memman.store.node import increment_access_count
    from memman.store.oplog import log_op

    query_str = ' '.join(query)
    db = _open_db(ctx)
    try:
        all_insights = get_all_active_insights(db)
        results = keyword_search(all_insights, query_str, limit)
        for ins, _score in results:
            increment_access_count(db, ins.id)
        log_op(db, 'search', '',
               f'q={query_str} hits={len(results)}')
        out = [
            {
                'id': ins.id,
                'content': ins.content,
                'category': ins.category,
                'importance': ins.importance,
                'tags': ins.tags,
                'score': score,
                }
            for ins, score in results
            ]
        _json_out(out)
    finally:
        db.close()


@cli.command()
@click.argument('id')
@click.pass_context
def forget(ctx: click.Context, id: str) -> None:
    """Soft-delete an insight."""
    from memman.store.node import soft_delete_insight
    from memman.store.oplog import log_op

    db = _open_db(ctx)
    try:
        def do_forget() -> None:
            soft_delete_insight(db, id)
            log_op(db, 'forget', id, '')

        db.in_transaction(do_forget)
        _json_out({
            'id': id,
            'status': 'deleted',
            'message': 'Insight soft-deleted successfully',
            })
    except ValueError as e:
        raise click.ClickException(str(e))
    finally:
        db.close()


@cli.command()
@click.argument('id')
@click.argument('content', nargs=-1, required=True)
@click.option('--cat', default='general', help='Category')
@click.option('--imp', default=3, type=int, help='Importance (1-5)')
@click.option('--tags', default='', help='Comma-separated tags')
@click.option('--source', default='user', help='Source')
@click.option('--entities', default='', help='Comma-separated entities')
@click.pass_context
def replace(ctx: click.Context, id: str, content: tuple[str, ...],
            cat: str, imp: int, tags: str, source: str,
            entities: str) -> None:
    """Replace an insight by ID with new content."""
    from memman.store.node import get_insight_by_id

    content_str = ' '.join(content)
    content_bytes = len(content_str.encode('utf-8'))
    if content_bytes > 8000:
        raise click.ClickException(
            f'content too long ({content_bytes} bytes, max 8000);'
            ' consider chunking into multiple remember calls')

    if cat not in VALID_CATEGORIES:
        raise click.ClickException(
            f'invalid category {cat!r}; valid: preference, decision,'
            ' fact, insight, context, general')
    if imp < 1 or imp > 5:
        raise click.ClickException(
            f'importance must be 1-5, got {imp}')

    db = _open_db(ctx)
    try:
        old = get_insight_by_id(db, id)
        if old is None:
            raise click.ClickException(
                f'insight {id} not found or already deleted')

        cat_src = ctx.get_parameter_source('cat')
        imp_src = ctx.get_parameter_source('imp')
        tags_src = ctx.get_parameter_source('tags')
        source_src = ctx.get_parameter_source('source')
        entities_src = ctx.get_parameter_source('entities')

        if cat_src != click.core.ParameterSource.COMMANDLINE:
            cat = old.category
        if imp_src != click.core.ParameterSource.COMMANDLINE:
            imp = old.importance
        if tags_src != click.core.ParameterSource.COMMANDLINE:
            tag_list = list(old.tags)
        else:
            tag_list = _parse_tags(tags)
        if source_src != click.core.ParameterSource.COMMANDLINE:
            source = old.source
        if entities_src != click.core.ParameterSource.COMMANDLINE:
            entity_list = list(old.entities)
        else:
            entity_list = _parse_entities(entities)

        now = datetime.now(timezone.utc)
        new_insight = Insight(
            id=str(uuid.uuid4()), content=content_str,
            category=cat, importance=imp, tags=tag_list,
            entities=entity_list, source=source,
            access_count=old.access_count,
            created_at=now, updated_at=now)

        data_dir_val = ctx.obj['data_dir']
        store_flag = ctx.obj['store']
        name = _resolve_store_name(data_dir_val, store_flag)

        _remember_impl(db, new_insight, content_str,
                       no_reconcile=True, replaced_id=id,
                       data_dir=data_dir_val,
                       store_name=name,
                       cat_explicit=True,
                       imp_explicit=True)
    finally:
        db.close()


@cli.command()
@click.argument('source_id')
@click.argument('target_id')
@click.option('--type', 'edge_type', default='semantic', help='Edge type')
@click.option('--weight', default=0.5, type=float, help='Edge weight')
@click.option('--meta', default='', help='JSON metadata')
@click.pass_context
def link(ctx: click.Context, source_id: str, target_id: str,
         edge_type: str, weight: float, meta: str) -> None:
    """Create a manual edge between two insights."""
    from memman.store.edge import insert_edge
    from memman.store.node import get_insight_by_id
    from memman.store.oplog import log_op

    if edge_type not in VALID_EDGE_TYPES:
        raise click.ClickException(
            f'invalid edge type {edge_type!r}')

    if weight < 0.0 or weight > 1.0:
        raise click.ClickException(
            'weight must be between 0.0 and 1.0')

    metadata: dict[str, str] = {}
    if meta:
        try:
            metadata = json.loads(meta)
        except json.JSONDecodeError as e:
            raise click.ClickException(
                f'invalid JSON metadata: {e}')
        if not isinstance(metadata, dict):
            raise click.ClickException(
                'metadata must be a JSON object, not '
                + type(metadata).__name__)
    metadata['created_by'] = 'claude'

    if source_id == target_id:
        raise click.ClickException(
            'cannot link an insight to itself')

    now = datetime.now(timezone.utc)
    db = _open_db(ctx)
    try:
        if get_insight_by_id(db, source_id) is None:
            raise click.ClickException(
                f'insight {source_id} not found')
        if get_insight_by_id(db, target_id) is None:
            raise click.ClickException(
                f'insight {target_id} not found')

        existing = db._query(
            'SELECT weight FROM edges'
            ' WHERE source_id = ? AND target_id = ? AND edge_type = ?',
            (source_id, target_id, edge_type)).fetchone()
        existing_weight = existing[0] if existing else None

        def do_link() -> None:
            insert_edge(db, Edge(
                source_id=source_id, target_id=target_id,
                edge_type=edge_type, weight=weight,
                metadata=metadata, created_at=now))
            insert_edge(db, Edge(
                source_id=target_id, target_id=source_id,
                edge_type=edge_type, weight=weight,
                metadata=metadata, created_at=now))
            log_op(db, 'link', source_id,
                   f'{source_id} <-> {target_id} ({edge_type})')

        db.in_transaction(do_link)
        actual = db._query(
            'SELECT weight FROM edges'
            ' WHERE source_id = ? AND target_id = ? AND edge_type = ?',
            (source_id, target_id, edge_type)).fetchone()
        actual_weight = actual[0] if actual else weight
        out = {
            'status': 'linked',
            'source_id': source_id,
            'target_id': target_id,
            'edge_type': edge_type,
            'weight': actual_weight,
            'metadata': metadata,
            }
        if existing_weight is not None and existing_weight > weight:
            out['warning'] = (
                f'existing weight {existing_weight} > requested'
                f' {weight}; kept higher')
        _json_out(out)
    finally:
        db.close()


@cli.command()
@click.argument('id')
@click.option('--edge', default='', help='Filter by edge type')
@click.option('--depth', default=2, type=int, help='Max traversal depth')
@click.pass_context
def related(ctx: click.Context, id: str, edge: str,
            depth: int) -> None:
    """Find connected insights via graph traversal."""
    from memman.graph.bfs import BFSOptions, bfs

    db = _open_db(ctx)
    try:
        nodes = bfs(db, id, BFSOptions(
            max_depth=depth, max_nodes=0, edge_filter=edge))
        out = []
        for n in nodes:
            entry: dict = {
                'id': n['insight'].id,
                'content': n['insight'].content,
                'category': n['insight'].category,
                'importance': n['insight'].importance,
                'depth': n['hop'],
                }
            if n.get('via_edge'):
                entry['via_edge_type'] = n['via_edge']
            out.append(entry)
        _json_out(out)
    finally:
        db.close()


@cli.group(invoke_without_command=True)
@click.pass_context
def queue(ctx: click.Context) -> None:
    """Inspect and manage the deferred-write queue."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(queue_list)


@queue.command('list')
@click.option('--limit', default=50, type=int)
@click.pass_context
def queue_list(ctx: click.Context, limit: int) -> None:
    """List recent queue rows."""
    from memman.queue import list_rows, open_queue_db, stats
    conn = open_queue_db(ctx.obj['data_dir'])
    try:
        _json_out({
            'stats': stats(conn),
            'rows': list_rows(conn, limit=limit),
            })
    finally:
        conn.close()


@queue.command('list-failed')
@click.option('--limit', default=50, type=int)
@click.pass_context
def queue_list_failed(ctx: click.Context, limit: int) -> None:
    """List failed queue rows."""
    from memman.queue import STATUS_FAILED, list_rows, open_queue_db
    conn = open_queue_db(ctx.obj['data_dir'])
    try:
        _json_out(list_rows(conn, status=STATUS_FAILED, limit=limit))
    finally:
        conn.close()


@queue.command('cat')
@click.argument('row_id', type=int)
@click.pass_context
def queue_cat(ctx: click.Context, row_id: int) -> None:
    """Print the full content of a queue row."""
    from memman.queue import get_row, open_queue_db
    conn = open_queue_db(ctx.obj['data_dir'])
    try:
        row = get_row(conn, row_id)
        if row is None:
            raise click.ClickException(f'queue row {row_id} not found')
        _json_out(row)
    finally:
        conn.close()


@queue.command('retry')
@click.argument('row_id', type=int)
@click.pass_context
def queue_retry(ctx: click.Context, row_id: int) -> None:
    """Re-queue a failed row."""
    from memman.queue import open_queue_db, retry_row
    conn = open_queue_db(ctx.obj['data_dir'])
    try:
        if not retry_row(conn, row_id):
            raise click.ClickException(
                f'queue row {row_id} not found or not in failed state')
        _json_out({'action': 'requeued', 'queue_id': row_id})
    finally:
        conn.close()


@queue.command('purge')
@click.option('--done', is_flag=True, default=False,
              help='Delete all rows with status=done')
@click.pass_context
def queue_purge(ctx: click.Context, done: bool) -> None:
    """Remove completed queue rows."""
    if not done:
        raise click.ClickException(
            'pass --done to confirm deletion of completed rows')
    from memman.queue import open_queue_db, purge_done
    conn = open_queue_db(ctx.obj['data_dir'])
    try:
        deleted = purge_done(conn)
        _json_out({'deleted': deleted})
    finally:
        conn.close()


@cli.group(invoke_without_command=True)
@click.pass_context
def store(ctx: click.Context) -> None:
    """Manage named memory stores."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(store_list)


@store.command('list')
@click.pass_context
def store_list(ctx: click.Context) -> None:
    """List all stores."""
    data_dir = ctx.obj['data_dir']
    stores = list_stores(data_dir)
    if not stores:
        click.echo(
            "  (no stores yet — run 'memman store create <name>'"
            " or any command to create default)")
        return
    active = _resolve_store_name(data_dir, ctx.obj['store'])
    for name in stores:
        prefix = '* ' if name == active else '  '
        click.echo(f'{prefix}{name}')


@store.command('create')
@click.argument('name')
@click.pass_context
def store_create(ctx: click.Context, name: str) -> None:
    """Create a new store."""
    data_dir = ctx.obj['data_dir']
    if not valid_store_name(name):
        raise click.ClickException(
            f'invalid store name {name!r}')
    if store_exists(data_dir, name):
        raise click.ClickException(
            f'store "{name}" already exists')
    sdir = store_dir(data_dir, name)
    db = open_db(sdir)
    db.close()
    click.echo(f'Created store "{name}"')


@store.command('set')
@click.argument('name')
@click.pass_context
def store_set(ctx: click.Context, name: str) -> None:
    """Set the active store."""
    data_dir = ctx.obj['data_dir']
    if not store_exists(data_dir, name):
        raise click.ClickException(
            f"store \"{name}\" does not exist"
            f" (use 'memman store create {name}' first)")
    write_active(data_dir, name)
    click.echo(f'Active store set to "{name}"')


@store.command('remove')
@click.argument('name')
@click.pass_context
def store_remove(ctx: click.Context, name: str) -> None:
    """Remove a store."""
    import shutil
    data_dir = ctx.obj['data_dir']
    if not store_exists(data_dir, name):
        raise click.ClickException(
            f"store \"{name}\" does not exist"
            f" (use 'memman store create {name}' first)")
    active = read_active(data_dir)
    if name == active:
        raise click.ClickException(
            f"cannot remove the active store \"{name}\""
            f" (switch first with 'memman store set <other>')")
    sdir = store_dir(data_dir, name)
    shutil.rmtree(sdir)
    click.echo(f'Removed store "{name}"')


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show database statistics."""
    from memman.store.node import get_stats

    db = _open_db(ctx)
    try:
        stats = get_stats(db)
        stats['db_path'] = db.path
        try:
            stats['db_size_bytes'] = pathlib.Path(db.path).stat().st_size
        except OSError:
            stats['db_size_bytes'] = 0
        _json_out(stats)
    finally:
        db.close()


@cli.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Run health checks on the database."""
    from memman.doctor import run_all_checks

    db = _open_db(ctx)
    try:
        result = run_all_checks(db, data_dir=ctx.obj['data_dir'])
        result['store'] = _resolve_store_name(
            ctx.obj['data_dir'], ctx.obj['store'])
        result['db_path'] = db.path
        _json_out(result)
    finally:
        db.close()


@cli.command()
@click.option('--limit', default=20, type=int, help='Max entries')
@click.option('--since', default='', help='Time window (e.g. 7d, 24h)')
@click.option('--group-by', 'group_by', default='',
              help='Group by field (operation)')
@click.option('--stats', is_flag=True, default=False,
              help='Show summary statistics')
@click.pass_context
def log(ctx: click.Context, limit: int, since: str, group_by: str,
        stats: bool) -> None:
    """Show operation log."""
    from memman.store.oplog import get_oplog, get_oplog_stats

    since_ts = ''
    if since:
        since_ts = _parse_since(since)

    db = _open_db(ctx)
    try:
        if stats or group_by:
            stats_data = get_oplog_stats(db, since_ts)
            _json_out(stats_data)
            return

        entries = get_oplog(db, limit, since_ts)
        if not entries:
            click.echo('No operations recorded yet.')
            return

        headers = ['TIME', 'OP', 'INSIGHT', 'DETAIL']
        sep = ['----', '--', '-------', '------']
        rows = []
        for e in entries:
            detail = e['detail']
            if len(detail) > 60:
                detail = detail[:57] + '...'
            rows.append([
                e['created_at'],
                e['operation'],
                e['insight_id'] or '',
                detail,
                ])

        all_rows = [headers, sep] + rows
        widths = [0] * 4
        for row in all_rows:
            for i, col in enumerate(row):
                widths[i] = max(widths[i], len(col))

        for row in all_rows:
            line = '  '.join(
                col.ljust(widths[i]) for i, col in enumerate(row))
            click.echo(line.rstrip())
    finally:
        db.close()


@cli.command()
@click.option('--threshold', default=0.5, type=float, help='EI threshold')
@click.option('--limit', default=20, type=int, help='Max candidates')
@click.option('--keep', default='', help='Insight ID to keep')
@click.option('--review', is_flag=True, default=False,
              help='Review stored insights for content quality issues')
@click.pass_context
def gc(ctx: click.Context, threshold: float, limit: int, keep: str,
       review: bool) -> None:
    """Garbage collection / retention lifecycle."""
    from memman.store.node import MAX_INSIGHTS, boost_retention
    from memman.store.node import get_insight_by_id, get_retention_candidates
    from memman.store.node import refresh_effective_importance
    from memman.store.node import review_content_quality
    from memman.store.oplog import log_op

    db = _open_db(ctx)
    try:
        if review:
            flagged = review_content_quality(db, limit)
            _json_out({
                'review_results': [{
                    'id': f['insight'].id,
                    'content': f['insight'].content,
                    'importance': f['insight'].importance,
                    'quality_warnings': f['quality_warnings'],
                    } for f in flagged],
                'total_flagged': len(flagged),
                'actions': {
                    'forget': 'memman forget <id>',
                    'keep': 'memman gc --keep <id>',
                    },
                })
            return

        if keep:
            ins = get_insight_by_id(db, keep)
            if ins is None:
                raise click.ClickException(
                    f'insight {keep} not found or already deleted')
            boost_retention(db, keep)
            ei = refresh_effective_importance(db, keep)
            new_access = ins.access_count + 3
            log_op(db, 'gc-keep', keep, f'access+3, ei={ei:.4f}')
            _json_out({
                'status': 'retained',
                'id': keep,
                'content': ins.content,
                'new_access': new_access,
                'effective_importance': ei,
                'immune': is_immune(ins.importance, new_access),
                })
            return

        candidates, total = get_retention_candidates(
            db, threshold, limit)

        out_candidates = []
        for c in candidates:
            ins = c['insight']
            out_candidates.append({
                'id': ins.id,
                'content': ins.content,
                'category': ins.category,
                'importance': ins.importance,
                'access_count': ins.access_count,
                'effective_importance': c['effective_importance'],
                'days_since_access': c['days_since_access'],
                'edge_count': c['edge_count'],
                'immune': c['immune'],
                })

        _json_out({
            'total_insights': total,
            'threshold': threshold,
            'candidates_found': len(candidates),
            'candidates': out_candidates,
            'max_insights': MAX_INSIGHTS,
            'actions': {
                'purge': 'memman forget <id>',
                'keep': 'memman gc --keep <id>',
                },
            })
    finally:
        db.close()


@cli.command()
@click.option('--format', 'fmt', default='dot', help='Output format: dot or html')
@click.option('-o', '--output', 'output_path', default='-', help='Output file (- for stdout)')
@click.pass_context
def viz(ctx: click.Context, fmt: str, output_path: str) -> None:
    """Export memman graph for visualization."""
    from memman.store.edge import get_all_edges
    from memman.store.node import get_all_active_insights

    db = _open_db(ctx)
    try:
        insights = get_all_active_insights(db)
        edges = get_all_edges(db)

        if fmt == 'dot':
            out = _render_dot(insights, edges)
        elif fmt == 'html':
            out = _render_html(insights, edges)
        else:
            raise click.ClickException(
                f'unsupported format: {fmt} (use dot or html)')

        if output_path in {'', '-'}:
            click.echo(out, nl=False)
        else:
            pathlib.Path(output_path).write_text(out)
            click.echo(f'written to {output_path}', err=True)
    finally:
        db.close()


@cli.group(invoke_without_command=True)
@click.pass_context
def embed(ctx: click.Context) -> None:
    """Manage embeddings."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(embed_status)


@embed.command('status')
@click.pass_context
def embed_status(ctx: click.Context) -> None:
    """Show embedding coverage statistics."""
    from memman.embed import get_client
    from memman.store.node import embedding_stats

    db = _open_db(ctx)
    try:
        ec = get_client()
        total, embedded = embedding_stats(db)
        coverage = f'{embedded * 100 // total}%' if total > 0 else '0%'
        _json_out({
            'total_insights': total,
            'embedded': embedded,
            'coverage': coverage,
            'embed_available': ec.available(),
            'model': ec.model,
            })
    finally:
        db.close()


@embed.command('backfill')
@click.pass_context
def embed_backfill(ctx: click.Context) -> None:
    """Embed all insights missing embeddings."""
    from memman.embed import get_client
    from memman.embed.vector import serialize_vector
    from memman.store.node import get_insights_without_embedding
    from memman.store.node import update_embedding

    db = _open_db(ctx)
    try:
        ec = get_client()
        missing = get_insights_without_embedding(db, 1000)
        if not missing:
            _json_out({
                'status': 'complete',
                'message': 'all insights already have embeddings',
                })
            return
        succeeded = 0
        failed = 0
        for ins in missing:
            try:
                vec = ec.embed(ins.content)
                blob = serialize_vector(vec)
                update_embedding(db, ins.id, blob)
                succeeded += 1
            except Exception:
                failed += 1
        _json_out({
            'status': 'backfill_complete',
            'succeeded': succeeded,
            'failed': failed,
            'model': ec.model,
            })
    finally:
        db.close()


@embed.command('run')
@click.argument('id')
@click.pass_context
def embed_run(ctx: click.Context, id: str) -> None:
    """Embed a single insight by ID."""
    from memman.embed import get_client
    from memman.embed.vector import serialize_vector
    from memman.store.node import get_insight_by_id, update_embedding

    db = _open_db(ctx)
    try:
        ec = get_client()
        ins = get_insight_by_id(db, id)
        if ins is None:
            raise click.ClickException(f'insight {id} not found')
        vec = ec.embed(ins.content)
        blob = serialize_vector(vec)
        update_embedding(db, id, blob)
        _json_out({
            'status': 'embedded',
            'id': id,
            'dimension': len(vec),
            'model': ec.model,
            })
    finally:
        db.close()


@cli.group()
def graph() -> None:
    """Graph management commands."""


@graph.command('reindex')
@click.option('--dry-run', is_flag=True, default=False,
              help='Show stats without modifying DB')
@click.pass_context
def graph_reindex(ctx: click.Context, dry_run: bool) -> None:
    """Recalculate auto-created graph edges."""
    from memman.graph.engine import reindex_auto_edges

    db = _open_db(ctx)
    try:
        stats = reindex_auto_edges(db, dry_run=dry_run)
        if dry_run:
            click.echo('Dry run (no changes):')
        else:
            click.echo('Reindex complete:')
        _json_out(stats)
    finally:
        db.close()


@cli.command()
@click.option('--target', default='', help='Target environment (claude-code | openclaw)')
@click.option('--eject', is_flag=True, default=False, help='Remove integration')
@click.option('--scheduler', is_flag=True, default=False,
              help='Install background enrichment scheduler (systemd/launchd)')
@click.option('--interval', default=900, type=int,
              help='Scheduler interval in seconds (default 900 = 15 min)')
@click.option('--dry-run', is_flag=True, default=False,
              help='Preview scheduler changes without writing files')
@click.pass_context
def setup(ctx: click.Context, target: str, eject: bool,
          scheduler: bool, interval: int, dry_run: bool) -> None:
    """Set up LLM CLI integration or background scheduler."""
    data_dir = ctx.obj['data_dir']

    if scheduler:
        from memman.setup.scheduler import install, uninstall
        if eject:
            result = uninstall(dry_run=dry_run)
        else:
            api_key = (os.environ.get('OPENROUTER_API_KEY')
                       or os.environ.get('MEMMAN_LLM_API_KEY'))
            if not api_key and not dry_run:
                click.echo(
                    'warning: OPENROUTER_API_KEY not set; scheduler unit'
                    ' will install but the worker will fail until you'
                    ' write OPENROUTER_API_KEY=... into ~/.memman/env'
                    ' (mode 600)', err=True)
            result = install(data_dir, interval_seconds=interval,
                             openrouter_api_key=api_key,
                             dry_run=dry_run)
        _json_out(result)
        return

    from memman.setup.claude import run_setup
    run_setup(data_dir, target=target, eject=eject)


@graph.command('rebuild')
@click.option('--dry-run', is_flag=True, default=False,
              help='Show counts without modifying DB')
@click.pass_context
def graph_rebuild(ctx: click.Context, dry_run: bool) -> None:
    """Re-enrich all insights through the full LLM pipeline."""
    from memman.embed import get_client
    from memman.graph.engine import MAX_LINK_BATCH, link_pending
    from memman.graph.semantic import build_embed_cache
    from memman.llm.client import get_llm_client
    from memman.store.node import count_pending_links, get_active_insight_ids
    from memman.store.node import reset_for_rebuild
    from memman.store.oplog import log_op

    db = _open_db(ctx)
    try:
        llm_client = get_llm_client()
        ec = get_client()

        all_ids = get_active_insight_ids(db)
        total_count = len(all_ids)

        if dry_run:
            _json_out({'total': total_count, 'dry_run': 1})
            return

        if total_count == 0:
            _json_out({'processed': 0, 'remaining': 0})
            return

        embed_cache = build_embed_cache(db)
        processed = 0

        bar = tqdm(
            total=total_count, desc='Rebuilding',
            unit='insight', file=sys.stderr,
            dynamic_ncols=True,
            disable=not sys.stderr.isatty())

        def _on_progress(stage: str, insight: Insight) -> None:
            preview = insight.content[:40].replace('\n', ' ')
            bar.set_description(f'{stage}: {preview}')
            if stage == 'done':
                bar.update(1)

        for i in range(0, total_count, MAX_LINK_BATCH):
            batch_ids = all_ids[i:i + MAX_LINK_BATCH]
            reset_for_rebuild(db, batch_ids)

            while True:
                count = link_pending(
                    db, embed_cache=embed_cache,
                    llm_client=llm_client, embed_client=ec,
                    on_progress=_on_progress)
                processed += count
                if count == 0:
                    break

        bar.set_description('Done')
        bar.close()

        remaining = count_pending_links(db)

        stats = {'processed': processed, 'remaining': remaining}
        log_op(db, 'rebuild', '', json.dumps(stats))
        _json_out(stats)
    finally:
        db.close()


def _node_label(i: Insight) -> str:
    """Return a short display label for a node."""
    content = i.content.replace('\n', ' ')
    if len(content) > 60:
        content = content[:60] + '...'
    return f'[{i.category}] {content}'


def _category_color(c: str) -> str:
    """Return a color for a category."""
    colors = {
        'decision': '#e74c3c', 'fact': '#3498db',
        'insight': '#9b59b6', 'preference': '#2ecc71',
        'context': '#f39c12',
        }
    return colors.get(c, '#95a5a6')


def _edge_color(t: str) -> str:
    """Return a color for an edge type."""
    colors = {
        'temporal': '#aaaaaa', 'semantic': '#3498db',
        'causal': '#e74c3c', 'entity': '#2ecc71',
        }
    return colors.get(t, '#cccccc')


def _render_dot(insights: list[Insight],
                edges: list[Edge]) -> str:
    """Render a DOT graph."""
    lines = [
        'digraph memman {',
        '  rankdir=LR;',
        ('  node [shape=box, style="filled,rounded",'
         ' fontsize=10, fontname="Helvetica"];'),
        '  edge [fontsize=8, fontname="Helvetica"];',
        '',
        ]

    active = {i.id for i in insights}

    for i in insights:
        label = _node_label(i).replace('"', '\\"')
        short_id = _trunc_id(i.id)
        color = _category_color(i.category)
        lines.append(
            f'  "{i.id}" [label="{short_id}: {label}",'
            f' fillcolor="{color}", fontcolor="white"];')

    lines.append('')
    for e in edges:
        if e.source_id not in active or e.target_id not in active:
            continue
        color = _edge_color(e.edge_type)
        sub_type = e.metadata.get('sub_type', '')
        edge_label = sub_type or e.edge_type
        lines.append(
            f'  "{e.source_id}" -> "{e.target_id}"'
            f' [label="{edge_label}", color="{color}",'
            f' fontcolor="{color}"];')

    lines.extend(('}', ''))
    return '\n'.join(lines)


def _js_str(s: str) -> str:
    """Return a JSON-encoded string safe for script embedding."""
    return json.dumps(s).replace('</', '<\\/')


def _render_html(insights: list[Insight], edges: list[Edge]) -> str:
    """Render an HTML vis.js interactive page."""
    active = {i.id for i in insights}

    node_parts = []
    for i in insights:
        short_id = _trunc_id(i.id)
        label = _node_label(i).replace('\n', ' ')
        title = i.content.replace('\n', '\\n')
        color = _category_color(i.category)
        node_parts.append(
            f'{{id:{_js_str(i.id)},label:{_js_str(short_id + ": " + label)},'
            f'title:{_js_str(title)},color:{_js_str(color)},'
            f'font:{{color:"white"}}}}')
    nodes_js = ',\n'.join(node_parts)

    edge_parts = []
    for e in edges:
        if e.source_id not in active or e.target_id not in active:
            continue
        color = _edge_color(e.edge_type)
        sub_type = e.metadata.get('sub_type', '')
        edge_label = sub_type or e.edge_type
        edge_parts.append(
            f'{{from:{_js_str(e.source_id)},to:{_js_str(e.target_id)},'
            f'label:{_js_str(edge_label)},'
            f'color:{{color:{_js_str(color)}}},'
            f'arrows:"to",font:{{color:{_js_str(color)},size:10}}}}')
    edges_js = ',\n'.join(edge_parts)

    return _HTML_TEMPLATE.replace('%NODES%', nodes_js).replace(
        '%EDGES%', edges_js)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MemMan Knowledge Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body { margin: 0; padding: 0; background: #1a1a2e; font-family: sans-serif; }
  #graph { width: 100vw; height: 100vh; }
  #legend { position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.7);
    color: white; padding: 12px; border-radius: 8px; font-size: 12px; }
  .leg-item { display: flex; align-items: center; margin: 4px 0; }
  .leg-dot { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
  .leg-line { width: 20px; height: 3px; margin-right: 8px; }
</style>
</head>
<body>
<div id="graph"></div>
<div id="legend">
  <b>Nodes</b>
  <div class="leg-item"><div class="leg-dot" style="background:#e74c3c"></div>decision</div>
  <div class="leg-item"><div class="leg-dot" style="background:#3498db"></div>fact</div>
  <div class="leg-item"><div class="leg-dot" style="background:#9b59b6"></div>insight</div>
  <div class="leg-item"><div class="leg-dot" style="background:#2ecc71"></div>preference</div>
  <div class="leg-item"><div class="leg-dot" style="background:#f39c12"></div>context</div>
  <div class="leg-item"><div class="leg-dot" style="background:#95a5a6"></div>general</div>
  <br><b>Edges</b>
  <div class="leg-item"><div class="leg-line" style="background:#aaaaaa"></div>temporal</div>
  <div class="leg-item"><div class="leg-line" style="background:#3498db"></div>semantic</div>
  <div class="leg-item"><div class="leg-line" style="background:#e74c3c"></div>causal</div>
  <div class="leg-item"><div class="leg-line" style="background:#2ecc71"></div>entity</div>
</div>
<script>
var nodes = new vis.DataSet([%NODES%]);
var edges = new vis.DataSet([%EDGES%]);
var container = document.getElementById("graph");
var data = { nodes: nodes, edges: edges };
var options = {
  physics: { solver: "forceAtlas2Based", forceAtlas2Based: { gravitationalConstant: -30 } },
  interaction: { hover: true, tooltipDelay: 100 },
  nodes: { shape: "box", margin: 8, borderWidth: 0, font: { size: 11 } },
  edges: { smooth: { type: "continuous" }, font: { size: 9 } }
};
new vis.Network(container, data, options);
</script>
</body>
</html>"""
