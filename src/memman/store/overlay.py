"""Branch stores: an empty SQLite store layered over its live parent.

A branch holds only its own writes. `OverlayBackend` answers each
Backend verb from the branch, the parent, or both, so recall, the drain
and the CLI use a branch as they use any store.

Notes
-----
- A store is a branch exactly when its meta holds `branch_parent`.
- Every id the branch holds, in any state, hides the parent row with
  that id.
- The parent opens with `read_only=True`, so a write to it raises on
  either backend.
- Every node write raises while the branch's meta holds
  `branch_merging`, which `branch.merge_branch` sets before it reads
  the branch.
"""

import dataclasses
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from types import TracebackType
from typing import Any, Self

from memman.embed.fingerprint import META_KEY, Fingerprint, swap_command
from memman.migrate import MigrateInsight
from memman.store import factory
from memman.store.backend import Backend, MetaStore, NodeStore, Oplog
from memman.store.backend import RecallSession, chain_head
from memman.store.errors import BackendError
from memman.store.model import EnrichmentCoverage, Id, Insight, NodeStats
from memman.store.model import OpLogEntry, OpLogStats, WorkerRun

BRANCH_PARENT = 'branch_parent'
BRANCH_MERGING = 'branch_merging'
BRANCH_CREATED_AT = 'branch_created_at'
BRANCH_TOKEN = 'branch_token'


def parent_token_key(branch: str) -> str:
    """The parent meta key that holds `branch`'s token.
    """
    return f'{BRANCH_TOKEN}:{branch}'


def token_refusal(branch: str, parent: str) -> str:
    """Refusal for a parent that does not hold `branch`'s token.
    """
    return (
        f'store {parent!r} does not hold the token of branch {branch!r}:'
        ' it was removed and recreated, points at another database, or'
        ' was restored from a backup older than the branch; fix the'
        ' parent if it points at the wrong database, else run memman'
        f' store drop {branch}, which lists the branch rows to remember'
        f' again in {parent!r}')


class OverlayNodeStore(NodeStore):
    """Node verbs of a branch, read through to its parent.
    """

    def __init__(self, overlay: 'OverlayBackend') -> None:
        self._overlay = overlay
        self._branch = overlay.branch.nodes

    @property
    def _parent(self) -> NodeStore:
        return self._overlay.parent.nodes

    def _require_unmerged(self) -> None:
        if self._overlay.meta.get(BRANCH_MERGING) is not None:
            raise BackendError(
                f'branch {self._overlay.name!r} is merging into'
                f' {self._overlay.parent_name!r}; re-run memman store merge'
                f' {self._overlay.name} to finish it')

    def _require_held(self, id: Id) -> None:
        if not self._overlay.branch_holds(id):
            raise BackendError(
                f'branch {self._overlay.name!r} holds no row {id};'
                ' a branch writes only its own rows')

    def _hold_for_retire(self, id: Id, *, refuse_retired: bool) -> bool:
        """Make the branch hold `id`, copying a current parent-only row.

        Parameters
        ----------
        id : Id
            The row a retirement targets.
        refuse_retired : bool
            Raise on a row the parent already replaced, in place of
            answering False.

        Returns
        -------
        bool
            True when the branch holds the row after the call. False
            for a missing row, a row the parent already forgot, and a
            row the parent already replaced unless `refuse_retired`.

        Raises
        ------
        BackendError
            `refuse_retired` is set and the parent already replaced the
            row. The message names the current head of its chain.
        """
        if self._overlay.branch_holds(id):
            return True
        row = self._parent.get_raw(id)
        if row is None or row.deleted_at is not None:
            return False
        if row.replaced_by is not None:
            if refuse_retired:
                raise BackendError(self._retired_message(
                    row.id, row.replaced_by))
            return False
        self._branch.insert_raw(row)
        return True

    def _retired_message(self, id: Id, successor: Id) -> str:
        """Refusal for a forget of a row the parent already replaced.

        Parameters
        ----------
        id : Id
            The parent row the forget targets.
        successor : Id
            The row's `replaced_by`.

        Returns
        -------
        str
            Names the row as retired in the parent and names the head
            of its chain, or says the chain ends in a forgotten row.
        """
        head = chain_head(self, successor)
        message = (
            f'insight {id} is already retired in'
            f' {self._overlay.parent_name} (replaced by {successor})')
        if head is None or head.deleted_at is not None:
            return f'{message}; its chain ends in a forgotten row'
        return f'{message}; forget {head.id} to drop the current claim'

    def insert(self, ins: Insight) -> None:
        self._require_unmerged()
        self._branch.insert(ins)

    def insert_raw(self, row: MigrateInsight) -> bool:
        self._require_unmerged()
        return self._branch.insert_raw(row)

    def get_raw(self, id: Id) -> MigrateInsight | None:
        row = self._branch.get_raw(id)
        return row if row is not None else self._parent.get_raw(id)

    def get(self, id: Id) -> Insight | None:
        if self._overlay.branch_holds(id):
            return self._branch.get(id)
        return self._parent.get(id)

    def get_include_deleted(self, id: Id) -> Insight | None:
        ins = self._branch.get_include_deleted(id)
        return ins if ins is not None else self._parent.get_include_deleted(id)

    def resolve_id(self, id_or_prefix: str) -> str:
        """Resolve a full id or a prefix across both stores.

        Raises
        ------
        ValueError
            The prefix matches more than one distinct id, in one store
            or across the two.
        """
        if self.get_include_deleted(id_or_prefix) is not None:
            return id_or_prefix
        matches = {
            self._branch.resolve_id(id_or_prefix),
            self._parent.resolve_id(id_or_prefix),
            } - {id_or_prefix}
        if len(matches) > 1:
            raise ValueError(
                f'prefix {id_or_prefix!r} matches {len(matches)} rows')
        return matches.pop() if matches else id_or_prefix

    def query(
            self, *, keyword: str = '', limit: int = 20) -> list[Insight]:
        hidden = self._branch.get_all_ids()
        # SQLite reads a negative limit as no limit.
        parent_limit = limit + len(hidden) if limit >= 0 else limit
        rows = self._branch.query(keyword=keyword, limit=limit) + [
            ins for ins in self._parent.query(
                keyword=keyword, limit=parent_limit)
            if ins.id not in hidden
            ]
        rows.sort(key=lambda ins: ins.created_at, reverse=True)
        return rows[:limit] if limit >= 0 else rows

    def soft_delete(self, id: Id) -> bool:
        with self._overlay.transaction():
            self._require_unmerged()
            if not self._hold_for_retire(id, refuse_retired=True):
                return False
            return self._branch.soft_delete(id)

    def soft_delete_current(self, id: Id) -> bool:
        with self._overlay.transaction():
            self._require_unmerged()
            if not self._hold_for_retire(id, refuse_retired=False):
                return False
            return self._branch.soft_delete_current(id)

    def mark_replaced(self, predecessor_id: Id, successor_id: Id) -> bool:
        with self._overlay.transaction():
            self._require_unmerged()
            if not self._hold_for_retire(predecessor_id, refuse_retired=False):
                return False
            return self._branch.mark_replaced(predecessor_id, successor_id)

    def predecessors(self, successor_id: Id) -> list[Insight]:
        hidden = self._branch.get_all_ids()
        rows = self._branch.predecessors(successor_id) + [
            ins for ins in self._parent.predecessors(successor_id)
            if ins.id not in hidden
            ]
        rows.sort(key=lambda ins: (ins.created_at, ins.id))
        return rows

    def replacement_integrity(self) -> dict[str, list[Id]]:
        result = self._branch.replacement_integrity()
        result['dangling'] = [
            id for id in result['dangling']
            if self._parent.get_include_deleted(
                self._branch.get_include_deleted(id).replaced_by) is None
            ]
        return result

    def update_enrichment(
            self, id: Id, *, summary: str, summary_model: str) -> None:
        self._require_unmerged()
        self._require_held(id)
        self._branch.update_enrichment(
            id, summary=summary, summary_model=summary_model)

    def count_active(self) -> int:
        """Rows recall sees: parent current rows not hidden, plus branch ones.
        """
        return (
            self._parent.count_active() - self._overlay.hidden_current_count()
            + self._branch.count_active())

    def count_total(self) -> int:
        return self._branch.count_total()

    def has_row_with_queue_uuid(self, queue_uuid: str) -> bool:
        return self._branch.has_row_with_queue_uuid(queue_uuid)

    def get_all_active(self) -> list[Insight]:
        hidden = self._branch.get_all_ids()
        rows = self._branch.get_all_active() + [
            ins for ins in self._parent.get_all_active()
            if ins.id not in hidden
            ]
        rows.sort(key=lambda ins: ins.created_at, reverse=True)
        return rows

    def stats(self) -> NodeStats:
        """Branch replaced and deleted counts, with recall's current count.
        """
        branch_stats = self._branch.stats()
        return dataclasses.replace(
            branch_stats, total_insights=self.count_active())

    def update_embedding(
            self, id: Id, vec: list[float], model: str) -> None:
        self._require_unmerged()
        self._require_held(id)
        self._branch.update_embedding(id, vec, model)

    def embedding_stats(self) -> tuple[int, int]:
        return self._branch.embedding_stats()

    def enrichment_coverage(self) -> EnrichmentCoverage:
        return self._branch.enrichment_coverage()

    def embedding_size_distribution(self) -> dict[int, int]:
        return self._branch.embedding_size_distribution()

    def stamp_enrich_attempted(self, id: Id) -> None:
        self._require_unmerged()
        self._require_held(id)
        self._branch.stamp_enrich_attempted(id)

    def stamp_enriched(self, id: Id) -> None:
        self._require_unmerged()
        self._require_held(id)
        self._branch.stamp_enriched(id)

    def get_pending_enrich_ids(self, *, limit: int) -> list[Id]:
        return self._branch.get_pending_enrich_ids(limit=limit)

    def get_active_ids(self) -> list[Id]:
        return self._branch.get_active_ids()

    def get_all_ids(self) -> set[Id]:
        return self._branch.get_all_ids()

    def count_pending_enrich(self) -> int:
        return self._branch.count_pending_enrich()

    def get_unenriched_attempted_ids(self, *, limit: int) -> list[Id]:
        return self._branch.get_unenriched_attempted_ids(limit=limit)

    def reset_for_rebuild(self, ids: list[Id]) -> None:
        self._require_unmerged()
        for id in ids:
            self._require_held(id)
        self._branch.reset_for_rebuild(ids)


class OverlayOplog(Oplog):
    """The branch's oplog, with recall's current count in `stats`.
    """

    def __init__(self, overlay: 'OverlayBackend') -> None:
        self._overlay = overlay
        self._branch = overlay.branch.oplog

    def log(
            self, *, operation: str, insight_id: Id, detail: str,
            before: dict[str, Any] | None = None,
            after: dict[str, Any] | None = None) -> None:
        self._branch.log(
            operation=operation, insight_id=insight_id, detail=detail,
            before=before, after=after)

    def maintenance_step(self) -> None:
        self._branch.maintenance_step()

    def trim_by_age(self) -> int:
        return self._branch.trim_by_age()

    def recent(
            self, *, limit: int = 20,
            since: str = '') -> list[OpLogEntry]:
        return self._branch.recent(limit=limit, since=since)

    def stats(self, *, since: str = '') -> OpLogStats:
        return dataclasses.replace(
            self._branch.stats(since=since),
            total_active=self._overlay.nodes.count_active())


class OverlayRecallSession(RecallSession):
    """One ranking over both stores, with the branch's hidden ids removed.
    """

    def __init__(
            self, branch: RecallSession, parent: RecallSession,
            hidden: set[Id], hidden_current: int) -> None:
        """Wrap one open session per store.

        Parameters
        ----------
        branch, parent : RecallSession
            Open sessions on the branch and its parent.
        hidden : set[Id]
            Every id the branch holds, dropped from the parent's lists.
        hidden_current : int
            How many of `hidden` are current parent rows: the most
            hidden ids a parent anchor list can hold.
        """
        self._branch = branch
        self._parent = parent
        self._hidden = hidden
        self._hidden_current = hidden_current

    def vector_anchors(
            self, query_vec: list[float], *,
            k: int = 10) -> list[tuple[Id, float]]:
        """The top k of both stores' anchors, no hidden id among them.
        """
        hits = self._branch.vector_anchors(query_vec, k=k) + [
            hit for hit in self._parent.vector_anchors(
                query_vec, k=k + self._hidden_current)
            if hit[0] not in self._hidden
            ]
        hits.sort(key=lambda hit: hit[1], reverse=True)
        return hits[:k]

    def similarities(
            self, query_vec: list[float]) -> dict[Id, float]:
        return {
            **self._parent.similarities(query_vec),
            **self._branch.similarities(query_vec),
            }

    def keyword_counts(
            self, query_tokens: set[str]) -> dict[Id, int]:
        return {
            **self._parent.keyword_counts(query_tokens),
            **self._branch.keyword_counts(query_tokens),
            }


class OverlayBackend(Backend):
    """A branch store read through to its parent.

    Parameters
    ----------
    name : str
        The branch's store name.
    branch : Backend
        The open branch store, always SQLite.
    parent_name : str
        The parent's store name, from the branch's `branch_parent`.
    data_dir : str
        Base memman data directory.

    Notes
    -----
    - The parent opens on the first call that reads it, so a plain add
      and the drain's enrich work while the parent is down.
    """

    nodes: OverlayNodeStore
    meta: MetaStore
    oplog: OverlayOplog

    def __init__(
            self, name: str, branch: Backend, parent_name: str,
            data_dir: str) -> None:
        self.name = name
        self.branch = branch
        self.parent_name = parent_name
        self._data_dir = data_dir
        self._parent: Backend | None = None
        self.nodes = OverlayNodeStore(self)
        self.meta = branch.meta
        self.oplog = OverlayOplog(self)

    @property
    def parent(self) -> Backend:
        """The parent backend, opened read-only on first use.

        Raises
        ------
        BackendError
            The parent cannot be opened. The message names it.
        """
        if self._parent is None:
            try:
                self._parent = factory.open_backend(
                    self.parent_name, self._data_dir,
                    read_only=True, create=False)
            except BackendError as exc:
                raise BackendError(
                    f'branch {self.name!r} cannot read its parent'
                    f' {self.parent_name!r}: {exc}') from exc
        return self._parent

    def hidden_current_count(self) -> int:
        """Branch-held ids that are current rows in the parent.
        """
        return len(
            self.branch.nodes.get_all_ids()
            & set(self.parent.nodes.get_active_ids()))

    def branch_holds(self, id: Id) -> bool:
        """True when the branch itself holds row `id`, in any state.
        """
        return self.branch.nodes.get_include_deleted(id) is not None

    @property
    def path(self) -> str:
        return self.branch.path

    def transaction(self) -> AbstractContextManager[None]:
        return self.branch.transaction()

    def reembed_lock(self, name: str) -> AbstractContextManager[bool]:
        return self.branch.reembed_lock(name)

    def swap_lock(self) -> AbstractContextManager[bool]:
        return self.branch.swap_lock()

    def swap_prepare(self, target_dim: int) -> None:
        self.branch.swap_prepare(target_dim)

    def iter_for_swap(
            self, cursor: str, batch: int) -> list[tuple[Id, str]]:
        return self.branch.iter_for_swap(cursor, batch)

    def write_swap_batch(
            self, items: list[tuple[Id, list[float]]]) -> None:
        self.branch.write_swap_batch(items)

    def swap_cutover(self, target: Fingerprint) -> None:
        self.branch.swap_cutover(target)

    def swap_abort(self) -> None:
        self.branch.swap_abort()

    @contextmanager
    def recall_session(self) -> Iterator[OverlayRecallSession]:
        """Yield one session over both stores.

        Raises
        ------
        BackendError
            The parent's embed fingerprint differs from the branch's,
            and the message names the branch swap that fixes it. Or
            the parent does not hold the branch's token, as after a
            parent removed and recreated under the same name, or after
            a merge that stopped past its parent commit, whose message
            names the merge re-run.
        """
        if self.parent.meta.get(META_KEY) != self.branch.meta.get(META_KEY):
            raise BackendError(
                f'store {self.name!r} and its parent {self.parent_name!r}'
                f' use different embed models; run {swap_command(self.name)}'
                " with the parent's model first")
        if (self.parent.meta.get(parent_token_key(self.name))
                != self.branch.meta.get(BRANCH_TOKEN)):
            # A merge past its parent commit removed the token; only a
            # re-run of that merge ends this state.
            self.nodes._require_unmerged()
            raise BackendError(token_refusal(self.name, self.parent_name))
        hidden = self.branch.nodes.get_all_ids()
        hidden_current = self.hidden_current_count()
        with self.branch.recall_session() as branch_session, \
                self.parent.recall_session() as parent_session:
            yield OverlayRecallSession(
                branch_session, parent_session, hidden, hidden_current)

    def integrity_check(self) -> dict[str, Any]:
        return self.branch.integrity_check()

    def close(self) -> None:
        try:
            self.branch.close()
        finally:
            if self._parent is not None:
                self._parent.close()
                self._parent = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None) -> None:
        self.close()

    def start_run(self) -> int | None:
        return self.branch.start_run()

    def beat_run(self, run_id: int | None) -> None:
        self.branch.beat_run(run_id)

    def finish_run(self, run_id: int | None) -> None:
        self.branch.finish_run(run_id)

    def recent_runs(self, *, limit: int) -> list[WorkerRun]:
        return self.branch.recent_runs(limit=limit)
