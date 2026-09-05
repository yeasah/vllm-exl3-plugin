"""A KV cache manager that keeps only part of a request's context resident.

Registered rather than patched: `KVCacheSpecRegistry.register` takes
out-of-tree specs and managers, so this is a plugin and vLLM is unmodified --
which matters given how much churn the KV subsystem has been under.

What it does each step is decide, from the shared policy, which of a request's
full blocks should be resident, and hand the rest back to the pool. The blocks
stay in `req_to_blocks` as `null_block`, so the list keeps its length and every
positional invariant the rest of the engine relies on: appends still land at
the end, `get_num_blocks_to_allocate` still counts correctly, and the token at
position p still belongs to index p // block_size.

**It does not reuse `_remove_blocks_in_range`.** That helper iterates backward
and stops at the first block already nulled, which is right for a sliding
window -- whose freed region only ever grows from the front -- and wrong here,
where the freed set is whatever the policy did not choose and can gain holes
anywhere. Calling it on a range whose tail is already null would free nothing
and report success.

Restores use the same shape the in-tree copy-on-write redirect does: reserve
the extra block in `get_num_blocks_to_allocate`, draw it in
`allocate_new_blocks`, and write it into the row *in place* rather than
appending. That keeps every positional invariant while still handing the block
to the worker through the ordinary new-block channel.

The worker learns which logical index each restored block belongs to without a
protocol change, because the pairing is implied: restored blocks are returned
before appended ones and in ascending index order, so the first `len(restored)`
new ids pair with `sorted(pending_restores)`. `restored[request_id]` records
that pairing on this side so the two can be checked against each other rather
than assumed equal.

Eviction is two-phase: a block chosen this step is freed on the *next* one, so
there is a step in which it is still allocated, still holds its contents, and
is already out of the view -- which is the window the worker copies it to the
host in. Freeing in the same pass that decided it would hand the block to
another request before anything had read it, and that failure is silent.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import replace as dataclasses_replace

from . import state as pager_state
from .policy import POLICIES


def register(spec_cls=None, manager_cls=None) -> None:
    """Register the paged spec and manager with vLLM. Idempotent."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    KVCacheSpecRegistry.register(
        spec_cls or PagedAttentionSpec,
        manager_cls or PagedAttentionManager,
        # Grouped with full attention: a paged layer allocates blocks the same
        # way and differs only in how many it keeps, so it must not be split
        # into a separate KV cache group.
        uniform_type_base_spec=FullAttentionSpec,
    )


def _spec_base():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return FullAttentionSpec


def make_spec_class():
    """Build `PagedAttentionSpec` against whatever `FullAttentionSpec` is.

    Defined lazily so importing this module does not require vLLM -- the policy
    is shared with code that runs without an engine.
    """

    @dataclass(frozen=True)
    class PagedAttentionSpec(_spec_base()):
        #: Full blocks a request may keep resident. 0 means unbounded, which
        #: makes this behave exactly as full attention -- the control arm.
        budget_blocks: int = 0
        sink_blocks: int = 2
        policy_name: str = "recency"

        @classmethod
        def merge(cls, specs):
            """Carry the paging fields through the group merge.

            `FullAttentionSpec.merge` rebuilds the spec by naming every field
            it knows about, so a subclass's own fields silently fall back to
            their defaults -- which for this one means `budget_blocks=0`, a
            policy of `full`, and a pager that quietly does nothing at all.
            Nothing raises; the run simply pages nothing and looks like a
            baseline. Found by a transfer counter reading zero, not by a check.
            """
            merged = super().merge(specs)
            first = specs[0]
            for spec in specs[1:]:
                if (spec.budget_blocks, spec.sink_blocks, spec.policy_name) != (
                        first.budget_blocks, first.sink_blocks,
                        first.policy_name):
                    raise ValueError(
                        "layers in one KV cache group disagree about paging")
            return dataclasses_replace(
                merged,
                budget_blocks=first.budget_blocks,
                sink_blocks=first.sink_blocks,
                policy_name=first.policy_name,
            )

    return PagedAttentionSpec


class PagedAttentionManager:
    """Mixin body; combined with `FullAttentionManager` in `build_manager`."""

    def _paged_init(self, kv_cache_spec) -> None:
        budget = getattr(kv_cache_spec, "budget_blocks", 0)
        sink = getattr(kv_cache_spec, "sink_blocks", 2)
        name = getattr(kv_cache_spec, "policy_name", "recency")
        self.policy = POLICIES[name](budget=budget, sink=sink) if budget else \
            POLICIES["full"]()
        #: row indices whose block was handed back and now lives on the host
        self.evicted: defaultdict[str, set[int]] = defaultdict(set)
        #: decided this step, freed the next one -- the window in which the
        #: worker copies the block out. Freeing in the same pass would hand the
        #: block to another request before anything had read it.
        self.pending_evictions: defaultdict[str, set[int]] = defaultdict(set)
        #: row indices the policy wants back but that have no block yet
        self.pending_restores: defaultdict[str, set[int]] = defaultdict(set)
        #: (row index, block id) for blocks restored in the last allocation,
        #: in the order they are handed to the worker
        self.restored: defaultdict[str, list] = defaultdict(list)
        self.blocks_freed = 0
        self.blocks_restored = 0
        self.state = pager_state.current()

    # -- the per-step hook -------------------------------------------------

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Free everything the policy did not choose, and note what it wants back.

        `processed_computed_tokens` is the committed prefix -- tokens that are
        certainly written -- which is the only count safe to free against, and
        the same clock the worker must use so the two sides agree.
        """
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return

        # Two-phase, and this is the phase that makes the transport safe: what
        # was decided last step is freed now, having had a step in which the
        # worker could copy it to the host. A block freed in the same pass that
        # decided it can be handed to another request before its contents have
        # been read anywhere.
        due = sorted(self.pending_evictions[request_id])
        self._free_indices(request_id, due)
        self.evicted[request_id] |= set(due)
        self.pending_evictions[request_id] = set()

        n_full = min(processed_computed_tokens // self.block_size, len(blocks))
        if n_full <= 0:
            return
        resident = set(self.policy.resident(n_full, processed_computed_tokens))

        null_id = self._null_block.block_id
        drop = {i for i in range(n_full)
                if i not in resident and blocks[i].block_id != null_id}
        want = {i for i in resident if blocks[i].block_id == null_id}

        self.pending_evictions[request_id] = drop
        self.evicted[request_id] -= want
        self.pending_restores[request_id] = want
        self._publish(request_id, processed_computed_tokens, drop)

    # -- restores ----------------------------------------------------------

    def get_num_blocks_to_allocate(self, request_id: str, *args, **kwargs) -> int:
        """Reserve the restores alongside the ordinary growth.

        Called after `remove_skipped_blocks` in `allocate_slots`, so
        `pending_restores` is already populated for this step. Reserving here
        and drawing in `allocate_new_blocks` is the same contract the partial
        prefix-cache hit uses for its CoW block.
        """
        base = super().get_num_blocks_to_allocate(request_id, *args, **kwargs)
        return base + len(self.pending_restores.get(request_id, ()))

    def allocate_new_blocks(self, request_id: str, *args, **kwargs) -> list:
        """Restored blocks first, then ordinary growth.

        Order is load-bearing: these become the request's new block ids in
        this order, and the worker pairs the leading ones with
        `sorted(pending_restores)` to recover which logical index each holds.
        """
        restored = self._restore_blocks(request_id)
        grown = super().allocate_new_blocks(request_id, *args, **kwargs)
        self._refresh(request_id)
        return restored + list(grown)

    def _restore_blocks(self, request_id: str) -> list:
        want = sorted(self.pending_restores.get(request_id, ()))
        self.restored[request_id] = []
        if not want:
            return []
        req_blocks = self.req_to_blocks[request_id]
        want = [i for i in want if i < len(req_blocks)]
        if not want:
            return []
        blocks = self.block_pool.get_new_blocks(len(want))
        for idx, block in zip(want, blocks):
            req_blocks[idx] = block
        self.restored[request_id] = [
            (idx, block.block_id) for idx, block in zip(want, blocks)
        ]
        self.evicted[request_id] -= set(want)
        self.pending_restores[request_id] = set()
        self.blocks_restored += len(want)
        published = self.state.get(request_id)
        if published is not None:
            published.restored = list(self.restored[request_id])
        return list(blocks)

    def _publish(self, request_id: str, num_computed: int, drop) -> None:
        """Record this step's eviction choice for the worker side.

        Only the choice. `row` and `resident` are filled in by `_refresh`
        after allocation, because this runs *before* it: a step that grows a
        block or takes one back would otherwise publish a view that predates
        the block the model is about to write into, and the guard's
        `write_target` check catches exactly that -- which is how this was
        found rather than shipped.
        """
        blocks = self.req_to_blocks[request_id]
        step = pager_state.RequestStep(
            evicting=[(i, blocks[i].block_id) for i in sorted(drop)],
            num_computed=num_computed,
        )
        self.state.block_size = self.block_size
        self.state.publish(request_id, step)

    def _refresh(self, request_id: str) -> None:
        """Publish the mapping and the view as they finally stand.

        Called at the end of allocation, which is the last thing that changes a
        request's blocks in a scheduling pass, so what the worker reads is what
        the worker will see.
        """
        step = self.state.get(request_id)
        if step is None:
            return
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return
        step.row = [b.block_id for b in blocks]
        step.resident = self.resident_indices(request_id, step.num_computed)

    def _free_indices(self, request_id: str, indices) -> None:
        """Hand specific row positions back to the pool, nulling them in place.

        Set-based on purpose; see the module docstring for why the in-tree
        range helper cannot be used here.
        """
        if not indices:
            return
        blocks = self.req_to_blocks[request_id]
        null = self._null_block
        freed = []
        for i in indices:
            block = blocks[i]
            if block.block_id == null.block_id:
                continue
            freed.append(block)
            blocks[i] = null
        if freed:
            self.block_pool.free_blocks(freed)
            self.blocks_freed += len(freed)

    # -- bookkeeping -------------------------------------------------------

    def resident_indices(self, request_id: str, num_computed: int) -> list[int]:
        """What the policy says should be resident, including the tail.

        The tail is the block holding the key this step is about to write. It
        is index `num_computed // block_size` and it is never a policy choice:
        it must exist, be resident, and be last. Callers must run this *after*
        allocation, since a growing request has no block at that index until
        then.
        """
        blocks = self.req_to_blocks.get(request_id, ())
        n_full = min(num_computed // self.block_size, len(blocks))
        keep = self.policy.resident(n_full, num_computed)
        tail = num_computed // self.block_size
        if tail < len(blocks):
            keep = [i for i in keep if i != tail] + [tail]
        return keep

    def free(self, request_id: str) -> None:
        self.evicted.pop(request_id, None)
        self.pending_evictions.pop(request_id, None)
        self.pending_restores.pop(request_id, None)
        self.restored.pop(request_id, None)
        self.state.drop(request_id)
        super().free(request_id)


def build_manager_class():
    """`FullAttentionManager` with the paging behaviour mixed in."""
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    class _PagedAttentionManager(PagedAttentionManager, FullAttentionManager):
        def __init__(self, kv_cache_spec, **kwargs):
            FullAttentionManager.__init__(self, kv_cache_spec, **kwargs)
            self._paged_init(kv_cache_spec)

    _PagedAttentionManager.__name__ = "PagedAttentionManager"
    return _PagedAttentionManager
