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

Scope, stated plainly: this evicts. Restores are *planned* here and not yet
performed, because a restored block needs a destination allocated inside the
accounting `get_num_blocks_to_allocate` does, and that is the next piece rather
than something to bolt on quietly. Until then a paged request loses context
permanently, which makes this an evictor with a pager's bookkeeping.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

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
        #: row indices the policy wants back but that have no block yet
        self.pending_restores: defaultdict[str, set[int]] = defaultdict(set)
        self.blocks_freed = 0

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
        n_full = min(processed_computed_tokens // self.block_size, len(blocks))
        if n_full <= 0:
            return
        resident = set(self.policy.resident(n_full, processed_computed_tokens))

        null_id = self._null_block.block_id
        drop = [i for i in range(n_full)
                if i not in resident and blocks[i].block_id != null_id]
        want = {i for i in resident if blocks[i].block_id == null_id}

        self._free_indices(request_id, drop)
        self.evicted[request_id] |= set(drop)
        self.evicted[request_id] -= want
        self.pending_restores[request_id] = want

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

        The worker calls the same policy with the same arguments; this exists
        so a caller on this side does not reimplement the tail rule.
        """
        blocks = self.req_to_blocks.get(request_id, ())
        n_full = min(num_computed // self.block_size, len(blocks))
        keep = self.policy.resident(n_full, num_computed)
        if n_full < len(blocks):
            keep = keep + [n_full]
        return keep

    def free(self, request_id: str) -> None:
        self.evicted.pop(request_id, None)
        self.pending_restores.pop(request_id, None)
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
