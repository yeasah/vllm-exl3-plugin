"""The paged KV cache manager's bookkeeping, without an engine or a GPU.

These run against a real `BlockPool` and a real `FullAttentionManager`
subclass, so they check the manager against vLLM's actual allocator rather
than a stand-in. What they are looking for, in order of how quietly each would
fail:

- blocks are genuinely returned to the pool, not merely nulled in the request's
  list. A manager that forgot `free_blocks` would look correct in every
  structural assertion and save no memory at all, which is the whole point.
- the list keeps its length and its positions, because the token at position p
  lives at index p // block_size and everything from slot mapping to appends
  depends on that staying true.
- the in-tree range helper is *not* used, since it stops at the first already
  nulled block and would silently free nothing on a second, non-contiguous
  eviction.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec  # noqa: E402

from kv_pager.manager import build_manager_class  # noqa: E402
from kv_pager.policy import Recency  # noqa: E402

BLOCK = 16


def make_manager(budget, sink=2, num_blocks=256, policy="recency"):
    spec = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=2, head_size=64, dtype=torch.bfloat16
    )
    object.__setattr__(spec, "budget_blocks", budget)
    object.__setattr__(spec, "sink_blocks", sink)
    object.__setattr__(spec, "policy_name", policy)
    pool = BlockPool(num_gpu_blocks=num_blocks, enable_caching=False,
                     hash_block_size=BLOCK)
    cls = build_manager_class()
    mgr = cls(
        spec,
        block_pool=pool,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK,
        needs_kv_cache_zeroing=False,
        enable_caching=False,
    )
    return mgr, pool


def give_blocks(mgr, pool, req_id, n):
    blocks = pool.get_new_blocks(n)
    mgr.req_to_blocks[req_id] = list(blocks)
    return blocks


def null_positions(mgr, req_id):
    null = mgr._null_block.block_id
    return [i for i, b in enumerate(mgr.req_to_blocks[req_id])
            if b.block_id == null]


def test_eviction_returns_blocks_to_the_pool():
    """The assertion that a structurally-correct but useless manager fails."""
    mgr, pool = make_manager(budget=8)
    give_blocks(mgr, pool, "r", 20)
    free_before = pool.get_num_free_blocks()

    mgr.remove_skipped_blocks("r", processed_computed_tokens=20 * BLOCK)

    kept = set(Recency(budget=8, sink=2).resident(20, 20 * BLOCK))
    assert null_positions(mgr, "r") == sorted(set(range(20)) - kept)
    assert pool.get_num_free_blocks() == free_before + (20 - len(kept))
    assert mgr.blocks_freed == 20 - len(kept)


def test_positions_and_length_survive_eviction():
    mgr, pool = make_manager(budget=8)
    blocks = give_blocks(mgr, pool, "r", 20)
    mgr.remove_skipped_blocks("r", processed_computed_tokens=20 * BLOCK)

    assert len(mgr.req_to_blocks["r"]) == 20, "the list must not shrink"
    kept = Recency(budget=8, sink=2).resident(20, 20 * BLOCK)
    for i in kept:
        assert mgr.req_to_blocks["r"][i].block_id == blocks[i].block_id, (
            "a surviving block moved position"
        )


def test_eviction_with_interior_holes_still_frees():
    """The failure the in-tree range helper would have produced.

    `_remove_blocks_in_range` iterates backward and breaks at the first block
    already nulled, so a range containing earlier evictions frees only the part
    above the topmost hole. Recency never produces that shape -- its dropped
    set is one contiguous middle -- so this uses `stress`, whose roaming window
    leaves holes scattered through the region it is not currently keeping.

    Written this way after the contiguous version passed under a deliberately
    broken manager: a test that names a failure it cannot produce is worse than
    no test, because it certifies the decision it was supposed to check.
    """
    mgr, pool = make_manager(budget=10, policy="stress")
    give_blocks(mgr, pool, "r", 24)

    mgr.remove_skipped_blocks("r", processed_computed_tokens=24 * BLOCK)
    holes = set(null_positions(mgr, "r"))
    assert holes, "stress should have evicted something"

    # Move the roaming window: the newly dropped indices now sit both above and
    # below blocks that are already null, so any range spanning them contains
    # holes.
    later = 24 * BLOCK + 6 * BLOCK
    free_before = pool.get_num_free_blocks()
    mgr.remove_skipped_blocks("r", processed_computed_tokens=later)

    kept = set(mgr.policy.resident(24, later))
    pending = set(mgr.pending_restores["r"])
    now_null = set(null_positions(mgr, "r"))

    # A block the policy wants back stays null until restores exist, so the
    # null set is the non-resident blocks *plus* the ones awaiting a restore.
    # Asserting the simpler identity would be asserting that restores work.
    assert now_null == (set(range(24)) - kept) | pending, (
        "the second eviction did not free exactly the non-resident blocks"
    )
    newly = (set(range(24)) - kept) - holes
    assert newly, "the window did not move; this asserts nothing"
    assert max(newly) < max(holes), (
        "no newly dropped block sits below an existing hole, so a backward "
        "scan would never meet one and this does not test the break"
    )
    assert pool.get_num_free_blocks() == free_before + len(newly), (
        "blocks past an interior hole were never returned to the pool"
    )

def test_full_policy_frees_nothing():
    """The control arm: a pager told to evict nothing must be inert."""
    mgr, pool = make_manager(budget=0)
    give_blocks(mgr, pool, "r", 20)
    free_before = pool.get_num_free_blocks()
    mgr.remove_skipped_blocks("r", processed_computed_tokens=20 * BLOCK)
    assert null_positions(mgr, "r") == []
    assert pool.get_num_free_blocks() == free_before
    assert mgr.blocks_freed == 0


def test_restores_are_planned_when_the_policy_asks_for_a_block_back():
    mgr, pool = make_manager(budget=8, policy="stress")
    give_blocks(mgr, pool, "r", 20)
    mgr.remove_skipped_blocks("r", processed_computed_tokens=20 * BLOCK)
    first = set(null_positions(mgr, "r"))
    assert first, "stress should have evicted something"

    # A later step: the roaming window has moved, so some evicted index is
    # wanted back. It has no block yet, so it must be reported as pending.
    mgr.remove_skipped_blocks("r", processed_computed_tokens=20 * BLOCK + 5 * BLOCK)
    pending = mgr.pending_restores["r"]
    assert pending, "stress must ask for a block back, or it tests nothing"
    assert all(i in first for i in pending), "asked back a block never evicted"
    assert not (pending & set(mgr.resident_indices("r", 25 * BLOCK)) - first)


def test_freeing_the_request_clears_paging_state():
    mgr, pool = make_manager(budget=8)
    give_blocks(mgr, pool, "r", 20)
    mgr.remove_skipped_blocks("r", processed_computed_tokens=20 * BLOCK)
    mgr.free("r")
    assert "r" not in mgr.evicted
    assert "r" not in mgr.pending_restores


def simulate(mgr, pool, req_id, blocks_total, budget):
    """Drive a request through the real allocate path, block by block.

    Mirrors `allocate_slots`' order -- `remove_skipped_blocks`, then ask how
    many blocks are needed, then draw them -- because the two overrides have to
    agree with each other and the only way to find out is to run them in the
    sequence the scheduler uses.

    `num_cached_block` is seeded so `get_num_blocks_to_allocate` takes the
    running-request fast path, which is the branch a decoding request is on.
    """
    mgr.num_cached_block[req_id] = 0
    peak_real = 0
    restored_total = 0
    for step in range(blocks_total):
        computed = step * BLOCK
        mgr.remove_skipped_blocks(req_id, processed_computed_tokens=computed)
        want = int(pool.get_num_free_blocks())
        n = mgr.get_num_blocks_to_allocate(req_id, computed + BLOCK, [], computed, computed, computed + BLOCK)
        drawn = mgr.allocate_new_blocks(req_id, computed + BLOCK, computed + BLOCK)
        assert len(drawn) == n, (
            f"step {step}: reserved {n} blocks and drew {len(drawn)} -- the "
            f"reservation and the draw have drifted apart"
        )
        assert want - pool.get_num_free_blocks() == n, (
            f"step {step}: the pool lost a different number of blocks than "
            f"were reserved"
        )
        restored_total += len(mgr.restored[req_id])
        null = mgr._null_block.block_id
        real = sum(1 for b in mgr.req_to_blocks[req_id] if b.block_id != null)
        peak_real = max(peak_real, real)
    return peak_real, restored_total


def test_reservation_matches_the_draw_across_a_long_run():
    mgr, pool = make_manager(budget=8, num_blocks=512, policy="stress")
    simulate(mgr, pool, "r", blocks_total=60, budget=8)


def test_paging_actually_bounds_the_blocks_held():
    """The assertion that makes this a pager rather than bookkeeping.

    A request 60 blocks long must never hold more than its budget plus the
    tail it is currently writing into. Without the eviction this is 60, so a
    manager that nulled entries without returning them, or reserved without
    freeing, fails here rather than somewhere subtle later.
    """
    budget = 8
    mgr, pool = make_manager(budget=budget, num_blocks=512, policy="stress")
    peak, restored = simulate(mgr, pool, "r", blocks_total=60, budget=budget)
    assert peak <= budget + 2, f"held {peak} blocks against a budget of {budget}"
    assert restored > 0, "nothing was ever restored; the fetch path is untested"


def test_restored_blocks_land_at_their_own_indices():
    mgr, pool = make_manager(budget=8, num_blocks=512, policy="stress")
    mgr.num_cached_block["r"] = 0
    give_blocks(mgr, pool, "r", 24)
    mgr.remove_skipped_blocks("r", processed_computed_tokens=24 * BLOCK)

    later = 24 * BLOCK + 6 * BLOCK
    mgr.remove_skipped_blocks("r", processed_computed_tokens=later)
    want = sorted(mgr.pending_restores["r"])
    assert want, "stress asked for nothing back"

    mgr.get_num_blocks_to_allocate("r", later + BLOCK, [], later, later, later + BLOCK)
    mgr.allocate_new_blocks("r", later + BLOCK, later + BLOCK)

    pairs = mgr.restored["r"]
    assert [i for i, _ in pairs] == want, "restores are not in ascending order"
    null = mgr._null_block.block_id
    for idx, block_id in pairs:
        got = mgr.req_to_blocks["r"][idx].block_id
        assert got == block_id != null, (
            f"index {idx} holds {got}, not the restored block {block_id}"
        )
    assert not (set(want) & set(null_positions(mgr, "r"))), (
        "a restored index is still null"
    )
