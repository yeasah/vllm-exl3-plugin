"""Catch a residency view that names a block it does not own.

This exists before the allocator does, deliberately. Freeing blocks under a
running request has a failure mode with nothing to announce it: the null
substitution never leaves the scheduler (`_remove_blocks_in_range` mutates
`req_to_blocks` in place, and `get_block_ids` ships only newly allocated
blocks), so the worker's row still names a freed block, and the pool is free to
hand that block to another request. Attention then reads someone else's KV. No
crash, no NaN -- plausible text, slightly wrong, for the rest of the
generation. R-SWA is safe from this only because its mask stops attention
reading those positions at all; a pager has to hold the invariant itself.

Four checks, each aimed at a different way to lose it:

    ownership     every block the kernel will read is still allocated to this
                  request (or is one the pager reserved for itself)
    write target  this step's own key goes into the *last* resident block --
                  the tail invariant, read off the real slot mapping rather
                  than recomputed, so a wrong write cannot agree with a wrong
                  expectation
    length        the number of blocks the kernel will read equals the number
                  the *policy* meant to be resident. Checking seq_len against
                  the row alone cannot fail -- that count is derived from
                  seq_len by division -- so the intended count has to come from
                  outside, and the pager always knows it
    exclusivity   two requests' resident sets do not overlap

Only the *resident prefix* is examined -- the first `cdiv(seq_len, block_size)`
entries. Everything past it is unread by construction (measured: a poisoned
tail changes nothing), so checking it would reject correct behaviour.

Only `ownership` needs the scheduler, and therefore an in-process engine
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`). The other three read nothing but the
worker's own step, so they run wherever the pager runs -- which matters because
the failure this guards against does not announce itself, and a check that only
runs during debugging is absent exactly when a corrupted run is being folded
into an accuracy number. Pass `scheduler=None` to get the three; `summary()`
reports which were active so a result can carry that rather than imply it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Violation:
    check: str
    req_id: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.check}] {self.req_id}: {self.detail}"


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class ResidencyGuard:
    """Checks a step's view against what the scheduler actually owns.

    `reserve` is the set of block ids the pager holds for itself -- a restored
    block has to land somewhere, and asking the scheduler for a destination
    would cost a scheduling round trip, so those ids are legitimately in a view
    without belonging to the request. Everything else in a view is a bug.
    """

    #: Checks that need no state beyond the step being executed.
    WORKER_LOCAL = ("write_target", "length", "exclusivity")
    #: ...and the one that needs the scheduler's allocation table.
    NEEDS_SCHEDULER = ("ownership",)

    def __init__(self, scheduler=None, reserve=(), group: int = 0):
        self.scheduler = scheduler
        self.reserve = set(reserve)
        self.group = group
        self.violations: list[Violation] = []
        self.checked = 0

    @property
    def active_checks(self) -> tuple[str, ...]:
        return self.WORKER_LOCAL + (
            self.NEEDS_SCHEDULER if self.scheduler is not None else ())

    def owned(self, req_id: str) -> set[int] | None:
        """Block ids the scheduler currently has allocated to `req_id`.

        Returns None when there is no scheduler to ask, or when it does not
        know the request -- neither is a violation. The worker can be a step
        ahead of the scheduler's view of a retired request.
        """
        if self.scheduler is None:
            return None
        mgr = self.scheduler.kv_cache_manager.coordinator.single_type_managers
        blocks = mgr[self.group].req_to_blocks.get(req_id)
        if blocks is None:
            return None
        return {b.block_id for b in blocks}

    def null_block_id(self) -> int:
        """The reserved id that means "not resident", or -1 if unknowable.

        -1 rather than a guess: `null_block` is conventionally block 0, and
        assuming that would make the guard quietly ignore a real block 0 in a
        view when it cannot ask.
        """
        if self.scheduler is None:
            return -1
        pool = self.scheduler.kv_cache_manager.coordinator.block_pool
        return pool.null_block.block_id

    def check_step(self, batch, block_table, seq_lens, slot_mapping, block_size,
                   intended):
        """One decode step. Returns the violations found, and records them.

        `intended` maps req_id to the number of blocks the policy meant to be
        resident this step. Without it the length check is vacuous, because the
        block count the kernel reads is *computed* from seq_len and therefore
        always agrees with it; the question worth asking is whether both agree
        with the decision that was made.

        `slot_mapping` is the real one the kernel will use, not a recomputed
        expectation -- the point of reading it is that a wrong write and a wrong
        idea of where the write should go cannot cancel out.
        """
        found: list[Violation] = []
        null = self.null_block_id()
        resident_sets: dict[str, set[int]] = {}

        for b in range(batch.num_reqs):
            if int(batch.num_scheduled_tokens[b]) != 1:
                continue                      # decode rows only
            req_id = batch.req_ids[b]
            seq_len = int(seq_lens[b])
            n_read = cdiv(seq_len, block_size)
            row = [int(x) for x in block_table[b][:n_read].tolist()]
            self.checked += 1

            # length
            want = intended.get(req_id)
            if want is not None and n_read != want:
                found.append(Violation(
                    "length", req_id,
                    f"seq_len {seq_len} makes the kernel read {n_read} block(s), "
                    f"but the policy meant {want} to be resident"))

            # ownership
            own = self.owned(req_id)
            if own is not None:
                stray = [x for x in row
                         if x not in own and x not in self.reserve and x != null]
                if stray:
                    found.append(Violation(
                        "ownership", req_id,
                        f"{len(stray)} block(s) in the view are not allocated to "
                        f"this request: {sorted(set(stray))[:6]}"))

            # write target: this step's key must land in the last resident block
            if slot_mapping is not None and n_read:
                start = int(batch.query_start_loc_np[b])
                slot = int(slot_mapping[start])
                if slot >= 0:
                    written = slot // block_size
                    if written != row[-1]:
                        found.append(Violation(
                            "write_target", req_id,
                            f"this step writes into block {written}, which is "
                            f"not the last resident block {row[-1]}"))

            resident_sets[req_id] = {x for x in row if x != null}

        # exclusivity
        ids = list(resident_sets.items())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                shared = ids[i][1] & ids[j][1]
                if shared:
                    found.append(Violation(
                        "exclusivity", ids[i][0],
                        f"shares {len(shared)} block(s) with {ids[j][0]}: "
                        f"{sorted(shared)[:6]}"))

        self.violations.extend(found)
        return found

    def summary(self) -> dict:
        by_check: dict[str, int] = {}
        for v in self.violations:
            by_check[v.check] = by_check.get(v.check, 0) + 1
        return {"steps_checked": self.checked,
                "violations": len(self.violations),
                "by_check": by_check,
                "active_checks": list(self.active_checks),
                "scheduler_visible": self.scheduler is not None,
                "first": [str(v) for v in self.violations[:3]]}
