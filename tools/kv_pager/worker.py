"""The worker half: move the bytes, then show the kernel the resident set.

Everything else in this package decides things. This is the part that acts, and
the order it acts in is the whole correctness argument:

    1. copy in    a block the manager restored this step holds whatever the
                  pool last left in it until this runs, and the forward is
                  about to read it
    2. copy out   a block chosen for eviction is still allocated and still
                  intact for exactly one step; this is that step
    3. view       hand the attention metadata builder its own shortened
                  `seq_lens` and block table
    4. guard      check the view names nothing it does not own

**The view goes to the metadata builder, never to `input_batch.seq_lens`.**
That field means two things at once: how many keys attention reads, and how far
through its prompt the request is -- `gpu/sample/sampler.py` classifies a row
with `seq_len < prefill_len` as still prefilling and emits no token for it. A
pager is exactly the thing that needs those two numbers to differ, so shortening
the shared tensor does not fail, it *hangs*: the row stops producing tokens, is
rescheduled forever, and the engine spins with the progress bar at zero.
`CommonAttentionMetadata` carries its own `seq_lens` and `block_table_tensor`
and has a `replace()`, so the kernel can be shown a shorter context while the
rest of the engine keeps seeing the real one.

It hooks `prepare_attn` *after* the real one, which is what keeps this step's
own KV write correct for free: the slot mapping was computed from the untouched
row, so the new key lands in the tail block, which is always resident and
always last. Nothing here has to reason about writes at all.

Steps 1 and 3 are both before the forward, which is safe for a reason worth
stating: the forward writes only the tail block, and an evicted block is never
the tail. If that ever stops being true -- a policy that evicts the block being
written -- the copy-out would race the write, so the resident set including the
tail is an invariant the guard checks rather than a convention.
"""

from __future__ import annotations

import torch

from . import state as pager_state
from .guard import ResidencyGuard
from .hosttier import HostTier


class WorkerPager:
    """Applies the published decisions to the KV cache and the kernel's view."""

    def __init__(self, host_slots: int, scheduler=None, verify=True):
        self.host_slots = host_slots
        self.scheduler = scheduler
        self.verify = verify
        self.state = pager_state.current()
        self.tier: HostTier | None = None
        self.guard: ResidencyGuard | None = None
        self.runner = None
        self.steps = 0
        self.copied_in = 0
        self.copied_out = 0
        self.missing_host_copy = 0
        #: steps where the manager's committed token count and the worker's
        #: disagree. The two sides derive the tail block from this number, so a
        #: lag between them shifts the whole view by a block.
        self.clock_mismatch = 0
        #: batch index -> (row, resident indices, seq_len) for this step
        self._plan: dict = {}

    def install(self):
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        if getattr(GPUModelRunner.prepare_attn, "_pager_hooked", False):
            return
        original = GPUModelRunner.prepare_attn

        def hooked(runner, input_batch, *args, **kwargs):
            self._attach(runner)
            self._wrap_builders(runner)
            self._sync_rows(runner, input_batch)
            out = original(runner, input_batch, *args, **kwargs)
            self.apply(runner, input_batch, out)
            return out

        hooked._pager_hooked = True
        GPUModelRunner.prepare_attn = hooked

    def _wrap_builders(self, runner):
        """Interpose on every full-attention metadata builder, once."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        for groups in runner.attn_groups:
            for group in groups:
                if not isinstance(group.kv_cache_spec, FullAttentionSpec):
                    continue
                for builder in group.metadata_builders:
                    if getattr(builder.build, "_pager_hooked", False):
                        continue
                    original = builder.build

                    def wrapper(*a, _orig=original, **kw):
                        key = "common_attn_metadata"
                        if key in kw:
                            kw[key] = self.view(kw[key])
                        elif a:
                            a = (self.view(a[0]),) + a[1:]
                        return _orig(*a, **kw)

                    wrapper._pager_hooked = True
                    builder.build = wrapper

    def _sync_rows(self, runner, batch) -> None:
        """Make the worker's own block table equal the manager's mapping.

        This has to run *before* the stock `prepare_attn`, because
        `compute_slot_mappings` indexes the worker's persistent row
        positionally to decide where this step's key is written -- and that row
        is not the logical mapping. Restored blocks reach the worker through
        the append channel, so its row grows faster than the context does, and
        index `p // block_size` stops meaning position `p`. The symptom is the
        new key landing inside a *restored* block, corrupting it, which the
        guard reports as a write to something other than the last resident
        block.

        Copying the manager's row over it fixes the slot mapping by
        construction rather than by compensating for it somewhere else, and
        resetting `num_blocks` keeps the next append landing at the logical
        end.
        """
        if not self.state.steps:
            return
        tables = runner.block_tables
        table = tables.block_tables[0]
        for b in range(batch.num_reqs):
            if int(batch.num_scheduled_tokens[b]) != 1:
                continue
            step = self.state.get(batch.req_ids[b])
            if step is None or not step.row:
                continue
            req_idx = int(batch.idx_mapping_np[b])
            row = torch.tensor(step.row, dtype=torch.int32, device="cpu")
            table.gpu[req_idx, :len(step.row)] = row.to(table.gpu.device)
            tables.num_blocks.np[0, req_idx] = len(step.row)
        tables.num_blocks.copy_to_uva()

    def view(self, common):
        """Replace the kernel's context length and block table with the view."""
        if not self._plan:
            return common
        seq_lens = common.seq_lens.clone()
        table = common.block_table_tensor.clone()
        for b, (row, resident, seq_len) in self._plan.items():
            if b >= table.shape[0]:
                continue
            for slot, idx in enumerate(resident):
                table[b][slot] = row[idx]
            seq_lens[b] = seq_len
        return common.replace(seq_lens=seq_lens, block_table_tensor=table)

    def _attach(self, runner):
        if self.tier is None:
            self.runner = runner
            self.tier = HostTier(runner.kv_caches, self.host_slots)
            self.guard = ResidencyGuard(self.scheduler)

    def apply(self, runner, batch, prepared):
        block_tables, slot_mappings = prepared
        if not block_tables:
            return
        self._plan = {}
        table = block_tables[0]
        slots = slot_mappings[0] if slot_mappings is not None else None
        block_size = runner.block_tables.kernel_block_sizes[0]
        caches = runner.kv_caches
        seq_lens = batch.seq_lens
        intended = {}

        for b in range(batch.num_reqs):
            req_id = batch.req_ids[b]
            step = self.state.get(req_id)
            if step is None or not step.resident:
                continue
            # Transport follows the manager's decisions wherever they are made;
            # the *view* is decode-only. Tying both to decode steps is what let
            # a block be freed during a prefill chunk with nothing having
            # copied it out -- silently, since the guard only inspects decode
            # rows too.
            decoding = int(batch.num_scheduled_tokens[b]) == 1
            self.steps += decoding
            computed = int(batch.num_computed_tokens_np[b])
            if step.num_computed != computed:
                self.clock_mismatch += 1

            # 1. copy in, before anything reads the restored blocks
            for idx, block_id in step.restored:
                key = (req_id, idx)
                if key in self.tier:
                    self.tier.load(key, caches, block_id)
                    self.copied_in += 1
                else:
                    # A restore with nothing behind it: the block was never
                    # evicted, or its host copy was dropped. Counted rather
                    # than ignored -- it means the model is about to read
                    # whatever the pool left there.
                    self.missing_host_copy += 1

            # 2. copy out, while the chosen blocks are still allocated
            for idx, block_id in step.evicting:
                self.tier.store((req_id, idx), caches, block_id)
                self.copied_out += 1

            if not decoding:
                continue

            # 3. the view -- recorded here, applied at the metadata builder
            row = step.row
            resident = self._resident_now(step, computed, block_size, len(row))
            self._validate(req_id, row, resident, len(caches[0]))
            tail_count = computed % block_size + 1
            seq_len = (len(resident) - 1) * block_size + tail_count
            self._plan[b] = (row, resident, seq_len)
            intended[req_id] = len(resident)

        # 4. and check what the kernel is about to be shown. The guard reads
        # the same view the builder will get, built here rather than from the
        # gathered table, which is deliberately left alone.
        if self.verify and self.guard is not None and intended:
            view_table = table.clone()
            view_seq = seq_lens.clone()
            for b, (row, resident, seq_len) in self._plan.items():
                for slot, idx in enumerate(resident):
                    view_table[b][slot] = row[idx]
                view_seq[b] = seq_len
            self.guard.check_step(batch, view_table, view_seq, slots,
                                  block_size, intended)

    def _resident_now(self, step, computed, block_size, row_len) -> list[int]:
        """The manager's choice of full blocks, with *this* step's tail.

        The two sides run on different clocks and must: the manager frees
        against the committed prefix (`total_computed_tokens -
        num_in_flight_tokens`), because an in-flight step is still reading
        blocks above it, while the view has to describe where this step's key
        is actually written. Taking the manager's set wholesale puts the tail
        one block behind whenever those differ, which is the `write_target`
        violation the guard reported.

        So: the policy's decision about full blocks is the manager's, since
        that is what it froze its freeing on, and everything from the
        manager's tail forward is kept unconditionally. Those indices are never
        freeable -- the manager only frees below its own tail -- so they are
        real, they hold real keys, and dropping them would make the newest
        tokens invisible rather than merely unpaged.
        """
        mgr_tail = step.num_computed // block_size
        tail = computed // block_size
        keep = [i for i in step.resident if i < mgr_tail]
        keep += [i for i in range(mgr_tail, tail + 1) if i < row_len]
        return keep

    def _validate(self, req_id, row, resident, num_blocks) -> None:
        """Fail where the mistake is, not where the GPU notices it.

        A bad block id written into the view is dereferenced by the attention
        kernel, so it surfaces as an asynchronous illegal access at whatever
        call happens to synchronise next -- which is somewhere else entirely.
        These are two comparisons per resident block against a Python list, and
        they turn that into an exception naming the request and the index.
        """
        bad = [i for i in resident if not 0 <= i < len(row)]
        if bad:
            raise IndexError(
                f"{req_id}: resident indices {bad[:4]} are outside a row of "
                f"{len(row)} blocks")
        ids = [row[i] for i in resident]
        bad_ids = [x for x in ids if not 0 <= x < num_blocks]
        if bad_ids:
            raise IndexError(
                f"{req_id}: block ids {bad_ids[:4]} are outside a pool of "
                f"{num_blocks} blocks")

    def summary(self) -> dict:
        out = {"steps": self.steps, "copied_in": self.copied_in,
               "copied_out": self.copied_out,
               "missing_host_copy": self.missing_host_copy,
               "clock_mismatch": self.clock_mismatch}
        if self.tier is not None:
            out["tier"] = self.tier.stats()
        if self.guard is not None:
            out["guard"] = self.guard.summary()
        return out
