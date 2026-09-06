# KV residency is a block-table rewrite: the gate on score-driven paging

A pager that manages KV residency by rewriting `req_to_blocks` only works if a
request's block table can be reordered without changing what the model computes.
The argument that it can is short: a decode step's attention output is
`sum_i softmax(q.k_i) v_i` over the resident set, softmax is permutation
invariant, and each cached K already carries the RoPE rotation for its own
position, so nothing downstream of the cache should be able to tell block 7 from
block 3. If that holds, residency is a block-table rewrite on stock attention
kernels -- no mask, no `null_block`, no FA4. If it does not, something reads
position from slot index and the whole plan changes shape.

`tools/blocktable_permute.py` measures it. **It holds**, on four checkpoints and
three attention backends, including with an fp8 KV cache and on an EXL3
checkpoint through this plugin's own path. It does *not* hold for
sliding-window layers, which is measured here rather than assumed.

## The instrument

The hook sits in `GPUModelRunner.prepare_attn`, the one point where a block
table row is settled and not yet read: `execute_model` has already flushed the
scheduler's staged appends into the persistent GPU rows, and the two consumers
-- the gather that hands the kernel its block table, and the triton kernel that
turns positions into KV slots -- both run inside it, from those same rows. A
rewrite there moves the attention read and the KV write together, and nothing
else has to be kept in sync, because appends are staged at `num_blocks[req]`
regardless of what order the earlier entries are in.

Only the *full* blocks move. The partial tail block stays where it is: a paged
kernel derives its valid-token count from `seq_len` minus the blocks before it,
and it is also where this step's own key lands. `num_computed_tokens_np` is the
scheduler's pre-step value, so `computed // block_size` never counts the block
the current token is about to be written into.

Five arms run against one engine, and four of them are controls:

| arm | what it does | what it is for |
|---|---|---|
| `control` | nothing | the reference |
| `identity` | the same GPU index copy, permutation = identity | the rewrite *mechanism* is inert |
| `permute` | full blocks reshuffled before every decode step | the test |
| `tailswap` | the partial tail swapped with a full block | a wrong table is detectable |
| `control2` | nothing, again | the engine is reproducible, and `tailswap` did not leak |

`tailswap` is what makes a `permute` pass mean anything: a test that only ever
reports "no change" cannot distinguish a permutation-invariant kernel from a
hook that never fired. `identity` closes the other end -- if the index copy
itself perturbed anything, no arm would mean anything. Both come back exactly
zero in every run below.

## Why the measurement is at layer 0 and not at the tokens

Comparing generated tokens does not work here, and this repo already knows why:
an execution-mode change alone flips 9 argmax decisions in 91 positions on
Laguna-XS ([tensor-parallel.md](tensor-parallel.md)), so demanding bit-identical
output would reject configurations that are correct. Permuting blocks changes
FlashAttention's accumulation order, so bitwise equality was never the
prediction.

The confound-free measurement is one layer deep. Generated token 0 comes out of
the prefill forward, which no arm touches. So at the first decode step both arms
have a bit-identical cache, a bit-identical input token and therefore bit-identical
Q, K and V, and the new key lands in the same physical slot either way. The only
difference in the universe at that instant is the order of the block ids, and
layer 0's attention output is the permutation's effect, measured rather than
inferred from what it does to logits sixteen layers later.

Distance is read off the bit patterns as an exact ulp count, so it is
scale-free. Two readings are taken, and both are relative rather than
thresholded against a number picked in advance:

- the largest difference anywhere, against **one representable step at that
  tensor's own scale** -- reordering the terms of a sum moves an fp32
  accumulator by parts in 1e7, so after rounding to bf16 it can change an
  element by one step and no more;
- how many elements moved past one step, against **what the negative control
  did in the same run**.

Elements near zero are excluded from the headline count. They are the
difference of signed terms that nearly cancelled, so their low bits carry no
information: the worst such element in the first passing run sat at -1.3e-6 on
a tensor of scale 0.375 and was 65 ulps away, while the largest *absolute*
difference in the whole tensor was 4.883e-4 -- exactly half a bf16 ulp at
magnitude 0.25, i.e. one rounding decision. Reporting the unrestricted ulp
maximum alone would have called that a failure.

## Results

`permute` and `tailswap` at the first rewritten step, layer 0 (the first
full-attention layer). "moved" counts elements past one representable step, out
of those carrying magnitude; `|d|` is the largest absolute difference anywhere
in the tensor, against one representable step at that tensor's scale.

| model / config | backend | `permute` moved | `permute` \|d\| vs step | `tailswap` moved | `tailswap` \|d\| |
|---|---|---|---|---|---|
| Llama-3.2-1B, bf16 KV, 2048 ctx | FLASH_ATTN | **0**/2102 | 4.88e-4 vs 3.40e-3 | 1405/2102 | 7.81e-3 |
| Llama-3.2-1B, bf16 KV, 256 ctx | FLASH_ATTN | 3/2548 | 1.22e-4 vs 2.14e-3 | 2087/2548 | 3.33e-2 |
| Llama-3.2-1B, **fp8 KV**, 2048 ctx | FLASHINFER | 134/1906 | 1.95e-3 vs 3.91e-3 | 1186/1906 | 7.81e-3 |
| Llama-3.2-1B-**exl3**, 2048 ctx | FLASH_ATTN | **0**/2292 | 1.95e-3 vs 3.60e-3 | 1518/2292 | 5.86e-3 |
| gemma-4-E2B-it, full-attn groups | TRITON_ATTN | **0**/6732 | 7.81e-3 vs 7.62e-2 | 4857/6732 | 7.45 |
| gemma-4-E2B-it, **sliding groups too** | TRITON_ATTN | 2684/2722 | **5.61** vs 8.89e-2 | 1848/2722 | 7.37 |

Every full-attention row passes both tests. The last row is the deliberate
counter-example and is discussed below.

Two things worth naming in that table. **fp8 is the noisiest passing
configuration** -- 134 elements moved, not zero, and the largest difference is
half a representable step rather than a tenth. That is FlashInfer's reduction,
not the fp8 values (the summands are bit-identical; only their order changed),
and it still sits an order of magnitude below the negative control. And **a
single wrong block is quiet**: `tailswap` misplaces one block out of 128, about
0.5% of the attention mass, so its margin over `permute` is a factor of ten at
2048 tokens and a factor of 700 at 256, where one block is 6% of the context.
A paging bug will degrade quality without announcing itself.

### Sliding-window groups are excluded, and that is not a precaution

A sliding-window kernel rebuilds key position from the block's index in the row
in order to apply its local mask, which is exactly the "reads position from slot
index" failure this test looks for. The hook skips those groups by default and
counts what it skipped (504 row-groups against 126 permuted, on gemma-4-E2B's
4:1 sliding-to-full layout), and the probe checks that the skipped layers
running *before* the first permuted one come back bit-identical -- which they do.

`--all-groups` permutes them anyway, and the result is unambiguous: layer 0's
output moves by 5.61 on a tensor of scale 11.4, with 2684 of 2722 elements past
one step. That is *more* disturbed than the negative control, and it costs KL
2.80 at the logits in the same step. So a hybrid model's global layers can be
paged and its sliding layers cannot -- which is not a real loss, since a
sliding-window layer's resident set is already bounded by its window.

### CUDA graphs: unresolved at this instrument's resolution

Forward hooks do not run inside a replayed CUDA graph, so under `--graphs` the
probe reports nothing and the finest measurement left is the first rewritten
step's logits. That is not enough. In the graphs run the permutation moves one
request's distribution by KL 3.7e-4 and a genuinely wrong block moves the
other's by 7.8e-5 -- the arms overlap, and the tool says so rather than
returning a verdict. Sixteen layers of amplification is too coarse an
instrument for the question.

This is a limit of the measurement, not evidence against the claim: the graphs
run's trajectories behave like the eager run's, and the block tables are the
same persistent GPU buffers either way. But the invariance is established in
eager mode only.

## What this settles, and what it does not

Settled: residency can be managed by rewriting a request's block table, on
stock FA2/FlashInfer/Triton paths, with fp8 KV, on a quantized EXL3 checkpoint,
without a mask or a `null_block` and without FA4. Block *index* carries no
positional meaning for full-attention layers.

Not settled, in rough order of how much it matters to the next phase:

1. **The allocator is still not involved.** The data path is proven below --
   a block survives being destroyed on the GPU and restored from host memory
   into a different physical block -- but every tool here imposes a view while
   vLLM's block pool still owns all of it. Reclaiming means the *scheduler*
   freeing blocks and re-allocating them, which is phase 2's work.
2. **Prefix caching was off for every run.** Its bookkeeping hashes a block by
   the token prefix leading to it, so a residency scheme that moves or drops
   blocks has to be reconciled with the cached-block registry. Untouched here.
3. **CUDA graphs**, per above -- and the eviction runs are eager-only too.
4. **One policy shape, one model, one budget.** Whether recency or a score
   keeps the right blocks is the phase-3 measurement; the needle here says only
   that the mechanism honours whatever the policy decides.
5. Single GPU only; no TP, no MLA backend, no mamba/hybrid state groups beyond
   gemma's sliding layers; decode steps only, never a prefill chunk.

## Fewer blocks than positions: the step the permutation test did not clear

Permutation proves block index carries no positional meaning. A pager needs
more than that: it drops blocks, so the row gets *shorter than the context*,
and the token at position `p` stops living at row index `p // block_size`.
`tools/blocktable_evict.py` measures whether the engine can be handed a request
in that state. **It can** -- 17 blocks stood in for 2076 positions, and the
model attended to exactly those and nothing else.

### `seq_lens` is overloaded, and that is what makes the naive approach hang

The obvious implementation -- lower `input_batch.seq_lens` to the resident
count and shorten the row -- does not fail, it *hangs*, with the engine
spinning on `execute_model` and the request never producing a token.
`gpu/sample/sampler.py` decides whether a row emits an output token by testing
`seq_len < prefill_len`, the condition that marks a chunked-prefill request as
not yet finished. So `seq_lens` means two things at once -- how many keys
attention should read, and how far through its prompt the request is -- and a
pager is exactly the thing that needs those two numbers to differ.

The seam that separates them is `AttentionMetadataBuilder.build`.
`CommonAttentionMetadata` carries its *own* `seq_lens` and `block_table_tensor`
and has a `replace()` helper, so the view can be handed to the kernel as a copy
while `input_batch.seq_lens` keeps saying what is really cached. The rest of
the engine then sees an ordinary request. Anything built on this should impose
residency there, not upstream of it.

Two further mechanics fall out for free, and both are worth keeping:

- the rewrite runs **after** the real `prepare_attn`, so `slot_mapping` was
  already computed from the untouched row -- this step's own key lands where it
  always would, and no slot surgery is needed anywhere;
- the view keeps the partial **tail block last**, the same invariant the
  permutation test needed, so the kernel's valid-token count for it stays
  `seq_len` minus the full blocks in front.

### What was checked

Only the per-step view is rewritten. The persistent rows, the scheduler and the
allocator are untouched, so **nothing is freed and no memory is saved**: this
asks whether the *kernel* can be told, which is the part that could have
derailed the plan. Reclaiming the blocks afterwards is ordinary work vLLM
already does on preemption.

Llama-3.2-1B, 2048-token context, budget 16 of 128 full blocks (2 sink + 14
most recent), FLASH_ATTN, bf16 KV:

| check | arm | result |
|---|---|---|
| the residency arithmetic is right | `viewfull` — every block resident, through the eviction path | **bit-identical** to control |
| a real budget runs at all | `evict` — 112 of 128 blocks dropped | runs; 17 blocks for 2072 positions |
| nothing past the resident prefix is read | `poison` — trailing row entries overwritten with a duplicate resident block | **bit-identical** to `evict` |
| the engine is still reproducible | `control2` | **bit-identical** to control |

`viewfull` is the load-bearing one. It drops nothing, so it must reproduce the
control exactly, and it is the only place a tail count or a `seq_len` that is
wrong by one would show up -- an off-by-one there is invisible in every arm
that actually evicts, because those have no reference to be wrong against.

### The needle: eviction is exact, not approximate

The numbers above cannot say whether the kernel attends to the resident set or
merely to *something*. A magic number planted at a known token offset can:
the block holding it is known, and two arms spend the **same budget** differing
only in whether that one block survives.

| arm | resident | answer |
|---|---|---|
| control | all 128 blocks | `918273` found |
| `keep_needle` | 16 blocks, one of them the needle's | `918273` found — and the generated tokens are *identical to the full-context run* |
| `drop_needle` | 16 blocks, that slot spent on recency instead | `3` — lost |

Retrieval flips with one block out of 128, which is what "attends to exactly
the resident set" means operationally. The `keep_needle` row is also the whole
premise of the pager in miniature: **87.5% of the KV dropped, output unchanged**,
because what was dropped was not being attended to.

The needle is planted at an exact block boundary deliberately -- one straddling
two blocks would survive the loss of either, and the arms would no longer
differ in exactly one thing.

## Design context carried forward

Recorded 2026-09-04 from a design session that preceded the measurements above.
Everything here is either superseded by them or still live, and it is labelled
either way rather than left to be re-derived.

### What the measurements superseded

The design was pinned on a choice between two branches, and **the trade-off
that made it a choice turned out to be false**:

- *Branch A, reorder and compact.* No mask, runs on FA2, fp8 available. Thought
  to break prefix-cache hashing, ALiBi and sliding window.
- *Branch B, holes plus `null_block` plus `mask_mod`.* Logical positions
  preserved, so prefix-cache hashing stays valid. Needs FA4 (not available on
  consumer hardware) or FlexAttention, which is 16-bit only -- and bf16 at 50%
  residency is exactly fp8 at 100%, so that carrier defeats the purpose.

The mechanism measured here is a third option with both sets of properties,
because residency is imposed as a **per-step view** rather than as a change to
anything durable. The authoritative `req_to_blocks`, the allocator and the
prefix-cache bookkeeping never see the compaction, so Branch B's best argument
survives intact under Branch A's mechanism: **a pager relocates blocks rather
than modifying them**, so a block's hash stays truthful and a hit on a
non-resident block is a fetch rather than a wrong answer. That is the property
that makes paging composable where TriAttention's compaction is not -- its
compacted block holds a different set of tokens than its hash claims, which is
unfixable in principle. Prefix caching was off in every run here, so this is a
structural argument and not yet a measurement.

Also superseded: FA4 and FlexAttention are not needed at all; ALiBi and T5
relative bias remain the genuine exceptions, on a rule worth keeping --
**position that is mixed into the hidden state before K is computed (RoPE,
learned or sinusoidal absolute, NoPE) survives reordering; position computed
from indices at attention time (ALiBi, T5 bias) does not.**

**And the coarse-granularity worry was an artefact of the transport, not of
paging.** A sweep of managed-memory fault migration put a 16-token granule at
4.1 GB/s and only reached the plateau near 1 MiB -- 1024 consecutive token
positions, which is a whole region of context rather than a page, and coarse
enough to wreck the sink structure the whole idea depends on. That curve is a
property of **UM page migration**, and the design measured here does not fault:
explicit pinned DMA's granularity floor is per-transfer overhead, which is much
smaller. Measured below: block-granular residency is back on the table, and the
policy scores blocks rather than spans.

### Still live

**The reclamation precedent is R-SWA, and it is in-tree.** Nothing here frees
anything: both tools impose views while the allocator keeps every block.
`RSWASpec` (`kv_cache_interface.py`) already does the missing half --
`_remove_blocks_in_range` (`single_type_kv_cache_manager.py`) frees blocks from
the **middle** of a running request and substitutes `block_pool.null_block` in
`req_to_blocks`, returning the memory to the pool. Two callers exist and the
distinction matters: sliding window evicts a head prefix, R-SWA evicts a gap.
The gap case is the pager's shape.

The open question underneath it is sharp, and it is the next mechanical thing to
find out: **R-SWA only ever slides forward, so it never restores a nulled slot.**
Whether the block manager supports re-populating a `null_block` slot on a
running request -- or whether R-SWA works precisely because it never asks -- is
untested, and it is the one asymmetry between "sliding window" and "pager".

**The fetch budget is absolute, not proportional.** At fp8 on the 27B, roughly
1,700 tokens per decode step costs 5% latency, *whatever the context length*.
So "keep 5% resident" is cheap at 32K and unaffordable at 300K, where 5% is
14,800 tokens and about 44% of a step. A policy has to bound **tokens fetched
per step**, not a fraction of context -- which is a different objective from the
one every published eviction method optimizes.

**Quest is the relevant literature; kvpress is the wrong shelf.** NVIDIA's
kvpress carries 40+ methods and a RULER harness, but every one of them prunes
permanently during prefill or compresses periodically during decode -- none
decides *per decoding step* which parts of the cache to use, because permanence
is baked into a compression suite's premise. Quest is the matching shape: per-page
min/max bounds on the keys give an upper bound on that page's attention for the
current query, and it selects top-k pages every step. Page-granular, query-aware,
non-destructive, re-decided each step. `ExpectedAttentionPress` (estimating
importance from the future-query distribution, the same family as TriAttention's
scorer) remains interesting as the *content* of the decision; Quest is the
*structure* of it.

**Policy constraints already measured, that a first policy should not rediscover:**

- **Aggregate a GQA group by mean, not max** -- 76.7% vs 69.6% of mass captured
  at a 5% budget on Qwen3-8B. Max follows whichever head is most confident;
  mean finds the consensus. A larger effect than the union tax itself (~1 point).
- **Per-layer budgets are probably right and a single global budget probably
  wrong**, since early layers are much more diffuse than late ones.
- **Hybrid stacks change the arithmetic**: Qwen3.8-27B is 16 attention layers of
  64, the rest GDN recurrent state that cannot be paged at all -- so there is 4x
  less to page than the parameter count suggests, and 4x less to save.
- Sliding-window layers are already bounded; there is nothing to page.
  MLA compresses KV into a shared latent and changes the object entirely.

**Concentration rises with context, and the score's margin over recency rises
with it** (Qwen3-8B, keep 5%, GQA-shared -- the numbers a pager actually gets):

| context | oracle | trig score | recency | score - recency |
|---|---|---|---|---|
| 4K | 87.8% | 76.7% | 70.7% | +6.0 |
| 8K | 86.7% | 70.8% | 59.5% | +11.3 |
| 16K | 91.3% | 80.7% | 68.3% | +12.4 |
| 32K | 92.7% | 78.1% | 62.9% | +15.2 |

The 8K dip appears in the **oracle** too, so it is a property of that prompt at
that length rather than of any policy -- the single-prompt caveat asserting
itself. The trend is the point: the regime where paging matters is the regime
where it works best, which is backwards from how these costs usually scale.

**A disk tier is coherent, and for very long contexts it is not optional.** A
1M-token context at fp8 is ~32 GB, so GPU plus host covers roughly half of one
on this box; disk measured 2.2 GB/s here (virtualised, so a floor rather than
the hardware's number). The observation that makes it plausible rather than
desperate: what makes a 4x RoPE-scaled 1M context unsatisfying as a *capability*
-- the model does not attend densely that far out -- is exactly what makes it
cheap to page. You are paging a region the model has already decided not to
look at.

## Freeing and restoring a block, and what the transport costs

The two questions phase 3 was gated on, both answered on 2026-09-05.

### A block can be freed and restored on a running request

`tools/kv_roundtrip.py`. Eviction has no reference, and neither does a
restore -- unless the whole cycle happens inside one decode step, where it
does. Each step, for one chosen block: its KV is copied out to pinned host
memory for every layer, **the GPU block is filled with garbage** (standing in
for the block being freed and handed to another request, which is what makes a
pager save memory and what is most likely to corrupt it), the host copy is
written into a *different* physical block, and the residency view is repointed
at the new address.

If every leg works the model reads exactly the bytes it would have read anyway,
from a new address, and the output is bit-identical. It is:

| config | backend | relocate | nocopy (same, restore skipped) |
|---|---|---|---|
| Llama-3.2-1B, bf16 KV | FLASH_ATTN | **bit-identical**, 31 round trips | diverges at step 1 |
| Llama-3.2-1B, **fp8 KV** | FLASHINFER | **bit-identical**, 31 round trips | diverges at step 1 |
| Llama-3.2-1B-**exl3** | FLASH_ATTN | **bit-identical**, 31 round trips | diverges at step 1 |

`nocopy` is what makes that mean something: identical in every respect except
that the restore is skipped, so the view names a block holding garbage. It
diverges immediately and the reference's own top token falls out of the top-20
entirely, so the destroy was real and the model was genuinely reading the
relocated block. The tool also checks, inside the run, that the overwrite
changed the source and that the source is *still* wreckage at the moment the
restore is read -- three assertions that each catch a different way for this
test to pass without meaning anything.

**Physical relocation is the point.** A pager that had to restore a block to
the address it came from would be a much weaker mechanism; this restores into
an unrelated block near the top of the pool and repoints the view, which works
for the same reason the permutation test worked -- block identity carries no
positional meaning.

The methodological trap, recorded because it cost a wrong answer first: **do
not re-read the source block each iteration.** The first version saved and
destroyed every step, so from step 2 on it was saving back the garbage the
previous step had written, and it reported the round trip as lossy. Worse is
the failure in the other direction -- a save/restore test that re-reads the
source will happily pass even when the restore does nothing, because the second
save reads back whatever the first restore wrote. Once a block is evicted its
only copy lives on the host, and the test has to model that.

Not covered: the block is relocated every step rather than left away for a
while, though nothing writes to an absent block; and the scheduler still owns
every block, so this is the data path and not the allocator integration.

### Explicit DMA does not care about locality, only about transfer size

`tools/kv_transport.py`, using vLLM's own `swap_blocks` and `swap_blocks_batch`
(the latter submits every copy in one `cuMemcpyBatchAsync`) rather than a
synthetic proxy. Geometry is per layer, because that is how a KV cache is laid
out: moving one block of context means one copy per attention layer. Defaults
model Qwen3.8-27B at fp8 -- 16 attention layers, 32 KiB per 16-token block.

| granule | UM faulting (2026-09-04) | explicit, one copy per block | explicit, one copy per run |
|---|---|---|---|
| 16 tok | 4.1 GB/s | **17.2** | 17.2 |
| 64 tok | — | **17.2** | 35.6 |
| 128 tok | 6.1 | **17.2** | 43.2 |
| 512 tok | 7.6 | **17.2** | 51.0 |
| 1024 tok | 12.6 | **17.2** | 52.8 |

Contiguous ceiling 54.4 GB/s. **The per-block column is flat**: scattering the
blocks across the pool costs nothing at all, which is the opposite of the
fault-driven curve that this design was previously being shaped around. What
costs is submission, and the whole table collapses to

    cost = copies x 1.30 us + bytes / 54 GB/s

confirmed across two geometries -- 1.31 us per copy at 32 KiB (Qwen3.8-27B fp8,
16 layers) and 1.26 us at 64 KiB (Qwen3-8B bf16, 36 layers), where the larger
copy raises the per-block rate to 26 GB/s exactly as a fixed overhead predicts.

**So granularity stops being a policy constraint and becomes a batching
optimization.** A policy may score 16-token blocks and pay nothing for
scattering them; coalescing runs into single descriptors is opportunistic
profit when the chosen set happens to contain them. Attention sinks cost one
block, not a 1024-token page -- which was the objection that made the coarse
granularity unacceptable.

In fetch budget, at a 5% latency allowance on a 20.7 ms decode step
(Qwen3.8-27B, fp8, 32 KiB/token across all layers):

| | tokens per step |
|---|---|
| fully scattered blocks | **543** |
| 128-token runs coalesced | 1,364 |
| contiguous | 1,718 |

The 8B at bf16 is far tighter -- 184 tokens/step scattered, because it spends
144 KiB per token across 36 layers. That is the arithmetic that makes fp8 and a
hybrid stack matter to a pager, rather than being incidental preferences.

## The allocator: design

Everything measured so far imposes a *view* — the kernel is told to attend to a
subset, while vLLM's block pool still owns every block, so no memory is saved.
This is the design for the part that actually reclaims, written after the
measurements rather than before so it inherits their constraints instead of
guessing at them.

### The finding that shapes the phasing: a recency pager never fetches

Worth stating first because it changes what phase 2 is for. A recency policy
keeps the last K blocks plus sinks, and its window only ever slides *forward*.
Blocks leave the resident set and are never asked for again, so after warmup the
**fetch rate is zero** — the machinery's restore path, which is the entire
difference between paging and eviction, would go untested by the very policy
chosen to test the machinery. A recency pager is StreamingLLM, and it is
StreamingLLM precisely because it never notices it was wrong.

Two consequences, and the second is architectural:

- Phase 2 needs a **`stress` policy** alongside `recency`: one that rotates
  residency deliberately, so the fetch path runs, the transport is exercised
  against the measured cost model, and the thrash curve gets measured. That is
  a test policy, not a shippable one, and it is the only way "policy error is a
  tunable cost" gets demonstrated rather than asserted.
- A real pager needs a **demand signal** — something resident that says "you
  will want this block that is not here". Recency has none by construction, and
  neither does any policy that only looks at what it kept. This is what Quest's
  per-page min/max key bounds actually are: kept resident when the block
  leaves, they give an upper bound on that page's attention for the current
  query without its keys. So Quest is not merely a better policy than recency,
  it is *the fault detector*, and without something in that shape a pager can
  only evict on a schedule and hope.

  The bounds are not free: `num_kv_heads x head_dim x 2` values per block per
  layer, which for Qwen3.8-27B (4 KV heads, head_dim 256) is 4 KiB in fp16
  against the block's own 32 KiB — **12.5% overhead, or 6.25% at fp8** — and it
  comes out of the residency budget. Worth measuring rather than assuming, and
  worth knowing before a policy is chosen.

### Architecture

Four parts, three of which already exist as measured code.

1. **A registered spec and manager.** `KVCacheSpecRegistry.register(spec_cls,
   manager_class=...)` supports out-of-tree specs, and
   `Attention.get_kv_cache_spec` is where a layer's spec is chosen — so the
   allocator can be a plugin rather than a fork patch. The manager subclasses
   `SingleTypeKVCacheManager`, bounds a request's GPU blocks to its budget, and
   frees the rest through `_remove_blocks_in_range`, which is R-SWA's existing
   code and already does exactly this for a middle gap.
2. **A host tier.** A pinned per-layer block pool, moved with `swap_blocks` at
   `copies x 1.30 us + bytes / 54 GB/s`. Sized by host RAM; the third tier
   (disk) is out of scope until this one works.
3. **The worker-side view** — built and measured, bit-exact when it drops
   nothing.
4. **An invariant checker** — built first, for the reason below.

### The decisions

**How the two sides agree, without a protocol change.** The scheduler owns
freeing; the worker owns the view; the null substitution never travels between
them. The cheapest correct answer is **one policy function, called on both
sides**, deterministic in inputs both already have — `num_computed_tokens`, the
budget, the block count. Not two implementations that agree by review: literally
one function, imported by both, which keeps the whole thing plugin-shaped with
no fork patch. This breaks the moment a policy is query-aware, because the
worker will know something the scheduler cannot — and *that* is when a residency
field in `SchedulerOutput` becomes necessary. Deferring it is deliberate, not an
oversight.

**Ordering, because the failure is silent.** A freed block that the view still
admits means attention reads another request's KV: plausible garbage, no crash,
no NaN. So the sequence is fixed — at step N the worker decides, copies the
block out to host, and drops it from the view; at step N+1 the scheduler frees
it. There is never a step in which the view names a block the pool has given
away. The invariant checker asserts exactly that (every id in the view is still
owned by this request), and it should be built *before* the freeing, because it
is the only thing that will notice when this is wrong.

**A reserve pool, so a fetch does not cost a step.** A restored block needs a
destination, and asking the scheduler for one costs a scheduling round trip —
which would make every miss a two-step stall rather than a transfer. The pager
should hold its own reserve of GPU blocks so a fetch is worker-local and
immediate. Size it at the per-step fetch budget plus slack: 543 tokens/step at
5% latency on the 27B at fp8 is ~34 blocks.

### The guard, which exists

`tools/kv_pager/guard.py`, built before the allocator because it is the only
thing that will notice when the allocator is wrong. Four checks, each on the
*resident prefix* only — everything past it is unread by construction, so
checking it would reject correct behaviour:

| check | catches |
|---|---|
| `ownership` | a view entry the request no longer owns — what a freed, reallocated block looks like from the worker's side |
| `write_target` | this step's own key landing anywhere but the last resident block, read off the real slot mapping rather than a recomputed expectation |
| `length` | `seq_len` and the row disagreeing with what the policy meant to be resident |
| `exclusivity` | two requests' resident sets overlapping |

`tools/kv_guard_selftest.py` injects each fault into a running engine at the
same point a pager would impose residency, and every one is caught while a
clean run reports nothing:

    clean        14 steps checked,    0 violations (none)
    stray        14 steps checked,   14 violations ['ownership']
    tailswap     14 steps checked,   14 violations ['write_target']
    shortview    14 steps checked,   14 violations ['length']
    overlap      14 steps checked,   14 violations ['exclusivity', 'ownership']

The clean row is not a formality: it says a correct full-residency view passes
all four, so the invariants match what vLLM actually does rather than what this
design assumes it does.

One of the four had to be repaired before it could fail at all. The `length`
check first compared `seq_len` against the block count *derived from* `seq_len`
by division, which is true by construction — a check that cannot fire. It only
became falsifiable once the policy's intended resident count was passed in from
outside, which is the general shape of the mistake: an invariant stated in
terms of one quantity twice.

### Keeping a pager bug out of an accuracy number

The failure this guards against is worse than a crash in one specific way: it
produces plausible text. A run corrupted by a stolen block does not stop, it
just becomes slightly wrong, and if that happens during a capability
measurement the damage is folded into the number and a design decision gets
made on it. So mechanism error has to be separable from policy cost by
construction, not by inspection.

Three layers, because no one of them is sufficient:

1. **The worker-local checks, always on.** Three of the four need nothing but
   the step being executed, so they run in a deployed pager and not only under
   a debugger. The self-test in worker-local mode shows exactly what that
   buys and what it does not:

       clean       0 violations                     stray       0 violations
       tailswap   14 violations ['write_target']    shortview  14 ['length']
       overlap     7 violations ['exclusivity']

   `stray` reporting nothing is the honest result, not a bug in the test: a
   block stolen by another request is precisely the fault the worker cannot
   see, because deciding it requires the scheduler's allocation table.

2. **A full-residency control arm on every quality measurement.** A pager
   configured to evict nothing must reproduce the unpaged baseline
   *bit-for-bit* — the same standard `viewfull` already meets. Any deviation is
   a mechanism bug and voids the measurement, which is what separates "paging
   costs two points" from "our pager has a bug", the confound that makes a
   corrupted accuracy number dangerous. Necessary and **not sufficient**: it
   cannot catch a bug that only manifests once eviction actually happens.

3. **Shadow checksums, sampled.** Evicted blocks already have an authoritative
   host copy, so corruption of their GPU image is harmless — it is refetched.
   The exposure is a *resident* block being stolen, and it is catchable
   out-of-process and cheaply: shadow a small random subset of resident blocks
   and verify them each step. That is the only one of the three that detects
   the damage during a real eviction run rather than its precondition.

And the provenance rule that makes the rest usable: **a quality number records
which layers were active**, the way `bench/` records the tree it was taken
from. `summary()` reports `active_checks` and `scheduler_visible` for exactly
this. A number without that record is not evidence about paging, because
nothing distinguishes it from a number taken with the guard switched off.

### The manager, so far

`tools/kv_pager/` — `policy.py` (the residency decision, a pure function both
sides import), `manager.py` (the spec and manager, registered rather than
patched), `guard.py`. Tested in `tests/test_kv_pager_manager.py` against a real
`BlockPool` and a real `FullAttentionManager` subclass, with no GPU and no
engine.

Restores work, and they use the shape the in-tree copy-on-write redirect
already uses: reserve the extra block in `get_num_blocks_to_allocate`, draw it
in `allocate_new_blocks`, write it into the row *in place*. The worker learns
which logical index each restored block belongs to with no protocol change,
because the pairing is implied — restored blocks are returned before grown ones
and in ascending index order, so the leading new ids pair with
`sorted(pending_restores)`, and `restored[request_id]` records that pairing on
this side so the two can be checked rather than assumed equal.

The assertion that makes it a pager rather than bookkeeping: a 60-block request
on a budget of 8 never holds more than **budget + 2** real blocks, across a run
that restores repeatedly. And the reservation matches the draw at every step,
which is the accounting bug that would otherwise surface as an OOM under load
rather than as a wrong answer.

**Eviction is two-phase**, which is what makes the transport safe: a block
chosen this step is freed on the *next* one, so there is a window in which it
is still allocated, still holds its contents, and is already out of the view.
That window is when the worker copies it out. Freeing in the pass that chose it
would hand the block to another request before anything had read it, and the
resulting corruption is silent — so the window is a tested property, not a
convention, and the constant it adds to peak residency is visible in the bound
test.

### The host tier

`tools/kv_pager/hosttier.py`. One pinned host buffer per layer, shaped like
that layer's GPU cache, with its own slot count. Its single contract is that a
block survives being freed and reused, and the tests exercise that directly on
real GPU tensors without an engine: store it, **overwrite the GPU block**,
restore it into a *different* physical block, require bit-exactness — the same
shape as the end-to-end round trip, because a save/restore test that does not
destroy the original in between passes even when the restore does nothing.

Copies go through tensor indexing rather than `swap_blocks`, deliberately.
`swap_blocks` derives byte offsets from a block stride and so assumes the cache
is block-major and contiguous, while indexing is correct whatever
`get_kv_cache_stride_order` did. It is the same cost class as the measured
per-block path (17.2 GB/s), and the batched primitive is worth adopting behind
a bit-exactness check rather than on the assumption that the layout is what it
looks like. They are also synchronous: the two-phase window is what makes
overlapping *possible*, but an async copy that has not landed before the pool
reassigns the block is precisely what the guard exists for, so the fast version
should arrive with an event to wait on rather than by deleting the wait.

### Wired, end to end

`tools/kv_pager/worker.py` and `state.py`, driven by `tools/kv_pager_run.py`.
The manager publishes its per-step decision to a small state object and the
worker consumes it — the worker never reaches into the manager, so replacing
that object with a field on `SchedulerOutput` is the only change a
multiprocess deployment needs.

    off     no pager                                    the reference
    full    the whole pager, evicting nothing           bit-identical to off
    paged   budget 16 of 64 blocks, stress policy       78 out, 30 back,
                                                        0 guard violations,
                                                        0 restores unbacked

`full` being bit-identical is the control arm the integrity section asked for,
and it is doing real work: manager publishing, row sync, view, transport
machinery and guard all active, with a policy that keeps everything resident.

Two bugs on the way, and the interesting thing is that neither was caught by
the thing built to catch bugs.

**The pager silently did nothing.** `FullAttentionSpec.merge` rebuilds the
group's spec by naming every field it knows about, so a subclass's own fields
fall back to their defaults — `budget_blocks=0`, policy `full`. Nothing raised;
the run simply paged nothing and looked like a baseline. It was found by a
*transfer counter reading zero*, which is worth noting: the guard checks that
what happens is legal, and a pager that does nothing is perfectly legal. The
counters are the instrument for "did anything happen at all", and a paging
result with no transfer counts attached is not evidence.

**The new key was landing inside a restored block.** Restored blocks reach the
worker through the append channel, so its row grows faster than the context
does and index `p // block_size` stops meaning position `p` — but
`compute_slot_mappings` indexes that row positionally. The guard's
`write_target` check reported it immediately and precisely, which is exactly
the case it was written for: a write going somewhere plausible and wrong. The
fix is to copy the manager's mapping over the worker's row before the stock
`prepare_attn` runs, so the slot mapping is right by construction instead of
compensated for downstream.

**The two sides run on different clocks, and must.** The manager frees against
`total_computed_tokens - num_in_flight_tokens`, because an in-flight step is
still reading blocks above that; the view has to describe where *this* step's
key is written. Taking the manager's resident set wholesale puts the tail a
block behind whenever they differ. So the policy's choice of full blocks comes
from the manager — that is what it froze its freeing on — and everything from
the manager's tail forward is kept unconditionally, those indices being
unfreeable by construction. The mismatch is expected and counted rather than
suppressed.

### Does a paged request still answer the question?

`tools/kv_pager_quality.py`. A magic number at a known token offset, so the
block holding it is known and an oracle can be told about it. Llama-3.2-1B,
2048 tokens of context, **budget 16 of 129 blocks — 12.4% resident**:

| arm | needle | output |
|---|---|---|
| `off` — no pager | **found** | the reference |
| `full` — pager, evicting nothing | **found** | bit-identical to `off` |
| `recency` — sinks + newest | **lost** | diverges at step 2 |
| `oracle` — recency + the needle's block | **found** | **tokens identical to the reference** |

That is the result the phasing was built to get to. At 12% residency the
machinery reproduces the full-context answer *token for token* when the right
blocks are kept, and loses it entirely when they are not — so **the mechanism
is not the limit, the policy is**. A quality number from here on is
attributable, which is the whole reason the control arm exists.

`recency` losing it is not a defect, it is what recency is: its window only
slides forward, it never asks for anything back, and a needle behind the window
is gone. That is StreamingLLM, and it is the baseline a scoring policy has to
beat.

Two limits found while building this, both worth more than the table.

**Paging is decode-only**, and the reason is not the one stated when that
restriction went in. The first version of this paragraph said a shortened
context breaks the causal mask. It does not: `flash_attn_varlen_func`
documents that "the causal mask is aligned to the bottom right corner", so
query `i` of a chunk sees keys `<= i + seqlen_k - seqlen_q`. Drop `D` keys from
the prefix and both sides move by `D`, which is the correct answer — and it has
to be, or chunked prefill would not work for any model. Compaction is
mask-safe.

What actually differs, in order of how much work each is:

1. **The tail is a run of blocks, not one.** A decode step writes one key into
   one partial block, and the view keeps that block last. A prefill chunk
   writes `N` keys spanning several blocks, so the invariant generalises to
   "the chunk's whole write span is resident, contiguous, in order, and last".
   The code hard-codes the single-block form. Mechanical, but it is not a
   one-line change and it is exactly where an off-by-one would be invisible.
2. **Block order becomes load-bearing.** This is the sharp one, because it
   retracts a property the whole design leans on. At decode the visible bound
   is `seqlen_k - 1` — everything — so order carries no meaning, which is what
   `blocktable_permute.py` measured. Under bottom-right alignment with `N > 1`
   the bound is *index-dependent*, so permuting retained blocks moves keys
   across it: some that should be visible get masked and vice versa. **The
   permutation result does not transfer to prefill.** The current
   implementation happens to preserve ascending order — restores are placed
   in-place, the view lists indices ascending — so this is not a live bug, but
   any policy that exploits order-freedom would be correct at decode and wrong
   at prefill.
3. **Prefix-cache hashing is live during prefill.** Blocks are hashed and
   registered as they fill, so evicting one means another request can hit that
   hash and be handed a block whose contents were freed. Decode-time paging
   with prefix caching off never meets this, and it is the genuinely new
   correctness argument rather than a generalisation of an old one.
4. **Nothing is measured.** Every result here — permutation invariance,
   eviction, restore, bit-exactness — was taken with a query length of one.

The cost of the restriction is real: peak residency includes the whole prompt,
so this bounds the *decode* footprint and not the prefill peak, which for long
prompts is where the binding constraint actually lives.

**Transport and the view are not the same schedule.** They were tied together
at first — both applied only on decode steps — and that let the manager evict
during a prefill chunk with nothing copying those blocks out, so they were
freed with their only copy still on the GPU. The guard did not catch it,
because the guard inspects decode rows too. It surfaced as one restore that
found nothing in the host tier, which is why `missing_host_copy` is counted
rather than assumed to be zero. Transport now follows the manager's decisions
wherever they are made; only the view is decode-only.

The round trip is exercised here (`oracle_late` evicts the needle's block and
asks for it back, and every restored block was backed by a host copy) but
cannot be *shown to change an answer*: a needle answer arrives in the first
token or two and the restore lands a few steps later. The end-to-end proof that
a restored block still means what it meant stays with `kv_roundtrip.py`, where
a destroyed and restored block gives bit-identical output through the model.

Two things worth keeping from building it.

**The in-tree range helper cannot be reused.** `_remove_blocks_in_range`
iterates backward and breaks at the first block already nulled — correct for a
sliding window, whose freed region only ever grows from the front, and wrong
for a pager, whose freed set is whatever the policy stopped choosing and can
gain holes anywhere. A range spanning an earlier eviction frees only the part
above the topmost hole, and reports success.

**A test that names a failure it cannot produce is worse than no test.** The
first version of the test above used `recency`, whose dropped set is one
contiguous middle — so the range helper works on it, and the test passed
against a deliberately broken manager while claiming to prove the decision not
to use it. It only became real once it used `stress`, whose roaming window
leaves holes scattered through the region it is not keeping, plus an assertion
that a newly dropped block actually sits below an existing hole so a backward
scan would meet one. Both mutations — the range helper, and forgetting
`free_blocks` entirely — are now caught by it.

### Plugin shape: what rests on a real seam, and what does not

Audited 2026-09-05, before moving this to its own repo, because the answer
decides whether the move is a rename or a redesign. It is mostly a rename.

| piece | seam | status |
|---|---|---|
| spec + manager | `KVCacheSpecRegistry.register` | supported, and documented for out-of-tree specs |
| the view | `register_backend(AttentionBackendEnum.X)`, overriding a backend with a subclass whose metadata builder applies it | supported, with an "override an existing attention backend" example in the registry |
| startup | `vllm.general_plugins` entry point | supported |
| **choosing the paged spec per layer** | `Attention.get_kv_cache_spec` | **patched.** `customize_spec` is the intended hook and exists, but on the full-attention path its result is only used to compute a page size — the layer then builds a plain `FullAttentionSpec` and returns that. Its own docstring calls it "a temporary compatibility API" and says "the end state is for the backend to build and return the spec directly, at which point this hook goes away" (vllm#42449). So this becomes supported upstream; until then it is one patched method |
| **worker row sync** | none | **patched.** `prepare_attn` is wrapped to copy the manager's mapping over the worker's row before `compute_slot_mappings` reads it positionally. Needed because restored blocks reach the worker through the append channel, so its row outgrows the context |

Two patches, one of which upstream is actively removing. The second is the one
to think about: it exists because the scheduler→worker protocol can only
*append* block ids, and a pager needs to say "this block now lives at index
`i`". That is the same protocol gap the design flagged as arriving when a
policy goes query-aware, showing up a phase early.

### What "done" looks like

One measurement, and it is the appliance's whole claim: **serve a context longer
than GPU KV capacity allows**, with output quality measured against the same
model served conventionally at a context that does fit. Everything else —
thrash curves, fetch rates, policy comparisons — is instrumentation around that.

## Reproducing

    tools/blocktable_permute.py run MODEL OUT.json [--kv fp8] [--graphs]
                                    [--ctx N] [--tokens N] [--all-groups]
    tools/blocktable_permute.py report OUT.json [OUT.json ...]

    tools/blocktable_evict.py   run MODEL OUT.json [--budget N] [--sink N]
                                    [--needle] [--needle-block N] [--diagnose]
    tools/blocktable_evict.py   report OUT.json [OUT.json ...]

    tools/kv_guard_selftest.py  MODEL [--ctx N] [--reqs N] [--worker-local]
    tools/kv_roundtrip.py       run MODEL OUT.json [--block N] [--kv fp8]
    tools/kv_transport.py       sweep [--layers N] [--block-bytes N]
    tools/kv_transport.py       verify

The engine is forced in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) because the
hook has to run in the same interpreter as the model runner, and prefix caching
is disabled so that a permuted block can never be handed to another request.
Prompts are wikitext-103, the same source `niah_kv.py` draws on, cut to lengths
that are deliberately not multiples of any plausible block size -- a request
whose prompt ends on a block boundary has no partial tail, which is the one case
the negative control cannot act on.

The runs above are in [data/kv-pager/](data/kv-pager/); all were taken on the
RTX 5070 Ti (sm120) box.

→ [TODO.md](../TODO.md) `kv-pager`, [turboquant-kv.md](turboquant-kv.md),
[triattention.md](triattention.md)
