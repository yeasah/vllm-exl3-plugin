# Block order does not matter: the gate on score-driven KV residency

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

1. **Eviction is not permutation.** A pager that drops blocks compacts the row,
   and then the token at position `p` no longer lives at row index
   `p // block_size` -- which is the assumption the slot-mapping kernel is
   built on, so the new token would be written to the wrong place. Compaction
   therefore needs its own slot-mapping path, and that is the first real design
   decision of the pager rather than something this test cleared.
2. **Prefix caching was off for every run.** Its bookkeeping hashes a block by
   the token prefix leading to it, so a residency scheme that moves or drops
   blocks has to be reconciled with the cached-block registry. Untouched here.
3. **CUDA graphs**, per above.
4. Single GPU only; no TP, no MLA backend, no mamba/hybrid state groups beyond
   gemma's sliding layers; decode steps only, never a prefill chunk.

## Reproducing

    tools/blocktable_permute.py run MODEL OUT.json [--kv fp8] [--graphs]
                                    [--ctx N] [--tokens N] [--all-groups]
    tools/blocktable_permute.py report OUT.json [OUT.json ...]

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
