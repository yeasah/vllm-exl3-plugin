"""TurboQuant's chunked continuation prefill must match the monolithic one.

`_continuation_prefill` attends a continuation chunk to the whole cached
context. Doing that in one piece makes prefill VRAM scale with context; the
chunked path slabs the cached prefix and merges the partial attentions by
log-sum-exp, which makes it scale with a byte budget instead
(`tq_prefill_workspace_mib`).

**Chunked-vs-unchunked prefill is not the equivalence to test**, which is why
this compares the two *implementations* instead. Measured on this model at 900
tokens: chunking the prefill at all moves logprobs by 4.4e-01 with a
turboquant cache reading exact K/V one way and the quantized cache the other,
and by 3.5e+00 with turboquant on top. Both are expected and neither is a bug,
so a test built on that comparison would need a 3.5-nat tolerance and would
catch nothing.

Both implementations here see byte-identical inputs -- same schedule, same
quantized cache, same chunk boundaries -- so the only difference is how the
attention is decomposed, and the bound is a couple of bf16 quanta.

Running both per call is deliberate rather than comparing two engines: it
removes the 20 layers of amplification between a wrong decomposition and the
logits, so a failure points at the call that caused it.

What this catches, verified by breaking each: a non-causal suffix chunk (213
quanta) and the rotated keys reshaped on the wrong axis (427). What it does
*not* catch is the accumulator dropping to bf16 -- six merges is too few for
the per-merge rounding to show -- and that is pinned instead by
deps/vllm/tests/kernels/turboquant/test_continuation_prefill_split.py, which
runs the decomposition on bare tensors at enough slabs to expose it.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    import torch

    HAVE_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    HAVE_CUDA = False

try:
    import exllamav3_ext  # noqa: F401

    HAVE_EXT = True
except ImportError:  # pragma: no cover
    HAVE_EXT = False

try:
    import vllm  # noqa: F401

    HAVE_VLLM = True
except ImportError:  # pragma: no cover
    HAVE_VLLM = False

MODEL = "turboderp/MiniCPM5-1B-exl3"
REVISION = "3.00bpw"

#: Small enough that a 900-token prompt needs several continuation chunks.
MAX_BATCHED = 256
#: Forced small so the slab is smaller than cached_len and the chunked path is
#: actually taken -- the default budget would keep this model monolithic.
SLAB = 128
#: bf16 has 8 significand bits, so one quantum at magnitude m is m * 2**-8.
#: The decomposition is exact in real arithmetic; what is left is the output
#: rounding, and the fp32 accumulator keeps it from growing with slab count.
MAX_ULP = 4.0


@unittest.skipUnless(
    HAVE_CUDA and HAVE_EXT and HAVE_VLLM, "needs CUDA, exllamav3 and vLLM"
)
class TestChunkedContinuationPrefill(unittest.TestCase):
    def test_chunked_matches_monolithic_per_call(self):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.v1.attention.backends import turboquant_attn as tqa

        impl = tqa.TurboQuantAttentionImpl
        monolithic = impl._continuation_prefill_monolithic
        chunked = impl._continuation_prefill_chunked
        observed: list[tuple[int, float, float]] = []

        def both(self, *args, **kwargs):
            reference = monolithic(self, *args)
            got = chunked(self, *args, slab=SLAB)
            delta = (got.float() - reference.float()).abs().max().item()
            ulp = reference.float().abs().max().item() * 2.0**-8
            observed.append((int(args[6]), delta, ulp))
            # Stay on the reference so later layers see identical inputs and a
            # divergence cannot be inherited from an earlier one.
            return reference

        impl._continuation_prefill = both
        try:
            tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
            filler = (
                "Quantization trades numerical precision for memory bandwidth. "
                "The trellis code spends bits unevenly across a block. "
            )
            ids = tok(filler * 60)["input_ids"][:900]

            llm = LLM(
                model=MODEL,
                revision=REVISION,
                kv_cache_dtype="turboquant_4bit_nc",
                max_model_len=4096,
                gpu_memory_utilization=0.85,
                enforce_eager=True,
                max_num_batched_tokens=MAX_BATCHED,
                max_num_seqs=1,
                enable_prefix_caching=False,
            )
            try:
                llm.generate(
                    {"prompt_token_ids": ids},
                    SamplingParams(temperature=0.0, max_tokens=1),
                )
            finally:
                del llm
        finally:
            del impl._continuation_prefill

        # A pass with no continuation chunks would be vacuous, and is easy to
        # cause by accident -- raising max_num_batched_tokens above the prompt
        # length is enough.
        self.assertGreater(len(observed), 0, "no continuation prefill was reached")
        cached_lens = sorted({c for c, _, _ in observed})
        self.assertTrue(
            any(c > SLAB for c in cached_lens),
            f"no call exceeded the slab, so nothing was chunked: {cached_lens}",
        )

        worst = max((d / u if u else 0.0, c) for c, d, u in observed)
        self.assertLessEqual(
            worst[0],
            MAX_ULP,
            f"chunked output differs by {worst[0]:.1f} bf16 quanta at "
            f"cached_len={worst[1]} (limit {MAX_ULP})",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
