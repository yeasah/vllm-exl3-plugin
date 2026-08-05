"""End-to-end: an EXL3 checkpoint generating tokens through vLLM.

This is the Phase 0 acceptance test. It is slow (~90s of engine startup per
revision) and needs a CUDA device, exllamav3's extension, and vLLM, so it is
skipped everywhere else.

Both revisions matter and for different reasons:

- 3.0bpw is uniform K=3, the simplest possible target.
- 3.5bpw mixes bit widths *inside* a layer (q=4, k=5, v=5), so it is the one
  that actually exercises the decision to keep merged-linear shards separate.
  If shards were being concatenated or assumed to share a bit width, this is
  where it would break.
"""

from __future__ import annotations

import os
import unittest

# Must be set before vLLM starts its engine core. Probing for a CUDA device
# below initializes CUDA in this process, and vLLM's default fork-based launch
# then dies with "Cannot re-initialize CUDA in forked subprocess". The standalone
# script form of this test does not hit it only because it never touches CUDA
# before constructing the engine.
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

MODEL = os.environ.get("EXL3_TEST_MODEL", "turboderp/Llama-3.2-1B-Instruct-exl3")


@unittest.skipUnless(
    HAVE_CUDA and HAVE_EXT and HAVE_VLLM, "needs CUDA, exllamav3 and vLLM"
)
class TestEndToEnd(unittest.TestCase):
    def _generate(self, revision: str):
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=MODEL,
            revision=revision,
            # exllamav3's kernels are fp16; EXL3 repos usually declare bfloat16.
            dtype="float16",
            enforce_eager=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            max_model_len=2048,
        )
        try:
            prompts = ["The capital of France is", "Q: What is 2 + 2?\nA:"]
            outs = llm.generate(
                prompts, SamplingParams(temperature=0.0, max_tokens=24)
            )
            return [o.outputs[0].text for o in outs]
        finally:
            del llm

    def _assert_coherent(self, texts):
        capital, arithmetic = texts
        # Weak assertions on purpose: this test is here to catch "loads but
        # emits garbage", which is the failure mode that survives a clean
        # startup. Layer-level exactness is pinned in test_kernels.py.
        self.assertIn("Paris", capital)
        self.assertIn("4", arithmetic)

    def test_uniform_bit_width(self):
        self._assert_coherent(self._generate("3.0bpw"))

    def test_mixed_bit_width(self):
        """3.5bpw has q=4, k=5, v=5 bits in the same merged QKV linear."""
        self._assert_coherent(self._generate("3.5bpw"))


if __name__ == "__main__":
    unittest.main()
