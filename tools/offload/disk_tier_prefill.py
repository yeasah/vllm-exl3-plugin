"""Disk tier timing: prefill and decode with part of each layer's experts read from the checkpoint on use.

usage: disk_tier_prefill.py MODEL [--share 0.5] [--slots 0.2] [--chunk 2048] [--chunks 4] [--decode 64]

One engine (eager, EXL3_DISK_TIER=SHARE, EXL3_EXPERT_SLOTS=SLOTS, no verification). A warm-up request first, then a
prefill of CHUNKS full chunks of real text (max_tokens=1) and, separately, a short prompt decoding DECODE tokens.
Reports wall time and where the disk-tier time went (the reads, and waits for the previous layer's copies out of
the pinned sources) against bytes read, for comparison with tier_replay.py's prediction at the same split. Set
SHARE to the fraction of experts in host rows *before* the slots take theirs: VRAM 20% / RAM 30% / disk 50% is
--slots 0.2 --share 0.5.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--share", default="0.5")
ap.add_argument("--slots", default="0.2")
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--chunks", type=int, default=4)
ap.add_argument("--decode", type=int, default=64)
ap.add_argument("--gpu-util", type=float, default=0.92)
ap.add_argument("--offload-gb", type=float, default=64)
a = ap.parse_args()

os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", EXL3_DISK_TIER=a.share, EXL3_EXPERT_SLOTS=a.slots)
os.environ.pop("EXL3_CACHE_VERIFY", None)

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

from vllm_exl3_plugin import cpu_offload  # noqa: E402

llm = LLM(a.model, enforce_eager=True, max_num_batched_tokens=a.chunk, max_model_len=a.chunk * a.chunks + 512,
          cpu_offload_gb=a.offload_gb, gpu_memory_utilization=a.gpu_util, seed=0, max_num_seqs=4,
          enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
docs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")
text = "\n\n".join(open(os.path.join(docs, n)).read() for n in sorted(os.listdir(docs)) if n.endswith(".md"))
ids = llm.get_tokenizer()(text)["input_ids"]
need = a.chunk * a.chunks + 600
ids = (ids * (need // len(ids) + 1))[:need]


def snap():
    store = cpu_offload._DISK["store"]
    return dict(reads=store.reads, bytes=store.bytes, read_s=cpu_offload._STATS.get("disk_read_s", 0.0),
                wait_s=cpu_offload._STATS.get("disk_wait_s", 0.0))


def run(name, prompt, sp):
    before, t = snap(), time.perf_counter()
    out = llm.generate([TokensPrompt(prompt_token_ids=prompt)], sp, use_tqdm=False)[0]
    took = time.perf_counter() - t
    after = snap()
    d = {k: after[k] - before[k] for k in after}
    gib = d["bytes"] / 2**30
    rate = d["bytes"] / d["read_s"] / 1e9 if d["read_s"] else 0
    print(f"{name}: {took:7.2f} s wall; disk {d['reads']} reads, {gib:.2f} GiB in {d['read_s']:.2f} s "
          f"({rate:.2f} GB/s), {d['wait_s']:.2f} s waiting on copies", flush=True)
    return out, took, d


run("warm-up", ids[:300], SamplingParams(temperature=0, max_tokens=4))
n = a.chunk * a.chunks
_, took, d = run(f"prefill {n} tokens", ids[600:600 + n], SamplingParams(temperature=0, max_tokens=1))
print(f"  per {a.chunk}-token chunk: {took / a.chunks:.2f} s wall, {d['bytes'] / a.chunks / 2**30:.2f} GiB and "
      f"{d['read_s'] / a.chunks:.2f} s from disk; {n / took:.0f} tok/s")
out, took, d = run(f"decode {a.decode} tokens", ids[:200], SamplingParams(temperature=0, max_tokens=a.decode,
                                                                           ignore_eos=True))
toks = len(out.outputs[0].token_ids)
print(f"  per token: {1e3 * took / toks:.1f} ms wall (incl. a 200-token prefill), {d['reads'] / toks:.1f} disk reads, "
      f"{1e3 * d['read_s'] / toks:.1f} ms reading")
