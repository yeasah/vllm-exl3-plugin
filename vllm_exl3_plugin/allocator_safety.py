"""Refuse to serve where expandable segments can corrupt memory.

PyTorch #196258: with expandable segments, `empty_cache()` and the allocator's release-and-retry unmap freed pages
on every device but synchronize only the current one, so a kernel still queued on another GPU writes into memory
that is gone (Xid 31) or has been remapped to something else (silent corruption). Open through torch 2.14.0;
`tools/torch-c10cuda-fix.sh` rebuilds libc10_cuda.so with the fix (pytorch/pytorch@3fda599) and leaves a stamp.

It needs a process holding torch memory on two or more devices. vLLM's workers are one device each -- measured on
a TP=2 MoE, with and without expandable segments, after load and at exit: no reservation on the peer device -- so
under vLLM this is a check on an invariant, but a configuration that breaks it is refused rather than served:
memory corruption is the worst way to fail. Vital in the sense of docs/vllm-patches.md, though not a patch.
"""

from __future__ import annotations

import hashlib
import os

import torch

#: The fix tools/torch-c10cuda-fix.sh applies, which names its stamp
FIX = "3fda5993c2572eca77dd53033f057dc0290f287f"


def c10_fix_present() -> bool:
    """Whether torch's libc10_cuda.so is the patched build: its stamp names this fix and holds the library's sha256."""
    lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libc10_cuda.so")
    stamp = f"{lib}.fix-{FIX}"
    try:
        with open(stamp) as f:
            want = f.read().split()[0]
        h = hashlib.sha256()
        with open(lib, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest() == want
    except (OSError, IndexError):
        return False


def expandable_segments_on() -> bool:
    """The live allocator setting where vLLM can read it, else the environment."""
    try:
        from vllm.device_allocator.alloc_conf import expandable_segments_enabled, expandable_segments_enabled_from_env
    except ImportError:
        conf = ",".join(os.environ.get(k, "") for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"))
        return "expandable_segments:true" in conf.replace(" ", "").lower()
    live = expandable_segments_enabled()
    return live if live is not None else expandable_segments_enabled_from_env()


def devices_with_memory() -> list[int]:
    """Devices this process's caching allocator holds memory on (reading the stats creates no context)."""
    if not torch.cuda.is_available():
        return []
    return [d for d in range(torch.cuda.device_count()) if torch.cuda.memory_reserved(d) > 0]


def check(where: str) -> None:
    """Raise if this process runs expandable segments on unpatched torch across more than one device."""
    if not expandable_segments_on():
        return
    devices = devices_with_memory()
    if len(devices) < 2 or c10_fix_present():
        return
    raise RuntimeError(
        f"Refusing to serve ({where}): expandable segments are on and this process holds torch memory on devices "
        f"{devices}, with torch {torch.__version__}'s unpatched libc10_cuda.so. PyTorch #196258 then lets a kernel "
        f"queued on one GPU write into memory freed and unmapped on another -- a crash at best, silent corruption at "
        f"worst. Either apply the fix (tools/torch-c10cuda-fix.sh, or a torch that includes pytorch/pytorch@3fda599), "
        f"or turn expandable segments off (remove expandable_segments from PYTORCH_CUDA_ALLOC_CONF / "
        f"PYTORCH_ALLOC_CONF)."
    )
