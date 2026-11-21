"""Routed experts read from the checkpoint on use: the disk tier below the expert cache's host rows.

docs/expert-cache.md, "A disk tier below host RAM" and its design note. The checkpoint's shards are the store: EXL3
MoE checkpoints keep each expert's projections as separate tensors, and one expert's three trellises (the only part
offloaded; its scales stay on the device) sit within ~1.24 MB of each other, separated only by their scales. So an
expert is one read: the aligned span covering its three trellises, O_DIRECT so the page cache neither helps nor costs
(whoever needs a disk tier has no RAM to spare for one), into a page-aligned bounce buffer, the trellises then copied
out to their destination (a host row, or the staging window's pinned source).

Offsets come from the shard headers, so nothing is read to index. A filesystem without O_DIRECT (tmpfs) is read
buffered instead.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import threading

import numpy
import torch

from .log import init_logger

logger = init_logger(__name__)

_LABELS = ("gate", "up", "down")
_ALIGN = 4096
_DTYPES = {"I16": numpy.int16, "F16": numpy.float16, "BF16": numpy.uint16, "F32": numpy.float32, "I32": numpy.int32}


class ExpertStore:
    """Read a routed expert's trellises straight from the checkpoint shards.

    `index[(prefix, expert)]` is (file, span start, span end, {label: (offset in span, nbytes, dtype, shape)}), for the
    trellis of each projection; `prefix` is the experts module's checkpoint key (`...layers.3.mlp.experts`). Reads are
    safe from several threads at once: each thread has its own bounce buffer, so prefill can keep a deep queue."""

    def __init__(self, root: str):
        self.root = root
        index_path = os.path.join(root, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path) as f:
                files = sorted(set(json.load(f)["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        self.index = {}
        self._fds = {}
        self._direct = {}
        spans = {}
        for name in files:
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n))
            base = 8 + n
            for key, info in header.items():
                if key == "__metadata__" or not key.endswith("_proj.trellis") or ".experts." not in key:
                    continue
                stem, _, rest = key.rpartition(".experts.")
                expert, _, proj = rest.partition(".")
                label = proj.split("_proj")[0]
                if not expert.isdigit() or label not in _LABELS:
                    continue
                a, b = info["data_offsets"]
                spans.setdefault((stem + ".experts", int(expert)), (name, {}))[1][label] = (
                    base + a, b - a, info["dtype"], tuple(info["shape"]))
        width = 0
        for k, (name, parts) in spans.items():
            if set(parts) != set(_LABELS):
                continue
            start = min(off for off, _, _, _ in parts.values())
            end = max(off + nb for off, nb, _, _ in parts.values())
            self.index[k] = (name, start, end, {lab: (off - start, nb, dt, sh) for lab, (off, nb, dt, sh) in parts.items()})
            width = max(width, end - start)
        # one aligned read covers [floor(start), ceil(end)), which can reach one page past the span on each side
        self._bounce_bytes = -(-(width + 2 * _ALIGN) // _ALIGN) * _ALIGN
        self._local = threading.local()
        self._lock = threading.Lock()
        self.reads = 0
        self.bytes = 0

    def __contains__(self, key) -> bool:
        return key in self.index

    def _fd(self, name: str) -> tuple[int, bool]:
        with self._lock:
            if name not in self._fds:
                path = os.path.join(self.root, name)
                try:
                    self._fds[name], self._direct[name] = os.open(path, os.O_RDONLY | os.O_DIRECT), True
                except OSError:
                    self._fds[name], self._direct[name] = os.open(path, os.O_RDONLY), False
                    logger.warning("expert store: %s does not take O_DIRECT; reading it buffered", path)
            return self._fds[name], self._direct[name]

    def _bounce(self) -> mmap.mmap:
        buf = getattr(self._local, "buf", None)
        if buf is None:
            buf = self._local.buf = mmap.mmap(-1, self._bounce_bytes)   # anonymous: page-aligned, as O_DIRECT needs
        return buf

    def read(self, prefix: str, expert: int, out: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """The expert's trellises: copied into `out`'s tensors (label -> a CPU tensor of the stored shape and dtype,
        typically a pinned host row or staging source), or into new CPU tensors."""
        name, start, end, parts = self.index[(prefix, expert)]
        fd, direct = self._fd(name)
        lo = start // _ALIGN * _ALIGN
        hi = -(-end // _ALIGN) * _ALIGN
        buf = self._bounce()
        view = memoryview(buf)[: hi - lo]
        got = os.preadv(fd, [view], lo)          # the same aligned read either way; only the descriptor differs
        if got < end - lo:
            # the shard's end: O_DIRECT returns what exists past an unaligned file size, which must still cover the span
            raise OSError(f"expert store: short read of {prefix}.{expert} from {name} ({got} of {end - lo} bytes)")
        raw = numpy.frombuffer(buf, dtype=numpy.uint8, count=hi - lo)
        result = {}
        for label, (off, nb, dt, shape) in parts.items():
            src = torch.from_numpy(raw[start - lo + off : start - lo + off + nb].view(_DTYPES[dt]).reshape(shape))
            if dt == "BF16":
                src = src.view(torch.bfloat16)
            dst = out.get(label) if out else None
            if dst is None:
                dst = torch.empty(shape, dtype=src.dtype)
            dst.copy_(src)
            result[label] = dst
        with self._lock:
            self.reads += 1
            self.bytes += hi - lo
        return result

    def close(self):
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()
