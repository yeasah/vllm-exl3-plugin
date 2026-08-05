"""Read a single module's tensors out of a remote safetensors checkpoint.

EXL3's interesting format variations (codebooks, bit widths, head quantization)
mostly live in repos far too large to download for a test -- gemma-4-12B at
3bpw is ~5 GB on disk and ~24 GB once Phase 0 dequantizes it. But validating the
*format* only needs one layer's tensors, which is a few MB.

safetensors makes that easy: the file is a JSON header giving every tensor's
byte range, followed by a flat data blob. HF serves both over HTTP range
requests, so pulling one q_proj out of a 5 GB shard costs one small GET.
"""

from __future__ import annotations

import json
import struct
import urllib.request

_DTYPES = {
    "F64": ("float64", 8),
    "F32": ("float32", 4),
    "F16": ("float16", 2),
    "BF16": ("bfloat16", 2),
    "I64": ("int64", 8),
    "I32": ("int32", 4),
    "I16": ("int16", 2),
    "I8": ("int8", 1),
    "U8": ("uint8", 1),
    "BOOL": ("bool", 1),
}

_HF = "https://huggingface.co"


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _resolve(repo: str, revision: str, filename: str) -> str:
    return f"{_HF}/{repo}/resolve/{revision}/{filename}"


def _shard_for(repo: str, revision: str, tensor_name: str) -> str:
    """Which safetensors file holds `tensor_name`."""
    try:
        index = json.loads(
            _get(_resolve(repo, revision, "model.safetensors.index.json"))
        )
    except Exception:
        return "model.safetensors"
    return index["weight_map"][tensor_name]


def _header(url: str) -> tuple[dict, int]:
    """The safetensors header dict and the byte offset where data begins."""
    n = struct.unpack("<Q", _get(url, 0, 7))[0]
    header = json.loads(_get(url, 8, 8 + n - 1))
    return header, 8 + n


def fetch_module_tensors(repo: str, revision: str, key: str, device="cuda:0") -> dict:
    """Every tensor named `<key>.<suffix>`, keyed by suffix.

    Returns torch tensors on `device`. Raises KeyError if the module has no
    trellis (i.e. is not EXL3-quantized in this checkpoint).
    """
    import torch

    shard = _shard_for(repo, revision, f"{key}.trellis")
    url = _resolve(repo, revision, shard)
    header, data_start = _header(url)

    prefix = key + "."
    out = {}
    for name, meta in header.items():
        if name == "__metadata__" or not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if "." in suffix:  # a deeper module that merely shares our prefix
            continue
        torch_dtype, itemsize = _DTYPES[meta["dtype"]]
        begin, end = meta["data_offsets"]
        raw = _get(url, data_start + begin, data_start + end - 1)
        expected = end - begin
        if len(raw) != expected:
            raise OSError(
                f"short read for {name}: got {len(raw)} bytes, wanted {expected}"
            )
        t = torch.frombuffer(bytearray(raw), dtype=getattr(torch, torch_dtype))
        # A 0-dim scalar (EXL3 stores mcg/mul1 that way) reshapes to ().
        out[suffix] = t.reshape(meta["shape"]).to(device)
    if "trellis" not in out:
        raise KeyError(f"{key} has no trellis in {repo}@{revision}")
    return out
