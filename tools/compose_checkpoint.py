#!/usr/bin/env python3
"""Build one EXL3 checkpoint out of the tensors of two or more others.

    tools/compose_checkpoint.py BASE OUT --from DONOR --take head

Start from `BASE`, then replace whole modules with `DONOR`'s copies of the same
modules. The motivating cases are the ones where a bit allocation is decided per
*category* rather than per tensor, and where redoing the quantization to change
one category costs hours of GPU time for tensors that were already right:

  - **Head bitrate.** `turboderp/Qwen3-0.6B-exl3` publishes `2.75bpw_H5` and
    `3.0bpw`, whose heads are K=5 and K=6. Composing them gives the two head
    arms over one identical body, which is the comparison the sweep in
    docs/qbench.md wants and the only way to get it without converting.
  - **Vision and MTP towers**, which exllamav3 already treats as separate knobs
    (`vision_bits`, `mtp_bits`) and which no body measurement covers.
  - **Non-routed tensors in an MoE model.** Upstream can boost everything that
    is not a routed expert; `turboderp/Laguna-XS-2.1-exl3@3.00bpw` ships it
    (routed experts K=3, `shared_expert` K=5). Whether that generalizes is open,
    and `--take all --keep experts` composes the arms to find out.
  - **Repairing a published checkpoint** whose head, vision tower or one broken
    tensor is wrong, without discarding the rest of a very expensive conversion.

**Modules move whole, never individual tensors.** A quantized linear is a
trellis plus `suh`, `svh`, an optional bias and an optional codebook selector,
and those are one indivisible unit: a trellis paired with the other checkpoint's
scales is not a worse model, it is noise, and nothing downstream can detect it.
So the selector language names modules, and every tensor of a selected module
travels together even where the two checkpoints store different suffix sets.

**What this does not do.** `quantization_config.json` is carried over rather
than recomputed -- per-module `tensor_storage` entries follow their modules, and
`head_bits` is corrected from the trellis actually written, but the top-level
`bits` keeps `BASE`'s value and no longer describes the file. That is safe for
loading (the plugin reads the safetensors index as ground truth and consults
`tensor_storage` only for module *presence*, which composition preserves) and is
recorded in the output's `composition` block rather than left to be discovered.
Checkpoints with several shards and no index are not handled; no publisher emits
that shape (see docs, and `tools/checkpoint_survey.py`).

**Restoring dense weights.** `--restore SEL` is the other direction: replace a
*quantized* base module with the **dense original** from the model the checkpoint
was converted from. The case it exists for is putting a bf16 vision tower back
into a checkpoint that quantized one -- `turboderp/Muse-Glimmer-30B-exl3` and
`turboderp/DeepSeek-V4-Flash-Vision-Exp-exl3` are the two families that do,
11 revisions between them, while 111 of the other 122 vision revisions on the Hub
already ship a dense tower. See docs/media-encoders.md.

That is a format change rather than a bit-width change, so it gets its own rule
and its own checks: the module's whole EXL3 storage is dropped (a surviving
`suh` would keep it reading as quantized), shapes are checked through
exllamav3's pad-to-128 rule rather than for equality, `tensor_storage` is
rewritten as dense and `vision_bits` is dropped once nothing vision-shaped is
quantized any more.

**`--verify` is what makes a restore publish-grade.** Matching names and shapes
do not establish that the dense tensor is the one the trellis was made from --
a different fine-tune, a transposition, or a scale convention baked into the
weights and recorded nowhere (docs/format-and-loading.md) all pass every
structural check and produce a checkpoint that loads. So a sample of the
replaced modules is dequantized and compared: a genuine restore measures ~0.17
relative error at K=3, a wrong tensor ~1.41, and the threshold sits at 0.6
between them. It needs a GPU and says so loudly when there is none.

Selectors, applied in the order given, later rules winning:

    --take SEL     take these modules from the current --from donor
    --restore SEL  replace these quantized base modules with the donor's dense
                   originals (donor = the unquantized model, not an EXL3 one)
    --keep SEL     revert these modules to BASE

`SEL` is a category name (`head`, `embed`, `vision`, `mtp`, `experts`,
`shared-experts`, `attn`, `mlp`, `body`, `quantized`, `all`), an fnmatch glob
over module keys (`model.layers.3.*`), or `re:` followed by a regex.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob as globmod
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_exl3_plugin import format  # noqa: E402

INDEX_NAME = "model.safetensors.index.json"

#: Suffixes that belong to a module rather than naming one. `format.EXL3_SUFFIXES`
#: covers EXL3 storage; `.weight` and `.bias` cover the dense tensors a checkpoint
#: stores beside it (norms, the embedding, and biases inside a quantized linear).
#: Anything else -- `gemma-4-26B-A4B`'s `router.scale`, whose tensor name *is* its
#: module key -- is its own module, which is why this falls through rather than
#: stripping a trailing component it does not recognize.
_GROUPING_SUFFIXES = (
    format.EXL3_SUFFIXES + format.BLOCKQ_SUFFIXES + (".weight", ".bias")
)


def module_key(name: str) -> str:
    """The module a checkpoint tensor belongs to.

    Reproduces the grouping `quantization_config.json` records in
    `tensor_storage`, but from the tensor names alone, so it also works on the
    modules that file omits -- Muse-Glimmer lists none of its 303 vision modules
    (see `EXL3Config._load_index_modules`), and those are exactly the ones this
    tool exists to move.
    """
    for suffix in _GROUPING_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def read_headers(d: str) -> dict[str, tuple[str, list[int], str]]:
    """Every tensor in a checkpoint: name -> (shard path, shape, dtype).

    Headers only. No weights are read, so planning a composition costs
    milliseconds regardless of model size and can be done on a laptop.
    """
    shards = sorted(globmod.glob(os.path.join(d, "*.safetensors")))
    if not shards:
        raise SystemExit(f"no .safetensors in {d}")
    if len(shards) > 1 and not os.path.exists(os.path.join(d, INDEX_NAME)):
        raise SystemExit(
            f"{d} has {len(shards)} shards and no {INDEX_NAME}. No published "
            "checkpoint uses that layout and this tool will not emit one; "
            "the index is what says which shards belong to the model."
        )
    out: dict[str, tuple[str, list[int], str]] = {}
    for path in shards:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            out[name] = (path, meta["shape"], meta["dtype"])
    return out


def modules_of(tensors) -> dict[str, list[str]]:
    """Group tensor names by owning module, preserving a stable order."""
    groups: dict[str, list[str]] = {}
    for name in tensors:
        groups.setdefault(module_key(name), []).append(name)
    for v in groups.values():
        v.sort()
    return groups


def trellis_bits(mod: str, group: list[str], tensors) -> int | None:
    """K for a quantized module, or None if it stores no trellis."""
    for name in group:
        if name.endswith(".trellis"):
            return format.bits_from_trellis_shape(tensors[name][1])
    return None


# --------------------------------------------------------------------------
# Selectors
# --------------------------------------------------------------------------

#: A routed expert, in every naming this checkpoint collection uses:
#: `layers.N.experts.N.gate_proj` (gemma-4-26B-A4B) and
#: `layers.N.mlp.experts.N.gate_proj` (Qwen3.5-35B-A3B, Laguna-XS-2.1). The
#: numbered component is what distinguishes a routed expert from the shared one.
_ROUTED = re.compile(r"(^|\.)experts\.\d+\.")

_CATEGORY = {
    "head": lambda m: m == "lm_head" or m.endswith(".lm_head"),
    "embed": lambda m: m.endswith("embed_tokens"),
    "vision": lambda m: bool(
        re.search(r"(^|\.)(vision|visual|vision_tower|vision_adapter|"
                  r"vision_projection|mm_projector|multi_modal|audio_tower)", m)
    ),
    "mtp": lambda m: m == "mtp" or m.startswith("mtp.") or ".mtp." in m,
    "experts": lambda m: bool(_ROUTED.search(m)),
    "shared-experts": lambda m: "shared_expert" in m,
    "attn": lambda m: bool(re.search(r"(^|\.)(self_attn|linear_attn|attn)\.", m)),
    "mlp": lambda m: bool(re.search(r"(^|\.)mlp\.", m)),
    "all": lambda m: True,
}


def select(sel: str, modules, quantized: set[str]) -> set[str]:
    """Resolve one selector against a checkpoint's module keys.

    `body` and `quantized` are defined here rather than in `_CATEGORY` because
    they need the set of modules that actually carry a trellis, which is a
    property of the checkpoint and not of the name.
    """
    if sel == "quantized":
        return set(quantized)
    if sel == "body":
        excluded = {"head", "embed", "vision", "mtp"}
        return {
            m for m in quantized
            if not any(_CATEGORY[c](m) for c in excluded)
        }
    if sel in _CATEGORY:
        return {m for m in modules if _CATEGORY[sel](m)}
    if sel.startswith("re:"):
        rx = re.compile(sel[3:])
        return {m for m in modules if rx.search(m)}
    if any(c in sel for c in "*?["):
        return {m for m in modules if fnmatch.fnmatchcase(m, sel)}
    if sel in modules:
        return {sel}
    raise SystemExit(
        f"selector {sel!r} matched no module and is not a known category "
        f"({', '.join(sorted(set(_CATEGORY) | {'body', 'quantized'}))}). "
        "Use a glob or 're:<regex>' for an exact set of modules."
    )


class _Rule(argparse.Action):
    """Record --from/--take/--keep in the order written.

    Ordering is the whole grammar: a `--take` binds to the most recent `--from`,
    and a later rule overrides an earlier one, so `--take all --keep experts`
    reads as what it does. argparse has no ordered multi-option form, so the
    three share one destination.
    """

    def __call__(self, parser, ns, values, option_string=None):
        ns.rules = (getattr(ns, "rules", None) or []) + [
            (option_string.lstrip("-"), values)
        ]


# --------------------------------------------------------------------------
# Compatibility
# --------------------------------------------------------------------------

#: config.json keys that describe the quantization rather than the model, and so
#: are allowed to differ between two checkpoints of the same model.
_CONFIG_IGNORED = {
    "quantization_config", "torch_dtype", "dtype", "transformers_version",
}


def load_config(d: str) -> dict:
    path = os.path.join(d, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _tie_word_embeddings(cfg) -> bool:
    """Whether a config declares tied embeddings, at either nesting level.

    Multimodal configs put it under `text_config`; either level counts. Same
    rule as `tools/quantize_embedding.py:is_tied`.
    """
    for scope in (cfg, cfg.get("text_config") or {}):
        if scope.get("tie_word_embeddings"):
            return True
    return False


def check_same_model(base_dir: str, donor_dir: str, base_cfg, donor_cfg, problems):
    """Refuse two checkpoints that are not the same model at the same shape.

    Composition is only meaningful when the two conversions describe one model,
    and the failure mode if they do not is a checkpoint that loads and is wrong
    -- so this is a hard error, not a warning. Shape agreement is re-checked
    per module below against the tensors themselves, which is the real
    authority; this catches the mistake earlier and says why in model terms.
    """
    if not base_cfg or not donor_cfg:
        problems.append(("warn", "one side has no config.json; "
                                 "cannot confirm the two are the same model"))
        return
    for key in sorted((set(base_cfg) | set(donor_cfg)) - _CONFIG_IGNORED):
        a, b = base_cfg.get(key), donor_cfg.get(key)
        if a != b:
            problems.append((
                "error" if key in ("architectures", "model_type") else "warn",
                f"config.json differs on {key!r}: base {a!r} vs donor {b!r}",
            ))


def check_module(mod, base_group, donor_group, base_t, donor_t, problems):
    """Whether one module can be lifted from the donor into the base.

    Bit width is *expected* to differ -- it is the point -- so the trellis is
    checked on its recovered (in, out) dimensions rather than its stored shape.
    Everything else must match exactly: `suh`/`svh` are per-feature vectors, and
    a length mismatch there means the two conversions disagree about the model,
    not about its bitrate.
    """
    bs = {n[len(mod):]: base_t[n][1] for n in base_group}
    ds = {n[len(mod):]: donor_t[n][1] for n in donor_group}
    for suffix in sorted(set(bs) | set(ds)):
        a, b = bs.get(suffix), ds.get(suffix)
        if suffix == ".trellis" and a and b:
            if format.dims_from_trellis_shape(a) != format.dims_from_trellis_shape(b):
                problems.append(("error", f"{mod}{suffix}: dimensions differ, "
                                          f"{format.dims_from_trellis_shape(a)} vs "
                                          f"{format.dims_from_trellis_shape(b)}"))
            continue
        if a is None or b is None:
            # A suffix on one side only. `.mcg` vs `.mul1` is the common and
            # benign case -- a per-tensor codebook selector, reported once in
            # aggregate below -- but a missing `.bias` changes the module's
            # arithmetic and must be loud.
            if suffix in (".mcg", ".mul1"):
                continue
            problems.append(("error", f"{mod}{suffix}: present in "
                                      f"{'base' if b is None else 'donor'} only"))
        elif a != b:
            problems.append(("error", f"{mod}{suffix}: shape {a} vs {b}"))


def codebook_of(group) -> str:
    for name in group:
        if name.endswith(".mcg"):
            return "mcg"
        if name.endswith(".mul1"):
            return "mul1"
    return "default"


def check_restore_module(mod, base_group, donor_group, base_t, donor_t, problems):
    """Whether a quantized base module can be replaced by the donor's dense one.

    Shapes are checked through exllamav3's padding rule, not for equality: a
    `Linear` pads both dimensions up to a multiple of 128 before quantizing, so
    a trellis legitimately describes a larger matrix than the model's own. The
    dense original is the *unpadded* one, and that is what should be stored --
    padding is an artifact of the trellis layout and nothing outside it wants
    the zero rows.
    """
    b = {n[len(mod):]: base_t[n][1] for n in base_group}
    d = {n[len(mod):]: donor_t[n][1] for n in donor_group}
    if ".trellis" not in b:
        problems.append(("error", f"{mod}: not quantized in the base, "
                                  "so there is nothing to restore"))
        return
    if ".weight" not in d:
        problems.append(("error", f"{mod}: the donor has no dense .weight "
                                  f"(it stores {sorted(d) or 'nothing'})"))
        return
    t_in, t_out = format.dims_from_trellis_shape(b[".trellis"])
    w = d[".weight"]
    if len(w) != 2:
        problems.append(("error", f"{mod}.weight: expected a 2-D matrix, got {w}"))
        return
    w_out, w_in = w
    if format.pad_dim(w_in) != t_in or format.pad_dim(w_out) != t_out:
        problems.append((
            "error",
            f"{mod}: dense weight is {w_out}x{w_in}, which does not pad to the "
            f"trellis's {t_out}x{t_in}. Different tensor, or the converter "
            "fused or split this one.",
        ))
    if ".bias" in b and ".bias" not in d:
        problems.append(("error", f"{mod}: base has a bias, donor does not"))


#: Relative Frobenius error above which a restore is not a dequantization of the
#: tensor it replaces. Genuine EXL3 error is far below this at every bit width
#: the format supports -- measured medians are 0.2942 at K=2, 0.1483 at K=3,
#: 0.0751 at K=4 and 0.0224 at K=6 (docs/qbench.md) -- while two unrelated
#: matrices of similar scale sit near sqrt(2). Nothing lands in between by
#: accident, so the threshold is a wide gap rather than a tuned value.
VERIFY_RFN_MAX = 0.6


def verify_restores(base_dir, base_mods, donors, restore, sample, problems):
    """Dequantize a sample of restored modules and compare against their donors.

    The guard that makes this mode publish-grade. Every other check here is on
    names and shapes, and the failures that matter most survive both: a donor
    that is a *different fine-tune* of the same architecture, a tensor the
    converter transposed, or a scale convention baked into the stored weights
    and recorded nowhere (docs/format-and-loading.md, "The checkpoint is not a
    complete description of itself" -- Laguna's `interm_div` is the precedent).
    All three produce a checkpoint that loads, and all three move the relative
    error from ~0.15 to ~1.4.

    Returns the per-module errors it measured. Needs a GPU, and says so rather
    than silently passing when there is none.
    """
    import torch
    from safetensors import safe_open

    from vllm_exl3_plugin import ops

    if not torch.cuda.is_available():
        problems.append((
            "warn",
            "--verify asked for but no CUDA device: the restored weights were "
            "NOT checked against the trellis they replace. Name and shape "
            "checks alone do not catch a wrong fine-tune or a baked scale "
            "convention. Re-run where there is a GPU before publishing.",
        ))
        return {}

    picks = sorted(restore)[:: max(1, len(restore) // max(sample, 1))][:sample]
    out = {}
    for mod in picks:
        donor_dir = restore[mod]
        d = donors[donor_dir]
        t = {}
        for name in base_mods[mod]:
            suffix = name[len(mod):]
            with safe_open(base_t_path(base_mods, mod, name), framework="pt") as h:
                t[suffix] = h.get_tensor(name)
        with safe_open(d["tensors"][f"{mod}.weight"][0], framework="pt") as h:
            ref = h.get_tensor(f"{mod}.weight")
        bits = format.bits_from_trellis_shape(list(t[".trellis"].shape))
        dense = ops.dense_weight(
            t[".trellis"].cuda(), t[".suh"].cuda(), t[".svh"].cuda(),
            bits, ".mcg" in t, ".mul1" in t,
        )
        o, i = ref.shape
        dense = dense[:o, :i].float()
        ref = ref.cuda().float()
        rfn = (torch.linalg.norm(dense - ref) / torch.linalg.norm(ref)).item()
        out[mod] = (rfn, bits)
        if rfn > VERIFY_RFN_MAX:
            problems.append((
                "error",
                f"{mod}: the donor's dense weight is not what the base's K={bits} "
                f"trellis decodes to (relative error {rfn:.3f}, expected well "
                f"under {VERIFY_RFN_MAX}). Wrong model, wrong revision, or a "
                "scale convention the checkpoint does not record.",
            ))
    return out


#: Filled in by main() so verify_restores can find which shard holds a tensor.
_BASE_TENSORS: dict = {}


def base_t_path(base_mods, mod, name) -> str:
    return _BASE_TENSORS[name][0]


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def raw_header(path):
    """Tensor records straight from a safetensors header: no torch, no data read.

    Returns {name: (dtype, shape, path, abs_start, abs_end)} plus the file's
    `__metadata__`. Offsets are absolute in the file, so a tensor can be copied
    without knowing anything else about it.
    """
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for name, rec in header.items():
        if name == "__metadata__":
            continue
        a, b = rec["data_offsets"]
        out[name] = (rec["dtype"], rec["shape"], path, base + a, base + b)
    return out, header.get("__metadata__")


#: Copy buffer. Large enough that the syscall overhead is irrelevant on a
#: multi-GiB shard, small enough to be invisible next to anything else here.
_COPY_CHUNK = 8 << 20


def write_shard(out_path, entries, metadata) -> None:
    """Write a safetensors file by copying tensor bytes, one buffer at a time.

    `entries` is an ordered list of (name, dtype, shape, source, start, end).

    **Never materializes a tensor.** The obvious implementation -- read the
    shard into a dict and hand it to `save_file` -- costs about twice the
    shard's size in RAM, which is fine for a 0.6B model and fatal for a 35B MoE
    whose shards are already most of the machine. Composition is a byte-level
    operation: nothing here needs to know what a trellis *means*, only where it
    starts and stops. So this builds the header from the source headers, assigns
    new offsets, and streams the payload across. Memory is one 8 MiB buffer
    whatever the model size, and it is faster too, since no dtype ever gets
    decoded.
    """
    header, off = {}, 0
    for name, dtype, shape, _src, a, b in entries:
        n = b - a
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [off, off + n]}
        off += n
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)   # keep the data blob 8-byte aligned
    handles: dict = {}
    try:
        with open(out_path, "wb") as w:
            w.write(len(blob).to_bytes(8, "little"))
            w.write(blob)
            for name, _dtype, _shape, src, a, b in entries:
                r = handles.get(src) or handles.setdefault(src, open(src, "rb"))
                r.seek(a)
                left = b - a
                while left:
                    chunk = r.read(min(left, _COPY_CHUNK))
                    if not chunk:
                        raise SystemExit(f"{src}: short read on {name}")
                    w.write(chunk)
                    left -= len(chunk)
    finally:
        for r in handles.values():
            r.close()


def link_or_copy(src: str, dst: str) -> bool:
    """Hardlink, resolving symlinks first. True if linked, False if copied.

    A Hugging Face snapshot is symlinks into `blobs/`, so linking the link
    yields something that dangles from anywhere else. Same reasoning as
    `tools/quantize_embedding.py`, and the same consequence when it falls back:
    an untouched shard becomes a full copy.
    """
    src = os.path.realpath(src)
    try:
        os.link(src, dst)
        return True
    except OSError:
        shutil.copy2(src, dst)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="checkpoint to start from")
    ap.add_argument("output", help="directory to write the composed checkpoint to")
    ap.add_argument("--from", dest="rules", metavar="DONOR", action=_Rule,
                    help="checkpoint that subsequent --take rules draw from")
    ap.add_argument("--take", dest="rules", metavar="SEL", action=_Rule,
                    help="take these modules from the current donor")
    ap.add_argument("--keep", dest="rules", metavar="SEL", action=_Rule,
                    help="revert these modules to the base")
    ap.add_argument("--restore", dest="rules", metavar="SEL", action=_Rule,
                    help="replace these QUANTIZED base modules with the "
                         "current donor's DENSE weights. The donor is an "
                         "unquantized checkpoint (the model the base was "
                         "converted from), not another EXL3 one.")
    ap.add_argument("--verify", type=int, default=4, metavar="N",
                    help="restore only: dequantize N restored modules and "
                         "check they match the dense weights replacing them. "
                         "Needs a GPU. 0 disables (see --help for why not).")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the checks, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="write despite errors. For deliberately odd "
                         "compositions; every error is still printed.")
    args = ap.parse_args()

    rules = getattr(args, "rules", None) or []
    if not rules:
        raise SystemExit("nothing to compose: give at least --from D --take SEL")

    base_t = read_headers(args.base)
    base_mods = modules_of(base_t)
    base_quant = format.quantized_module_keys(base_t)
    base_cfg = load_config(args.base)

    # --- resolve rules into an assignment of module -> donor dir ------------
    donors: dict[str, dict] = {}
    assign: dict[str, str] = {}
    restore: dict[str, str] = {}
    current: str | None = None
    for kind, value in rules:
        if kind == "from":
            current = os.path.normpath(value)
            if current not in donors:
                t = read_headers(current)
                donors[current] = {
                    "tensors": t,
                    "modules": modules_of(t),
                    "quantized": format.quantized_module_keys(t),
                    "config": load_config(current),
                }
            continue
        if kind == "keep":
            for m in select(value, base_mods, base_quant):
                assign.pop(m, None)
                restore.pop(m, None)
            continue
        if current is None:
            raise SystemExit(f"--{kind} {value!r} before any --from")
        if kind == "restore":
            # Selected from the BASE, because "restore" is a statement about
            # what the base has quantized. Selecting from the donor would match
            # its whole dense model, which is every module there is.
            d = donors[current]
            chosen = select(value, base_mods, base_quant) & base_quant
            if not chosen:
                raise SystemExit(
                    f"--restore {value!r} matched no *quantized* base module. "
                    "Restoring is for modules the base stores as a trellis; "
                    "anything already dense needs no restoring."
                )
            absent = sorted(m for m in chosen if m not in d["modules"])
            if absent:
                raise SystemExit(
                    f"--restore {value!r}: {len(absent)} selected module(s) do "
                    f"not exist in {current}, e.g. {absent[:3]}. Either it is "
                    "not the model this checkpoint was converted from, or the "
                    "converter split or fused those tensors and the dense "
                    "originals are under different names."
                )
            for m in chosen:
                restore[m] = current
                assign.pop(m, None)
            continue
        d = donors[current]
        chosen = select(value, d["modules"], d["quantized"])
        missing = sorted(chosen - set(base_mods))
        if missing:
            raise SystemExit(
                f"--take {value!r} from {current} selected {len(missing)} "
                f"module(s) the base does not have, e.g. {missing[:3]}. "
                "Composition replaces modules; it cannot add them."
            )
        for m in chosen:
            assign[m] = current
            restore.pop(m, None)

    if not assign and not restore:
        raise SystemExit("the rules selected no modules; nothing would change")

    # --- checks -------------------------------------------------------------
    problems: list[tuple[str, str]] = []
    for path, d in donors.items():
        check_same_model(args.base, path, base_cfg, d["config"], problems)
        only_base = set(base_mods) - set(d["modules"])
        only_donor = set(d["modules"]) - set(base_mods)
        if only_base or only_donor:
            problems.append((
                "warn",
                f"{os.path.basename(path)}: module sets differ "
                f"({len(only_base)} only in base, {len(only_donor)} only in donor); "
                "unselected differences do not travel",
            ))

    global _BASE_TENSORS
    _BASE_TENSORS = base_t

    for mod, donor_dir in sorted(restore.items()):
        d = donors[donor_dir]
        check_restore_module(mod, base_mods[mod], d["modules"][mod], base_t,
                             d["tensors"], problems)

    verified = {}
    if restore and args.verify and not any(lv == "error" for lv, _ in problems):
        verified = verify_restores(args.base, base_mods, donors, restore,
                                   args.verify, problems)
    elif restore and not args.verify:
        problems.append((
            "warn",
            "--verify 0: restored weights were not checked against the trellis "
            "they replace. Name and shape checks do not catch a wrong "
            "fine-tune or an unrecorded scale convention.",
        ))

    mixed_codebook = 0
    for mod, donor_dir in sorted(assign.items()):
        d = donors[donor_dir]
        check_module(mod, base_mods[mod], d["modules"][mod], base_t, d["tensors"],
                     problems)
        if codebook_of(base_mods[mod]) != codebook_of(d["modules"][mod]):
            mixed_codebook += 1

    if mixed_codebook:
        problems.append((
            "warn",
            f"{mixed_codebook} module(s) change codebook between base and donor. "
            "The selector is per tensor and travels with the trellis, so this "
            "loads and is correct -- but the output mixes codebooks and the "
            "top-level 'codebook' field in quantization_config.json describes "
            "only the base's.",
        ))

    # On a tied model the head is also what the *lookup* is served from, so
    # replacing it changes more than the logits GEMM. Strictly a serving-path
    # effect: the stored bf16 `embed_tokens.weight` is a separate module and is
    # not touched by `--take head` (every EXL3 checkpoint ships one even when
    # tied, and `tools/quantize_embedding.py` reads exactly that tensor to build
    # `bq_*`, so blockq generation is unaffected either way). What changes is
    # which tensor reaches the embedding module at load: for a tied checkpoint
    # `EXL3Config.get_cache_scale_mapper` drops the dense embedding unread and
    # renames `lm_head.*` onto the embedding prefix. Worth saying because the
    # effect then shows up as a changed lookup, which is the harder one to
    # attribute to a rule that only named the head.
    head_mods = {m for m in assign if m == "lm_head" or m.endswith(".lm_head")}
    if head_mods and _tie_word_embeddings(base_cfg):
        blockq = format.blockq_module_keys(base_t)
        extra = (" This checkpoint carries bq_*, which serves the lookup "
                 "instead, so the swap is confined to the logits GEMM."
                 if blockq else
                 " With no bq_* here, the default path serves the lookup from "
                 "the trellis being replaced (EXL3_DENSE_EMBED=1 serves the "
                 "dense embedding instead).")
        problems.append((
            "warn",
            "the base declares tie_word_embeddings, so lm_head also backs the "
            "token embedding on the serving path. The stored bf16 "
            "embed_tokens.weight is NOT modified." + extra,
        ))

    versions = {os.path.basename(p): (json.load(open(q)).get("version")
                if os.path.exists(q := os.path.join(p, "quantization_config.json"))
                else None)
                for p in [args.base, *donors]}
    if len(set(versions.values())) > 1:
        problems.append((
            "warn",
            f"exllamav3 versions differ across sources ({versions}). Conversion "
            "conventions are not all recorded in the checkpoint -- see "
            "docs/format-and-loading.md, 'The checkpoint is not a complete "
            "description of itself' -- so a cross-version composition can be "
            "silently wrong in ways no shape check sees. Verify the output.",
        ))

    # --- report -------------------------------------------------------------
    print(f" -- base   {args.base}  ({len(base_t)} tensors, "
          f"{len(base_mods)} modules, {len(base_quant)} quantized)")
    for path in donors:
        print(f" -- donor  {path}")

    by_donor: dict[str, list[str]] = {}
    for mod, donor_dir in assign.items():
        by_donor.setdefault(donor_dir, []).append(mod)

    delta = 0
    for donor_dir, mods in sorted(by_donor.items()):
        d = donors[donor_dir]
        buckets: dict[tuple, int] = {}
        for mod in mods:
            a = trellis_bits(mod, base_mods[mod], base_t)
            b = trellis_bits(mod, d["modules"][mod], d["tensors"])
            buckets[(a, b)] = buckets.get((a, b), 0) + 1
            delta += (sum(_nbytes(d["tensors"][n]) for n in d["modules"][mod])
                      - sum(_nbytes(base_t[n]) for n in base_mods[mod]))
        print(f" -- taking {len(mods)} module(s) from "
              f"{os.path.basename(os.path.normpath(donor_dir))}:")
        for (a, b), n in sorted(buckets.items(), key=lambda kv: (kv[0][0] or 0,
                                                                kv[0][1] or 0)):
            what = "dense" if a is None and b is None else f"K={a} -> K={b}"
            print(f"      {n:6d}  {what}")
        sample = sorted(mods)[:3]
        print(f"      e.g. {', '.join(sample)}"
              f"{' ...' if len(mods) > 3 else ''}")
    if restore:
        by_r: dict[str, list[str]] = {}
        for mod, donor_dir in restore.items():
            by_r.setdefault(donor_dir, []).append(mod)
        for donor_dir, mods in sorted(by_r.items()):
            d = donors[donor_dir]
            buckets: dict[int, int] = {}
            for mod in mods:
                k = trellis_bits(mod, base_mods[mod], base_t)
                buckets[k] = buckets.get(k, 0) + 1
                dense = [n for n in d["modules"][mod]
                         if n.endswith((".weight", ".bias"))]
                delta += (sum(_nbytes(d["tensors"][n]) for n in dense)
                          - sum(_nbytes(base_t[n]) for n in base_mods[mod]))
            print(f" -- restoring {len(mods)} module(s) to dense from "
                  f"{os.path.basename(os.path.normpath(donor_dir))}:")
            for k, n in sorted(buckets.items()):
                print(f"      {n:6d}  K={k} -> dense")
            print(f"      e.g. {', '.join(sorted(mods)[:3])}"
                  f"{' ...' if len(mods) > 3 else ''}")
        if verified:
            worst = max(verified.items(), key=lambda kv: kv[1][0])
            print(f" -- verified {len(verified)} restored module(s) against the "
                  f"trellis; worst relative error {worst[1][0]:.4f} "
                  f"(K={worst[1][1]}, {worst[0]})")
    print(f" -- size delta {delta / 2**20:+.1f} MiB")

    errors = [m for lvl, m in problems if lvl == "error"]
    for lvl, m in problems:
        print(f" {'!!' if lvl == 'error' else '..'} {m}", file=sys.stderr)
    if errors and not args.force:
        raise SystemExit(f"{len(errors)} error(s); refusing to write. "
                         "--force writes anyway.")

    if args.dry_run:
        print(" -- dry run, nothing written")
        return

    write(args, base_t, base_mods, donors, assign, restore, versions)


def _nbytes(entry) -> int:
    _, shape, dtype = entry
    n = 1
    for x in shape:
        n *= x
    return n * _DTYPE_BYTES.get(dtype, 2)


_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1,
                "I16": 2, "I32": 4, "I64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def write(args, base_t, base_mods, donors, assign, restore, versions) -> None:
    """Write the output, preserving the base's shard layout.

    Shards are the unit of work: one holding no replaced tensor is hardlinked,
    so composing a head onto a 30B model rewrites one shard and links the rest.
    The base's file names and index-or-not are kept exactly, which is both the
    cheapest thing and the only attested one -- the layout is a shape some
    publisher already emits, because it is the shape the base arrived in.
    """
    from safetensors import safe_open   # only to enumerate a hardlinked shard

    src, dst = args.base, args.output
    if os.path.exists(dst) and os.listdir(dst):
        raise SystemExit(f"{dst} exists and is not empty")
    os.makedirs(dst, exist_ok=True)

    # Which base tensors go away, and which donor tensors arrive, per shard.
    # A module's donor tensors land in the shard its base trellis lived in, so
    # a suffix set that differs between the two (`.mcg` vs `.mul1`) moves with
    # it rather than stranding a name the index would have to invent a home for.
    replaced: dict[str, set[str]] = {}
    incoming: dict[str, list[tuple[str, str]]] = {}
    for mod, donor_dir in assign.items():
        shard = base_t[base_mods[mod][0]][0]
        replaced.setdefault(shard, set()).update(base_mods[mod])
        for name in donors[donor_dir]["modules"][mod]:
            incoming.setdefault(shard, []).append(
                (name, donors[donor_dir]["tensors"][name][0]))

    # A restore drops the module's whole EXL3 storage and brings only the dense
    # tensors back. Everything the trellis needed -- suh, svh, the codebook
    # selector -- is meaningless without it and must not survive, or the module
    # still looks quantized to `format.quantized_module_keys`.
    for mod, donor_dir in restore.items():
        shard = base_t[base_mods[mod][0]][0]
        replaced.setdefault(shard, set()).update(base_mods[mod])
        for name in donors[donor_dir]["modules"][mod]:
            if name.endswith((".weight", ".bias")):
                incoming.setdefault(shard, []).append(
                    (name, donors[donor_dir]["tensors"][name][0]))

    weight_map: dict[str, str] = {}
    linked = copied = rewritten = 0
    for shard in sorted({v[0] for v in base_t.values()}):
        out = os.path.join(dst, os.path.basename(shard))
        if shard not in replaced:
            if link_or_copy(shard, out):
                linked += 1
            else:
                copied += 1
            with safe_open(shard, framework="pt") as h:
                for k in h.keys():
                    weight_map[k] = os.path.basename(shard)
            continue
        rewritten += 1
        drop = replaced[shard]
        src_hdr, meta = raw_header(shard)
        entries = [(k, *rec) for k, rec in src_hdr.items() if k not in drop]
        seen: dict = {}
        for name, donor_shard in incoming[shard]:
            if donor_shard not in seen:
                seen[donor_shard] = raw_header(donor_shard)[0]
            entries.append((name, *seen[donor_shard][name]))
        print(f" -- rewriting {os.path.basename(shard)} "
              f"(-{len(drop)} +{len(incoming[shard])} tensors)")
        write_shard(out, entries, meta or {"format": "pt"})
        for e in entries:
            weight_map[e[0]] = os.path.basename(shard)

    # Everything that is not weights travels from the base unchanged.
    for name in os.listdir(src):
        srcf = os.path.join(src, name)
        if not os.path.isfile(srcf):
            continue
        if name.endswith(".safetensors") or name == INDEX_NAME:
            continue
        if name == "quantization_config.json":
            continue
        link_or_copy(srcf, os.path.join(dst, name))

    if os.path.exists(os.path.join(src, INDEX_NAME)):
        # Regenerated rather than copied: a module whose donor stores `.mul1`
        # where the base stored `.mcg` changes which tensor names exist, and a
        # weight_map naming a tensor that is not there fails the load.
        with open(os.path.join(src, INDEX_NAME)) as f:
            meta = json.load(f).get("metadata", {}) or {}
        meta["total_size"] = sum(
            os.path.getsize(os.path.join(dst, n))
            for n in sorted(set(weight_map.values()))
        )
        with open(os.path.join(dst, INDEX_NAME), "w") as f:
            json.dump({"metadata": meta, "weight_map": weight_map}, f, indent=2)

    quant_after = format.quantized_module_keys(
        [n for n in base_t if module_key(n) not in restore]
    )
    write_quantization_config(args, dst, donors, assign, restore, versions,
                              quant_after)

    if copied:
        print(f" !! {copied} shard(s) were COPIED, not hardlinked -- output and "
              f"source are on different filesystems. Write the output beside "
              f"the base to keep composition cheap.", file=sys.stderr)
    print(f" -- {rewritten} shard(s) rewritten, {linked} hardlinked")
    print(f" -- wrote {dst}")


def write_quantization_config(args, dst, donors, assign, restore, versions,
                              base_quant_after=frozenset()) -> None:
    """Carry the base's `quantization_config.json` over, with the moved modules.

    Not a recomputation. Each replaced module's `tensor_storage` entry is
    *copied from the donor's own file*, where it was already correct, so the map
    keeps describing the tensors that are actually present. `head_bits` is
    corrected because it is the one top-level field this tool routinely
    falsifies and the one a reader is most likely to trust. `bits` is left
    alone and said to be stale in the `composition` block, because the honest
    value is a size-weighted average over the whole file and computing it here
    would be inventing a number upstream computes differently.
    """
    def load_qc(d):
        p = os.path.join(d, "quantization_config.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    qc = load_qc(args.base)
    if not qc:
        return
    ts = qc.get("tensor_storage")
    moved_unlisted = 0
    for mod, donor_dir in sorted(assign.items()):
        dts = (load_qc(donor_dir).get("tensor_storage") or {})
        if ts is None:
            continue
        if mod in dts:
            ts[mod] = dts[mod]
        elif mod in ts:
            # The donor's file omits a module its checkpoint really has --
            # Muse-Glimmer does this for its entire vision tower. Leaving the
            # base's entry would describe the wrong tensors, so drop it and let
            # the index speak, which is what the loader prefers anyway.
            del ts[mod]
            moved_unlisted += 1
        else:
            moved_unlisted += 1

    # A restored module is dense now, so its entry must stop claiming EXL3
    # storage. Described the way the checkpoint describes its own dense tensors
    # (shape/n_bytes/dtype, no `quant_format`) rather than deleted, so the map
    # still covers the module -- `EXL3Config` reads absence as "unknown", and
    # the index is what it trusts either way.
    restored_dense = 0
    for mod, donor_dir in sorted(restore.items()):
        d = donors[donor_dir]
        if ts is None:
            break
        entry = {}
        for name in d["modules"][mod]:
            if not name.endswith((".weight", ".bias")):
                continue
            _, shape, dtype = d["tensors"][name]
            n = 1
            for x in shape:
                n *= x
            entry[name] = {
                "shape": list(shape),
                "n_bytes": n * _DTYPE_BYTES.get(dtype, 2),
                "dtype": "torch." + dtype.lower().replace("bf16", "bfloat16")
                         .replace("f16", "float16").replace("f32", "float32"),
            }
        ts.pop(mod, None)
        if entry:
            ts[mod] = {"stored_tensors": entry}
            restored_dense += 1

    # `vision_bits` describes storage that may no longer exist. Dropped only
    # when nothing vision-shaped is quantized any more, because a partial
    # restore leaves the field true of what remains.
    if restore and "vision_bits" in qc:
        left = [m for m in base_quant_after if _CATEGORY["vision"](m)]
        if not left:
            qc.pop("vision_bits")
        else:
            qc.setdefault("composition", {})
            qc["composition"]["vision_bits_partial"] = len(left)

    head = next((m for m in assign if m == "lm_head" or m.endswith(".lm_head")), None)
    if head is not None:
        donor_t = donors[assign[head]]["tensors"]
        for name in donors[assign[head]]["modules"][head]:
            if name.endswith(".trellis"):
                qc["head_bits"] = format.bits_from_trellis_shape(donor_t[name][1])

    qc["composition"] = {
        "tool": "vllm-exl3-plugin tools/compose_checkpoint.py",
        "base": os.path.abspath(args.base),
        "rules": [f"--{k} {v}" for k, v in getattr(args, "rules", [])],
        "modules_replaced": len(assign),
        "modules_restored_to_dense": len(restore),
        "source_versions": versions,
        "stale": ["bits"] + (["codebook"] if "codebook" in qc else []),
        "note": "bits is inherited from the base and no longer describes this "
                "file; per-module tensor_storage entries follow their modules.",
    }
    if moved_unlisted:
        qc["composition"]["modules_not_in_tensor_storage"] = moved_unlisted
    with open(os.path.join(dst, "quantization_config.json"), "w") as f:
        json.dump(qc, f, indent=2)


if __name__ == "__main__":
    main()
