"""Checkpoints the gate needs but nobody publishes.

Every other entry names a `repo@revision` and lets the Hub resolve it. The
block-quantized embedding has no such checkpoint: `tools/quantize_embedding.py`
produces one from a published checkpoint, and no output of it has been
published. So an entry may instead name a **fixture** -- a recipe for deriving a
checkpoint from one the Hub does hold -- which this module resolves to a local
directory before capture.

**Derived rather than published, on purpose.** Producing the MiniCPM5-1B fixture
takes 3.1s and is byte-reproducible (the encoder runs on CPU; two runs a week
apart in different processes agree on the sha256 of all three tensors). That is
faster than downloading one would be, needs no account to run the gate, and --
the part that decides it -- puts `tools/quantize_embedding.py` under the gate
too. A published fixture would freeze the producer's output at the moment it was
uploaded and never exercise the producer again, which is the wrong half of the
system to stop testing: the tool is what rewrites real checkpoints.

The tool is invoked as a **subprocess**, not imported. The gate should exercise
the command line a user runs, including its argument handling and its output
layout, not a function underneath it.

## Staleness

A cached fixture is keyed on everything that decides its contents: the base
`repo@revision`, the recipe, and a digest of the encoder's own source. Change
`blockq.py` and the next `check` builds a new fixture rather than silently
serving the old one -- which matters because the alternative failure is a gate
that passes while testing a checkpoint the current code would not produce.

The key is deliberately *not* a hash of the produced bytes: that would be
circular, identifying what was built rather than what should have been.
Reproducibility is instead reported, as `digest`, into the capture record, so
`check` can tell "the fixture changed" apart from "the build regressed" -- the
same distinction `src.*.diff_sha` draws for the patch stack.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Bumped when a recipe's output would change for reasons the source digest
#: cannot see -- a different tool invocation, say. Changing it invalidates every
#: cached fixture of that recipe.
RECIPE_VERSION = 1

#: Files whose contents decide what a `blockq` fixture holds. The tool and the
#: format module: between them they are the encoder.
BLOCKQ_SOURCES = (
    "tools/quantize_embedding.py",
    "vllm_exl3_plugin/blockq.py",
    "vllm_exl3_plugin/format.py",
)


def cache_root() -> str:
    """Where derived checkpoints live. Never inside the repo -- they are large,
    and `bench/expected/` is deliberately the only thing a run writes in-tree."""
    return os.environ.get(
        "BENCH_FIXTURES",
        os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "vllm-exl3-plugin",
            "bench-fixtures",
        ),
    )


def _source_digest(relpaths) -> str:
    h = hashlib.sha256()
    for rel in relpaths:
        h.update(rel.encode())
        with open(os.path.join(ROOT, rel), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def key(kind: str, model: str, revision: str) -> str:
    """Identity of a fixture: base checkpoint, recipe, and the encoder's source.

    A cached directory under this name was produced by exactly this code from
    exactly this checkpoint, so it can be reused without a staleness check.
    """
    if kind != "blockq":
        raise SystemExit(f"unknown fixture recipe: {kind!r}")
    slug = f"{model}@{revision}".replace("/", "-").replace("@", "-at-")
    return f"{slug}-{kind}-v{RECIPE_VERSION}-{_source_digest(BLOCKQ_SOURCES)}"


def path(kind: str, model: str, revision: str) -> str:
    return os.path.join(cache_root(), key(kind, model, revision))


def digest(fixture_dir: str) -> str:
    """A content digest of what the recipe added, for the capture record.

    Recorded rather than enforced at build time: it answers "is this the same
    checkpoint the baseline was taken against", which is a question `check`
    asks, not one the builder can.
    """
    from safetensors import safe_open

    h = hashlib.sha256()
    shards = sorted(
        f for f in os.listdir(fixture_dir) if f.endswith(".safetensors")
    )
    for name in shards:
        with safe_open(os.path.join(fixture_dir, name), framework="pt") as f:
            for tensor in sorted(k for k in f.keys() if ".bq_" in k):
                h.update(tensor.encode())
                h.update(f.get_tensor(tensor).numpy().tobytes())
    return h.hexdigest()[:16]


def _snapshot(model: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision)


def ensure(kind: str, model: str, revision: str, quiet: bool = False) -> str:
    """Resolve a fixture to a local checkpoint directory, building it if needed."""
    dest = path(kind, model, revision)
    if os.path.isdir(dest):
        return dest

    base = _snapshot(model, revision)
    os.makedirs(cache_root(), exist_ok=True)
    # Build beside the destination and rename, so an interrupted build cannot
    # leave a half-written checkpoint that the next run would happily serve.
    staging = f"{dest}.building-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    if not quiet:
        print(f"     building fixture {os.path.basename(dest)}", flush=True)
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "quantize_embedding.py"),
         base, staging],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if proc.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise SystemExit(
            f"fixture build failed for {model}@{revision} ({kind}), "
            f"exit {proc.returncode}:\n{proc.stdout}"
        )
    os.replace(staging, dest)
    return dest
