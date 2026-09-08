"""On-disk cache for embeddings encoded during load (`EXL3_BLOCKQ_ON_LOAD`).

`EXL3_BLOCKQ_ON_LOAD` buys the block-quantized embedding's memory saving on a
checkpoint nobody has repaired, at the cost of encoding the dense matrix every
time the engine starts. That cost is real -- Qwen3.5-9B's 248320x4096 takes 5.8 s,
which is model loading going from 1.96 s to 9.99 s -- and it lands squarely on the
one thing a profiling harness does over and over. The encode is a pure function of
the dense tensor, so it should be paid once per checkpoint, not once per engine.

What identifies an entry, and why each part is in the key:

  - **The checkpoint's resolved revision**, not the repo name. EXL3 repos publish
    one branch per bit rate and branches get re-pushed; `turboderp/X-exl3@3.0bpw`
    is not a stable description of any particular embedding matrix. The commit
    hash transformers actually resolved (`hf_config._commit_hash`) is.
  - **The tensor's name.** One embedding per model is the common case, not a
    guarantee: multimodal checkpoints already carry more than one embedding
    matrix, and if this path ever serves a second one, two entries under one key
    would silently hand a model the wrong matrix.
  - **Vocabulary and hidden size, block size and bit width.** These are implied
    by the two above and are in the key anyway, so that changing `BLOCKQ_BLOCK`
    or `BLOCKQ_BITS` invalidates rather than misreads. Shapes are re-checked on
    read regardless; the key is what makes the check a miss instead of an error.

A local directory has no revision to key on, so it is identified by a digest of
its own files' names, sizes and mtimes. That is weaker than a commit hash --
`touch` invalidates, and a rewrite that preserved both size and mtime would not
-- but it is the property a local checkout actually has, and it fails toward a
re-encode rather than toward a wrong matrix.

**Entries hold the whole vocabulary, and the reader slices.** Encoding is
row-independent (`blockq.encode`'s reductions are all inside a row), so a rank's
shard of the encoding equals the encoding of its shard, and one entry serves
every tensor-parallel arrangement -- which matters here precisely because
sweeping TP is a thing the profiling this exists for does. The cost is that a
*cold* load at TP=N has each rank encode the full vocabulary rather than its
1/N. That is once per checkpoint, and it is why the uncached path in
`embedding.py` still encodes only the local rows.

**Nothing here is allowed to be fatal.** A read-only or full `~/.cache`, a
truncated entry, a version of this file that wrote a layout this one does not
understand -- each is a cache miss and a log line, never a failure to serve.
"""

from __future__ import annotations

import hashlib
import os

import torch

from . import env, format
from .log import init_logger

logger = init_logger(__name__)

#: Bumped when the *file* layout changes in a way an older reader would
#: misinterpret. Part of the key, so a bump orphans old entries rather than
#: reading them; they are inert bytes under `root()` and can simply be deleted.
CACHE_VERSION = 1

#: Warn once per failing directory rather than once per tensor per rank.
_WARNED: set[str] = set()


def enabled() -> bool:
    """`EXL3_BLOCKQ_CACHE=0` turns the cache off; on-load encoding still works."""
    return env.get("BLOCKQ_CACHE", "1") != "0"


def root() -> str:
    """Where entries live. `~/.cache/vllm-exl3-plugin/` is already this
    project's pile (`bench/fixtures.py` puts derived checkpoints beside it), and
    `XDG_CACHE_HOME` is honoured for the same reason it is there."""
    return env.get(
        "BLOCKQ_CACHE_DIR",
        os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "vllm-exl3-plugin",
            "blockq-embeddings",
        ),
    )


def checkpoint_identity(model_name: str | None, commit_hash: str | None) -> str | None:
    """A string that changes whenever the checkpoint's tensors could have.

    `commit_hash` is the commit transformers resolved, not the revision the user
    asked for. The two differ in the case this project sees constantly: EXL3
    repos publish one branch per bit rate, `X-exl3@3.00bpw` is what gets served,
    and that branch gets re-pushed. Keying on the branch name would hand a
    re-quantized checkpoint the previous one's embedding, silently.

    `None` means "cannot say", and every caller treats that as "do not cache" --
    the same refusal `EXL3Config._skip_hub_lookup` makes rather than papering
    over an unresolved revision with `"main"`.
    """
    if not model_name:
        return None
    if os.path.isdir(model_name):
        return _local_identity(model_name)
    if commit_hash:
        return f"{model_name}@{commit_hash}"
    return None


#: Files whose identity stands in for a local checkout's. The weights, plus the
#: two files that decide how they are read -- a `config.json` edit can change
#: which tensor is the embedding without touching a shard.
_LOCAL_IDENTITY_FILES = ("config.json", "model.safetensors.index.json")


def _local_identity(path: str) -> str | None:
    h = hashlib.sha256()
    real = os.path.realpath(path)
    h.update(real.encode())
    try:
        names = sorted(
            n
            for n in os.listdir(real)
            if n.endswith(".safetensors") or n in _LOCAL_IDENTITY_FILES
        )
        if not names:
            return None
        for name in names:
            # Follows symlinks deliberately: a Hub snapshot directory is a tree
            # of links into content-addressed blobs, and the blob is the thing
            # whose identity we want.
            st = os.stat(os.path.join(real, name))
            h.update(f"|{name}:{st.st_size}:{st.st_mtime_ns}".encode())
    except OSError:
        return None
    return f"{os.path.basename(real.rstrip(os.sep))}@local-{h.hexdigest()[:16]}"


def _slug(text: str) -> str:
    """A filename-safe, human-recognisable prefix. Only for reading `ls` output:
    the digest beside it is what actually distinguishes two entries.

    Truncated from the right, so what survives is the repository name rather
    than the tail of a commit hash -- `ls` on this directory should say which
    models are in it.
    """
    safe = "".join(c if (c.isalnum() or c in "._-") else "-" for c in text)
    return safe[:64].strip("-") or "entry"


class Entry:
    """One cached encoding: where it lives, and what it claims to be."""

    def __init__(self, path: str, meta: dict[str, str]):
        self.path = path
        self.meta = meta


def entry(identity: str, tensor: str, rows: int, hidden: int) -> Entry:
    """Locate the entry for one embedding tensor of one checkpoint."""
    fields = {
        "cache_version": str(CACHE_VERSION),
        "checkpoint": identity,
        "tensor": tensor,
        "rows": str(rows),
        "hidden": str(hidden),
        "block": str(format.BLOCKQ_BLOCK),
        "bits": str(format.BLOCKQ_BITS),
    }
    digest = hashlib.sha256(
        "|".join(f"{k}={v}" for k, v in fields.items()).encode()
    ).hexdigest()[:16]
    path = os.path.join(root(), f"{_slug(identity)}-{digest}.safetensors")
    # safetensors metadata is str -> str. Carried so an entry can be identified
    # from the file alone -- the name is a digest, and a stale pile is otherwise
    # opaque to anything but this function.
    meta = dict(fields)
    meta["format"] = "pt"
    return Entry(path, meta)


def read(
    entry: Entry, *, rows: int, hidden: int, row_start: int, row_stop: int
) -> dict[str, torch.Tensor] | None:
    """The entry's `[row_start:row_stop)` rows, or `None` for any kind of miss.

    Read as a slice rather than loaded whole: at TP>1 each rank wants its own
    rows and nothing else, and safetensors can serve that without materializing
    the vocabulary. The shape check is against `format.blockq_shapes`, so an
    entry written by a different format -- however it got there -- is a miss and
    not a wrong answer.
    """
    if not os.path.exists(entry.path):
        return None
    try:
        from safetensors import safe_open

        want = format.blockq_shapes(rows, hidden)
        out: dict[str, torch.Tensor] = {}
        with safe_open(entry.path, framework="pt", device="cpu") as f:
            # A tensor the format needs and the entry lacks raises out of
            # `get_slice`, which the handler below turns into a miss. There is
            # no separate check for it: one that has never been seen to fire is
            # a comment, and this one could not say more than the exception.
            for name, shape in want.items():
                sliced = f.get_slice(name)
                if tuple(sliced.get_shape()) != tuple(shape):
                    logger.warning(
                        "Ignoring block-quantized embedding cache entry %s: '%s' "
                        "has shape %s, expected %s.",
                        entry.path, name, list(sliced.get_shape()), list(shape),
                    )
                    return None
                out[name] = sliced[row_start:row_stop]
        return out
    except Exception as exc:
        logger.warning(
            "Could not read the block-quantized embedding cache entry %s (%s); "
            "encoding instead.", entry.path, exc,
        )
        return None


def write(entry: Entry, tensors: dict[str, torch.Tensor]) -> bool:
    """Store a full-vocabulary encoding. Returns whether it landed.

    Written to a temporary name and renamed, so a reader never sees a partial
    file and two ranks racing to fill the same entry cannot interleave. They
    write identical bytes -- the encode is on CPU precisely so that it is
    reproducible -- so last-writer-wins is not a correctness question.
    """
    directory = os.path.dirname(entry.path)
    tmp = f"{entry.path}.{os.getpid()}.tmp"
    try:
        from safetensors.torch import save_file

        os.makedirs(directory, exist_ok=True)
        save_file(tensors, tmp, metadata=entry.meta)
        os.replace(tmp, entry.path)
        return True
    except Exception as exc:
        if directory not in _WARNED:
            _WARNED.add(directory)
            logger.warning(
                "Could not write the block-quantized embedding cache under %s "
                "(%s). Serving is unaffected; the encode will be repeated on "
                "every load until this is fixed, or set EXL3_BLOCKQ_CACHE=0.",
                directory, exc,
            )
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False
