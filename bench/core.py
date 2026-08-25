"""Capture and comparison numerics, shared by `bench/` and `tools/tp_compare.py`.

Both tools ask the same question in different framings -- "do these two engine
configurations compute the same thing?" -- so the numerics live here once.
`tp_compare` compares two live captures against each other (is TP=2 within the
eager-vs-graphs noise floor?); `bench` compares a live capture against a
baseline committed to the repo (did a dependency bump change what we serve?).

The measurement is teacher-forced, and that choice is load-bearing. Sampled-token
comparison conflates numerical error with model confidence, and once two runs
diverge they are on *different contexts*, so everything after the first
difference compares different prompts rather than different arithmetic. Feeding
both configurations a fixed token sequence and reading `prompt_logprobs` scores
every position against an identical context.

What a capture records, and why each part earns its place:

- **prompt token ids**, so a comparison can refuse to run when the two sides were
  not asked the same question. Detokenized text hides special tokens; ids do not.
- **top-k logprobs at every prompt position**, the sensitive channel. A monotonic
  error -- a missing logit scale, a dropped soft cap -- leaves every argmax
  untouched and is *completely invisible* in generated tokens. vLLM's
  Transformers backend dropped MuseGlimmer's `output_multiplier` and
  `final_logit_softcapping` while producing 40 of 40 greedy tokens identical to
  the correct model; only the logprobs moved. See docs/transformers-backend.md.
- **greedy continuation ids**, the coarse channel, and the human-readable one. A
  dropped embedding norm in the same backend changed 7 of 7 tokens.
- **reported weight bytes**, which no logit comparison can reach. The tied
  embedding path serves a model's embedding from its quantized `lm_head` and
  never loads the fp16 `embed_tokens`; if a vLLM change breaks `tie_weights` or
  the tied-skip mapper, the model still produces correct logits and silently
  costs a GiB. That is the project's whole thesis, so it is gated.
"""

from __future__ import annotations

import math

#: Deliberately spans the confidence range, because that is the axis that makes
#: token-based comparison lie. The factual prompt is near-deterministic; the
#: open-ended one leaves the model genuinely uncertain, and is long enough to
#: contribute most of the scored positions.
PROMPTS = [
    "What is the capital of France?",
    "I am comparing three approaches to quantizing large language models: "
    "trellis coding, group-wise integer quantization, and low-rank adapters "
    "applied post-training. For each one, explain the core idea, where the "
    "error comes from, and which hardware makes it fast.",
]


def _git(path: str, *args: str) -> str | None:
    import subprocess

    try:
        out = subprocess.run(("git", "-C", path) + args, capture_output=True,
                             text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # pragma: no cover - probing must never fail a run
        return None


#: Baselines are this suite's *output*, and they live inside the tree whose
#: provenance a capture records. Without excluding them, blessing dirties the
#: thing it is describing: the first entry writes a baseline, so the second
#: records `dirty_files: 1`, the third `2`, and a full bless can never record a
#: clean plugin state no matter how clean the checkout was. Provenance is about
#: the code that produced a measurement, not the measurement.
_PROVENANCE_EXCLUDE = (":(exclude)bench/expected",)


def source_provenance(path: str, exclude: tuple[str, ...] = ()) -> dict | None:
    """What code a source tree actually held, which its package version may not say.

    Installed version strings are not reliable here and one of ours is actively
    wrong: vLLM is an editable install off a detached HEAD at `v0.27.0`, and
    reports `0.27.1.dev0+ge50f7d369.d20260810`. The hash and date are right; the
    base version is not -- it appears to be whatever was newest when the build
    was made. `git describe` says `v0.27.0-dirty`, which is the truth.

    The worse problem is invisible rather than wrong. This project applies
    `patches/` to vLLM as **unstaged working-tree changes**, and no version
    string can see those. Two baselines could carry an identical `vllm` field
    and have been produced by different patch stacks -- which, for a gate whose
    whole job is spanning a dependency bump, is the difference most likely to
    matter. `diff_sha` is what distinguishes them.

    Limits worth knowing: the digest covers tracked modifications only, so
    `untracked` is reported separately (an untracked `.py` inside a package
    does change behaviour), and the digest identifies a patch stack without
    describing it -- to see what changed, diff the trees. All three respect
    `exclude`; a field that did not would let the suite describe its own
    output, which is what the exclusion exists to prevent.
    """
    head = _git(path, "rev-parse", "HEAD")
    if head is None:
        return None  # not a git checkout: a wheel install, or a tarball
    scope = ("--", ".") + exclude if exclude else ()
    status = _git(path, "status", "--porcelain", *scope) or ""
    dirty = [ln for ln in status.splitlines() if ln.strip()]
    diff = _git(path, "diff", "HEAD", *scope)
    diff_sha = None
    if diff:
        import hashlib

        diff_sha = hashlib.sha256(diff.encode()).hexdigest()[:12]
    # Scoped like `status` and `diff` above, and for the same reason -- the
    # exclusion is not only about *modified* baselines. A bless that creates
    # entries writes files that are untracked at the moment they are counted
    # (capture opens its output before recording the environment), so an
    # unscoped count climbs 1, 2, ... across the new entries of one run and
    # bakes into their baselines a number no clean checkout can reproduce.
    # Found 2026-08-24, on the first bless to add entries rather than overwrite
    # them, which is why it had never shown.
    untracked = _git(path, "ls-files", "--others", "--exclude-standard",
                     *scope) or ""
    return {
        # No `--dirty`: `git describe` takes no pathspec, so it would report the
        # tree as dirty on the strength of the baselines this suite just wrote --
        # contradicting `dirty_files: 0` beside it. Let `describe` answer "which
        # commit" and let the two fields below answer "what is uncommitted".
        "describe": _git(path, "describe", "--tags", "--always"),
        "head": head[:10],
        "dirty_files": len(dirty),
        "diff_sha": diff_sha,
        "untracked": len([ln for ln in untracked.splitlines() if ln.strip()]),
    }


def _source_trees() -> dict:
    """Locate the three trees whose contents decide what a baseline means."""
    import os

    trees = {"plugin": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
    for name in ("vllm", "exllamav3"):
        try:
            module = __import__(name)
            # <repo>/<package>/__init__.py -> <repo>
            trees[name] = os.path.dirname(os.path.dirname(os.path.abspath(module.__file__)))
        except Exception:  # pragma: no cover
            continue
    return trees


def environment() -> dict:
    """What the machine will admit about itself.

    This is **evidence, not identity**. A perf baseline is only meaningful on the
    platform that produced it, and no amount of introspection identifies a
    platform reliably: firmware, host BIOS, thermal headroom, a noisy neighbour
    on a shared host and the hypervisor's own scheduling all move throughput and
    none of them are visible from in here. So the operator supplies the identity
    (see `bench/run.py --platform`) and this exists to cross-check it -- if the
    tag says one box and the GPU name says another, something is mislabelled and
    the numbers should not be trusted.

    Best-effort throughout: a missing field is recorded as None rather than
    failing a run, since none of this is load-bearing on its own.
    """
    env: dict = {}
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
            major, minor = torch.cuda.get_device_capability(0)
            env["capability"] = f"{major}.{minor}"
    except Exception:  # pragma: no cover - probing must never fail a run
        pass
    try:
        import vllm

        # Kept for continuity, but do not read it as the truth -- see
        # `source_provenance`. This string says 0.27.1.dev0 for a checkout
        # detached at v0.27.0, and cannot see our unstaged patches at all.
        env["vllm_reported"] = vllm.__version__
    except Exception:  # pragma: no cover
        pass
    # Load-bearing Python dependencies that are neither vLLM nor a git tree we
    # track. `transformers` is the biggest: two entries run *its* model code
    # through the Transformers backend, gemma-4's config handling has been
    # version-sensitive enough to need a patch, and every tokenizer comes from
    # it -- so a change there can move logprobs with nothing else moving.
    # `compressed-tensors` decides how a quantized checkpoint is interpreted and
    # is a vLLM dependency in its own right. Found missing 2026-08-25, when a
    # dry-run install of llm-compressor turned out to want transformers
    # *downgraded* 5.15.0 -> 5.14.1 underneath a freshly blessed set.
    for pkg in ("transformers", "compressed-tensors"):
        try:
            import importlib.metadata as _md

            env[f"pkg.{pkg}"] = _md.version(pkg)
        except Exception:  # pragma: no cover - probing must never fail a run
            env[f"pkg.{pkg}"] = None
    for name, path in _source_trees().items():
        prov = source_provenance(
            path, _PROVENANCE_EXCLUDE if name == "plugin" else ()
        )
        if prov is not None:
            env[f"src.{name}"] = prov
    try:
        import subprocess

        env["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip().splitlines()[0]
    except Exception:  # pragma: no cover
        env["driver"] = None
    return env


def environment_diff(a: dict, b: dict) -> list[str]:
    """Fields that differ between two `environment()` records.

    Descends one level into the `src.*` records, so a changed patch stack reports
    as `src.vllm.diff_sha: ... -> ...` rather than as two opaque dicts.
    """
    lines = []
    for k in sorted(set(a) | set(b)):
        av, bv = a.get(k), b.get(k)
        if av == bv:
            continue
        if isinstance(av, dict) and isinstance(bv, dict):
            for sub in sorted(set(av) | set(bv)):
                if av.get(sub) != bv.get(sub):
                    lines.append(f"{k}.{sub}: {av.get(sub)!r} -> {bv.get(sub)!r}")
        else:
            lines.append(f"{k}: {av!r} -> {bv!r}")
    return lines


def split_environment_diff(lines: list[str]) -> tuple[list[str], list[str]]:
    """Separate drift worth a warning from drift that is just development.

    The plugin's own tree moves with every commit, so reporting it at the same
    volume as everything else would put a warning on essentially every run --
    and a warning that always fires is one you stop reading, which is the
    failure this whole file is trying to avoid elsewhere.

    So it is still reported, because knowing the baseline came from different
    plugin code is real context when reading a failure, just not as an alarm.
    Dependency and machine drift keep the alarm: those are the changes a bump
    gate exists to notice, and neither should happen quietly.
    """
    notable = [ln for ln in lines if not ln.startswith("src.plugin.")]
    plugin = [ln for ln in lines if ln.startswith("src.plugin.")]
    return notable, plugin


def report_environment_diff(base: dict, fresh: dict, indent: str = "     ") -> None:
    """Print drift between two `environment()` records, alarming selectively."""
    notable, plugin = split_environment_diff(environment_diff(base, fresh))
    for line in notable:
        print(f"{indent}! {line}")
    if plugin:
        summary = "; ".join(
            ln.split(": ", 1)[0].removeprefix("src.plugin.") for ln in plugin
        )
        print(f"{indent}  (plugin tree differs from baseline: {summary} -- "
              "expected during development)")


#: Chat templates that inject the current date make a baseline expire at
#: midnight. Muse-Glimmer's does exactly that -- `strftime_now('%Y-%m-%d')` in
#: its system preamble -- so its ids changed the day after blessing and `check`
#: correctly refused to compare, for a reason that has nothing to do with the
#: build under test. A gate that goes red on the calendar is a gate people learn
#: to ignore.
#:
#: The templates that take this offer it as an override (`current_date is
#: defined`); templates that do not use it ignore the extra context variable, so
#: passing it unconditionally is safe. The value is arbitrary and only has to
#: never change.
#: Spellings differ per family and there is no common one: Muse-Glimmer takes
#: `current_date`/`knowledge_cutoff`, Llama 3.x takes `date_string`. Both reach
#: for `strftime_now` when the override is absent, so both were rotting nightly
#: until this existed. Add the spelling when adding a model whose template does
#: the same -- `"strftime_now" in tok.chat_template` is the check.
PINNED_TEMPLATE_VARS = {
    "current_date": "2026-01-01",
    "knowledge_cutoff": "2026-01-01",
    "date_string": "01 Jan 2026",
}


def prompt_ids(tok, text: str) -> list[int]:
    """Chat-templated token ids for one prompt.

    `apply_chat_template` returns a BatchEncoding for some tokenizers and a plain
    list for others, and iterating a BatchEncoding yields its *keys* -- so the
    shape has to be normalized rather than assumed.
    """
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        **PINNED_TEMPLATE_VARS,
    )
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def capture_prompts(llm, tok, k: int = 20, new_tokens: int = 24) -> list[dict]:
    """Score every prompt position and greedily continue, for each prompt.

    One `generate` call per prompt does both: `prompt_logprobs` scores the fixed
    context, `max_tokens` produces the continuation. Greedy (temperature 0) so
    the continuation is a property of the model rather than of a seed.
    """
    from vllm import SamplingParams

    out = []
    for text in PROMPTS:
        ids = prompt_ids(tok, text)
        params = SamplingParams(
            temperature=0.0, max_tokens=new_tokens, prompt_logprobs=k, logprobs=k
        )
        result = llm.generate({"prompt_token_ids": ids}, params)[0]

        steps = []
        for pos in result.prompt_logprobs or []:
            if pos is None:  # first position has no predecessor to score
                steps.append(None)
                continue
            steps.append({str(t): round(lp.logprob, 6) for t, lp in pos.items()})

        gen = result.outputs[0]
        out.append(
            {
                "ids": ids,
                "steps": steps,
                "generated_ids": [int(t) for t in gen.token_ids],
                "generated_text": tok.decode(list(gen.token_ids)),
            }
        )
    return out


def kl(p: dict, q: dict) -> float:
    """KL(P||Q) over P's support, renormalized.

    Both sides are truncated to top-k, so Q may not cover all of P. Restricting
    to the shared support and renormalizing keeps this finite; it understates
    divergence when the top-k sets disagree, which is itself reported separately.
    """
    shared = [t for t in p if t in q]
    if not shared:
        return float("nan")
    zp = math.log(sum(math.exp(p[t]) for t in shared))
    zq = math.log(sum(math.exp(q[t]) for t in shared))
    total = 0.0
    for t in shared:
        lp, lq = p[t] - zp, q[t] - zq
        total += math.exp(lp) * (lp - lq)
    return total


def compare_prompt(pa: dict, pb: dict) -> dict:
    """Per-position divergence between two captures of the same prompt.

    Returns `{"comparable": False}` when the two sides were not asked the same
    question, rather than reporting a divergence that means nothing.
    """
    if pa["ids"] != pb["ids"]:
        return {"comparable": False}

    kls, dtop, disagree, n = [], [], 0, 0
    for sa, sb in zip(pa["steps"], pb["steps"]):
        if not sa or not sb:
            continue
        n += 1
        kls.append(kl(sa, sb))
        ta = max(sa, key=sa.get)
        tb = max(sb, key=sb.get)
        if ta != tb:
            disagree += 1
        if ta in sb:
            dtop.append(abs(sa[ta] - sb[ta]))

    finite = [v for v in kls if v == v]
    return {
        "comparable": True,
        "positions": n,
        "argmax_disagreements": disagree,
        "kl_max": max(finite) if finite else 0.0,
        "kl_mean": sum(finite) / len(finite) if finite else 0.0,
        "dlogprob_max": max(dtop) if dtop else 0.0,
        "dlogprob_mean": sum(dtop) / len(dtop) if dtop else 0.0,
        "generated_match": pa.get("generated_ids") == pb.get("generated_ids"),
    }
