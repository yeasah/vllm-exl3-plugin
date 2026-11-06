---
name: check-upstream-before-patching-vllm
description: "Before writing any patch against vLLM or exllamav3, check what upstream already did to that file — we are pinned far behind and have duplicated their work repeatedly"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0014c7cb-d253-41bd-9304-bdba01e6a0e4
  modified: 2026-08-17T13:23:06.662Z
---

Before writing a patch to a pinned dependency (vLLM, exllamav3), check upstream
first:

    git -C ~/git/vllm fetch origin main
    git -C ~/git/vllm log --oneline HEAD..origin/main -- <the file you're about to edit>

**Why:** we have now duplicated upstream work three times. `retire-gemma4-patch`
(vLLM landed 70b84f0 generically), the Transformers-backend embedding fix (vLLM
PR #51247 did it better, by rebasing the subclass's MRO onto
`VocabParallelEmbedding` rather than substituting the module), and the logit
softcap (PR #52173). Each time the diagnosis was ours and worth having; only the
fix was wasted. On 2026-08-16 the checkout was 568 commits behind and the check
would have taken one command.

The user's framing, which is the durable part: this project spends its worry on
breakage arriving *from below* on a version bump, but what has actually cost us
is **being left behind**. Weigh staleness as a real risk, not just churn.

**It cuts both ways, and the other direction is a "submarine bug".** Our local
vLLM (`~/git/vllm`) is detached at a **release tag** (`v0.27.0`), while the vast
box builds from **main**. So a bug living on main is invisible locally for as
long as we stay on tags. On 2026-08-16 that bit twice with one bug: vLLM #49990
(2026-08-05) stores huggingface_hub's `ResolvedRevision` in `ModelConfig`, and
that class is a `str` subclass whose **string value does not survive pickle** —
it comes back as `"main"`. V1 pickles the config into the EngineCore subprocess,
so every repo served off a non-default branch breaks, which is every EXL3
checkpoint. The user hit it weeks earlier on a remote main build, abandoned that
build rather than debug it remotely, and assumed it had since been fixed; local
never saw it because `v0.27.0` predates the commit.

**The same discipline applies to vLLM's CLI and API surface, which churns hard.**
Do not suggest a flag from memory — grep the pinned tree. On 2026-08-17 I offered
`--max-num-encoder-input-tokens`, which has been removed with no obvious
replacement; meanwhile `--mm-processor-cache-gb 0` does *not* bound the encoder
cache, and `--limit-mm-per-prompt` has grown feature-size on top of counts
(`'{"image": {"count": 1, "width": 512, "height": 512}}'`). Module paths move too
(`vllm.transformers_utils.tokenizer` → `vllm.tokenizers.registry`).

**How to apply:** run the check before editing, not after — and if upstream has
already fixed it, say so and take their fix rather than carrying our own. Note
that a superseded fix does not invalidate the diagnostic work that found it;
report those separately. When a symptom appears only in one environment, compare
what the two are actually built from before assuming the code differs. And treat
"we'll deal with it at the next bump" as a decision that needs a written home —
`bench/` is now that home, since it catches this class in ~90s. See
[[vllm-fork-is-the-dependency]], [[vast-8x3090-test-box]] and
[[bench-suite-purpose]].
