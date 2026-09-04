# Reproductions for upstream reports

Self-contained scripts behind claims in [docs/upstream.md](../../docs/upstream.md).
Each one must run for someone who has never heard of this project — no EXL3, no
plugin, no checkpoint of ours — because that is the difference between a report a
maintainer can act on and one they have to take on trust.

| script | reproduces |
|---|---|
| `ct_embed_quantize.py` | llm-compressor's own documented embedding recipe, on `EleutherAI/pythia-160m` (their example family). Run in a **separate venv**: llmcompressor pins `transformers <= 5.14.1` and would downgrade a serving environment. |
| `ct_embed_serve.py` | loading that checkpoint in vLLM. Stock v0.28.0 raises `no module or parameter named 'embed_in.weight_packed'`; with the `vllm-embed-quant-config` commit in `deps/vllm` it loads and generates. |

```
python -m venv ~/.venv-llmcompressor
~/.venv-llmcompressor/bin/pip install llmcompressor
~/.venv-llmcompressor/bin/python tools/repro/ct_embed_quantize.py
python tools/repro/ct_embed_serve.py          # serving venv
```
