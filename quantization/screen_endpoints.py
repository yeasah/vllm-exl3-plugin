#!/usr/bin/env python3
"""Screen an OpenAI-compatible endpoint's providers before generating traces with `--backend openai`.

Each provider is pinned alone (OpenRouter's provider.only) and put through ctx_trace's own screening: is a
/completions prompt completed raw (no chat template of the provider's, no reasoning split out), and does each
sampling parameter change what it must change. Without --providers, every endpoint OpenRouter lists for the
model is screened. Without --sampling-profile, every parameter is tested; with one, only those it sets to a
non-neutral value, which is what a trace run would require.

    quantization/screen_endpoints.py -t Qwen/Qwen3.8-27B -m qwen/qwen3.8-27b \\
        [--providers mancer/fp8,parasail/fp8] [--sampling-profile quantization/sampling/Qwen__Qwen3.8-27B.json]

-t is the repo (or path) whose tokenizer and chat template render the prompts: the model as published, so a
provider is screened on the same prompts a trace would send it. Costs a few dozen short requests per provider.
"""
import argparse
import concurrent.futures
import io
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "deps", "exllamav3"))
TOKENIZER_FILES = ["*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"]      # as quant.py's


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("-t", "--tokenizer", required=True, help="repo or path with the model's config, tokenizer and template")
    p.add_argument("-m", "--model", required=True, help="the endpoint's model id, e.g. qwen/qwen3.8-27b")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--providers", default=None, help="comma-separated provider.only entries (default: every endpoint listed)")
    p.add_argument("--body", type=json.loads, default={}, help="JSON merged into every request")
    p.add_argument("--sampling-profile", default=None, help="test only the parameters this profile sets non-neutral")
    p.add_argument("--concurrency", type=int, default=8, help="providers screened at once")
    p.add_argument("-o", "--output", default=None, help="write every verdict and the endpoint listing as JSON")
    args = p.parse_args()

    # ctx_trace's backend reports as it goes; only the main thread's lines reach the terminal
    real = sys.stdout
    class MainOnly(io.TextIOBase):
        def write(self, t):
            return real.write(t) if threading.current_thread() is threading.main_thread() else len(t)
        def flush(self):
            real.flush()
    sys.stdout = MainOnly()

    import ctx_trace as ct
    from exllamav3 import Config, Tokenizer
    from huggingface_hub import snapshot_download

    path = args.tokenizer if os.path.isdir(args.tokenizer) else snapshot_download(args.tokenizer, allow_patterns=TOKENIZER_FILES)
    tok = Tokenizer.from_config(Config.from_directory(path))
    tv = ct.probe_template_vars(tok, dict(ct.DEFAULT_TEMPLATE_VARS))
    if args.sampling_profile:
        prof = ct.load_profile(args.sampling_profile)
        used = {ct.VLLM_KEYS.get(k, k) for m in prof["modes"].values() for k, v in m.items()
                if ct.VLLM_KEYS.get(k, k) in ct.OPENAI_NEUTRAL and v not in ct.OPENAI_NEUTRAL[ct.VLLM_KEYS.get(k, k)]}
    else:
        used = ct.OPENAI_SAMPLING - {"temperature"}

    def backend(pin):
        body = dict(args.body, **({"provider": {**args.body.get("provider", {}), "only": [pin]}} if pin else {}))
        return ct.OpenAIBackend({"base_url": args.base_url, "model": args.model, "api_key_env": args.api_key_env,
                                 "body": body, "tokens": "text" if "openrouter.ai" in args.base_url else "auto"},
                                tok, tv, os.devnull)

    listing = backend(None).info.get("endpoints") or {}
    eps = {e["tag"]: e for e in listing.get("endpoints", [])}
    pins = args.providers.split(",") if args.providers else list(eps) or [None]
    print(f"screening {len(pins)} provider(s) of {args.model} for {', '.join(sorted(used)) or 'temperature only'}", flush=True)

    def screen(pin):
        b = backend(pin)
        try:
            b.screen(used)
        except SystemExit:
            pass
        return pin, next(iter(b.screened.values()))

    rows = []
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as ex:
        for pin, v in ex.map(screen, pins):
            e = eps.get(pin, {})
            tests = v["tests"]
            if not tests:
                verdict = "not raw" if "rendered or tokenized" in v.get("why", "") or "split reasoning" in v.get("why", "") \
                    else "error"
            else:
                verdict = "pass" if v["passed"] else "refused"
            ignored = [k for k, t in tests.items() if not t["ok"] and "inconclusive" not in t]
            unsure = [k for k, t in tests.items() if "inconclusive" in t]
            rows.append({"pin": pin, "provider": v["provider"], "quantization": e.get("quantization"), "verdict": verdict,
                         "ignores": ignored, "untested": unsure, "why": v.get("why"), "tests": tests})
            print(f"  {str(pin):28s} {str(e.get('quantization') or ''):8s} {verdict:8s} "
                  f"{'ignores ' + ', '.join(ignored) if ignored else ''}{'  could not test ' + ', '.join(unsure) if unsure else ''}"
                  f"{'  ' + v['why'][:120] if verdict in ('not raw', 'error') else ''}", flush=True)
    order = {"pass": 0, "refused": 1, "not raw": 2, "error": 3}
    rows.sort(key=lambda r: (order[r["verdict"]], len(r["ignores"]), str(r["pin"])))
    ok = [r["pin"] for r in rows if r["verdict"] == "pass"]
    print(f"\npassed: {', '.join(map(str, ok)) or 'none'}")
    if ok:
        print(f'--openai-args \'{json.dumps({"base_url": args.base_url, "model": args.model, "api_key_env": args.api_key_env, "body": dict(args.body, provider={"only": ok})})}\'')
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "base_url": args.base_url, "tested": sorted(used), "results": rows,
                       "endpoints": listing}, f, indent=1)


if __name__ == "__main__":
    main()
