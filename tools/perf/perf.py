#!/usr/bin/env python3
"""Serving benchmark for the recipe tables: starts `vllm serve` per test config, runs `vllm bench serve` against it at
each concurrency (sharegpt for decode, random for prefill), and renders results/<model>/results.md. Run it from its
working directory (results/ and the ShareGPT file live there): `perf.py --testfile tests.json`; with no testfile it
only re-renders.

Under concurrency, a request's TTFT depends on what it is concurrent with. Two controls keep that comparable across
configs: at concurrency > 1 the random run's output length holds each request's decode time at a fixed multiple of its prefill time
(`--decode-ratio`: prefill and decode rate from the config's 1x random run, the decode rate scaled by how sharegpt's
slows at that concurrency), so a
config that decodes faster, or prefills slower, does not see more overlapping prefills; and the first wave of requests, which always starts as simultaneous prefills, is
left out of the per-request means (`--save-detailed` results). Each bench run also sends one warm-up request from
its own sample first (`--warmups`): the random dataset's shared prefix differs between invocations, so only that
caches it, and it warms the expert cache and JIT paths. Results saved without per-request data render as before.
"""

import glob
import subprocess
import argparse
import json
import yaml
import os
import sys
import re
import shlex
import socket
import threading
import traceback
import pynvml
import datetime
from huggingface_hub import hf_hub_download, snapshot_download

_PROMPTS = { "cat": "Draw a picture of a cute cat using SVG" }

def run_tests(args):
    with open(args.testfile, "r") as f:
        tests = yaml.safe_load(f)
        mysterious_bullshit = tests.get("mysterious_bullshit", 0.0)
        for test in tests["tests"]:
            if not args.tests or re.search(args.tests, test["model"]):
                model = f"{test['model']}@{test['revision']}"
                try:
                    run_test(args, test, mysterious_bullshit)
                except Exception as e:
                    print(f"*** TEST FAILED: {e} ***")
                    traceback.print_exc()

def run_test(args, test, mysterious_bullshit):
    # get GPU info, count and target utilization
    vram = test.get("vram", 15)
    count = pynvml.nvmlDeviceGetCount()
    num_gpus = 0
    total_vram = 0
    avail_vram = 0
    devname = None
    for i in range(count):
        num_gpus += 1
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        devname = pynvml.nvmlDeviceGetName(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_vram += mem.total / 1024**3
        avail_vram += mem.free / 1024**3
        if avail_vram > vram:
            break
    if avail_vram < vram:
        raise Exception(f"Total VRAM available {avail_vram} GiB is insufficient for test, {vram} GiB needed")
    total_vram -= mysterious_bullshit / 1024
    utilization = vram / total_vram

    base_model = test["model"].split("/")[-1]
    tag = f'{test["revision"]}_{devname.replace(' ', '-')}x{num_gpus}_{vram}G'
    if "suffix" in test:
        tag += f"_{test['suffix']}"
    results_dir = f'results/{base_model}/{tag}'
    done = os.path.isfile(f'{results_dir}/done')

    test['vllm_args'] = { "kv_cache_dtype": args.default_kv_dtype } | test.get('vllm_args', {})
    info = { "devname": devname,
             "prompts": test.get("prompts", _PROMPTS),
             "config": test,
             "tag": tag,
             "num_gpus": num_gpus,
             "avail_vram": avail_vram,
             "total_vram": total_vram,
             "utilization": utilization }
        
    if not done:
        seqs_list = test.get("seqs", [ 1, 4 ])

        os.makedirs(results_dir, exist_ok=True)
        # A server already on the port (another suite, a leftover) would answer every bench request: ours would fail
        # to bind and the bench would quietly measure that one instead (2026-11-27: a test run's requests reached a
        # live suite's server as 404s)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(3)
            if probe.connect_ex(("127.0.0.1", int(args.port))) == 0:
                raise Exception(f"port {args.port} is already in use; stop that server or pass --port")
        print(f"*** STARTING VLLM FOR {test['model']}@{test['revision']} ACROSS {num_gpus} GPUS AT UTILIZATION {utilization:.2f} FOR {vram} GiB VRAM ***")
        #model_path = os.path.dirname(hf_hub_download(repo_id=test['model'], revision=test['revision'], filename="tokenizer.json"))
        model_path = snapshot_download(repo_id=test['model'], revision=test['revision'])
        print()

        vllm_env = { "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True" } | test.get('env', {})
        vargs = []
        for k,v in test['vllm_args'].items():
            vargs.append(f'--{k.replace("_","-")}')
            if v is not None:
                if isinstance(v, dict):
                    vargs.append(json.dumps(v))
                else:
                    vargs.append(str(v))
        srv = subprocess.Popen(["vllm", "serve",
                                test["model"],
                                "--gpu-memory-utilization", str(utilization),
                                "--tensor-parallel-size", str(num_gpus),
                                "--revision", test["revision"],
                                "--cpu-offload-gb", str(test.get("host_ram", 0) / num_gpus),
                                "--cpu-offload-params", "experts",
                                "--port", str(args.port),
                                "--language-model-only",
                                "--max-num-seqs", str(max(seqs_list))] + vargs,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               bufsize=0,
                               text=True,
                               env=os.environ.copy() | vllm_env)
        # The server's output is drained for the whole run, not only until startup: once nobody reads the pipe, the
        # server blocks in write() as soon as its log fills the pipe buffer (64 KiB, about 20 minutes of periodic
        # stats lines) and every request stalls with the GPU idle. Two 2026-11-27 runs hung that way at S4 random.
        started = threading.Event()

        def pump():
            with open(f"{results_dir}/serve.log", "w") as serve_log:
                for logline in srv.stdout:
                    serve_log.write(logline)
                    serve_log.flush()
                    if "Application startup complete." in logline:
                        started.set()
            started.set()   # EOF: the server exited

        pumper = threading.Thread(target=pump, name="serve-log", daemon=True)
        pumper.start()
        try:
            started.wait()
            if srv.poll() is not None:
                raise Exception(f"vLLM exited with return code {srv.returncode} before startup completed, see log")
            vllm = parse_log(f'{results_dir}/serve.log', info)
            if vllm.get('max_host_ram', 0) < test.get("host_ram", 0):
                raise Exception(f"vLLM found insufficient host RAM {vllm.get('max_host_ram', 0):.1f} / {test.get("host_ram", 0):.1f} GiB")
            with open(f"{results_dir}/test.json", "w") as f:
                json.dump(info, f)
            print(f"*** VLLM STARTED ***")
            info['vllm'] = vllm
            for seqs in seqs_list:
                run_bench(args, results_dir, model_path, seqs, info)
            if args.cat:
                for cat_name,prompt in info["prompts"].items():
                    print(f"*** GENERATING {cat_name} SVG ***")
                    run_cat(args, cat_name, prompt, results_dir, info)
            print("*** TERMINATING VLLM ***")
        finally:
            srv.terminate()
            srv.wait()
            pumper.join(timeout=60)

    with open(f"{results_dir}/done", "w") as f:
        f.close()
    
def run_cat(args, cat_name, prompt, results_dir, info):
    cat_file = f'{results_dir}/{cat_name}.svg'
    if not os.path.isfile(cat_file):
        with open(f"{cat_file}.log", "w") as cat:
            res = subprocess.run(["vllm", "chat",
                                "--stats",
                                "--url", f"http://localhost:{args.port}/v1",
                                "-q", prompt],
                                stdout=cat)
            if res.returncode == 0:
                cat_data = res.stdout
                with open(f"{cat_file}.log", "r") as cat:
                    cat_data = cat.read()
                    svg = re.findall(r"<svg[^`]*</svg>", cat_data)
                    if svg:               
                        with open(cat_file, "w") as cat:
                            cat.write(svg[-1])

def _trimmed_means(j, seqs):
    """Per-request means from --save-detailed results: (mean input tokens, mean output tokens, mean TTFT s, TPOT s),
    or None without per-request data. Above concurrency 1 the first `seqs` requests by start time are left out: they
    always begin as simultaneous prefills, whatever the config, so they say more about vllm bench than about the
    server; the rest arrive one at a time as seats free up. TPOT is weighted by tokens -- total decode time over total
    decode tokens -- not vLLM's per-request mean, in which an 8-token answer, all start-up and an MTP drafter that
    has not settled, counts as much as an 800-token one."""
    starts, ttfts, lats = j.get("start_times"), j.get("ttfts"), j.get("latencies")
    ins, outs = j.get("input_lens"), j.get("output_lens")
    if not (starts and ttfts and lats and ins and outs):
        return None
    keep = sorted(range(len(starts)), key=lambda i: starts[i])[seqs if seqs > 1 else 0:]
    keep = [i for i in keep if outs[i] and outs[i] > 1 and ttfts[i] is not None]
    if not keep:
        return None
    n = len(keep)
    tpot = sum(lats[i] - ttfts[i] for i in keep) / sum(outs[i] - 1 for i in keep)
    return (sum(ins[i] for i in keep) / n, sum(outs[i] for i in keep) / n,
            sum(ttfts[i] for i in keep) / n, tpot)


def parse_results(j, seqs):
    # guard: make sure concurrency was at least reached
    if j["max_concurrent_requests"] < seqs:
        raise Exception(f"ERROR: configured concurrency {seqs} not reached: max concurrent {j['max_concurrent_requests']}")

    # guard: make sure every request completed
    if j["failed"]:
        raise Exception(f"ERROR: {j['failed']} requests failed")

    # wall clock time per request
    total_reqs = j["completed"]
    total_s = j['duration']
    req_total_s = total_s / total_reqs

    trimmed = _trimmed_means(j, seqs)
    if trimmed:
        # the first wave dropped above concurrency 1; the pacing (req_total_s) still spans the whole run
        mean_in, mean_out, req_prefill_s, req_tpot_s = trimmed
        in_tok, out_tok = mean_in * total_reqs, mean_out * total_reqs
    else:
        out_tok = j['total_output_tokens']
        in_tok = j['total_input_tokens']
        req_tpot_s = j['mean_tpot_ms'] / 1000.0
        req_prefill_s = j['mean_ttft_ms'] / 1000.0

    # request-level decode stats
    req_gen_s = req_tpot_s * (out_tok / total_reqs - 1)
    
    # generation and prefill share of request time
    req_s = req_gen_s + req_prefill_s
    gen_share = req_gen_s / req_s
    prefill_share = req_prefill_s / req_s
    
    # approx request concurrency
    mean_concurrency = req_s / req_total_s
    gen_tps = out_tok / (total_s * gen_share)
    # Prefill is reported per request: the rate one prompt is ingested at under this load, mean input over mean
    # TTFT. The share model's aggregate (in_tok / (total_s * prefill_share)) reduces to concurrency x this, which
    # credits every stream with prefilling at once when the others are mostly decoding -- a property of the request
    # pattern, not a rate the server can sustain. Decode stays aggregate: the streams do decode concurrently.
    prefill_tps = (in_tok / total_reqs) / req_prefill_s
    
    accept = j.get('spec_decode_acceptance_rate', 0.0)

    return { "prefill_tps": prefill_tps,
             "gen_tps": gen_tps,
             "date": datetime.datetime.strptime(j['date'], '%Y%m%d-%H%M%S'),
             "accept": accept,
             "concur": mean_concurrency,
             "trimmed": bool(trimmed) and seqs > 1,
             "raw": j,
             "seqs": seqs }

def parse_log(logfile, test):
    info = {}
    with open(logfile, "r") as log:
        for logline in log.readlines():
            if "WDDM budget exceeded" in logline:
                raise Exception(logline)
            m = re.search(r"non-default args: (.*)", logline)
            if m:
                info['vllm_args'] = m.group(1)
            m = re.search(r" Available RAM: ([0-9.]+) GiB", logline)
            if m and 'max_host_ram' not in info:
                info['max_host_ram'] = float(m.group(1)) - 3
            m = re.search(r"expert placement: VRAM ([0-9.]+) GiB \(([0-9.]+)%\), host RAM ([0-9.]+) GiB \(([0-9.]+)%\) in ([0-9.]+) GiB pinned \(([0-9.]+) GiB of spare rows to swap through, ([0-9.]+) GiB idle\), disk ([0-9.]+) GiB \(([0-9.]+)%\)", logline)
            if m:
                if 'expert_vram' not in info:
                    info |= { "expert_vram": 0, 
                              "expert_vram_pct": 0,
                              "expert_host": 0,
                              "expert_host_idle": 0,
                              "expert_host_pct": 0,
                              "expert_disk": 0,
                              "expert_disk_pct": 0 }
                info['expert_vram'] += float(m.group(1))
                info['expert_vram_pct'] += float(m.group(2)) / test['num_gpus']
                info['expert_host'] += float(m.group(5)) - float(m.group(7))
                info['expert_host_idle'] += float(m.group(7))
                info['expert_host_pct'] += float(m.group(4)) / test['num_gpus']
                info['expert_disk'] += float(m.group(8))
                info['expert_disk_pct'] += float(m.group(9)) / test['num_gpus']
            m = re.search(r"max_seq_len=([0-9]+)", logline)
            if m and 'max_model_len' not in info:
                # the engine's final length; with --max-model-len auto vLLM re-fits it after warmup, so the last wins
                info['max_model_len'] = int(m.group(1))
            m = re.search(r"Auto-fit max_model_len: using max_model_len=([0-9]+)", logline)
            if m:
                info['max_model_len'] = int(m.group(1))
            m = re.search(r"Extended KV cache to [0-9]+ blocks \(([0-9.]+) GiB\)", logline)
            if m:
                info['kv_cache_gb'] = float(m.group(1))
            m = re.search(r"GPU KV cache size: ([0-9,]+) tokens", logline)
            if m:
                info['kv_cache'] = int(m.group(1).replace(',' , ''))
            m = re.search(r"on host: ([0-9.]+) GiB pinned", logline)
            if m:
                # technically could be blockq too
                if 'exl3_offload' not in info:
                    info['exl3_offload'] = 0
                info['exl3_offload'] += float(m.group(1))
            m = re.search(r"cpu-offload: ([0-9.]+) GiB of EXL3 weights", logline)
            if m:
                if 'exl3_offload' not in info:
                    info['exl3_offload'] = 0
                info['exl3_offload'] += float(m.group(1))
            m = re.search(r"Total CPU offloaded parameters: ([0-9.]+)", logline)
            if m:
                if 'dense_offload' not in info:
                    info['dense_offload'] = 0
                info['dense_offload'] += float(m.group(1))
            m = re.search(r"Model loading took ([0-9.]+) GiB memory", logline)
            if m:
                if 'weights_vram' not in info:
                    info['weights_vram'] = 0
                info['weights_vram'] += float(m.group(1))
            m = re.search(r"Free memory on device \(([0-9.]+)/([0-9.]+) GiB\) on startup. Desired GPU memory utilization is \([0-9.]+, ([0-9.]+) GiB\). Actual usage is ([0-9.]+) GiB for consumed memory \(weights \+ non-torch\), ([0-9.]+) GiB for peak activation, and ([0-9.]+) GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=[0-9.]+` \([0-9.]+ GiB\) to fit into requested memory, or `--kv-cache-memory=[0-9.]+` \([0-9.]+ GiB\) to fully utilize gpu memory. Current kv cache memory in use is ([0-9.]+) GiB.", logline)
            if m:
                if 'startup_vram_free' not in info:
                    info |= { "startup_vram_free": 0, "startup_vram_avail": 0, "vram_used": 0, "vram_target": 0 }
                info['startup_vram_free'] += float(m.group(1))
                info['startup_vram_avail'] += float(m.group(2))
                info['vram_used'] += float(m.group(4)) + float(m.group(5)) + float(m.group(6)) + float(m.group(7))
                info['vram_target'] += float(m.group(3))
    return info

def format_notes(test, res_random, res_sharegpt):
    #date = res_sharegpt['date'].strftime("%-m/%-d/%y %-H:%M")
    date = res_sharegpt['date'].strftime("%-m/%-d/%y")
    out_len = res_sharegpt['raw']['total_output_tokens'] / res_sharegpt['raw']['completed']
    return f"{res_sharegpt['raw']['num_prompts']} x {out_len:.0f} ({date})"

def build_results(args, model, testdir, results):
    with open(f'{testdir}/test.json', "r") as f:
        info = json.load(f)
        vllm = parse_log(f'{testdir}/serve.log', info)
        desired_host_ram = info['config'].get("host_ram", 0)
        avail_host_ram = vllm.get('max_host_ram', 0)
        if avail_host_ram < desired_host_ram:
            raise Exception(f"{info['tag']}: VLLM FOUND INSUFFICIENT HOST RAM {avail_host_ram:.1f} / {desired_host_ram:.1f} GiB")
        devname = info['devname']
        if info['num_gpus'] > 1:
            devname += f' x{info["num_gpus"]}'
        devname = devname.replace("NVIDIA ", "")
        devname = devname.replace("GeForce ", "")

        spec = None
        if "speculative_config" in info["config"].get("vllm_args", {}):
            spec = info["config"]["vllm_args"]["speculative_config"].get("num_speculative_tokens", 0)

        res_type = "normal"
        if "expert_vram" in vllm:
            if spec:
                res_type = "disk_spec"
            else:
                res_type = "disk"
        elif spec:
            res_type = "spec"

        for seqs in info['config'].get('seqs', [1,4]):
            res_sharegpt = {}
            results_file = f'{testdir}/bench_S{seqs}_sharegpt.json'
            with open(results_file, 'r') as f:
                j = json.load(f)
                res_sharegpt = parse_results(j, seqs)
            res_random = {}
            results_file = f'{testdir}/bench_S{seqs}_random.json'
            with open(results_file, 'r') as f:
                j = json.load(f)
                res_random = parse_results(j, seqs)
            long_note = ""
            long_file = f'{testdir}/bench_S{seqs}_long.json'
            if os.path.isfile(long_file):
                with open(long_file, 'r') as f:
                    j = json.load(f)
                    res_long = parse_results(j, seqs)
                    long_frac = res_long['gen_tps'] / res_sharegpt['gen_tps']
                    long_note = f" ({long_frac*100:.0f}% at {res_long['raw']['total_input_tokens'] / 1000:.0f}K)"

            host_ram = vllm.get("dense_offload", 0) + vllm.get("exl3_offload", 0)
            res_name = info['config']['revision']
            if 'suffix' in info['config']:
                res_name = f"{res_name} ({info['config']['suffix']})"
            vals = [ devname, res_name ]
            if res_type == "disk" or res_type == "disk_spec":
                vals += [ f"{vllm.get('vram_used'):.1f} GiB ({vllm['expert_vram_pct']:.1f}%)",
                        f"{host_ram - vllm['expert_host_idle']:.1f} GiB ({vllm['expert_host_pct']:.1f}%)",
                        f"{vllm['expert_disk']:.1f} GiB ({vllm['expert_disk_pct']:.1f}%)" ]
            else:
                vals += [ f"{vllm.get('vram_used'):.1f} GiB",
                        f"{host_ram:.1f} GiB" ]
            vals += [ f"{vllm['kv_cache'] / 1024:.0f}K ({vllm['kv_cache_gb']:.1f} GiB {info['config']['vllm_args']['kv_cache_dtype']})",
                    f"{seqs}",
                    f"{res_random['prefill_tps']:.0f} t/s",
                    f"{res_sharegpt['gen_tps']:.0f} t/s{long_note}" ]

            if res_type == "spec" or res_type == "disk_spec":
                vals.append(f"{spec} tok ({res_sharegpt['accept']:.1f}%)")
            if args.results_notes:
                vals.append(format_notes(info, res_random, res_sharegpt))
            results[res_type].append(vals)

        svgs = info.get('prompts', {"cat": "Draw a picture of a cute cat using SVG"} )
        for svg_name, svg_prompt in svgs.items():
            catfile = f"{testdir}/{svg_name}.svg.log"
            if os.path.isfile(catfile):
                catdir = os.path.basename(testdir)
                with open(catfile, "r") as cat:
                    for line in cat:
                        m = re.search(r"\(([0-9]+) tokens in ([0-9.]+)s\)", line)
                        if m:
                            tokens = m.group(1)
                            seconds = m.group(2)
                            results["svgs"].append(f"""
The following {svg_name} was produced by `{info['config']['model']}` at `{info['config']['revision']}` on a {devname} in {seconds} seconds, using {tokens} tokens (including thoughts, if any)

<img src="{catdir}/{svg_name}.svg" alt="{svg_prompt}">
""")

def display_result(args, model, tests):

    results = { "normal": [],
                "disk": [],
                "spec": [],
                "disk_spec": [],
                "svgs": [] }
    res_str = f"## Benchmarks for {model}\n"
    for testdir in tests:
        try:
            build_results(args, model, testdir, results)
        except Exception as e:
            print(f"SKIPPING TEST {testdir}: {e}")
    for res_type, res_list in results.items():
        if res_list and res_type != "svgs":
            res_str += "\n"
            if res_type == "disk" or res_type == "disk_spec":
                hdr = [ "Platform", "Type", "VRAM", "Host RAM", "Host Disk", "KV", "Req.", "Prefill", "Output" ]
            else:
                hdr = [ "Platform", "Type", "VRAM", "Host RAM", "KV", "Req.", "Prefill", "Output" ]
            if res_type == "disk_spec" or res_type == "spec":
                hdr += [ "Draft" ]
            if args.results_notes:
                hdr += [ "Notes" ]
            res_str += "|" + "|".join(hdr) + "|\n"
            res_str += ("|---" * len(hdr)) + "|\n"
            for res in res_list:                            
                res_str += "|" + "|".join(res) + "|\n"

    res_str += """
- **Platform**: The GPU(s) that generated the benchmark.
- **Type**: The configuration type: quantization, etc.
- **VRAM**: Total VRAM used per vLLM's accounting. For MoEs, the share of experts resident in VRAM is shown in percent.
- **Host RAM**: Total host RAM used for offload. For MoEs, the share of experts resident in host RAM is shown in percent.
- **Host Disk**: For MoEs which serve some experts from disk, the size and percent of this amount.
- **KV**: KV size in tokens as served, the data type, and the total VRAM used.
- **Req**: Number of concurrent requests.
- **Prefill**: The average prefill rate for each request, not including concurrency.
- **Output**: The total output rate, including concurrency. Output rate at longer context is shown as a percent.
- **Draft**: Details of draft model configuration and measured acceptance rate.
"""

    if args.results_notes:
        res_str += "**Notes**: n x m -> n is the number of prompts used to measure output rate, m is the average output length in tokens.\n"

    if results["svgs"]:
        res_str += "\n## SVGs\n"
        for svg in results["svgs"]:
            res_str += svg
    return res_str

def display_results(args):
    done = {}
    for dir in sorted(glob.glob("results/*/*")):
        if os.path.isdir(dir):
            donefile=f"{dir}/done"
            if os.path.isfile(donefile):
                try:
                    testdir = os.path.dirname(donefile)
                    with open(f'{testdir}/test.json', "r") as f:
                        info = json.load(f)
                        base_model = info["config"]["model"].split("/")[-1]
                        if base_model not in done:
                            done[base_model] = []
                        done[base_model].append(testdir)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
            else:
                print(f"SKIPPING INCOMPLETE {dir}")

    for model,tests in done.items():
        print(f"PROCESSING MODEL {model}")
        try:
            with open(f'results/{model}/results.md', "w") as out:
                out.write(display_result(args, model, tests))        
        except Exception as e:
            print("SKIPPING MODEL {model}: {e}")
        
def random_output_len(args, results_dir, results_file, seqs):
    """The random run's mean output length. Under concurrency, a request's TTFT depends on how many other prompts are
    prefilling beside it, which depends on how long the others spend decoding relative to prefilling: a config that
    decodes twice as fast, or prefills half as fast, sees far more overlapping prefills at a fixed output length. So
    at seqs > 1 the output length targets a fixed ratio of decode time to prefill time per request,
    `--decode-ratio`: prefill time from this config's own 1x random run (realistic prompts, nothing overlapping),
    decode time from the 1x random run's TPOT scaled by how sharegpt's slows at this concurrency. The 1x run itself keeps
    --output-tokens: with nothing concurrent it can decode at whatever rate it has. Recorded next to the result."""
    if seqs <= 1 or not args.decode_ratio:
        return args.output_tokens
    s1 = f"{results_dir}/bench_S1_random.json"
    if not os.path.isfile(s1):
        print(f"*** RANDOM x{seqs}: no 1x random run to scale against; keeping {args.output_tokens} output tokens ***")
        return args.output_tokens
    with open(s1) as f:
        j1 = json.load(f)
    # seconds of prefill per input token, unloaded
    prefill_s_per_tok = (j1["mean_ttft_ms"] / 1000.0) / (j1["total_input_tokens"] / j1["completed"])
    prefill_s = prefill_s_per_tok * (args.prefix + args.input_tokens)
    # The decode rate the random run will see: random content decodes unlike real text (degenerate output, other
    # routing and draft acceptance), so the 1x random run's TPOT, scaled by how this config's decode slows at this
    # concurrency on real text (sharegpt here against sharegpt at 1x). Neither alone will do: sharegpt has the wrong
    # content, and the 1x random rate would scale each config by its unloaded speed, which concurrency changes
    # unevenly (MTP gains far more at 1 stream than at 4).
    with open(f"{results_file}_sharegpt.json") as f:
        sharegpt_tpot_ms = json.load(f)["mean_tpot_ms"]
    s1_sharegpt = f"{results_dir}/bench_S1_sharegpt.json"
    tpot_ms, how = sharegpt_tpot_ms, "sharegpt"
    if j1.get("mean_tpot_ms") and os.path.isfile(s1_sharegpt):
        with open(s1_sharegpt) as f:
            s1_sharegpt_tpot_ms = json.load(f)["mean_tpot_ms"]
        if s1_sharegpt_tpot_ms:
            tpot_ms = j1["mean_tpot_ms"] * sharegpt_tpot_ms / s1_sharegpt_tpot_ms
            how = f"1x random {j1['mean_tpot_ms']:.1f} x sharegpt {sharegpt_tpot_ms:.1f}/{s1_sharegpt_tpot_ms:.1f}"
    output_len = max(64, min(args.max_output_tokens, round(args.decode_ratio * prefill_s * 1000 / tpot_ms)))
    with open(f"{results_file}_random.scale.json", "w") as f:
        json.dump({"decode_ratio": args.decode_ratio, "s1_prefill_s": prefill_s, "tpot_ms": tpot_ms,
                   "tpot_from": how, "output_len": output_len}, f)
    print(f"*** RANDOM x{seqs}: {output_len} output tokens, {args.decode_ratio:g}x the {prefill_s:.2f} s unloaded "
          f"prefill at {tpot_ms:.1f} ms/token ({how}) ***")
    return output_len

def truncated_dataset(path, model_path, prompt_len, out_path):
    """A copy of a custom dataset whose prompts are cut to `prompt_len` tokens of this model's tokenizer, from the end
    (the instruction leads the prompt). vllm bench cannot do it: its --input-len maps only to --random-input-len, and
    the custom dataset sends each prompt as stored, tokenizing it only to count."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    with open(path) as f, open(out_path, "w") as out:
        for line in f:
            item = json.loads(line)
            ids = tok(item["prompt"], add_special_tokens=False)["input_ids"]
            cut = prompt_len
            text = tok.decode(ids[:cut])
            # decoding and re-encoding can merge or split tokens at the cut; trim until it fits
            while cut > 0 and len(tok(text, add_special_tokens=False)["input_ids"]) > prompt_len:
                cut -= 16
                text = tok.decode(ids[:cut])
            out.write(json.dumps(item | {"prompt": text}) + "\n")
    print(f"*** LONG PROMPT: cut to {prompt_len} tokens of {len(ids)} ***")
    return out_path

def run_bench(args, results_dir, model_path, seqs, info):
    num_prompts = args.num_prompts * seqs
    results_file = f'{results_dir}/bench_S{seqs}'

    for bench in "long", "sharegpt", "random":
        if bench == "long" and (seqs > 1 or not args.long_prompt or 'kv_cache' not in info['vllm']):
            continue
        outfile = f"{results_file}_{bench}.json"
        if not os.path.isfile(outfile):
            print(f"*** RUNNING BENCH FOR {bench} x{seqs} ***")
            with open(f"{results_file}_{bench}.log", "w") as log:
                bench_args = ["vllm", "bench", "serve",
                                "--model", info["config"]["model"],
                                "--max-concurrency", str(seqs),
                                "--tokenizer", model_path,
                                "--host", "localhost",
                                "--port", str(args.port),
                                "--seed", str(args.seed),
                                "--save-result", "--save-detailed", "--result-filename", outfile ]
                if bench == "long":
                    # only one long prompt without warmup, sized to fit KV, the context length and templating
                    room = min(info['vllm']['kv_cache'], info['vllm'].get('max_model_len', info['vllm']['kv_cache']))
                    prompt_len = min(room, args.long_prompt) - args.sharegpt_output_tokens - 128
                    bench_args += [ "--backend", "openai-chat",
                                    "--num-prompts", "1",
                                    "--endpoint", "/v1/chat/completions",
                                    "--dataset-name", "custom",
                                    "--custom-output-len", str(args.sharegpt_output_tokens),
                                    "--dataset-path", truncated_dataset(args.long_dataset, model_path, prompt_len,
                                                                        f"{results_file}_long.prompt.jsonl") ]
                if bench == "sharegpt":
                    bench_args += [ "--backend", "openai-chat",
                                    "--num-prompts", str(num_prompts * 2),
                                    "--endpoint", "/v1/chat/completions",
                                    "--dataset-name", "sharegpt",
                                    # One warm-up request: warms JIT paths and the expert cache
                                    "--num-warmups", str(args.warmups),
                                    # every request the same budget, ending at this model's own EOS: by default each
                                    # is capped at the length of the dataset's recorded GPT answer, as short as 8 tokens
                                    "--sharegpt-output-len", str(args.sharegpt_output_tokens),
                                    "--dataset-path", args.sharegpt_dataset ]
                if bench == "random":
                    output_len = random_output_len(args, results_dir, results_file, seqs)
                    bench_args += [ "--random-prefix-len", str(args.prefix),
                                    "--random-range-ratio", f'{{"input": {args.rand}, "output": {args.rand}}}',
                                    # One warm-up request, the run's own first prompt: the random dataset's shared prefix
                                    # is drawn after the per-request lengths, so it differs between invocations and only a
                                    # request from this run's sample caches it; it also warms JIT paths and the expert cache
                                    "--num-warmups", str(args.warmups),
                                    "--num-prompts", str(num_prompts),
                                    "--output-len", str(output_len),
                                    "--input-len", str(args.input_tokens) ]
                    if args.random_ignore_eos:
                        # random-token prompts have so far decoded every requested token anyway (Qwen3.6-35B-A3B: 2497 of
                        # 2497 at S1, 17199 of 17199 at S4); this makes the decode length a guarantee, not a property of
                        # how rarely a model emits EOS after noise, since the scaling above relies on it
                        bench_args += [ "--ignore-eos" ]
                res = subprocess.run(bench_args,
                                    stdout=log,
                                    stderr=subprocess.STDOUT,
                                    text=True)
                if res.returncode != 0:
                    raise Exception(f"vllm bench {bench} returned {res.returncode}")

def main():
    pynvml.nvmlInit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--testfile", help="testfile to run. if not specified, just regenerates results")
    parser.add_argument("--tests", help="restrict tests to matching pattern")
    parser.add_argument("--results-notes", action=argparse.BooleanOptionalAction, help="include extra notes in results")
    parser.add_argument("--long-dataset", default="datasets/Little_Brother.jsonl")
    parser.add_argument("--long-prompt", type=int, default=65536)
    parser.add_argument("--port", default=13373)
    parser.add_argument("--rand", type=float, default=0.5)
    parser.add_argument("--prefix", type=int, default=4096)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=1024,
                        help="random dataset output length at concurrency 1, and at higher concurrency with --decode-ratio 0")
    parser.add_argument("--decode-ratio", type=float, default=12.0,
                        help="at concurrency > 1, scale the random dataset's output length so each request decodes for "
                             "this many times its prefill time (prefill time and decode rate from the 1x random run, the "
                             "decode rate scaled to this concurrency by sharegpt's); 0 keeps --output-tokens. 12 is about "
                             "what 1024 tokens gave before")
    parser.add_argument("--max-output-tokens", type=int, default=8192,
                        help="ceiling on the scaled output length")
    parser.add_argument("--sharegpt-dataset", default="datasets/ShareGPT_V3_unfiltered_cleaned_split.json")
    parser.add_argument("--sharegpt-output-tokens", type=int, default=512,
                        help="sharegpt requests' output budget; each ends at the model's EOS or here, instead of at the "
                             "length of the dataset's recorded answer")
    parser.add_argument("--warmups", type=int, default=1,
                        help="warm-up requests per bench run, from the run's own first prompt (vllm bench --num-warmups)")
    parser.add_argument("--random-ignore-eos", default=True, action=argparse.BooleanOptionalAction,
                        help="random dataset requests always decode their full output length (they have so far "
                             "anyway; this guarantees it)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--default-kv-dtype", default="nvfp4")
    parser.add_argument("--cat", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.testfile:
        run_tests(args)
    display_results(args)

if __name__ == "__main__":
    main()
