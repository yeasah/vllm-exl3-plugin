#!/usr/bin/env python3
"""Screen a box before you spend the rental.

`checkpoint_survey.py` screens a checkpoint before you spend the bandwidth; this
screens a *host* before you spend hours of GPU time on it. It exists because of a
distinction the project had to learn twice: for an output-based benchmark (qbench,
`capability-suite`) almost nothing about the host matters, while for a throughput
benchmark almost everything does.

What can change the tokens a model emits is a short list -- GPU architecture and
driver decide which kernels vLLM selects, GPU *count* decides the tensor-parallel
degree and therefore cross-rank reduction order, VRAM decides KV cache size and so
the scheduling that batches requests, and uncorrected ECC errors corrupt weights
silently. Everything else -- PCIe width, clocks, host CPU, RAM, cooling -- moves
throughput and leaves the tokens alone. So this tool does not just dump statistics:
it *classifies* them, and when comparing two boxes it says which differences can
move a result and which can only move a stopwatch.

Deliberately stdlib-only and single-file: it has to run on a bare box before torch
or vLLM are installed, which is exactly when the answer is still cheap to act on.
`scp tools/host_survey.py box: && python3 host_survey.py` is the intended use.

Exit status: 0 usable, 1 disqualifying (uncorrected ECC, no GPU visible, or a
comparison mismatch in an output-relevant field), 2 usable with warnings.

    python3 host_survey.py                  # human-readable report
    python3 host_survey.py --json > box.json
    python3 host_survey.py --compare box.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys

#: Fields where a difference between two boxes can change the *tokens produced*.
#: Everything not listed here is throughput-only. Keeping this list explicit is the
#: whole point of the tool -- it is the project's claim about what portability means,
#: written where it can be checked rather than remembered.
OUTPUT_RELEVANT = (
    "gpu_name",
    "gpu_count",
    "compute_cap",
    "driver_version",
    "cuda_version",
    "vram_mib",
    "ecc_uncorrected",
)

_SMI_FIELDS = [
    "name",
    "compute_cap",
    "memory.total",
    "driver_version",
    "ecc.mode.current",
    "ecc.errors.corrected.aggregate.total",
    "ecc.errors.uncorrected.aggregate.total",
    "pcie.link.gen.max",
    "pcie.link.width.max",
    "persistence_mode",
]


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    return out.stdout if out.returncode == 0 else None


def _na(v: str) -> str | None:
    """nvidia-smi says 'N/A' for anything the card does not implement (ECC on
    consumer parts, most notably). Absent and zero are different answers."""
    v = v.strip()
    return None if v in ("N/A", "[N/A]", "", "Not Supported", "[Not Supported]") else v


def collect() -> dict:
    data: dict = {"host": platform.node(), "kernel": platform.release()}

    if shutil.which("nvidia-smi") is None:
        data["error"] = "nvidia-smi not found"
        return data

    raw = _run(["nvidia-smi", f"--query-gpu={','.join(_SMI_FIELDS)}",
                "--format=csv,noheader"])
    gpus = []
    for line in (raw or "").strip().splitlines():
        if not line.strip():
            continue
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(_SMI_FIELDS):
            continue
        g = dict(zip(_SMI_FIELDS, vals))
        gpus.append(g)
    data["gpus"] = gpus

    if not gpus:
        data["error"] = "no GPU reported by nvidia-smi"
        return data

    data["gpu_count"] = len(gpus)
    data["gpu_name"] = gpus[0]["name"]
    data["compute_cap"] = gpus[0]["compute_cap"]
    data["driver_version"] = gpus[0]["driver_version"]
    mem = _na(gpus[0]["memory.total"]) or ""
    m = re.match(r"(\d+)", mem)
    data["vram_mib"] = int(m.group(1)) if m else None

    # Heterogeneous multi-GPU is its own disqualifier: TP across unlike cards is
    # not a configuration this project has ever validated.
    data["gpu_homogeneous"] = len({g["name"] for g in gpus}) == 1

    def _ecc_sum(key: str):
        vals = [_na(g[key]) for g in gpus]
        if all(v is None for v in vals):
            return None  # card has no ECC at all -- not the same as "zero errors"
        return sum(int(v) for v in vals if v is not None and v.isdigit())

    data["ecc_mode"] = _na(gpus[0]["ecc.mode.current"])
    data["ecc_corrected"] = _ecc_sum("ecc.errors.corrected.aggregate.total")
    data["ecc_uncorrected"] = _ecc_sum("ecc.errors.uncorrected.aggregate.total")

    # Throughput-only, collected so a slow box can be diagnosed rather than guessed at
    data["pcie_gen"] = _na(gpus[0]["pcie.link.gen.max"])
    data["pcie_width"] = _na(gpus[0]["pcie.link.width.max"])
    data["persistence_mode"] = _na(gpus[0]["persistence_mode"])

    # The header spells this differently across driver generations ("CUDA Version"
    # on older ones, "CUDA UMD Version" since 6xx), so accept either.
    header = _run(["nvidia-smi"]) or ""
    m = re.search(r"CUDA(?:\s+UMD)?\s+Version:\s*([0-9.]+)", header)
    data["cuda_version"] = m.group(1) if m else None

    # Optional enrichment: absent on a fresh box, and that is fine
    for mod, key in (("torch", "torch"), ("vllm", "vllm")):
        try:
            data[key] = __import__(mod).__version__
        except Exception:
            data[key] = None
    return data


def fingerprint(data: dict) -> str:
    """Short digest over the output-relevant fields only.

    Two boxes with the same fingerprint should produce the same tokens for the same
    checkpoint and flags; two with different fingerprints may not, and the comparison
    below says which field moved.
    """
    payload = json.dumps({k: data.get(k) for k in OUTPUT_RELEVANT}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def verdict(data: dict) -> tuple[int, list[str]]:
    """Exit code plus the reasons, worst first."""
    bad, warn = [], []
    if data.get("error"):
        return 1, [data["error"]]
    if data.get("ecc_uncorrected"):
        bad.append(f"{data['ecc_uncorrected']} uncorrected ECC errors -- refuse this box")
    if not data.get("gpu_homogeneous", True):
        bad.append("GPUs are not identical; TP across unlike cards is unvalidated here")
    if data.get("ecc_corrected"):
        warn.append(
            f"{data['ecc_corrected']} corrected ECC errors -- the card is healing "
            "faults; watch it, and prefer another box for a multi-day run")
    if data.get("ecc_mode") is None:
        warn.append("no ECC on this GPU: silent corruption cannot be detected here")
    if data.get("persistence_mode") == "Disabled":
        warn.append("persistence mode disabled (throughput only)")
    return (1 if bad else 2 if warn else 0), bad + warn


def report(data: dict) -> None:
    if data.get("error"):
        print(f"  !! {data['error']}")
        return
    n = data["gpu_count"]
    print(f"  host          {data['host']}  (kernel {data['kernel']})")
    print(f"  fingerprint   {fingerprint(data)}   <- equal fingerprints should give equal tokens")
    print()
    print("  output-relevant (a difference here can change results)")
    print(f"    gpu           {n} x {data['gpu_name']}")
    print(f"    compute cap   {data['compute_cap']}")
    print(f"    vram          {data['vram_mib']} MiB each")
    print(f"    driver        {data['driver_version']}    cuda {data['cuda_version']}")
    ecc = data["ecc_uncorrected"]
    print(f"    ecc           mode={data['ecc_mode']} corrected={data['ecc_corrected']} "
          f"uncorrected={ecc}")
    print()
    print("  throughput-only (will not change what the model emits)")
    print(f"    pcie          gen{data['pcie_gen']} x{data['pcie_width']}")
    print(f"    persistence   {data['persistence_mode']}")
    print(f"    torch/vllm    {data['torch']} / {data['vllm']}")
    print()
    code, notes = verdict(data)
    for line in notes:
        print(f"  {'!!' if code == 1 else '--'} {line}")
    print(f"  verdict       {['usable', 'REFUSE', 'usable, with warnings'][code]}")


def compare(data: dict, baseline: dict) -> int:
    out_diff, perf_diff = [], []
    keys = sorted(set(data) | set(baseline))
    for k in keys:
        if k in ("gpus", "host", "kernel"):
            continue
        a, b = baseline.get(k), data.get(k)
        if a != b:
            (out_diff if k in OUTPUT_RELEVANT else perf_diff).append((k, a, b))
    print(f"  baseline {fingerprint(baseline)}  ->  this box {fingerprint(data)}")
    print()
    if out_diff:
        print("  !! output-relevant differences -- results are NOT comparable:")
        for k, a, b in out_diff:
            print(f"       {k:16s} {a}  ->  {b}")
    else:
        print("  -- no output-relevant differences: results are comparable")
    if perf_diff:
        print("\n  -- throughput-only differences (fine for capability, not for perf):")
        for k, a, b in perf_diff:
            print(f"       {k:16s} {a}  ->  {b}")
    return 1 if out_diff else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit the raw survey as JSON")
    ap.add_argument("--fingerprint", action="store_true",
                    help="print only the output-relevant digest")
    ap.add_argument("--compare", metavar="FILE",
                    help="compare against a survey JSON, classifying each difference")
    args = ap.parse_args()

    data = collect()
    if args.fingerprint:
        print(fingerprint(data))
        return verdict(data)[0]
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return verdict(data)[0]
    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)
        rc = compare(data, baseline)
        return rc or verdict(data)[0]
    report(data)
    return verdict(data)[0]


if __name__ == "__main__":
    sys.exit(main())
