"""Second stress batch: "forgotten data" -- content agents and users handle constantly that the
bundled corpus does not carry. Same scoring as build.py: 2048-token rows of non-special tokens,
the last 1024 scored (the split is done here). Measurement only: the logs, errors and terminal
families come from this host and are not calibration material.

CommitPackFT families take file contents (new_contents) from the eval-split repositories
(ctx_trace's crc32 rule) where the diff kind samples that language, so the diff calibration
never saw these files; languages the diff kind does not sample take any repository."""
import glob, json, os, random, subprocess, sys, tempfile, zlib
from transformers import AutoTokenizer

OUT = os.environ.get("STRESS_OUT", os.path.dirname(os.path.abspath(__file__)))
S = "/home/ypell/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-9B/snapshots/489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
WORK = "/home/bulk/ypell/quant_work"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "deps", "exllamav3"))
from ctx_trace import DIFF_LANGS, DIFF_LICENSES
L, N = 2048, 8
rng = random.Random(11)
tok = AutoTokenizer.from_pretrained(S)
BANNED = set(tok.added_tokens_decoder)
enc = lambda t: tok(t, add_special_tokens = False, split_special_tokens = True)["input_ids"]
rows = []

def add(family, text, n = N, **meta):
    ids = enc(text)
    slots = [k for k in range(len(ids) // L) if not BANNED & set(ids[k * L:(k + 1) * L])]
    if len(slots) < n:
        print(f" !! {family}: {len(slots)} clean windows of {n}"); n = len(slots)
    for k in sorted(rng.sample(slots, n)):
        w = ids[k * L:(k + 1) * L]
        rows.append({"family": "forgotten", "kind": family, **meta, "input_ids": w[:L // 2], "response_ids": w[L // 2:]})

from huggingface_hub import HfFileSystem
fs = HfFileSystem()
def commitpack_files(lang, want_chars = 400_000, max_lines = 4000):
    held_out = lang in DIFF_LANGS
    out, n, seen = [], 0, set()
    with fs.open(f"datasets/bigcode/commitpackft/data/{lang}/data.jsonl") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line: break
            r = json.loads(line); repo = r["repos"].split(",")[0]
            if r["license"] not in DIFF_LICENSES or (held_out and zlib.crc32(repo.encode()) % 5 != 0):
                continue
            body = r["new_contents"]
            if not body.strip() or body in seen or max(map(len, body.splitlines() or [""])) > 400:
                continue
            seen.add(body); out.append(body); n += len(body)
            if n > want_chars: break
    print(f" -- {lang}: {len(out)} files, {n // 1000} k chars ({'held-out repos' if held_out else 'any repo'})")
    return "\n\n".join(out)

# --- controls --------------------------------------------------------------------------------
from datasets import load_dataset
add("control_wikitext", "".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = "validation")["text"]))
add("control_python", commitpack_files("python"))

# --- CommitPackFT file contents ---------------------------------------------------------------
for lang, fam in [("sql", "sql"), ("yaml", "yaml"), ("toml", "toml"), ("ini", "ini"), ("json", "json"), ("csv", "csv"),
                  ("jupyter-notebook", "notebook"), ("markdown", "markdown"), ("dockerfile", "dockerfile"), ("makefile", "makefile"),
                  ("restructuredtext", "rst")]:
    add(fam, commitpack_files(lang))

# --- logs: our own runs (convert, qbench, ctx_trace), progress bars and all ---------------------
logs = sorted(glob.glob(f"{WORK}/*/*.log") + glob.glob(f"{WORK}/*/logs/*.log")); rng.shuffle(logs)
add("run_logs", "\n".join(open(p, errors = "replace").read()[-20000:] for p in logs[:120]))

# --- errors: harvested Python tracebacks, plus real compiler and interpreter errors -------------
tb = []
for p in logs:
    t = open(p, errors = "replace").read()
    i = t.find("Traceback (most recent call last)")
    while i >= 0:
        j = t.find("\n\n", i); tb.append(t[i:j if j > 0 else i + 6000]); i = t.find("Traceback (most recent call last)", i + 1)
SNIPPETS = ["import json; json.loads('{\"a\": [1, 2,}')", "import collections; collections.OrderedDict(1, 2, 3)",
            "d = {'a': 1}; print(d['b'])", "import os; os.listdir('/nonexistent/dir')", "x = [1, 2]; x[5]",
            "import re; re.compile('(?P<x>a)(?P<x>b)')", "int('12abc')", "import math; math.sqrt(-1)",
            "def f(n): return f(n + 1)\nf(0)", "import datetime; datetime.date(2026, 2, 30)",
            "from decimal import Decimal; Decimal('1.2.3')", "import struct; struct.unpack('<I', b'ab')",
            "class A: pass\nA().missing()", "import subprocess; subprocess.run(['false'], check = True)",
            "import torch; torch.zeros(2, 3) @ torch.zeros(4, 5)", "import numpy as np; np.zeros((2, 3)).reshape(4, 4)"]
for sn in SNIPPETS:
    tb.append(subprocess.run([sys.executable, "-c", sn], capture_output = True, text = True).stderr)
CPP = ["#include <vector>\nint main() { std::vector<int> v; v.push_back(\"x\"); return v.size() }",
       "#include <map>\n#include <string>\nint main() { std::map<std::string, int> m; m[1] = 2; auto x = m.find(3); }",
       "template <typename T> struct S { T t; };\nint main() { S<void> s; undefined_fn(s); }",
       "#include <memory>\nstruct B { virtual void f() = 0; };\nint main() { auto p = std::make_unique<B>(); p->g(); }"]
with tempfile.TemporaryDirectory() as d:
    for k, src in enumerate(CPP):
        f = os.path.join(d, f"broken_{k}.cpp"); open(f, "w").write(src)
        tb.append(subprocess.run(["g++", "-std=c++20", "-c", f, "-o", os.devnull], capture_output = True, text = True).stderr)
rng.shuffle(tb)
add("errors", "\n\n".join(tb * 3))       # few distinct errors: repeated to fill windows, at shuffled offsets

# --- terminal output: harmless commands on this host ---------------------------------------------
CMDS = ["ls -la /usr/lib64 /usr/share/doc", "df -h", "lsblk", "ps aux --sort=-%mem", "pip list --format=columns",
        "git -C /home/ypell/git/vllm-exl3-plugin/deps/vllm log --stat -n 60", "systemctl list-units --type=service --no-pager",
        "rpm -qa --qf '%{NAME} %{VERSION}-%{RELEASE} %{ARCH}\\n'", "nvidia-smi", "free -h", "uname -a", "lscpu", "mount",
        "find /usr/share/fonts -maxdepth 2", "du -sh /usr/share/* 2>/dev/null"]
term = "".join(f"$ {c}\n" + subprocess.run(c, shell = True, capture_output = True, text = True, errors = "replace").stdout[:30000] for c in CMDS)
add("terminal", term)

json.dump({"rows": rows, "source": "quantization/stress/forgotten.py",
           "vocab_size": json.load(open(f"{WORK}/_orn9_agent/eval_wild.json"))["vocab_size"]},
          open(f"{OUT}/forgotten.json", "w"))
from collections import Counter
print(len(rows), "rows", dict(Counter(r["kind"] for r in rows)))
