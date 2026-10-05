"""Stress-test traces: content chosen to miss the calibration corpus, scored at every position
(raw-text style) except the render rows. Non-special tokens only: no added token appears in a
scored span, and typed markup is tokenized with special-token parsing off.

Families: control (held-out English wikitext), F1 repetition, F3 scripts/symbols, F4 data
formats, F5 structural, F6 long context (separate trace, for position buckets).

Not a portable tool: most F3-F5 sources are files on the host it was written on (/usr/share,
a VS Code server's bundles, the dnf history database, the deps/vllm git log). Wikipedia,
wikitext and pg19 come from the Hub. Results and the scoring layout: docs/calibration.md
"Where a calibration misses". The traces it writes score every position; the run used the
last 1024 of each 2048-token row and 256-token windows of the long rows (see the doc)."""
import base64, glob, io, json, os, random, sqlite3, subprocess
from transformers import AutoTokenizer

OUT = os.environ.get("STRESS_OUT", os.path.dirname(os.path.abspath(__file__)))
S = "/home/ypell/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-9B/snapshots/489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
VLLM = "/home/ypell/git/vllm-exl3-plugin/deps/vllm"
VSC = sorted(glob.glob("/home/ypell/.vscode-server/cli/servers/Stable-*/server/extensions"))[-1]
L, LONG, N = 2048, 32768, 8
rng = random.Random(7)
tok = AutoTokenizer.from_pretrained(S)
BANNED = set(tok.added_tokens_decoder)
VOCAB = json.load(open("/home/bulk/ypell/quant_work/_orn9_agent/eval_wild.json"))["vocab_size"]   # ctx_trace convention for this model

def enc(text):
    return tok(text, add_special_tokens = False, split_special_tokens = True)["input_ids"]

def windows(text, n = N, length = L):
    """n non-overlapping token windows at random offsets of one tokenized corpus."""
    ids = enc(text)       # windows holding an added token (e.g. a literal <think> in a diff) are skipped
    slots = [k for k in range(len(ids) // length) if not BANNED & set(ids[k * length:(k + 1) * length])]
    assert len(slots) >= n, f"corpus too short: {len(ids)} tokens, {len(slots)} clean windows for {n} x {length}"
    return [ids[k * length:(k + 1) * length] for k in sorted(rng.sample(slots, n))]

rows_short, rows_long = [], []
def raw(family, kind, ids, out = rows_short, **meta):
    assert not BANNED & set(ids), f"added token in {family}/{kind}"
    out.append({"family": family, "kind": kind, **meta, "input_ids": ids[:1], "response_ids": ids[1:]})

def files_text(paths, limit = 4_000_000):
    out, n = [], 0
    for p in paths:
        try:
            t = open(p, encoding = "utf8", errors = "strict").read()
        except Exception:
            continue
        out.append(t); n += len(t)
        if n > limit: break
    return "\n".join(out)

# --- control: held-out English (wikitext-2 validation, the eval pool) ------------------------
from datasets import load_dataset
wt = "".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = "validation")["text"])
for ids in windows(wt): raw("control", "wikitext", ids)

# --- F1 repetition ----------------------------------------------------------------------------
for w in [" the", ".", "0", " cat", "\n", "a", "的", " =="]:
    t = enc(w); raw("F1", "rep_token", (t * L)[:L], unit = w)
vocab = [i for i in range(len(tok)) if i not in BANNED and tok.decode([i]).strip()]
for _ in range(N):
    cyc = rng.sample(vocab, rng.randint(2, 8)); raw("F1", "rep_cycle", (cyc * L)[:L], period = len(cyc))
sents = [s.strip() + "." for s in wt.split(". ") if 60 < len(s) < 200 and "@" not in s and "=" not in s]
for s in rng.sample(sents, N):
    t = enc(s + " "); raw("F1", "rep_phrase", (t * (L // len(t) + 1))[:L])

# --- F3 scripts and symbols -------------------------------------------------------------------
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
def wiki(lang):
    d = f"datasets/wikimedia/wikipedia/20231101.{lang}"
    path = sorted(fs.ls(d, detail = False))[0]
    arts = pq.ParquetFile(fs.open(path)).read_row_group(0, columns = ["text"]).column("text").to_pylist()
    arts = [a for a in arts if len(a) > 1500]; rng.shuffle(arts)
    text, ids = "", []
    for a in arts:
        text += a + "\n\n"
        if len(text) > 6 * L * 4: break
    return enc(text)
for tier, langs in (("wiki_mid", ["ru", "ar", "hi", "el", "he", "th", "fa", "uk"]),
                    ("wiki_rare", ["ka", "hy", "am", "bo", "my", "km", "si", "dv"])):
    for lg in langs:
        ids = wiki(lg); assert len(ids) >= L + 200, (lg, len(ids))
        o = rng.randrange(0, len(ids) - L); raw("F3", tier, ids[o:o + L], lang = lg)
trees = [subprocess.run(["tree", "-a", "-L", "4", r], capture_output = True, text = True).stdout for r in
         ["/usr/share/doc", "/usr/share/icons", "/home/ypell/.venv/lib64/python3.12/site-packages/torch",
          "/usr/share/locale", f"{VLLM}/vllm", "/usr/share/X11", "/usr/share/emacs", "/usr/lib64/python3.12"]]
for t in trees:
    ids = enc(t); o = rng.randrange(0, max(1, len(ids) - L)); raw("F3", "tree", ids[o:o + L])
def mathify(s):
    styles = [(0x1D434, 0x1D44E), (0x1D400, 0x1D41A), (0x1D56C, 0x1D586), (0x1D538, 0x1D552)]   # italic, bold, bold fraktur, double-struck
    up, lo = rng.choice(styles)
    hole = {0x1D455: "ℎ", 0x1D53A: "ℂ", 0x1D53F: "ℍ", 0x1D545: "ℕ", 0x1D547: "ℙ", 0x1D548: "ℚ", 0x1D549: "ℝ", 0x1D551: "ℤ"}
    out = []
    for c in s:
        cp = (up + ord(c) - 65) if "A" <= c <= "Z" else (lo + ord(c) - 97) if "a" <= c <= "z" else None
        out.append(c if cp is None else hole.get(cp, chr(cp)))
    return "".join(out)
for ids in windows(mathify(wt[:400_000]), 4): raw("F3", "math_alnum_synth", ids)
EMO = [chr(c) for r in ((0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F680, 0x1F6FF), (0x1F900, 0x1F9FF)) for c in range(*r)]
for _ in range(4):
    parts = []
    while len(parts) < 3000:
        k = rng.random()
        parts.append("‍".join(rng.choices(EMO, k = rng.randint(2, 4))) if k < 0.15 else rng.choice(EMO) + ("️" if k < 0.3 else ""))
        if rng.random() < 0.2: parts.append(rng.choice([" ", "\n"]))
    ids = enc("".join(parts)); raw("F3", "emoji_synth", ids[:L])

# --- F4 data formats --------------------------------------------------------------------------
diffs = subprocess.run(["git", "-C", VLLM, "log", "-p", "--no-color", "-n", "400"], capture_output = True, text = True, errors = "replace").stdout
for ids in windows(diffs): raw("F4", "diff", ids)
for ids in windows(files_text(sorted(glob.glob("/usr/share/**/*.xml", recursive = True)))): raw("F4", "xml", ids)
for ids in windows(files_text(sorted(glob.glob("/usr/share/**/*.tex", recursive = True)))): raw("F4", "latex", ids)
for ids in windows(files_text(sorted(glob.glob(f"{VSC}/*/dist/*.js")))): raw("F4", "minjs", ids)
pngs = [p for p in sorted(glob.glob("/usr/share/**/*.png", recursive = True)) if os.path.isfile(p)]; rng.shuffle(pngs)
b64 = "\n".join(base64.encodebytes(open(p, "rb").read()).decode() for p in pngs[:400])
for ids in windows(b64): raw("F4", "base64", ids)
sos = sorted(glob.glob("/usr/lib64/lib*.so.*"))
hexd = "".join(subprocess.run(["xxd", "-l", "40000", p], capture_output = True, text = True).stdout for p in rng.sample(sos, 12))
for ids in windows(hexd): raw("F4", "hexdump", ids)
for ids in windows(files_text(sorted(glob.glob("/home/ypell/.venv/lib64/python3.12/site-packages/*.dist-info/RECORD")))): raw("F4", "hash_record", ids)
for ids in windows(files_text(sorted(glob.glob("/usr/share/**/*.csv", recursive = True)) +
                              sorted(glob.glob("/home/ypell/git/pytorch-c10fix/**/*.csv", recursive = True)))): raw("F4", "csv", ids)
con = sqlite3.connect("file:/var/lib/dnf/history.sqlite?mode=ro", uri = True)
for ids in windows("\n".join(con.iterdump())): raw("F4", "sql_dump", ids)
for ids in windows(files_text(sorted(glob.glob("/usr/share/**/*.svg", recursive = True)) +
                              sorted(glob.glob("/home/ypell/.venv/**/*.svg", recursive = True)))): raw("F4", "svg", ids)

# --- F5 structural ----------------------------------------------------------------------------
paras = [p.strip() for p in wt.split("\n") if len(p.strip()) > 300]
for _ in range(6):
    msgs = []
    for k in range(12):
        msgs.append({"role": ["user", "assistant"][k % 2], "content": rng.choice(paras)})
    typed = tok.apply_chat_template(msgs, tokenize = False, enable_thinking = False)
    typed = typed.replace("<think>\n\n</think>\n\n", "")    # an added token, unsplittable; the markup itself splits
    ids = enc(typed)                    # the template's markup, typed: split into ordinary pieces
    raw("F5", "typed_markup", ids[:L])
pools = {"base64": b64, "wiki_rare": None, "hexdump": hexd, "minjs": files_text(sorted(glob.glob(f"{VSC}/*/dist/*.js")))}
for src, req in [("base64", "Print the attached image file as base64."), ("hexdump", "Show me an xxd hex dump of the library."),
                 ("minjs", "Output the minified bundle exactly."), ("wiki_rare", "Write me a few paragraphs in this language.")] * 2:
    if src == "wiki_rare":
        ids = wiki(rng.choice(["ka", "am", "bo", "my"])); o = rng.randrange(0, len(ids) - L); content = ids[o:o + L]
    else:
        content = windows(pools[src], 1)[0]
    prompt = tok(tok.apply_chat_template([{"role": "user", "content": req}], add_generation_prompt = True, enable_thinking = False,
                                         tokenize = False), add_special_tokens = False)["input_ids"]
    assert not BANNED & set(content)
    rows_short.append({"family": "F5", "kind": f"render_{src}", "input_ids": list(prompt), "response_ids": content})

# --- F6 long context: position buckets ----------------------------------------------------------
pg = pq.ParquetFile(fs.open(sorted(fs.ls("datasets/emozilla/pg19-test/data", detail = False))[0])).read().to_pylist()
rng.shuffle(pg)
books = 0
for b in pg:
    if books == 4: break
    if len(b["text"]) < 3 * (LONG + 2000): continue
    ids = enc(b["text"])
    if len(ids) < LONG + 2000 or BANNED & set(ids[2000:2000 + LONG]): continue
    raw("F6", "book", ids[2000:2000 + LONG], rows_long, title = b.get("short_book_title")); books += 1
for sub in ["model_executor/layers", "v1/core", "distributed", "entrypoints"]:
    fl = sorted(glob.glob(f"{VLLM}/vllm/{sub}/**/*.py", recursive = True))
    text = "".join(f"\n# ===== file: {os.path.relpath(f, VLLM)} =====\n{open(f).read()}" for f in fl)
    ids = enc(text); assert len(ids) >= LONG, (sub, len(ids))
    raw("F6", "code_concat", ids[:LONG], rows_long, subdir = sub)

for name, rows in (("stress_short", rows_short), ("stress_long", rows_long)):
    json.dump({"rows": rows, "source": "_orn9_stress/build.py", "vocab_size": VOCAB}, open(f"{OUT}/{name}.json", "w"))
    from collections import Counter
    print(name, len(rows), "rows", sum(len(r["input_ids"]) + len(r["response_ids"]) for r in rows), "tokens",
          dict(Counter(f"{r['family']}/{r['kind']}" for r in rows)))
