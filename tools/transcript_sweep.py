#!/usr/bin/env python3
"""Mine the project's own conversation transcripts for knowledge never written down.

Working sessions establish more than they record. The chatty, tangent-heavy mode
that produces the good ideas is exactly the mode that skips the write-up, so
findings accumulate in `~/.claude/projects/<project>/*.jsonl` and are buried by
every later exchange. This is the tool for going back for them.

Run twice by hand before it existed (2026-08-23 and 2026-09-04), which is the
threshold this project uses for building the thing rather than redoing it.

    transcript_sweep.py extract [--since TS] [--out DIR] [--no-summaries]
    transcript_sweep.py signature [--since TS] [--context N]
    transcript_sweep.py check "phrase" ...        (or --from FILE)
    transcript_sweep.py mark [--at TS] [--note TEXT]

**The workflow, and why it is in that order.**

1. `extract` pools every session file, keeps human and assistant prose, drops
   tool traffic, and writes `all.md`, `user.md` and numbered chunks. It reports
   the size up front so the cost is known before committing to the read.
2. **Read `user.md` in full first.** It is roughly a fifth of the volume and
   carries most of the signal: the domain facts, the corrections, the decisions
   and the complaints all originate there, and the assistant side is mostly
   elaboration on them. On the 2026-09-04 sweep that was 373k of 1.66M chars.
3. `signature` covers the assistant side cheaply, surfacing the passages that
   carry a finding's linguistic markers rather than reading all of it.
4. **`check` every candidate before writing any of it down.** This is the step
   that earns the tool. On the second sweep it killed more candidates than it
   passed -- the cumem allocator's cost, FlashInfer's workspace, `boundary:N,M`
   and TurboQuant's prefill transient were all already recorded, and one
   "pending" correction had already been made. A sweep that skips this produces
   duplicates and re-litigates settled questions.
5. Route what survives: project facts to `docs/`, stack facts to the field
   notes, open work to `TODO.md`. See the `ecosystem-field-notes` memory.
6. `mark` records the boundary so the next sweep starts where this one stopped.

**The pool should span related projects, not just related sessions.** The
highest-value thing a sweep finds is a gap *between* pieces of work, and in a
stack of several repos those gaps cross repo boundaries: a decision taken while
working on one and never carried to another leaves no trace in either one's
history. Keeping the whole stack under a single Claude Code project keeps them
in one pool, which is a reason not to split projects for tidiness as repos
multiply. The kv-pager is the worked example -- designed in this repo, moved to
`vllm-virtualkv-plugin`, evidence left behind here on purpose -- and a
per-repo pool could not see that arc.

The cost is that the pool only grows, so `--since` and `mark` stop being a
convenience and become the thing that keeps a sweep affordable.

**Two mechanics that are not optional.**

*Dedupe by content, not by session.* Sessions fork on `--resume` and on
interrupts, so the same exchange appears in several files under different ids.
The 2026-09-04 sweep was 31% duplicate before hashing. Pooling everything and
keeping the earliest instance of each (role, text) also puts concurrent sessions
in one chronological narrative.

*Chunk to fit the reader.* 170k-char chunks exceeded the Read tool's token
budget and had to be sliced by hand mid-sweep; the default here is sized to fit.

Compaction summaries are kept by default and tagged `[SUMMARY]`. They are
enormous and redundant with the prose around them, but they enumerate what was
already committed, which makes them the cheapest available cross-check against
recording something twice. `--no-summaries` drops them.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(REPO, "docs", "data", "sweeps.json")
FIELD_NOTES = os.environ.get(
    "FIELD_NOTES", os.path.expanduser("~/notes/vllm-ecosystem-field-notes.md")
)
SESSIONS = os.environ.get(
    "CLAUDE_SESSIONS",
    os.path.expanduser("~/.claude/projects/" + REPO.replace("/", "-")),
)

# Turns that are machinery rather than conversation. The compaction summary is
# deliberately not here -- it is tagged instead, see the module docstring.
SYNTHETIC = (
    "<system-reminder", "<command-name", "<local-command", "<user-prompt",
    "<ide_", "<task-notification", "[SYSTEM NOTIFICATION", "[Request interrupted",
    "Caveat: The messages below",
)
SUMMARY_MARK = "This session is being continued from a previous conversation"

#: Phrases that tend to sit next to something established and never recorded.
#: Deliberately over-inclusive: a signature pass is a filter over 1M+ chars, and
#: the cost of a false positive is one skimmed paragraph.
SIGNATURES = (
    r"turns out", r"worth (noting|knowing|recording|saying)", r"it is not in the docs",
    r"undocumented", r"the (real|actual) (reason|cause|mechanism)", r"in fact",
    r"which is why", r"the honest", r"nobody (says|documents|mentions)",
    r"silently", r"surpris(ing|ed|ingly)", r"counter-?intuitive",
    r"contrary to", r"the docs (say|claim|do not)", r"measured", r"confirmed",
    r"refuted", r"was wrong", r"I was wrong", r"correction", r"the tell",
    r"gotcha", r"trap", r"the catch", r"note for", r"for the record",
)


def sessions() -> list[str]:
    files = sorted(glob.glob(os.path.join(SESSIONS, "*.jsonl")))
    if not files:
        raise SystemExit(f"no session files under {SESSIONS}")
    return files


def text_of(message) -> str:
    """Human-visible prose from a transcript message, tool traffic dropped."""
    content = message.get("content") if isinstance(message, dict) else message
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        b.get("text", "") for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


def harvest(since: str, keep_summaries: bool = True):
    """Every unique prose turn at or after `since`, oldest first.

    Keyed on (role, whitespace-normalised text) rather than on session id, so a
    forked or resumed session contributes its exchanges once. The earliest
    timestamp wins, which keeps the fork's own ordering rather than the copy's.
    """
    seen: dict[str, tuple] = {}
    for path in sessions():
        sid = os.path.basename(path)[:8]
        with open(path, errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                ts = rec.get("timestamp")
                if not ts or ts < since or rec.get("type") not in ("user", "assistant"):
                    continue
                body = (text_of(rec.get("message")) or "").strip()
                if not body or any(body.startswith(p) for p in SYNTHETIC):
                    continue
                summary = SUMMARY_MARK in body[:400]
                if summary and not keep_summaries:
                    continue
                key = hashlib.sha1(
                    (rec["type"] + re.sub(r"\s+", " ", body)).encode()
                ).hexdigest()
                body = re.sub(r"\n{3,}", "\n\n", body)
                row = (ts, sid, rec["type"], summary, body)
                if key not in seen or ts < seen[key][0]:
                    seen[key] = row
    return sorted(seen.values())


def header(ts, sid, role, summary) -> str:
    tag = " [SUMMARY]" if summary else ""
    return f"\n\n===== {ts[:19]} [{sid}] {role.upper()}{tag} =====\n"


def cmd_extract(args) -> None:
    rows = harvest(args.since, keep_summaries=not args.no_summaries)
    if not rows:
        raise SystemExit(f"nothing since {args.since}")
    os.makedirs(args.out, exist_ok=True)
    users = [r for r in rows if r[2] == "user"]

    def write(name, subset):
        with open(os.path.join(args.out, name), "w") as fh:
            for ts, sid, role, summary, body in subset:
                fh.write(header(ts, sid, role, summary) + body + "\n")
        return sum(len(r[4]) for r in subset)

    total = write("all.md", rows)
    ucount = write("user.md", users)

    def chunk_out(subset, prefix):
        """Split to files a single read can hold -- both streams, since the
        user-only stream is the one the workflow opens first and it outgrew a
        read on the second sweep."""
        held, n, size = [], 0, 0
        for row in subset:
            block = header(*row[:4]) + row[4] + "\n"
            if size + len(block) > args.chunk and held:
                n += 1
                open(os.path.join(args.out, f"{prefix}{n:02d}.md"), "w").write("".join(held))
                held, size = [], 0
            held.append(block)
            size += len(block)
        if held:
            n += 1
            open(os.path.join(args.out, f"{prefix}{n:02d}.md"), "w").write("".join(held))
        return n

    n = chunk_out(rows, "")
    un = chunk_out(users, "user-")

    print(f"since {args.since}  ->  {args.out}")
    print(f"  {len(rows)} unique turns, {total/1e6:.2f}M chars, {n} chunks")
    print(f"  user turns: {len(users)} ({ucount/1e3:.0f}k chars, "
          f"{100*ucount/total:.0f}% of the prose) in {un} chunks "
          f"-- read user-*.md first")
    summaries = sum(1 for r in rows if r[3])
    if summaries:
        print(f"  {summaries} compaction summaries kept (tagged [SUMMARY]); "
              f"they enumerate what was already committed")


def cmd_signature(args) -> None:
    """Assistant-side passages carrying a finding's linguistic markers."""
    pattern = re.compile("|".join(SIGNATURES), re.I)
    hits = 0
    for ts, sid, role, summary, body in harvest(args.since):
        if role != "assistant" or summary:
            continue
        lines = body.splitlines()
        marked = {i for i, line in enumerate(lines) if pattern.search(line)}
        if not marked:
            continue
        keep, last = [], -99
        for i in sorted(marked):
            lo, hi = max(0, i - args.context), min(len(lines), i + args.context + 1)
            if lo <= last:
                keep[-1] = (keep[-1][0], hi)
            else:
                keep.append((lo, hi))
            last = hi
        hits += 1
        print(header(ts, sid, role, summary).strip())
        for lo, hi in keep:
            print("\n".join(lines[lo:hi]))
            print("  ...")
    print(f"\n# {hits} assistant turns matched", file=sys.stderr)


def cmd_check(args) -> None:
    """Is this already written down? Run before recording anything.

    Searches the places a finding could legitimately already live. A hit is not
    proof of duplication -- read it -- but a miss is good evidence the thing is
    genuinely unrecorded.
    """
    phrases = list(args.phrase)
    if args.from_file:
        phrases += [
            ln.strip() for ln in open(args.from_file)
            if ln.strip() and not ln.startswith("#")
        ]
    if not phrases:
        raise SystemExit("nothing to check")
    targets = sorted(glob.glob(os.path.join(REPO, "docs", "*.md")))
    targets += [os.path.join(REPO, "TODO.md"), os.path.join(REPO, "README.md")]
    targets += sorted(glob.glob(os.path.join(REPO, "agent-memory", "*.md")))
    notes = FIELD_NOTES if os.path.exists(FIELD_NOTES) else None

    for phrase in phrases:
        try:
            where = subprocess.run(
                ["grep", "-rliE", phrase, *targets], capture_output=True, text=True
            ).stdout.split()
        except Exception as exc:                      # pragma: no cover
            raise SystemExit(f"grep failed: {exc}")
        names = ",".join(sorted(os.path.basename(w) for w in where)) or "-"
        note_hits = 0
        if notes:
            out = subprocess.run(
                ["grep", "-ciE", phrase, notes], capture_output=True, text=True
            ).stdout.strip()
            note_hits = int(out) if out.isdigit() else 0
        verdict = "RECORDED  " if (where or note_hits) else "unrecorded"
        tail = f"  notes:{note_hits}" if notes else ""
        print(f"{verdict}  {phrase[:44]:46} {names}{tail}")


def load_state() -> list:
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return []


def cmd_mark(args) -> None:
    at = args.at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    head = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    state = load_state()
    state.append({"ended": at, "commit": head, "note": args.note or ""})
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(state, open(STATE, "w"), indent=2)
    print(f"recorded sweep boundary {at} at {head}  ({STATE})")


def default_since() -> str:
    state = load_state()
    return state[-1]["ended"] if state else "1970-01-01T00:00:00"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="pool, dedupe and chunk the transcripts")
    e.add_argument("--since", default=None)
    e.add_argument("--out", default=os.path.join(REPO, ".sweep"))
    e.add_argument("--chunk", type=int, default=80_000,
                   help="chars per chunk; default fits one read")
    e.add_argument("--no-summaries", action="store_true")
    e.set_defaults(func=cmd_extract)

    g = sub.add_parser("signature", help="assistant passages that look like findings")
    g.add_argument("--since", default=None)
    g.add_argument("--context", type=int, default=2)
    g.set_defaults(func=cmd_signature)

    c = sub.add_parser("check", help="is it already recorded? run before writing")
    c.add_argument("phrase", nargs="*")
    c.add_argument("--from-file", dest="from_file")
    c.set_defaults(func=cmd_check)

    m = sub.add_parser("mark", help="record where this sweep stopped")
    m.add_argument("--at")
    m.add_argument("--note")
    m.set_defaults(func=cmd_mark)

    args = ap.parse_args()
    if getattr(args, "since", "sentinel") is None:
        args.since = default_since()
    args.func(args)


if __name__ == "__main__":
    main()
