#!/usr/bin/env python3
"""Mirror a working tree on fast local disk into a slow archive volume, in the background.

    sync_outputs.py SRC DST                 loop: every --interval s, copy settled files SRC -> DST
    sync_outputs.py SRC DST --once          one pass
    sync_outputs.py --restore DST SRC       one pass back: archive -> local, files missing or differing in size

Made for runpod, where /workspace is geesefs over S3 (docs: memory runpod-workspace-is-s3-fuse):
working from it starves the GPUs, so quant.py works on the container disk and this copies the
results over as they appear. What the volume allows shapes how:
- contents only (copyfile): it refuses chmod and utime;
- each file lands under a temporary name and is renamed into place, so an interrupted copy
  never leaves a truncated file under the real name;
- without settable mtimes the archive cannot say what it already has, so the copied
  (size, mtime) of each file is recorded on the source side (SRC/.sync_state.json); a file
  rewritten at the same size is still copied again;
- only settled files (unchanged for --settle s) are copied, and a file that changed during its
  copy is not recorded, so the next pass takes it again.
Nothing is deleted from the archive. Work directories, logit caches and qbench's partial
results are never copied: they are regenerable or transient, and the bulk of the I/O.
"""
import argparse, fnmatch, json, os, shutil, sys, time

EXCLUDE = ["_work*", "_logit_cache", "_qbench_parts", "__pycache__", "*.sync-tmp", ".sync_state.json"]
STATE = ".sync_state.json"

def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def walk(root, exclude):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not any(fnmatch.fnmatch(x, p) for p in exclude)]
        for f in files:
            if not any(fnmatch.fnmatch(f, p) for p in exclude):
                yield os.path.relpath(os.path.join(d, f), root)

def copy_one(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".sync-tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)

def sync_pass(src, dst, settle, exclude):
    state_path = os.path.join(src, STATE)
    try:
        with open(state_path) as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {}
    copied, nbytes, t0 = 0, 0, time.time()
    for rel in walk(src, exclude):
        s = os.path.join(src, rel)
        try:
            st = os.stat(s)
        except FileNotFoundError:
            continue
        key = [st.st_size, st.st_mtime_ns]
        if state.get(rel) == key or time.time() - st.st_mtime < settle:
            continue
        try:
            copy_one(s, os.path.join(dst, rel))
        except OSError as e:
            print(f"{stamp()} !! {rel}: {e}", flush=True)
            continue
        st2 = os.stat(s)
        if [st2.st_size, st2.st_mtime_ns] == key:        # unchanged during the copy
            state[rel] = key
            copied += 1; nbytes += st.st_size
            tmp = state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, state_path)
    if copied:
        print(f"{stamp()} synced {copied} file(s), {nbytes / 2**30:.2f} GiB, in {time.time() - t0:.0f} s", flush=True)
    return copied

def restore(archive, local, exclude):
    """Copy the archive back; record each restored file as already synced, so the next sync
    pass does not send it straight back."""
    state_path = os.path.join(local, STATE)
    try:
        with open(state_path) as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {}
    copied, nbytes = 0, 0
    for rel in walk(archive, exclude):
        a, l = os.path.join(archive, rel), os.path.join(local, rel)
        size = os.stat(a).st_size
        if not (os.path.isfile(l) and os.stat(l).st_size == size):
            copy_one(a, l)
            copied += 1; nbytes += size
        st = os.stat(l)
        state[rel] = [st.st_size, st.st_mtime_ns]
    os.makedirs(local, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f)
    print(f"{stamp()} restored {copied} file(s), {nbytes / 2**30:.2f} GiB", flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--restore", action="store_true", help="copy src (the archive) back into dst (local)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--settle", type=float, default=120, help="copy files unchanged for this many seconds")
    ap.add_argument("--exclude", action="append", default=[], help="extra name patterns to skip")
    args = ap.parse_args()
    exclude = EXCLUDE + args.exclude
    if args.restore:
        restore(args.src, args.dst, exclude)
        return
    print(f"{stamp()} syncing {args.src} -> {args.dst} every {args.interval:.0f} s (settle {args.settle:.0f} s)", flush=True)
    while True:
        sync_pass(args.src, args.dst, args.settle, exclude)
        if args.once:
            return
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
