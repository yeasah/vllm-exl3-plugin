"""Tabulate a concurrency grid: per server and concurrency, aggregate t/s and the expert cache's
VRAM hit rate over that phase's stats windows. Run where the logs are, in the server's timezone.
usage: concurrency_table.py OUT PHASES   OUT is the grid script's stdout, PHASES its
       "name n start end" file; server logs are p4b_<name>.log beside them"""
import collections, datetime, re, sys
out, phases = open(sys.argv[1]).read(), [l.split() for l in open(sys.argv[2])]
year = datetime.date.today().year
rate = {}  # (name, n) -> aggregate t/s
name = None
for line in out.splitlines():
    if line.startswith("=== "): name = line[4:].strip()
    m = re.match(r"concurrency (\d+): .* = ([\d.]+) t/s aggregate, median per request ([\d.]+)", line)
    if m: rate[(name, int(m[1]))] = (float(m[2]), float(m[3]))
stat = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d).*VRAM hit ([\d.]+)%.*?(\d+) promotions.*?(\d+) steps had decodes, (\d+) learned from, (\d+) staged")
hits = collections.defaultdict(list)
logs = {}
for nm, n, a, b in phases:
    if nm not in logs:
        try: logs[nm] = open(f"p4b_{nm}.log").read().splitlines()
        except FileNotFoundError: logs[nm] = []
    for line in logs[nm]:
        m = stat.search(line)
        if not m: continue
        t = datetime.datetime.strptime(f"{year}-{m[1]}", "%Y-%m-%d %H:%M:%S").timestamp()
        if int(a) + 20 <= t <= int(b):  # skip the first windows, which straddle the previous phase
            hits[(nm, int(n))].append((float(m[2]), int(m[3]), int(m[6])))
ns = sorted({int(p[1]) for p in phases})
names = list(dict.fromkeys(p[0] for p in phases))
print(f"{'server':16s}" + "".join(f"{f'n={n}':>22s}" for n in ns))
for nm in names:
    cells = []
    for n in ns:
        r = rate.get((nm, n)); h = hits.get((nm, n))
        hit = f"{sum(x[0] for x in h) / len(h):4.0f}%" if h else "    "
        cells.append(f"{r[0]:7.1f} t/s {hit} {r[1]:4.0f}" if r else f"{'—':>22s}")
    print(f"{nm:16s}" + "".join(f"{c:>22s}" for c in cells))
print("cells: aggregate decode t/s, mean VRAM hit over the phase's stats windows, median per-request t/s")
