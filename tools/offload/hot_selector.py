"""Replay cache policies over a topk trace. Each hit = expert read from VRAM; miss = read over the link."""
import sys, numpy as np, collections
Z = np.load(sys.argv[1] + ".npz", allow_pickle=True)
tags, layers, topk = Z["tags"], Z["layers"], Z["topk"]
E = 1 << int(np.ceil(np.log2(int(max(t.max() for t in topk)) + 1)))
L = int(layers.max()) + 1
ev = []  # (job, kind, phase, layer, ids[rows,k])
for tag, l, t in zip(tags, layers, topk):
    if tag == "none": continue
    j, kind = tag.split(":")
    ev.append((int(j), kind, "prefill" if t.shape[0] > 1 else "decode", int(l), t))
print(f"E={E} L={L} calls={len(ev)} rows decode={sum(e[4].shape[0] for e in ev if e[2]=='decode')/L:.0f} "
      f"prefill={sum(e[4].shape[0] for e in ev if e[2]=='prefill')/L:.0f} (tokens)  k={ev[0][4].shape[1]}")
FR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
def hist(sel):
    h = np.zeros((L, E))
    for e in ev:
        if sel(e): np.add.at(h[e[3]], e[4].ravel(), 1)
    return h
dec = hist(lambda e: e[2] == "decode"); pre = hist(lambda e: e[2] == "prefill")

import json
N = int(sys.argv[2])  # experts to offload per layer
pats = []
for l in range(L):
    cold = dec[l].argsort()[:N].tolist()
    pats.append("re:layers" + chr(92) + f".{l}" + chr(92) + "..*experts" + chr(92) + ".(" + "|".join(map(str, sorted(cold))) + ")" + chr(92) + ".")
open("cold.txt", "w").write(" ".join(pats)); print(len(pats), pats[0][:90])
