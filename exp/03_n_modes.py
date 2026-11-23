"""How many modes? Free energy against K for one chain (default: the latest).

The modes are computed once; each K refits the leading K of them. The free energy should flatten and then
rise once extra modes stop paying for their Occam factor.
usage: python exp/03_n_modes.py [CHAIN]
Writes results/n_modes/<day>.json and <day>.png.
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import chains, day_of, out_dir, dump
from volsurf import Config, build_market
from volsurf.fit import fit_baseline, fit_modes, baseline
from volsurf.modes import compute_modes
from volsurf.plotting import STYLE, BLUE, INK

path = sys.argv[1] if len(sys.argv) > 1 else chains()[-1]
cfg = Config(n_modes=40)
mk = build_market(path, cfg)
b = fit_baseline(mk, cfg)
md = compute_modes(mk, baseline(mk, b["q"], cfg), cfg)
Ks = [k for k in (4, 8, 12, 16, 20, 24, 28, 32, 40) if k <= md["V"].shape[1]]
res = {}
for K in Ks:
    f = fit_modes(mk, b["q"], md["V"], cfg, K=K)
    res[K] = dict(F=f["F"], nll=float(f["nll"]), tau=f["tau"])
    print(f"  K={K:2d}  F {f['F']:9.1f}  -lnL {f['nll']:9.1f}  tau {f['tau']:.2f}")
d = out_dir("n_modes"); day = day_of(path)
dump(dict(baseline_F=b["F"], scan=res), d / f"{day}.json")
plt.rcParams.update(STYLE)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(Ks, [res[k]["F"] for k in Ks], "o-", color=BLUE, ms=6)
ax.set_xlabel("number of modes K"); ax.set_ylabel("free energy (lower is better)")
ax.set_title(f"SPX {day}: free energy against the number of modes", loc="left", color=INK)
fig.tight_layout(); fig.savefig(d / f"{day}.png", dpi=150)
