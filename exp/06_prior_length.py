"""Prior correlation length of the local-variance correction, scored by free energy.

The smoothness prior couples neighbouring strike splines over ell_x ATM standard deviations. A long ell_x smooths
away sharp features (event smiles, W shapes); a short one lets the wings wander. The free energy decides.
usage: python exp/06_prior_length.py [CHAIN ...]        (default: every chain in DATA_DIR)
Writes results/prior_length/prior_length.json and prior_length.md.
"""
import sys
from _common import chains, day_of, out_dir, dump, markdown_table
from volsurf import Config, run_day

files = sys.argv[1:] or chains()
ells = (0.6, 1.0, 1.6, 2.5)
out = {}
for ell in ells:
    cfg = Config().with_(ell_x=ell); out[ell] = {}
    for path in files:
        r = run_day(path, cfg, relin=False, log=lambda *_: None)
        out[ell][day_of(path)] = dict(F=r["fit"]["F"], inside=r["checks"]["inside_all"])
        print(f"  ell_x={ell}: {day_of(path)} F {r['fit']['F']:.1f}", flush=True)
d = out_dir("prior_length")
dump({str(k): v for k, v in out.items()}, d / "prior_length.json")
days = [day_of(p) for p in files]
rows = [[e] + [f"{out[e][dd]['F']:.1f}" for dd in days] + [f"{sum(out[e][dd]['F'] for dd in days):.1f}"] for e in ells]
(d / "prior_length.md").write_text(markdown_table(rows, ["ell_x"] + days + ["total F"]))
print((d / "prior_length.md").read_text())
