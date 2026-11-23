"""Local-vol jumps at the expiries: skew interpolated linearly in T against PCHIP in theta.

Cheap: only the SSVI baselines are fitted. Writes results/skew_interp/jumps.json and jumps.md.
"""
import numpy as np
from _common import chains, day_of, out_dir, dump, markdown_table
from volsurf import Config, build_market
from volsurf.fit import fit_baseline
from volsurf.ssvi import SSVISlices

xs = np.linspace(-3.0, 2.0, 101); out, rows = {}, []
for path in chains():
    cfg = Config(); mk = build_market(path, cfg); q = fit_baseline(mk, cfg)["q"]; day = day_of(path); out[day] = {}
    for kind in ("linear", "pchip"):
        b = SSVISlices(mk["T"], q, kind); jumps = []
        for Ti in mk["T"][1:-1]:
            lv = [100 * np.sqrt(b.v0(xs * np.sqrt(b.theta(t)), t)[0]) for t in (Ti * (1 - 2e-3), Ti * (1 + 2e-3))]
            jumps.append(float(np.abs(lv[1] - lv[0]).max()))
        out[day][kind] = jumps
    rows.append([day, f"{max(out[day]['linear']):.2f}", f"{max(out[day]['pchip']):.2f}"])
d = out_dir("skew_interp")
dump(out, d / "jumps.json")
(d / "jumps.md").write_text(markdown_table(rows, ["day", "max jump, linear in T (vol pts)",
                                                  "max jump, PCHIP in theta (vol pts)"]))
print((d / "jumps.md").read_text())
