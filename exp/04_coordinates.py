"""Spline coordinates, scored by free energy summed over all chains.

Stage 1, strike coordinate: x = k / sqrt(theta(T)) uniform, or warped as asinh(c x) / c for c = 1, 2, 4.
Stage 2, maturity clock: ln T or ln theta(T), with 4 or 5 maturity splines.
Writes results/coordinates/coordinates.json and coordinates.md. Slow: 8 configurations x every chain.
"""
from _common import chains, day_of, out_dir, dump, markdown_table
from volsurf import Config, run_day
from volsurf.config import SPX_STUDY

base = SPX_STUDY
configs = {
    "uniform x, ln T, 4": base,
    "warp c=1, ln T, 4": base.with_(warp=1.0),
    "warp c=2, ln T, 4": base.with_(warp=2.0),
    "warp c=4, ln T, 4": base.with_(warp=4.0),
    "uniform x, ln T, 5": base.with_(n_t=5),
    "uniform x, ln theta, 4": base.with_(t_clock="lntheta"),
    "uniform x, ln theta, 5": base.with_(t_clock="lntheta", n_t=5),
}
files = chains(); out = {}
for name, cfg in configs.items():
    out[name] = {}
    for path in files:
        r = run_day(path, cfg, relin=False, log=lambda *_: None)
        out[name][day_of(path)] = dict(F=r["fit"]["F"], inside=r["checks"]["inside_all"],
                                       wing_maxima=r["checks"]["put_wing"]["local_maxima"])
    tot = sum(v["F"] for v in out[name].values())
    print(f"  {name:26s} total F {tot:9.1f}")
d = out_dir("coordinates")
dump(out, d / "coordinates.json")
days = [day_of(p) for p in files]
rows = [[n] + [f"{out[n][dd]['F']:.1f}" for dd in days] + [f"{sum(out[n][dd]['F'] for dd in days):.1f}"]
        for n in sorted(out, key=lambda n: sum(out[n][dd]["F"] for dd in days))]
(d / "coordinates.md").write_text(markdown_table(rows, ["coordinates"] + days + ["total"]))
print((d / "coordinates.md").read_text())
