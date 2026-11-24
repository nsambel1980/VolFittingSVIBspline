"""Fit every chain in DATA_DIR with the default configuration.

Writes results/days/<YYYYMMDD>/{parameters,summary,config}.json, modes.npz, fit_vs_bidask.png and
localvol_slices.png, plus results/days/localvol_3d.png and results/days/summary.md across days.
"""
import numpy as np
from _common import chains, day_of, out_dir, markdown_table
from volsurf import Config, run_day, save_day
from volsurf.config import SPX_STUDY
from volsurf.fit import baseline
from volsurf.plotting import fit_vs_bidask, localvol_slices, localvol_3d

cfg = SPX_STUDY
models, labels, rows = [], [], []
for path in chains():
    day = day_of(path); print(f"{day}:")
    res = run_day(path, cfg)
    out = save_day(res, cfg, out_dir("days", day))
    mk, model = res["mk"], res["model"]
    fit_vs_bidask(mk, model, out / "fit_vs_bidask.png", f"SPX {mk['quote_time']}: fit relative to mid vol",
                  baseline=baseline(mk, res["base"]["q"], cfg))
    localvol_slices([model], [f"SPX {mk['quote_time']}"], out / "localvol_slices.png")
    models.append(model); labels.append(f"SPX {mk['quote_time']}")
    c = res["checks"]
    rows.append([day, f"{res['fit']['F']:.1f}", f"{100 * c['inside_all']:.1f}%", f"{c['atmf_gap_volpts']:.3f}",
                 f"{c['skew_gap']:.1e}", f"{c['min_density_ratio']:.1e}", f"{c['min_price_change']:.1e}",
                 f"{max(c['expiry_jumps_volpts']):.2f}", f"{res['minutes']:.1f}"])
    print(f"  inside {100 * c['inside_all']:.1f}%, {res['minutes']:.1f} min")

localvol_3d(models, labels, out_dir("days") / "localvol_3d.png")
(out_dir("days") / "summary.md").write_text(markdown_table(rows, [
    "day", "free energy", "inside bid-ask", "ATMF gap (vol pts)", "skew gap", "min density / max",
    "min price change", "max expiry jump (vol pts)", "minutes"]))
print(open(out_dir("days") / "summary.md").read())
