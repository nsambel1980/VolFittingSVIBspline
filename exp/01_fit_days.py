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
    rows.append([day, len(mk["k"]), res["fit"]["K"], f"{100 * c['inside_all']:.1f}%", f"{c['price_rms_h']:.2f}",
                 c["band_misses"], c["ask_breaches"], f"{c['atmf_gap_volpts']:.3f}",
                 "yes" if c["arbitrage_free"] else f"no: {list(c['arbitrage_failures'])}",
                 f"{max(c['expiry_jumps_volpts']):.2f}", f"{res['minutes']:.1f}"])
    print(f"  inside {100 * c['inside_all']:.1f}%, {res['minutes']:.1f} min")

localvol_3d(models, labels, out_dir("days") / "localvol_3d.png")
(out_dir("days") / "summary.md").write_text(markdown_table(rows, [
    "day", "quotes", "modes", "inside bid-ask", "price RMS (half-spreads)", "band misses", "ask breaches",
    "ATMF gap (vol pts)", "arbitrage-free", "max expiry jump (vol pts)", "minutes"]))
print(open(out_dir("days") / "summary.md").read())
