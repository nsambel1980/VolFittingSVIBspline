"""Command line.

  python -m volsurf fit CHAIN.json --out DIR [--modes DIR/modes.npz] [--n-modes 24] [--no-relin] [--no-plots]
  python -m volsurf fit-vols VOLS.csv --out DIR [--quote-time 2026-11-19T15:22] [--s-noise 0.0005] [...]

The CSV for fit-vols has one row per strike: expiry (YYYY-MM-DD, needs --quote-time) or T (years);
strike + forward, or k = ln(K/F); iv_bid + iv_ask, or iv_mid [+ half_spread]; vols as decimals.
"""
import argparse
import datetime as dt
import numpy as np
from .config import Config
from .market import build_market_from_vols, load_vol_csv
from .pipeline import run_day, save_day
from .fit import baseline


def main(argv=None):
    p = argparse.ArgumentParser(prog="volsurf")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit one Cboe option chain (prices)")
    f.add_argument("chain")
    v = sub.add_parser("fit-vols", help="fit implied vols from a CSV")
    v.add_argument("csv")
    v.add_argument("--quote-time", help="ISO time of the quotes; needed when the CSV gives expiry dates")
    v.add_argument("--s-noise", type=float, help="vol noise added to the bid-ask likelihood (default from Config)")
    for q in (f, v):
        q.add_argument("--out", required=True)
        q.add_argument("--modes", help="modes.npz from an earlier fit, reused instead of recomputing")
        q.add_argument("--n-modes", type=int, default=Config().n_modes)
        q.add_argument("--no-relin", action="store_true"); q.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)
    cfg = Config(n_modes=a.n_modes)
    if a.cmd == "fit-vols" and a.s_noise is not None:
        cfg = cfg.with_(s_noise=a.s_noise)
    modes = None
    if a.modes:
        d = np.load(a.modes); modes = dict(V=d["V"], kappa=d["kappa"])
    if a.cmd == "fit":
        source = a.chain
    else:
        qt = dt.datetime.fromisoformat(a.quote_time) if a.quote_time else None
        source = build_market_from_vols(load_vol_csv(a.csv), cfg, quote_time=qt)
    res = run_day(source, cfg, modes=modes, relin=not a.no_relin)
    out = save_day(res, cfg, a.out)
    if not a.no_plots:
        from .plotting import fit_vs_bidask, localvol_slices
        name = str(res["mk"]["quote_time"] or a.csv)
        fit_vs_bidask(res["mk"], res["model"], out / "fit_vs_bidask.png", f"{name}: fit relative to mid vol",
                      baseline=baseline(res["mk"], res["base"]["q"], cfg))
        localvol_slices([res["model"]], [name], out / "localvol_slices.png")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
