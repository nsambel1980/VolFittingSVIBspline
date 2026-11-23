"""Command line:  python -m volsurf fit CHAIN.json --out DIR [--modes DIR/modes.npz] [--n-modes 24] [--no-plots]"""
import argparse
import numpy as np
from .config import Config
from .pipeline import run_day, save_day
from .fit import baseline


def main(argv=None):
    p = argparse.ArgumentParser(prog="volsurf")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit one Cboe option chain")
    f.add_argument("chain"); f.add_argument("--out", required=True)
    f.add_argument("--modes", help="modes.npz from an earlier fit, reused instead of recomputing")
    f.add_argument("--n-modes", type=int, default=Config().n_modes)
    f.add_argument("--no-relin", action="store_true"); f.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)
    cfg = Config(n_modes=a.n_modes)
    modes = None
    if a.modes:
        d = np.load(a.modes); modes = dict(V=d["V"], kappa=d["kappa"])
    res = run_day(a.chain, cfg, modes=modes, relin=not a.no_relin)
    out = save_day(res, cfg, a.out)
    if not a.no_plots:
        from .plotting import fit_vs_bidask, localvol_slices
        fit_vs_bidask(res["mk"], res["model"], out / "fit_vs_bidask.png",
                      f"SPX {res['mk']['quote_time']}: fit relative to mid vol",
                      baseline=baseline(res["mk"], res["base"]["q"], cfg))
        localvol_slices([res["model"]], [str(res["mk"]["quote_time"])], out / "localvol_slices.png")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
