"""One day, end to end: market -> SSVI baseline -> modes -> fit -> relinearize -> checks."""
import json
import time
from pathlib import Path
import numpy as np
from .config import Config
from .market import build_market, build_market_from_vols
from .fit import fit_baseline, fit_modes, baseline
from .modes import compute_modes
from .relinearize import relinearize
from .diagnostics import summary


def run_day(source, cfg=Config(), modes=None, relin=True, log=print, q_init=None):
    """Fit one surface. `source`: a chain path, or a market dict from build_market / build_market_from_vols.
    `modes`: a previous day's modes (dict with V, kappa) to reuse when the coefficient layout matches;
    `q_init`: a previous baseline (e.g. yesterday's) used as an extra start."""
    t0 = time.time()
    mk = source if isinstance(source, dict) else build_market(source, cfg)
    log(f"  {len(mk['k'])} quotes on {mk['nT']} expiries")
    base = fit_baseline(mk, cfg, q_init=q_init)
    log(f"  baseline: -lnL {base['nll']:.1f}, F {base['F']:.1f}, term-structure scale {base['smooth']}")
    fresh = compute_modes(mk, baseline(mk, base["q"], cfg), cfg) if modes is None else None
    if modes is not None and modes["V"].shape[0] != fresh_D(mk, base, cfg):
        log("  given modes do not match this market's coefficient layout: recomputing")
        modes = None; fresh = compute_modes(mk, baseline(mk, base["q"], cfg), cfg)
    if modes is None:
        modes = fresh
        log(f"  modes: {modes['n_admissible']} admissible of {modes['n_coefficients']}; "
            f"kappa >= {cfg.kappa_min}: {int(np.sum(modes['kappa'] >= cfg.kappa_min))}")
    fit = fit_modes(mk, base["q"], modes["V"], cfg, smooth=base["smooth"], kappa=modes.get("kappa"))
    log(f"  fit K={fit['K']}: -lnL {fit['nll']:.1f}, F {fit['F']:.1f}, tau {fit['tau']:.2f}")
    out = dict(mk=mk, base=base, modes=modes, fit=fit)
    final = fit
    if relin:
        rl = relinearize(mk, cfg, fit)
        out["relin"] = rl; final = rl
        log(f"  relinearized: gaps {rl['history'][-1]['gaps']}, -lnL {rl['history'][-1]['nll']:.1f}")
        model = rl["model"]
    else:
        from .dupire import LocalVarSurface
        model = LocalVarSurface(mk, baseline(mk, fit["q"], cfg), cfg, a=fit["a"])
    out["model"] = model
    out["checks"] = summary(mk, model)
    out["q"], out["a"] = final["q"], final["a"]
    out["minutes"] = (time.time() - t0) / 60
    return out


def fresh_D(mk, base, cfg):
    from .dupire import LocalVarSurface
    return LocalVarSurface(mk, baseline(mk, base["q"], cfg), cfg).D


def fit_vols(expiries, cfg=Config(), quote_time=None, spot=None, **kw):
    """Fit from implied vols instead of prices: build_market_from_vols(expiries, ...) then run_day.
    `expiries`: one dict per expiry (see build_market_from_vols), or a CSV path read by load_vol_csv."""
    if isinstance(expiries, (str, Path)):
        from .market import load_vol_csv
        expiries = load_vol_csv(expiries)
    return run_day(build_market_from_vols(expiries, cfg, quote_time=quote_time, spot=spot), cfg, **kw)


def parameters(res):
    """The fitted parameters, as plain numbers."""
    mk, b = res["mk"], res["model"].base
    return dict(quote_time=None if mk["quote_time"] is None else str(mk["quote_time"]), spot=mk["spot"],
                expiries=[q["label"] for q in mk["info"]],
                T=[float(t) for t in mk["T"]],
                forward=[None if q.get("F") is None else float(q["F"]) for q in mk["info"]],
                discount=[None if q.get("D") is None else float(q["D"]) for q in mk["info"]],
                atmf_vol=[float(x) for x in np.sqrt(b.th_i / mk["T"])], atmf_total_variance=[float(x) for x in b.th_i],
                atm_skew_baseline=[float(x) for x in b.sk_i], eta=float(b.eta), gamma=float(b.gam),
                term_structure_scale=float(res["base"]["smooth"]),
                q=[float(x) for x in res["q"]], coefficients=[float(x) for x in res["a"]], tau=float(res["fit"]["tau"]))


def save_day(res, cfg, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fit, base = res["fit"], res["base"]
    stats = dict(baseline=dict(nll=base["nll"], F=base["F"], term_structure_scale=base["smooth"],
                               scale_scan=base["scan"]),
                 fit=dict(K=fit["K"], nll=float(fit["nll"]), F=fit["F"], tau=fit["tau"], steps=fit["steps"]),
                 relinearization=[dict(stage=h["stage"], atmf_gap_volpts=h["gaps"][0], skew_change=h["gaps"][1],
                                       nll=float(h["nll"])) for h in res.get("relin", {}).get("history", [])],
                 checks=res["checks"], minutes=res["minutes"])
    json.dump(parameters(res), open(out / "parameters.json", "w"), indent=1)
    json.dump(stats, open(out / "summary.json", "w"), indent=1)
    json.dump(cfg.to_dict(), open(out / "config.json", "w"), indent=1)
    np.savez(out / "modes.npz", V=res["modes"]["V"], kappa=res["modes"]["kappa"])
    return out
