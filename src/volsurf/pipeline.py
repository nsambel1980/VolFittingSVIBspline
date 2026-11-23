"""One day, end to end: market -> SSVI baseline -> modes -> fit -> relinearize -> checks."""
import json
import time
from pathlib import Path
import numpy as np
from .config import Config
from .market import build_market
from .fit import fit_baseline, fit_modes, baseline
from .modes import compute_modes
from .relinearize import relinearize
from .diagnostics import summary


def run_day(path, cfg=Config(), modes=None, relin=True, log=print):
    """Fit one option chain. `modes`: a previous day's modes (dict with V) to reuse; None computes them."""
    t0 = time.time()
    mk = build_market(path, cfg)
    base = fit_baseline(mk, cfg)
    log(f"  baseline: -lnL {base['nll']:.1f}, F {base['F']:.1f}")
    if modes is None:
        modes = compute_modes(mk, baseline(mk, base["q"], cfg), cfg)
        log(f"  modes: {modes['n_admissible']} admissible of {modes['n_coefficients']}; "
            f"kappa > 1: {int(np.sum(modes['kappa'] > 1))}")
    fit = fit_modes(mk, base["q"], modes["V"], cfg)
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


def parameters(res):
    """The fitted parameters, as plain numbers."""
    mk, b = res["mk"], res["model"].base
    return dict(quote_time=str(mk["quote_time"]), spot=mk["spot"],
                expiries=[f"20{q['ymd'][:2]}-{q['ymd'][2:4]}-{q['ymd'][4:]}" for q in mk["info"]],
                T=[float(t) for t in mk["T"]],
                forward=[float(q["F"]) for q in mk["info"]], discount=[float(q["D"]) for q in mk["info"]],
                atmf_vol=[float(x) for x in np.sqrt(b.th_i / mk["T"])], atmf_total_variance=[float(x) for x in b.th_i],
                atm_skew=[float(x) for x in b.sk_i], eta=float(b.eta), gamma=float(b.gam),
                q=[float(x) for x in res["q"]], coefficients=[float(x) for x in res["a"]], tau=float(res["fit"]["tau"]))


def save_day(res, cfg, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fit, base = res["fit"], res["base"]
    stats = dict(baseline=dict(nll=base["nll"], F=base["F"]),
                 fit=dict(K=fit["K"], nll=float(fit["nll"]), F=fit["F"], tau=fit["tau"], steps=fit["steps"]),
                 relinearization=[dict(stage=h["stage"], atmf_gap_volpts=h["gaps"][0], skew_gap=h["gaps"][1],
                                       nll=float(h["nll"])) for h in res.get("relin", {}).get("history", [])],
                 checks=res["checks"], minutes=res["minutes"])
    json.dump(parameters(res), open(out / "parameters.json", "w"), indent=1)
    json.dump(stats, open(out / "summary.json", "w"), indent=1)
    json.dump(cfg.to_dict(), open(out / "config.json", "w"), indent=1)
    np.savez(out / "modes.npz", V=res["modes"]["V"], kappa=res["modes"]["kappa"])
    return out
