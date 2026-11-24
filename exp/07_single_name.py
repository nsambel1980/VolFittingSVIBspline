"""Single-name stress harness: a synthetic listed chain with weeklies, an earnings jump and ask-only wings,
fitted on ALL rows and scored with the metrics of the VolLab comparison.

  price RMS / h      root-mean-square price error in half-spreads on every out-of-the-money row:
                     two-sided -> (c - mid) / h;  ask-only -> max(c - ask, 0) / h  (h = ask / 2 there)
  band / ask misses  two-sided rows priced outside [bid, ask];  ask-only rows priced above the ask
  IV error           RMS model - true implied vol on two-sided rows (vol points)
  strike / week bend RMS second differences of local vol: 0.025 log-strike apart at fixed time, and at 1/4,
                     1/2, 3/4 of each expiry interval scaled to a 7-day spacing; on supported / weak cells
  shock move         RMS change of local vol after refitting with mids moved by +-0.1 half-spread
usage: python exp/07_single_name.py [--src DIR] [--label NAME] [--shocks] [--config key=value ...]
Writes results/single_name/<label>.json.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(Path(__file__).resolve().parents[1] / "src"))
ap.add_argument("--label", default="current")
ap.add_argument("--shocks", action="store_true")
ap.add_argument("--config", nargs="*", default=[])
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "single_name"))
a = ap.parse_args()
sys.path.insert(0, a.src)

import numpy as np                                                    # noqa: E402
from volsurf import Config, run_day                                   # noqa: E402
from volsurf.market import build_market_from_rows                     # noqa: E402
from volsurf.black import implied_w, call                             # noqa: E402
import importlib.util                                                 # noqa: E402
_spec = importlib.util.spec_from_file_location(                       # the current generator, also for old code
    "volsurf.synthetic", Path(__file__).resolve().parents[1] / "src" / "volsurf" / "synthetic.py")
_syn = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_syn)
book = _syn.book


def parse(kv):
    k, v = kv.split("=", 1)
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, v


def config(expiries):
    kw = dict(expiries=expiries)
    for k, v in map(parse, a.config):
        kw[k] = tuple(v) if isinstance(v, list) else v
    return Config(**kw)


def model_calls(model, s, k):
    return model.calls(s, k) if hasattr(model, "calls") else call(k, model.w_exact(s, k))


def score(bk, res):
    mk, model = res["mk"], res["model"]
    F = {q.get("ymd", q["label"]): q["F"] for q in mk["info"]}
    ymds = [y for _, y in bk["expiries"]]
    truth = {(r["ymd"], r["K"]): r for r in bk["true_rows"]}
    err, misses, asks, ivs, n = [], 0, 0, [], 0
    for s, ymd in enumerate(ymds):
        f = F[ymd]
        rows = [r for r in bk["rows"] if r["ymd"] == ymd and (r["cp"] == "P") == (r["K"] < 100.0)]
        K = np.array([r["K"] for r in rows]); k = np.log(K / f)
        intr = np.maximum(1 - np.exp(k), 0.0)
        bid = np.array([r["bid"] for r in rows]) / f + intr; ask = np.array([r["ask"] for r in rows]) / f + intr
        one = np.array([r["bid"] == 0 for r in rows])
        c = model_calls(model, s, k)
        h = np.where(one, (ask - intr) / 2, (ask - bid) / 2)
        mid = 0.5 * (bid + ask)
        r = np.where(one, np.maximum(c - ask, 0) / h, (c - mid) / h)
        err.append(r); n += len(r)
        misses += int(np.sum(~one & ((c < bid) | (c > ask)))); asks += int(np.sum(one & (c > ask)))
        T = mk["T"][s]; tw = ~one
        iv_m = np.sqrt(implied_w(k[tw], c[tw]) / T)
        iv_t = np.array([np.sqrt(implied_w(np.log(r_["K"] / 100.0), truth[(ymd, r_["K"])]["c"]) / T)
                         for r_ in np.array(rows)[tw]])
        ivs.append(100 * (iv_m - iv_t))
    e = np.concatenate(err); iv = np.concatenate(ivs)
    return dict(rows=n, price_rms_h=float(np.sqrt(np.mean(e ** 2))), band_misses=misses, ask_breaches=asks,
                iv_rms_volpts=float(np.sqrt(np.mean(iv ** 2))), iv_max_volpts=float(np.abs(iv).max()))


KS = np.arange(-0.6, 0.401, 0.025)


def lv_cells(model):
    """Local vol at the quarter / half / three-quarter times of every interval, on a fixed log-strike grid."""
    out = []
    Tn = model.Tn
    for i in range(len(Tn) - 1):
        a_, b_ = Tn[i], Tn[i + 1]; ts = [a_ + (b_ - a_) * f for f in (0.25, 0.5, 0.75)]
        lv, sup = [], []
        for t in ts:
            th = model.base.theta(t); x = KS / np.sqrt(th)
            lv.append(100 * model.local_vol(t, x)); sup.append(model.supported(t, x))
        out.append(dict(ts=ts, lv=np.array(lv), sup=np.array(sup)))
    return out


def roughness(cells):
    sb_s, sb_w, wb_s, wb_w = [], [], [], []
    for c in cells:
        lv, sup = c["lv"], c["sup"]
        b = lv[:, :-2] - 2 * lv[:, 1:-1] + lv[:, 2:]; m = sup[:, :-2] & sup[:, 1:-1] & sup[:, 2:]
        sb_s += list(b[m]); sb_w += list(b[~m])
        dt_ = c["ts"][1] - c["ts"][0]
        wb = (lv[0] - 2 * lv[1] + lv[2]) * (7 / 365 / dt_) ** 2; m = sup.all(0)
        wb_s += list(wb[m]); wb_w += list(wb[~m])
    rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) else None
    return dict(strike_bend_supported=rms(sb_s), strike_bend_weak=rms(sb_w), week_bend_supported=rms(wb_s),
                week_bend_weak=rms(wb_w), lv_max=float(max(np.nanmax(c["lv"]) for c in cells)))


def fit(shock=0.0):
    bk = book(shock=shock)
    cfg = config(bk["expiries"])
    t0 = time.time()
    try:
        mk = build_market_from_rows(bk["rows"], bk["spot"], bk["quote_time"], cfg)
        res = run_day(mk, cfg, log=lambda *x: None)
    except Exception as ex:                                            # report, do not hide
        return bk, None, dict(error=f"{type(ex).__name__}: {ex}"[:300],
                              where=traceback.format_exc().strip().splitlines()[-3][:200],
                              seconds=time.time() - t0)
    return bk, res, dict(seconds=time.time() - t0)


out = dict(label=a.label, config=a.config)
bk, res, meta = fit(); out["base"] = meta
if res is not None:
    out["base"].update(score(bk, res))
    cells = lv_cells(res["model"]); out["base"].update(roughness(cells))
    ch = res["checks"]
    out["base"].update({k: ch[k] for k in ch if k in ("arbitrage_free", "min_density_ratio", "min_price_change",
                                                       "delivered_butterfly", "delivered_calendar",
                                                       "baseline_calendar", "baseline_floored")})
    out["base"]["K"] = int(res["fit"]["K"]); out["base"]["F"] = float(res["fit"]["F"])
    if a.shocks:
        for sgn in (+1, -1):
            _, r2, m2 = fit(0.1 * sgn); key = "shock+" if sgn > 0 else "shock-"
            if r2 is None:
                out[key] = m2; continue
            c2 = lv_cells(r2["model"])
            d_s, d_w = [], []
            for c, cc in zip(cells, c2):
                d = cc["lv"] - c["lv"]; m = c["sup"] & cc["sup"]
                d_s += list(d[m]); d_w += list(d[~m])
            out[key] = dict(seconds=m2["seconds"], lv_move_supported=float(np.sqrt(np.mean(np.square(d_s)))),
                            lv_move_weak=float(np.sqrt(np.mean(np.square(d_w)))))
Path(a.out).mkdir(parents=True, exist_ok=True)
json.dump(out, open(Path(a.out) / f"{a.label}.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
