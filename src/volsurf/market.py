"""Market inputs from a Cboe delayed-quotes option chain (JSON).

Per expiry:
  * forward F and discount factor D from put-call parity, C - P = D (F - K), by a weighted regression on
    strikes near spot (no rate or dividend inputs needed);
  * out-of-the-money quotes (puts below F, calls above) converted to normalized calls and implied total
    variance at log-forward-moneyness k = ln(K / F), for bid, ask and mid;
  * strikes kept within the configured range of x = k / sqrt(theta_ATM) and thinned to n_per_expiry strikes
    evenly spaced in x for the fit; every quoted strike in range is kept for out-of-sample checks.

The returned "market" is a dict:
  T (nT,)            years to expiry (calendar, to 9:30 ET for AM-settled SPX, 16:00 otherwise)
  k, i (N,)          log-moneyness and expiry index of each fitted quote
  z (N,)             mid total variance;  iv, iv_bid, iv_ask (N,) mid / bid / ask implied vols
  theta_atm (nT,)    ATM total variance from the quotes
  full [nT]          dicts with every quoted strike in range: k, x, K, iv_bid, iv_ask, iv_mid
  info [nT]          forward, discount factor, implied rate, parity fit quality
  quote_time, spot
"""
import json
import datetime as dt
import numpy as np
from .black import implied_w


def load_chain(path):
    with open(path) as f:
        d = json.load(f)
    ltt = d["data"].get("last_trade_time")
    quote_time = dt.datetime.fromisoformat(ltt[:19]) if ltt else None
    rows = []
    for o in d["data"]["options"]:
        s = o["option"]
        rows.append(dict(root=s[:-15], ymd=s[-15:-9], cp=s[-9], K=int(s[-8:]) / 1000.0,
                         bid=float(o["bid"]), ask=float(o["ask"])))
    return dict(rows=rows, quote_time=quote_time, spot=float(d["data"]["current_price"]))


def years_to_expiry(root, ymd, quote_time):
    day = dt.datetime.strptime(ymd, "%y%m%d")
    settle = day + (dt.timedelta(hours=9, minutes=30) if root == "SPX" else dt.timedelta(hours=16))
    return (settle - quote_time).total_seconds() / (365.0 * 86400)


def forward_discount(calls, puts, spot, band=0.08):
    """Weighted regression of mid(C) - mid(P) on K near spot. Returns F, D, number of strikes, weighted rms."""
    K, y, wt = [], [], []
    for k in sorted(set(calls) & set(puts)):
        cb, ca = calls[k]; pb, pa = puts[k]
        if abs(k / spot - 1) < band and cb > 0 and pb > 0 and ca > cb and pa > pb:
            K.append(k); y.append(0.5 * (cb + ca) - 0.5 * (pb + pa))
            wt.append(1.0 / ((ca - cb) ** 2 + (pa - pb) ** 2))
    K, y, wt = np.array(K), np.array(y), np.array(wt)
    A = np.vstack([np.ones_like(K), K]).T
    coef = np.linalg.solve(A.T @ (wt[:, None] * A), A.T @ (wt * y))
    D = -coef[1]; F = coef[0] / D
    res = y - A @ coef
    return F, D, len(K), float(np.sqrt(np.average(res ** 2, weights=wt)))


def build_market(path, cfg, quote_time=None):
    chain = load_chain(path)
    qt = quote_time or chain["quote_time"]
    if qt is None:
        raise ValueError("no quote time in the file; pass quote_time")
    return build_market_from_rows(chain["rows"], chain["spot"], qt, cfg)


def build_market_from_rows(rows, spot, quote_time, cfg):
    ks, idx, z, ivs, ivb, iva, Ts, info, fulls = [], [], [], [], [], [], [], [], []
    for (root, ymd) in cfg.expiries:
        calls = {r["K"]: (r["bid"], r["ask"]) for r in rows if r["root"] == root and r["ymd"] == ymd and r["cp"] == "C"}
        puts = {r["K"]: (r["bid"], r["ask"]) for r in rows if r["root"] == root and r["ymd"] == ymd and r["cp"] == "P"}
        if not calls or not puts:
            raise ValueError(f"expiry {root} {ymd} not in the chain")
        T = years_to_expiry(root, ymd, quote_time)
        F, D, npar, par_rms = forward_discount(calls, puts, spot, cfg.parity_band)
        Ks, cmid, cbid, cask = [], [], [], []
        for K in sorted(set(calls) | set(puts)):
            b, a = (puts if K < F else calls).get(K, (0.0, 0.0))
            if b <= 0 or a <= b:
                continue
            off = 0.0 if K >= F else 1.0 - K / F          # put -> call by parity (normalized)
            Ks.append(K); cbid.append(b / (D * F) + off); cask.append(a / (D * F) + off)
            cmid.append(0.5 * (a + b) / (D * F) + off)
        Ks = np.array(Ks); k = np.log(Ks / F)
        w_m, w_b, w_a = (implied_w(k, np.array(c)) for c in (cmid, cbid, cask))
        th = float(np.interp(0.0, k, w_m))
        x = k / np.sqrt(th)
        ok = (x >= cfg.quote_x_lo) & (x <= cfg.quote_x_hi) & (w_b > 1e-9) & (w_a > w_b)
        full = dict(k=k[ok], x=x[ok], K=Ks[ok], iv_bid=np.sqrt(w_b[ok] / T), iv_ask=np.sqrt(w_a[ok] / T),
                    iv_mid=np.sqrt(w_m[ok] / T))
        cand = np.where(ok)[0]
        targets = np.linspace(cfg.quote_x_lo, cfg.quote_x_hi, cfg.n_per_expiry)
        pick = sorted(set(cand[np.argmin(np.abs(x[cand][None, :] - targets[:, None]), axis=1)]))
        full["used"] = np.isin(full["k"], k[pick]); fulls.append(full)
        s = len(Ts)
        ks.append(k[pick]); idx.append(np.full(len(pick), s)); z.append(w_m[pick])
        ivs.append(np.sqrt(w_m[pick] / T)); ivb.append(np.sqrt(w_b[pick] / T)); iva.append(np.sqrt(w_a[pick] / T))
        Ts.append(T)
        info.append(dict(root=root, ymd=ymd, T=T, F=F, D=D, rate=-np.log(D) / T, theta=th, n=len(pick),
                         n_parity=npar, parity_rms=par_rms, atm_vol=np.sqrt(th / T)))
    return dict(T=np.array(Ts), k=np.concatenate(ks), i=np.concatenate(idx), z=np.concatenate(z),
                iv=np.concatenate(ivs), iv_bid=np.concatenate(ivb), iv_ask=np.concatenate(iva), nT=len(Ts),
                theta_atm=np.array([q["theta"] for q in info]), info=info, full=fulls,
                quote_time=quote_time, spot=spot)


def describe(mk):
    lines = [f"quote time {mk['quote_time']}, spot {mk['spot']:.2f}, {len(mk['z'])} fitted quotes"]
    for q in mk["info"]:
        lines.append(f"  {q['root']} 20{q['ymd'][:2]}-{q['ymd'][2:4]}-{q['ymd'][4:]}  T={q['T']:.3f}y  "
                     f"F={q['F']:.2f}  r={100*q['rate']:.2f}%  ATMF vol={100*q['atm_vol']:.2f}%  quotes={q['n']}")
    return "\n".join(lines)
