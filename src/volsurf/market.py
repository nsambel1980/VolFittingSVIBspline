"""Market inputs: from a Cboe delayed-quotes option chain (JSON), or directly from implied vols.

From prices (build_market, build_market_from_rows):
Per expiry:
  * forward F and discount factor D from put-call parity, C - P = D (F - K), by a weighted regression on
    strikes near spot (no rate or dividend inputs needed);
  * out-of-the-money quotes (puts below F, calls above) converted to normalized calls and implied total
    variance at log-forward-moneyness k = ln(K / F), for bid, ask and mid;
  * strikes kept within the configured range of x = k / sqrt(theta_ATM) and thinned to n_per_expiry strikes
    evenly spaced in x for the fit; every quoted strike in range is kept for out-of-sample checks.

From implied vols (build_market_from_vols, load_vol_csv): bid / ask (or mid) vols per strike and the forward
per expiry are given; the same strike range and thinning are applied.

Either way the returned "market" is the same dict:
  T (nT,)            years to expiry (calendar, to 9:30 ET for AM-settled SPX, 16:00 otherwise)
  k, i (N,)          log-moneyness and expiry index of each fitted quote
  z (N,)             mid total variance;  iv, iv_bid, iv_ask (N,) mid / bid / ask implied vols
  theta_atm (nT,)    ATM total variance from the quotes
  full [nT]          dicts with every quoted strike in range: k, x, K, iv_bid, iv_ask, iv_mid
  info [nT]          label, forward, discount factor and parity fit quality (when known)
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
    slices, info = [], []
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
        sl = _slice(k, w_b, w_m, w_a, T, cfg, K=Ks)
        slices.append(sl)
        info.append(dict(label=f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]}", root=root, ymd=ymd, T=T, F=F, D=D,
                         rate=-np.log(D) / T, theta=sl["theta"], n=len(sl["pick"]), n_parity=npar,
                         parity_rms=par_rms, atm_vol=np.sqrt(sl["theta"] / T)))
    return _assemble(slices, info, quote_time, spot)


def _slice(k, w_b, w_m, w_a, T, cfg, K=None):
    """One expiry: ATM variance, strikes in range, and the n_per_expiry strikes used in the fit."""
    o = np.argsort(k); k, w_b, w_m, w_a = k[o], w_b[o], w_m[o], w_a[o]
    K = np.exp(k) if K is None else np.asarray(K, float)[o]
    th = float(np.interp(0.0, k, w_m))
    if not (k[0] < 0 < k[-1]):
        raise ValueError(f"expiry T={T:.4f}: quotes must bracket the forward (k from {k[0]:.3f} to {k[-1]:.3f})")
    x = k / np.sqrt(th)
    ok = (x >= cfg.quote_x_lo) & (x <= cfg.quote_x_hi) & (w_b > 1e-9) & (w_a > w_b)
    full = dict(k=k[ok], x=x[ok], K=K[ok], iv_bid=np.sqrt(w_b[ok] / T), iv_ask=np.sqrt(w_a[ok] / T),
                iv_mid=np.sqrt(w_m[ok] / T))
    cand = np.where(ok)[0]
    targets = np.linspace(cfg.quote_x_lo, cfg.quote_x_hi, cfg.n_per_expiry)
    pick = np.array(sorted(set(cand[np.argmin(np.abs(x[cand][None, :] - targets[:, None]), axis=1)])))
    full["used"] = np.isin(full["k"], k[pick])
    return dict(T=T, theta=th, full=full, pick=pick, k=k[pick], w_m=w_m[pick], w_b=w_b[pick], w_a=w_a[pick])


def _assemble(slices, info, quote_time, spot):
    cat = lambda key, f=lambda sl, v: v: np.concatenate([f(sl, sl[key]) for sl in slices])
    vol = lambda sl, w: np.sqrt(w / sl["T"])
    return dict(T=np.array([sl["T"] for sl in slices]), k=cat("k"),
                i=np.concatenate([np.full(len(sl["k"]), s) for s, sl in enumerate(slices)]),
                z=cat("w_m"), iv=cat("w_m", vol), iv_bid=cat("w_b", vol), iv_ask=cat("w_a", vol), nT=len(slices),
                theta_atm=np.array([sl["theta"] for sl in slices]), info=info, full=[sl["full"] for sl in slices],
                quote_time=quote_time, spot=spot)


def build_market_from_vols(expiries, cfg, quote_time=None, spot=None, min_half_spread=1e-5):
    """Market from implied vols instead of prices. `expiries` is a list with one dict per expiry:

        T                    years to expiry (float), or
        expiry               a date / datetime / 'YYYY-MM-DD', with quote_time given (AM 9:30 settlement
                             if root == 'SPX', else 16:00)
        k                    log forward-moneyness ln(K/F), or
        strike + forward     strikes and the forward F
        iv_bid, iv_ask       bid and ask implied vols (decimals: 0.2 = 20%), or
        iv_mid               mid vols only; then half_spread (vol, scalar or per strike) if known, else the
                             spread is treated as ~0 and the likelihood is Gaussian with width cfg.s_noise,
                             so set s_noise to the size of your marking error
        label                optional name for plots and parameters.json
        discount             optional discount factor, stored for reference only

    Every quote is taken as given: no put-call parity, no out-of-the-money selection. The strike range
    (quote_x_lo, quote_x_hi) and the thinning to n_per_expiry strikes are as for prices.
    """
    slices, info = [], []
    for j, e in enumerate(sorted(expiries, key=lambda e: _years(e, quote_time))):
        T = _years(e, quote_time)
        if "k" in e:
            k = np.asarray(e["k"], float); F = e.get("forward")
            K = np.asarray(e["strike"], float) if "strike" in e else None
        elif "strike" in e and "forward" in e:
            K = np.asarray(e["strike"], float); F = float(e["forward"]); k = np.log(K / F)
        else:
            raise ValueError("each expiry needs k, or strike and forward")
        if "iv_bid" in e and "iv_ask" in e:
            b, a = np.asarray(e["iv_bid"], float), np.asarray(e["iv_ask"], float)
            m = np.asarray(e["iv_mid"], float) if "iv_mid" in e else 0.5 * (b + a)
        elif "iv_mid" in e:
            m = np.asarray(e["iv_mid"], float)
            h = np.maximum(np.broadcast_to(np.asarray(e.get("half_spread", 0.0), float), m.shape), min_half_spread)
            b, a = m - h, m + h
        else:
            raise ValueError("each expiry needs iv_bid and iv_ask, or iv_mid")
        if np.nanmedian(m) > 3:
            raise ValueError("vols look like percentages; pass decimals (0.2 for 20%)")
        good = np.isfinite(k) & np.isfinite(b) & np.isfinite(a) & np.isfinite(m) & (b > 0) & (a > b)
        sl = _slice(k[good], b[good] ** 2 * T, m[good] ** 2 * T, a[good] ** 2 * T, T, cfg,
                    K=None if K is None else K[good])
        slices.append(sl)
        lab = e.get("label") or (str(e["expiry"])[:10] if "expiry" in e else f"T={T:.3f}y")
        info.append(dict(label=lab, T=T, F=None if F is None else float(F), D=e.get("discount"),
                         theta=sl["theta"], n=len(sl["pick"]), atm_vol=np.sqrt(sl["theta"] / T)))
    return _assemble(slices, info, quote_time, spot)


def _years(e, quote_time):
    if "T" in e:
        return float(e["T"])
    if quote_time is None:
        raise ValueError("expiry given as a date: pass quote_time, or give T in years")
    d = e["expiry"]
    if isinstance(d, str):
        d = dt.datetime.fromisoformat(d)
    if not isinstance(d, dt.datetime):
        d = dt.datetime(d.year, d.month, d.day)
    if d.hour == 0 and d.minute == 0:
        d = d + (dt.timedelta(hours=9, minutes=30) if e.get("root") == "SPX" else dt.timedelta(hours=16))
    return (d - quote_time).total_seconds() / (365.0 * 86400)


def load_vol_csv(path):
    """Read implied vols from a CSV with a header row, one row per strike. Columns:
        expiry (YYYY-MM-DD) or T;  strike + forward, or k;  iv_bid + iv_ask, or iv_mid [+ half_spread];
        optional: root, label, discount.
    Returns the list of per-expiry dicts that build_market_from_vols takes."""
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no rows")
    key = "T" if "T" in rows[0] else "expiry"
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    out = []
    for g, rs in groups.items():
        e = {key: float(g) if key == "T" else g}
        for col in ("root", "label"):
            if col in rs[0] and rs[0][col]:
                e[col] = rs[0][col]
        for col in ("forward", "discount"):
            if col in rs[0] and rs[0][col]:
                e[col] = float(rs[0][col])
        for col in ("k", "strike", "iv_bid", "iv_ask", "iv_mid", "half_spread"):
            if col in rs[0]:
                e[col] = np.array([float(r[col]) if r[col] != "" else np.nan for r in rs])
        out.append(e)
    return out


def describe(mk):
    spot = f", spot {mk['spot']:.2f}" if mk.get("spot") is not None else ""
    lines = [f"quote time {mk['quote_time']}{spot}, {len(mk['z'])} fitted quotes"]
    for q in mk["info"]:
        fwd = f"F={q['F']:.2f}  " if q.get("F") is not None else ""
        rate = f"r={100*q['rate']:.2f}%  " if q.get("rate") is not None else ""
        lines.append(f"  {q['label']}  T={q['T']:.3f}y  {fwd}{rate}ATMF vol={100*q['atm_vol']:.2f}%  quotes={q['n']}")
    return "\n".join(lines)
