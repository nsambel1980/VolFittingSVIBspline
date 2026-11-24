"""Market inputs: from an option chain (prices), or from implied vols. Prices are the primary data.

From prices (build_market, build_market_from_rows):
Per expiry:
  * forward F and discount factor D from put-call parity, C - P = D (F - K), by a weighted regression on
    strikes near spot (no rate or dividend inputs needed);
  * out-of-the-money quotes (puts below F, calls above) as normalized call prices c = C / (D F) (puts by
    parity) at log-forward-moneyness k = ln(K / F): bid, ask and mid, kept as prices. Ask-only quotes (no bid)
    have the intrinsic value as their lower bound; bid-only quotes have no upper bound (inf).
  * every row is used by default (n_per_expiry = 0, unlimited strike range); a strike range and thinning to
    n_per_expiry strikes evenly spaced in x = k / sqrt(theta_ATM) are available.
From implied vols (build_market_from_vols, load_vol_csv): bid / ask (or mid) vols per strike are converted to
the same normalized call prices on the way in.

Either way the returned "market" is the same dict:
  T (nT,)                 years to expiry
  k, i (N,)               log-moneyness and expiry index of each fitted quote
  c_bid, c_ask, c_mid (N,) normalized call bounds and mid (c_ask = inf: bid only; c_mid = nan: one-sided)
  h (N,)                  half-spread in normalized price (ask-only: half the ask's time value)
  one_sided (N,)          True for ask-only / bid-only quotes
  iv, iv_bid, iv_ask (N,) the same in implied vol, for display only (nan / 0 / inf where undefined)
  theta_atm (nT,)         ATM total variance from the two-sided quotes
  full [nT]               dicts with every quote in range: k, x, K, c_bid, c_ask, c_mid, iv_*, one_sided, used
  info [nT]               label, forward, discount factor and parity fit quality (when known)
  quote_time, spot
"""
import json
import datetime as dt
import numpy as np
from .black import implied_w, call


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


def chain_expiries(rows, quote_time, cfg):
    """All (root, yymmdd) in the rows with at least cfg.min_days to expiry and quotes on both sides of spot."""
    out = []
    for root, ymd in sorted({(r["root"], r["ymd"]) for r in rows}, key=lambda e: e[1]):
        if years_to_expiry(root, ymd, quote_time) * 365 < cfg.min_days:
            continue
        out.append((root, ymd))
    return tuple(out)


def build_market_from_rows(rows, spot, quote_time, cfg):
    slices, info = [], []
    expiries = cfg.expiries if cfg.expiries else chain_expiries(rows, quote_time, cfg)
    for (root, ymd) in expiries:
        calls = {r["K"]: (r["bid"], r["ask"]) for r in rows if r["root"] == root and r["ymd"] == ymd and r["cp"] == "C"}
        puts = {r["K"]: (r["bid"], r["ask"]) for r in rows if r["root"] == root and r["ymd"] == ymd and r["cp"] == "P"}
        if not calls or not puts:
            raise ValueError(f"expiry {root} {ymd} not in the chain")
        T = years_to_expiry(root, ymd, quote_time)
        F, D, npar, par_rms = forward_discount(calls, puts, spot, cfg.parity_band)
        Ks, cbid, cask = [], [], []
        for K in sorted(set(calls) | set(puts)):
            b, a = (puts if K < F else calls).get(K, (0.0, 0.0))
            b = b if b > 0 else 0.0; a = a if a > 0 else np.inf
            if (b == 0.0 and not np.isfinite(a)) or a <= b:
                continue
            off = 0.0 if K >= F else 1.0 - K / F          # put -> call by parity (normalized)
            Ks.append(K); cbid.append(b / (D * F) + off); cask.append(a / (D * F) + off)
        Ks = np.array(Ks); k = np.log(Ks / F)
        sl = _slice(k, np.array(cbid), np.array(cask), T, cfg, K=Ks)
        slices.append(sl)
        info.append(dict(label=f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]}", root=root, ymd=ymd, T=T, F=F, D=D,
                         rate=-np.log(D) / T, theta=sl["theta"], n=len(sl["pick"]), n_parity=npar,
                         parity_rms=par_rms, atm_vol=np.sqrt(sl["theta"] / T)))
    return _assemble(slices, info, quote_time, spot)


def _ivs(k, c, T):
    """Implied vols of normalized calls where defined (time value > 0): nan otherwise; inf stays inf."""
    intr = np.maximum(1 - np.exp(k), 0.0)
    ok = np.isfinite(c) & (c > intr + 1e-14) & (c < 1.0)
    w = np.full(len(k), np.nan)
    if ok.any():
        w[ok] = implied_w(k[ok], c[ok], hi=50.0)
    w = np.where(np.isfinite(c) & (c <= intr + 1e-14), 0.0, w)
    return np.where(np.isinf(c), np.inf, np.sqrt(w / T))


def _slice(k, cb, ca, T, cfg, K=None):
    """One expiry from normalized call bounds: ATM variance, quotes in range, the quotes used in the fit.
    cb = intrinsic for ask-only quotes, ca = inf for bid-only quotes. Two-sided quotes set the ATM variance;
    with thinning, a one-sided quote is used only where no two-sided one is close to the target."""
    o = np.argsort(k); k, cb, ca = k[o], cb[o], ca[o]
    K = np.exp(k) if K is None else np.asarray(K, float)[o]
    intr = np.maximum(1 - np.exp(k), 0.0)
    cb = np.maximum(cb, intr)
    two = (cb > intr + 1e-12) & np.isfinite(ca) & (ca > cb)
    cm = np.where(two, 0.5 * (cb + ca), np.nan)
    if two.sum() < 2 or not (k[two][0] < 0 < k[two][-1]):
        raise ValueError(f"expiry T={T:.4f}: two-sided quotes must bracket the forward")
    near = np.argsort(np.abs(k[two]))[:6]                                  # ATM variance from the nearest mids
    kn, cn = k[two][near], cm[two][near]; o2 = np.argsort(kn)
    wn = implied_w(kn[o2], cn[o2], hi=50.0)
    th = float(np.interp(0.0, kn[o2], wn)) if kn.min() < 0 < kn.max() else float(np.mean(wn))
    x = k / np.sqrt(th)
    ok = (x >= cfg.quote_x_lo) & (x <= cfg.quote_x_hi) & (ca > cb)
    if not cfg.one_sided:
        ok &= two
    h = np.where(two, 0.5 * (ca - cb), np.where(np.isfinite(ca), 0.5 * (ca - intr), np.nan))
    if np.any(~np.isfinite(ca)):                                            # bid-only: scale from the bid
        h = np.where(~np.isfinite(ca), 0.5 * np.maximum(cb - intr, 1e-6), h)
    h = np.maximum(h, cfg.min_half_spread)
    ivb, iva, ivm = _ivs(k, cb, T), _ivs(k, ca, T), _ivs(k, cm, T)
    full = dict(k=k[ok], x=x[ok], K=K[ok], c_bid=cb[ok], c_ask=ca[ok], c_mid=cm[ok], h=h[ok],
                iv_bid=ivb[ok], iv_ask=iva[ok], iv_mid=ivm[ok], one_sided=~two[ok])
    cand = np.where(ok)[0]
    if cfg.n_per_expiry and cfg.n_per_expiry < len(cand):
        lo_, hi_ = max(cfg.quote_x_lo, x[cand].min()), min(cfg.quote_x_hi, x[cand].max())
        targets = np.linspace(lo_, hi_, cfg.n_per_expiry)
        gap = (hi_ - lo_) / max(cfg.n_per_expiry - 1, 1)
        cost = np.abs(x[cand][None, :] - targets[:, None]) + 0.5 * gap * (~two[cand])[None, :]
        pick = np.array(sorted(set(cand[np.argmin(cost, axis=1)])))
    else:
        pick = cand
    full["used"] = np.isin(full["k"], k[pick])
    return dict(T=T, theta=th, full=full, pick=pick, k=k[pick], cb=cb[pick], ca=ca[pick], cm=cm[pick],
                h=h[pick], one=~two[pick], ivb=ivb[pick], iva=iva[pick], ivm=ivm[pick])


def _assemble(slices, info, quote_time, spot):
    cat = lambda key: np.concatenate([sl[key] for sl in slices])
    return dict(T=np.array([sl["T"] for sl in slices]), k=cat("k"),
                i=np.concatenate([np.full(len(sl["k"]), s) for s, sl in enumerate(slices)]),
                c_bid=cat("cb"), c_ask=cat("ca"), c_mid=cat("cm"), h=cat("h"), one_sided=cat("one"),
                iv=cat("ivm"), iv_bid=cat("ivb"), iv_ask=cat("iva"), nT=len(slices),
                theta_atm=np.array([sl["theta"] for sl in slices]), info=info, full=[sl["full"] for sl in slices],
                quote_time=quote_time, spot=spot)


def build_market_from_vols(expiries, cfg, quote_time=None, spot=None, min_half_spread=1e-5):
    """Market from implied vols instead of prices; the vols are converted to normalized call prices.
    `expiries` is a list with one dict per expiry:

        T                    years to expiry (float), or
        expiry               a date / datetime / 'YYYY-MM-DD', with quote_time given (AM 9:30 settlement
                             if root == 'SPX', else 16:00)
        k                    log forward-moneyness ln(K/F), or
        strike + forward     strikes and the forward F
        iv_bid, iv_ask       bid and ask implied vols (decimals: 0.2 = 20%); NaN or 0 for a missing side
                             (ask-only / bid-only quotes, used when cfg.one_sided), or
        iv_mid               mid vols only, with half_spread (vol, scalar or per strike) if known; otherwise
                             a tiny spread, so the fit treats the mid as exact up to the noise in the likelihood
        label                optional name for plots and parameters.json
        discount             optional discount factor, stored for reference only

    Every quote is taken as given: no put-call parity, no out-of-the-money selection. The strike range
    (quote_x_lo, quote_x_hi) and thinning (n_per_expiry) are as for prices.
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
        elif "iv_mid" in e:
            m = np.asarray(e["iv_mid"], float)
            h = np.maximum(np.broadcast_to(np.asarray(e.get("half_spread", 0.0), float), m.shape), min_half_spread)
            b, a = m - h, m + h
        else:
            raise ValueError("each expiry needs iv_bid and iv_ask, or iv_mid")
        ref = np.where(np.isfinite(a) & (a > 0), a, b)
        if np.nanmedian(np.where(np.isfinite(ref), ref, np.nan)) > 3:
            raise ValueError("vols look like percentages; pass decimals (0.2 for 20%)")
        b = np.where(np.isfinite(b) & (b > 0), b, 0.0)                 # missing bid: ask-only
        a = np.where(np.isfinite(a) & (a > 0), a, np.inf)               # missing ask: bid-only
        good = np.isfinite(k) & (a > b) & ((b > 0) | np.isfinite(a))
        if not cfg.one_sided:
            good &= (b > 0) & np.isfinite(a)
        k, b, a = k[good], b[good], a[good]; K = None if K is None else K[good]
        ca = np.where(np.isinf(a), np.inf, call(k, np.where(np.isinf(a), 1.0, a) ** 2 * T))   # no ask: inf
        sl = _slice(k, call(k, b ** 2 * T), ca, T, cfg, K=K)
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
    lines = [f"quote time {mk['quote_time']}{spot}, {len(mk['k'])} fitted quotes"]
    for q in mk["info"]:
        fwd = f"F={q['F']:.2f}  " if q.get("F") is not None else ""
        rate = f"r={100*q['rate']:.2f}%  " if q.get("rate") is not None else ""
        lines.append(f"  {q['label']}  T={q['T']:.3f}y  {fwd}{rate}ATMF vol={100*q['atm_vol']:.2f}%  quotes={q['n']}")
    return "\n".join(lines)
