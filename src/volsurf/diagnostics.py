"""Checks on a fitted exact surface: arbitrage, bid-ask coverage, ATMF / skew gaps, put-wing shape, expiry jumps."""
import numpy as np
from .black import density_from_calls


def coverage(mk, model):
    """Share of ALL quotes in range (not only the fitted ones) whose model price is inside [bid, ask]
    (ask-only: at most the ask; bid-only: at least the bid)."""
    per = []
    for s in range(model.nT):
        f = mk["full"][s]; c = model.calls(s, f["k"])
        per.append((c >= f["c_bid"] - 1e-15) & (c <= f["c_ask"] + 1e-15))
    return float(np.concatenate(per).mean()), [float(p.mean()) for p in per]


def price_score(mk, model):
    """Price errors in half-spreads on every quote in range, as in a desk's quote check:
    two-sided (c - mid) / h; ask-only max(c - ask, 0) / h; bid-only max(bid - c, 0) / h.
    Also counts: two-sided quotes outside [bid, ask], ask-only quotes above the ask, bid-only below the bid."""
    e, band, ask_b, bid_b, n = [], 0, 0, 0, 0
    for s in range(model.nT):
        f = mk["full"][s]; c = model.calls(s, f["k"]); h = f["h"]
        two = ~f["one_sided"]; ask_only = f["one_sided"] & np.isfinite(f["c_ask"]); bid_only = ~np.isfinite(f["c_ask"])
        r = np.where(two, (c - np.nan_to_num(f["c_mid"])) / h,
                     np.where(ask_only, np.maximum(c - f["c_ask"], 0) / h, np.maximum(f["c_bid"] - c, 0) / h))
        e.append(r); n += len(r)
        band += int(np.sum(two & ((c < f["c_bid"]) | (c > f["c_ask"]))))
        ask_b += int(np.sum(ask_only & (c > f["c_ask"]))); bid_b += int(np.sum(bid_only & (c < f["c_bid"])))
    e = np.concatenate(e)
    return dict(rows=n, price_rms_h=float(np.sqrt(np.mean(e ** 2))), band_misses=band, ask_breaches=ask_b,
                bid_breaches=bid_b)


def arbitrage(model, x=(-8.0, 6.0)):
    """Static-arbitrage checks, on the solver grid and on the delivered surface (what iv_exact returns).

    min_density_ratio      solver prices: smallest density / peak density (negative = butterfly; ~1e-10 rounding)
    min_price_change       solver prices: smallest change of any call over any time step (negative = calendar)
    delivered_butterfly    delivered calls: smallest butterfly C(K-) - 2C(K) + C(K+) on the solver's strike grid,
                           x = k / sqrt(theta) in `x`, normalized by C(F); in strike K, not log-strike
    delivered_calendar     delivered calls: smallest C(T_{i+1}, k) - C(T_i, k) at the same k (normalized)
    baseline_calendar      SSVI baseline: smallest increase of total variance between neighbouring maturities
                           on a fine grid out to |x| = 10 (negative = the baseline itself has calendar arbitrage)
    baseline_floored       share of solver nodes where the baseline local variance had to be floored
    """
    _, hist = model.full_solve(model.a_ref, all_steps=True)
    dens = [density_from_calls(model.cref[s], model.kg) for s in range(model.nT)]
    kg, K = model.kg, np.exp(model.kg)
    bf, cal, prev = [], [], None
    for s in range(model.nT):
        c = model.calls(s, kg); atm = float(model.calls(s, np.array([0.0]))[0])
        sel = (kg / np.sqrt(model.base.th_i[s]) > x[0]) & (kg / np.sqrt(model.base.th_i[s]) < x[1])
        slope = np.diff(c) / np.diff(K)
        b = np.diff(slope) * np.diff(K)[:-1]                  # = C(K-) - 2C(K) + C(K+) on a uniform strike grid
        bf.append(float(b[sel[1:-1]].min() / atm))
        if prev is not None:
            cal.append(float((c - prev)[sel].min()))
        prev = c
    return dict(min_density_ratio=float(min(d.min() / d.max() for d in dens)),
                min_price_change=float(np.min(np.diff(hist, axis=0))),
                delivered_butterfly=float(min(bf)), delivered_calendar=float(min(cal)) if cal else 0.0,
                baseline_calendar=model.base.calendar_check(), baseline_floored=float(model.frac_floored))


# min_density_ratio is a grid-accuracy check: far out in the wings (10+ sd) the stretched solver grid leaves
# relative density errors of ~1e-6 of the peak, i.e. butterflies worth < 1e-8 of the forward. The delivered
# butterfly and calendar checks below are the economic ones and keep strict tolerances.
ARB_TOL = dict(min_density_ratio=-1e-5, min_price_change=-1e-9, delivered_butterfly=-1e-9,
               delivered_calendar=-1e-9, baseline_calendar=-1e-10)


def arbitrage_ok(arb):
    """True when every check in `arb` is above its tolerance. Floored baseline nodes are reported
    (baseline_floored) but are not arbitrage: a floor raises a non-positive local variance to a tiny positive
    one, which keeps the solve arbitrage-free."""
    return all(arb[k] >= v for k, v in ARB_TOL.items())


def arbitrage_failures(arb):
    """The checks that fail, with their values and tolerances."""
    return {k: (arb[k], v) for k, v in ARB_TOL.items() if arb[k] < v}


def local_vol_ratio(model):
    nx = model.n_x
    nu = np.array([model.nu_basis(t) @ model.a_ref[nx:] for t in model.tg])
    return float(np.exp(nu.min())), float(np.exp(nu.max()))


def put_wing(model, maturities=(0.1, 0.25, 0.5, 1.0, 2.0), x=(-3.5, -1.0)):
    """Local maxima of local vol in the put wing (SPX should have none) and the largest local vol there."""
    xs = np.linspace(x[0], x[1], 251); maxima = 0; vmax = 0.0
    for t in maturities:
        lv = 100 * model.local_vol(t, xs)
        d = np.diff(lv); maxima += int(np.sum((d[:-1] > 0) & (d[1:] <= 0))); vmax = max(vmax, float(lv.max()))
    return dict(local_maxima=maxima, max_local_vol=vmax)


def expiry_jumps(model, x=(-3.0, 2.0)):
    """Largest local-vol jump (vol points) across each interior expiry."""
    xs = np.linspace(x[0], x[1], 101); out = []
    for Ti in model.Tn[1:-1]:
        lv = []
        for t in (Ti * (1 - 2e-3), Ti * (1 + 2e-3)):
            k = xs * np.sqrt(model.base.theta(t)); v0, _ = model.base.v0(k, t)
            nu = model.nu_basis(t, k) @ model.a_ref[model.n_x:]
            lv.append(100 * np.sqrt(v0 * np.exp(nu)))
        out.append(float(np.max(np.abs(lv[1] - lv[0]))))
    return out


def summary(mk, model):
    from .relinearize import gap_summary
    inside, by = coverage(mk, model)
    gaps = gap_summary(model)
    b = model.base; ch = b.checks()
    arb = arbitrage(model)
    return dict(inside_all=inside, inside_by_expiry=by, atmf_gap_volpts=gaps[0], skew_change=gaps[1],
                **price_score(mk, model),
                **arb, arbitrage_free=arbitrage_ok(arb), arbitrage_failures=arbitrage_failures(arb),
                calendar_ok=bool(ch["calendar_ok"]), local_var_ratio=local_vol_ratio(model), put_wing=put_wing(model),
                expiry_jumps_volpts=expiry_jumps(model), first_slice_min_coef=float(model.a_ref[:model.n_x].min()),
                butterfly_ok=bool(ch["butterfly_ok"].all()),
                wing_slopes_put=[float(x) for x in ch["wing_slope_put"]],
                wing_slopes_call=[float(x) for x in ch["wing_slope_call"]])
