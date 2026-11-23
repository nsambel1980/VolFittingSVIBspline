"""Checks on a fitted exact surface: arbitrage, bid-ask coverage, ATMF / skew gaps, put-wing shape, expiry jumps."""
import numpy as np
from .black import density_from_calls


def coverage(mk, model):
    """Share of ALL quoted strikes in range (not only the fitted ones) whose model vol is inside bid-ask."""
    per = []
    for s in range(model.nT):
        f = mk["full"][s]; m = model.iv_exact(s, f["k"])
        per.append((m >= f["iv_bid"]) & (m <= f["iv_ask"]))
    return float(np.concatenate(per).mean()), [float(p.mean()) for p in per]


def arbitrage(model):
    """Butterfly: smallest density / peak density over the expiries (negative = violation, ~1e-10 is rounding).
    Calendar: smallest change of any call price over any time step of the solver (negative = violation)."""
    _, hist = model.full_solve(model.a_ref, all_steps=True)
    dens = [density_from_calls(model.cref[s], model.kg) for s in range(model.nT)]
    return dict(min_density_ratio=float(min(d.min() / d.max() for d in dens)),
                min_price_change=float(np.min(np.diff(hist, axis=0))))


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
    return dict(inside_all=inside, inside_by_expiry=by, atmf_gap_volpts=gaps[0], skew_gap=gaps[1],
                **arbitrage(model), local_var_ratio=local_vol_ratio(model), put_wing=put_wing(model),
                expiry_jumps_volpts=expiry_jumps(model), first_slice_min_coef=float(model.a_ref[:model.n_x].min()),
                butterfly_ok=bool(ch["butterfly_ok"].all()),
                wing_slopes_put=[float(x) for x in ch["wing_slope_put"]],
                wing_slopes_call=[float(x) for x in ch["wing_slope_call"]])
