"""One round of relinearization at the fitted solution.

The mode constraints hold to first order only; with corrections of realistic size the surface's actual ATMF
level drifts slightly from its parameter theta_i. One round fixes that:
  1. project the coefficients back onto {actual ATMF = theta_i, mass and forward kept} with the smallest step
     in the prior metric (a few Newton steps),
  2. recompute the modes as the null space of the constraints AT THAT POINT, ranked by information,
  3. refit (q, c') with a = a_c + U' c' and the same prior 1/2 a^T P1 a / tau^2.
"""
import numpy as np
from .dupire import LocalVarSurface
from .modes import rank_modes
from .likelihood import QuoteLikelihood
from .fit import baseline, lm_fit


def gaps_and_rows(model):
    """Actual minus parameter ATMF level (relative) and ATM skew per expiry, their linear rows in the full
    coefficient space, and the first-slice conservation rows. Same +-2dk stencil for model and baseline."""
    h = 2 * model.dk; kk = np.array([-h, 0.0, h]); L, S, gl, gs = [], [], [], []
    th_atm = model.mk["theta_atm"]
    for s in range(model.nT):
        w0 = model.base.w0(kk, model.Tn[s]); w = model.w_exact(s, kk)
        gl.append((w[1] - w0[1]) / th_atm[s]); gs.append((w[2] - w[0] - (w0[2] - w0[0])) / (2 * h))
        dw = model.interp(model.Phi[s], kk) / model.base.vega_w(kk, model.Tn[s])[:, None]
        L.append(dw[1] / th_atm[s]); S.append((dw[2] - dw[0]) / (2 * h))
    return np.r_[gl, gs], np.vstack([L, S]), model.cons_rows


def gap_summary(model):
    """Largest |actual - parameter| over expiries: ATMF in vol points, ATM skew of total variance."""
    g, _, _ = gaps_and_rows(model)
    nT = model.nT; th = model.mk["theta_atm"]
    atm_vol = 100 * np.abs(g[:nT]) * th / (2 * np.sqrt(th * model.Tn))
    return float(atm_vol.max()), float(np.abs(g[nT:]).max())


def project(mk, cfg, q, a, P1, iters=4):
    for _ in range(iters):
        m = LocalVarSurface(mk, baseline(mk, q, cfg), cfg, a=a)
        g, C, E = gaps_and_rows(m)
        nT = m.nT
        Cf = np.vstack([C[:nT], E]); r = np.r_[g[:nT], np.zeros(len(E))]       # ATMF level only
        Pi = np.linalg.inv(P1)
        a = a - Pi @ Cf.T @ np.linalg.solve(Cf @ Pi @ Cf.T, r)
    return a, LocalVarSurface(mk, baseline(mk, q, cfg), cfg, a=a)


def relinearize(mk, cfg, fit, log=None):
    """fit: output of fit.fit_modes. Returns q, a (full coefficients), the model and the gap history."""
    q, a, tau, K = fit["q"].copy(), fit["a"].copy(), fit["tau"], fit["K"]
    lik = QuoteLikelihood(mk, cfg); smooth = fit.get("smooth", 0.0)
    model = LocalVarSurface(mk, baseline(mk, q, cfg), cfg, a=a); P1 = model.P1
    history = [dict(stage="fit", gaps=gap_summary(model), nll=lik(model.c_ref_q))]
    for rnd in range(cfg.relin_rounds):
        a, model = project(mk, cfg, q, a, P1)
        _, C, E = gaps_and_rows(model)
        Up, _ = rank_modes(model, np.vstack([C[:model.nT], E]), lik)
        Up = Up[:, :K]
        cur, q, cp, steps = lm_fit(mk, cfg, q, Up, np.zeros(K), tau, a_offset=a.copy(), prior_matrix=P1, log=log,
                                   smooth=smooth)
        a = a + Up @ cp
        model = LocalVarSurface(mk, baseline(mk, q, cfg), cfg, a=a)
        history.append(dict(stage=f"round {rnd + 1}", gaps=gap_summary(model), nll=lik(model.c_ref_q), steps=steps))
    return dict(q=q, a=a, tau=tau, model=model, history=history)
