"""Fitting: the SSVI baseline, then baseline parameters + K mode coefficients on the exact surface.

Objective     -ln L(bid/ask) + SSVI penalty + 1/2 a^T P1 a / tau^2 + 1/2 a^T Q a   (a = U c; Q: fixed wing anchor)
Optimizer     Levenberg-Marquardt; every trial step re-solves Dupire and recomputes the mode responses
Free energy   Laplace approximation at the optimum, tau integrated over a log grid; flat unit boxes on the
              baseline parameters (the Jacobian is scaled by their prior widths)
"""
import numpy as np
from scipy.optimize import minimize, lsq_linear
from scipy.special import logsumexp
from .ssvi import SSVISlices, initial_q, bounds, widths, penalty, psi_law
from .likelihood import QuoteLikelihood
from .dupire import LocalVarSurface


def num_hess(f, x, h):
    d = len(x); H = np.zeros((d, d)); f0 = f(x)
    for a in range(d):
        ea = np.zeros(d); ea[a] = h[a]
        H[a, a] = (f(x + ea) - 2 * f0 + f(x - ea)) / h[a] ** 2
        for b in range(a + 1, d):
            eb = np.zeros(d); eb[b] = h[b]
            H[a, b] = H[b, a] = (f(x + ea + eb) - f(x + ea - eb) - f(x - ea + eb) + f(x - ea - eb)) / (4 * h[a] * h[b])
    return H


def baseline(mk, q, cfg):
    return SSVISlices(mk["T"], q, cfg.skew_interp)


def jac_baseline(mk, q, cfg):
    """d w / d(q / width) at the quotes, from the SSVI formula."""
    nT = mk["nT"]; wd = widths(q, nT); J = np.zeros((len(mk["k"]), len(q)))
    for a in range(len(q)):
        h = np.zeros(len(q)); h[a] = 1e-6 * max(abs(q[a]), 1e-3)
        J[:, a] = (baseline(mk, q + h, cfg).w_at_quotes(mk) - baseline(mk, q - h, cfg).w_at_quotes(mk)) / (2 * h[a]) * wd[a]
    return J


# ---------------------------------------------------------------- zeroth order
def fit_baseline(mk, cfg):
    nT = mk["nT"]; lik = QuoteLikelihood(mk, cfg.s_noise); th = mk["theta_atm"]
    obj = lambda q: lik(baseline(mk, q, cfg).w_at_quotes(mk)) + penalty(q, nT)
    lo, hi = bounds(th)
    starts = (initial_q(th), initial_q(th, 1.6, 0.45, -0.8), initial_q(th, 0.8, 0.3, -0.5))
    best = None
    for q0 in starts:
        r = minimize(obj, q0, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(maxiter=20000, maxfun=200000))
        if best is None or r.fun < best.fun:
            best = r
    q = best.x; w = baseline(mk, q, cfg).w_at_quotes(mk)
    H = num_hess(lambda u: obj(q + u * widths(q, nT)), np.zeros(len(q)), np.full(len(q), 1e-4))
    F = lik(w) - 0.5 * len(q) * np.log(2 * np.pi) + 0.5 * np.linalg.slogdet(H)[1]
    return dict(q=q, F=float(F), nll=float(lik(w)))


# ---------------------------------------------------------------- free energy
def free_energy(model, lik, Jn, c, tau_grid, prior_center=None, U=None):
    """Laplace free energy at the current point, tau integrated on a log grid.
    Prior on the K mode coefficients: 1/2 |c + b|^2 / tau^2 + 1/2 c^T Qc c (b = prior_center, default 0;
    Qc = U^T Q U, the fixed wing anchor)."""
    d, K = Jn.shape[1], len(c)
    b = np.zeros(K) if prior_center is None else prior_center
    Qc = np.zeros((K, K)) if U is None else U.T @ model.Q @ U
    gw, _, W = lik.newton_weights(model.w_ref_q)
    X = np.hstack([Jn, model.A]); f0 = lik(model.w_ref_q); th0 = np.r_[np.zeros(d), c]
    out = []
    for g in tau_grid:
        Pk = np.eye(K) / np.exp(2 * g) + Qc
        P = np.zeros((d + K, d + K)); P[d:, d:] = Pk
        H = X.T @ (W[:, None] * X) + P
        grad_prior = np.r_[np.zeros(d), (c + b) / np.exp(2 * g) + Qc @ c]
        dth = np.linalg.solve(H, -(X.T @ gw) - grad_prior)
        th = th0 + dth; dl = X @ dth
        cc = th[d:] + b
        fq = f0 + gw @ dl + 0.5 * np.sum(W * dl ** 2) + 0.5 * cc @ cc / np.exp(2 * g) + 0.5 * th[d:] @ Qc @ th[d:]
        out.append(fq + 0.5 * np.linalg.slogdet(H)[1] - 0.5 * np.linalg.slogdet(Pk)[1] - 0.5 * d * np.log(2 * np.pi))
    out = np.array(out); j = int(np.argmin(out))
    F = -(logsumexp(-out) + np.log(tau_grid[1] - tau_grid[0]) - np.log(tau_grid[-1] - tau_grid[0]))
    return float(F), float(np.exp(tau_grid[j]))


def tau_grid(cfg):
    lo, hi, n = cfg.tau_grid
    return np.linspace(np.log(lo), np.log(hi), int(n))


# ---------------------------------------------------------------- Levenberg-Marquardt on the exact surface
def lm_fit(mk, cfg, q, U, c, tau, a_offset=None, prior_matrix=None, log=None):
    """Minimize -ln L + penalty + 1/2 (a_offset + U c)^T P (a_offset + U c) / tau^2 over (q, c).
    Default (no offset, P = P1 with U^T P1 U = I) is the plain prior 1/2 |c|^2 / tau^2."""
    nT, d, K = mk["nT"], len(q), U.shape[1]
    lik = QuoteLikelihood(mk, cfg.s_noise); lo, hi = bounds(mk["theta_atm"])
    a0 = np.zeros(U.shape[0]) if a_offset is None else a_offset

    def build(q_, c_):
        return LocalVarSurface(mk, baseline(mk, q_, cfg), cfg, a=a0 + U @ c_, U=U, c=c_) if a_offset is not None \
            else LocalVarSurface(mk, baseline(mk, q_, cfg), cfg, U=U, c=c_)

    cur = build(q, c)
    P = cur.P1 if prior_matrix is None else prior_matrix
    M = P / tau ** 2 + cur.Q                                       # smoothness / tau^2 + fixed wing anchor
    Pc = U.T @ M @ U; bvec = U.T @ M @ a0
    prior = lambda c_: 0.5 * (a0 + U @ c_) @ M @ (a0 + U @ c_)
    phi = lambda m_, q_, c_: lik(m_.w_ref_q) + penalty(q_, nT) + prior(c_)
    f_cur = phi(cur, q, c); lam = 1e-2; small = 0; steps = 0
    Lc = np.linalg.cholesky(Pc)
    for _ in range(cfg.max_iter):
        Jn = jac_baseline(mk, q, cfg); gw, W, _ = lik.newton_weights(cur.w_ref_q)
        Am = cur.A                                                  # responses along U at the current point
        r_lin = -gw / W + Am @ c
        X = np.hstack([Jn, Am])
        dg = np.sqrt(lam * (np.sum(W[:, None] * X ** 2, axis=0) + 1e-12))
        Xs = np.vstack([np.sqrt(W)[:, None] * X, np.hstack([np.zeros((K, d)), Lc.T]), np.diag(dg)])
        ys = np.r_[np.sqrt(W) * r_lin, -np.linalg.solve(Lc, bvec), dg * np.r_[np.zeros(d), c]]
        th = lsq_linear(Xs, ys).x
        q_try = np.clip(q + th[:d] * widths(q, nT), lo, hi); c_try = th[d:]
        trial = build(q_try, c_try); f_new = phi(trial, q_try, c_try)
        if not (np.isfinite(f_new) and f_new < f_cur):
            lam *= 5
            if lam > 1e7:
                break
            continue
        gain = (f_cur - f_new) / abs(f_cur); steps += 1
        cur, q, c, f_cur = trial, q_try, c_try, f_new; lam = max(lam / 3, 1e-6)
        if log:
            log(f"    step {steps}: -lnL = {lik(cur.w_ref_q):.1f}")
        small = small + 1 if gain < cfg.gain_tol else 0
        if small >= 3:
            break
    return cur, q, c, steps


def fit_modes(mk, q0, V, cfg, K=None, log=None):
    """Fit baseline parameters + the first K modes; tau chosen by free energy (a few rounds)."""
    K = cfg.n_modes if K is None else K
    U = V[:, :K]; q = np.asarray(q0, float).copy(); c = np.zeros(K); tau = 1.0
    grid = tau_grid(cfg); lik = QuoteLikelihood(mk, cfg.s_noise); total = 0
    for _ in range(cfg.tau_rounds):
        cur, q, c, steps = lm_fit(mk, cfg, q, U, c, tau, log=log); total += steps
        F, tau_new = free_energy(cur, lik, jac_baseline(mk, q, cfg), c, grid, U=U)
        if abs(np.log(tau_new / tau)) < 0.25:
            break
        tau = tau_new
    return dict(model=cur, q=q, c=c, U=U, a=U @ c, tau=tau, F=F, nll=lik(cur.w_ref_q), steps=total, K=K)
