"""Fitting: the SSVI baseline, then baseline parameters + K mode coefficients on the exact surface.

Objective     -ln L(bid/ask, in price) + SSVI penalty + term-structure prior R_s(q)
              + 1/2 a^T P1 a / tau^2 + 1/2 a^T Q a      (a = U c; Q: fixed wing anchor)
Baseline      SSVI slices by L-BFGS-B from several starts; the scale s of the term-structure prior is chosen by
              the baseline's Laplace evidence on a grid
Optimizer     Levenberg-Marquardt; every trial step re-solves Dupire and recomputes the mode responses
Free energy   Laplace approximation at the optimum, tau integrated over a log grid; flat unit boxes on the
              baseline parameters (the Jacobian is scaled by their prior widths)
Modes         as many as carry information: kappa >= kappa_min (or a fixed n_modes)
"""
import numpy as np
from scipy.optimize import minimize, lsq_linear
from scipy.special import logsumexp
from .ssvi import SSVISlices, initial_q, bounds, widths, penalty, psi_law, smoothness, unpack, ssvi_w, \
    project_calendar
from .black import call, vega_w
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


def slice_prices(mk, q):
    """Normalized call prices of the SSVI slices at the fitted quotes, straight from the slice parameters
    (no interpolation objects: this is the inner loop of the baseline fit)."""
    th, sk, psi, rho, eta, gam = unpack(q, mk["nT"]); i = mk["i"]
    return call(mk["k"], ssvi_w(mk["k"], th[i], psi[i], rho[i]))


def slice_jacobian(mk, q):
    """Exact d c / d q at the fitted quotes (N x (2 nT + 2)), by the chain rule through
    w(theta, psi, rho), psi(theta, eta, gamma) and rho = s / psi."""
    nT = mk["nT"]; k = mk["k"]; i = mk["i"]
    th, sk, psi, rho, eta, gam = unpack(q, nT)
    t, p, r = th[i], psi[i], rho[i]
    u = p * k + t * r; R = np.sqrt(u ** 2 + t ** 2 * (1 - r ** 2))
    w = 0.5 * (t + r * p * k + R)
    w_t = 0.5 * (1 + (u * r + t * (1 - r ** 2)) / R)
    w_p = 0.5 * (r * k + u * k / R)
    w_r = 0.5 * (p * k + (u * t - t ** 2 * r) / R)
    free = (np.abs(sk / psi) < 0.999)[i]                              # rho not clipped
    r_s = np.where(free, 1 / p, 0.0); r_p = np.where(free, -sk[i] / p ** 2, 0.0)
    dpsi_th = p * ((1 - gam) / t + (gam - 1) / (1 + t))
    w_psi = w_p + w_r * r_p                                            # total derivative through psi
    vg = vega_w(k, w)
    J = np.zeros((len(k), 2 * nT + 2))
    rows = np.arange(len(k))
    J[rows, i] = vg * (w_t + w_psi * dpsi_th)
    J[rows, nT + i] = vg * w_r * r_s
    J[:, 2 * nT] = vg * w_psi * p / eta
    J[:, 2 * nT + 1] = vg * w_psi * p * np.log((1 + t) / t)
    return J


def jac_baseline(mk, q, cfg):
    """d c / d(q / width) at the quotes (prices), from the SSVI formula."""
    return slice_jacobian(mk, q) * widths(q, mk["nT"])[None, :]


def _num_grad(f, x, h=1e-7):
    g = np.zeros(len(x)); f0 = f(x)
    for a in range(len(x)):
        e = np.zeros(len(x)); e[a] = h * max(abs(x[a]), 1e-2)
        g[a] = (f(x + e) - f0) / e[a]
    return g


def _hess_from_grad(grad, x, h):
    d = len(x); H = np.zeros((d, d))
    for a in range(d):
        e = np.zeros(d); e[a] = h[a]
        H[:, a] = (grad(x + e) - grad(x - e)) / (2 * h[a])
    return 0.5 * (H + H.T)


# ---------------------------------------------------------------- zeroth order
def fit_baseline(mk, cfg, q_init=None, log=None):
    """SSVI slices fitted to the quotes, with the term-structure prior; its scale by Laplace evidence.
    q_init: a previous fit (e.g. yesterday's) used as an extra start."""
    nT = mk["nT"]; lik = QuoteLikelihood(mk, cfg); th = mk["theta_atm"]; T = mk["T"]
    lo, hi = bounds(th)

    def objective(q, sc):
        """value and gradient: data (exact gradient) + penalty and term-structure prior (cheap, numerical)."""
        c = slice_prices(mk, q); f, g, _ = lik.terms(c)
        extra = lambda x: penalty(x, nT) + smoothness(x, T, sc)
        val = float(np.sum(f)) + extra(q)
        return val, slice_jacobian(mk, q).T @ g + _num_grad(extra, q)

    run = lambda sc, q0: minimize(objective, np.clip(q0, lo, hi), args=(sc,), jac=True, method="L-BFGS-B",
                                  bounds=list(zip(lo, hi)), options=dict(maxiter=5000))
    scales = sorted(cfg.smooth_grid) if nT > 1 else [0.0]
    mid = scales[len(scales) // 2]
    starts = [initial_q(th), initial_q(th, 1.6, 0.45, -0.8), initial_q(th, 0.8, 0.3, -0.5)]
    if q_init is not None and len(q_init) == 2 * nT + 2:
        starts.append(np.asarray(q_init, float))
    best = min((run(mid, q0) for q0 in starts), key=lambda r: r.fun)
    scan = {}
    for sc in scales:
        r = run(sc, best.x); q = project_calendar(np.clip(r.x, lo, hi), nT)
        wd = widths(q, nT)
        H = _hess_from_grad(lambda u: objective(q + u * wd, sc)[1] * wd, np.zeros(len(q)), np.full(len(q), 1e-4))
        sign, ld = np.linalg.slogdet(H)
        F = objective(q, sc)[0] - 0.5 * len(q) * np.log(2 * np.pi) + 0.5 * (ld if sign > 0 else np.inf)
        scan[sc] = dict(q=q, F=float(F), nll=float(lik(slice_prices(mk, q))))
        if log:
            log(f"    term-structure scale {sc}: F {F:.1f}")
    sc = min(scan, key=lambda v: scan[v]["F"])
    out = dict(scan[sc]); out["smooth"] = sc
    out["scan"] = {float(k): dict(F=v["F"], nll=v["nll"]) for k, v in scan.items()}
    return out


# ---------------------------------------------------------------- free energy
def free_energy(model, lik, Jn, c, tau_grid, prior_center=None, U=None):
    """Laplace free energy at the current point, tau integrated on a log grid.
    Prior on the K mode coefficients: 1/2 |c + b|^2 / tau^2 + 1/2 c^T Qc c (b = prior_center, default 0;
    Qc = U^T Q U, the fixed wing anchor)."""
    d, K = Jn.shape[1], len(c)
    b = np.zeros(K) if prior_center is None else prior_center
    Qc = np.zeros((K, K)) if U is None else U.T @ model.Q @ U
    gw, _, W = lik.newton_weights(model.c_ref_q)
    X = np.hstack([Jn, model.A]); f0 = lik(model.c_ref_q); th0 = np.r_[np.zeros(d), c]
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
def lm_fit(mk, cfg, q, U, c, tau, a_offset=None, prior_matrix=None, log=None, smooth=0.0):
    """Minimize -ln L + penalty + 1/2 (a_offset + U c)^T P (a_offset + U c) / tau^2 over (q, c).
    Default (no offset, P = P1 with U^T P1 U = I) is the plain prior 1/2 |c|^2 / tau^2."""
    nT, d, K = mk["nT"], len(q), U.shape[1]
    lik = QuoteLikelihood(mk, cfg); lo, hi = bounds(mk["theta_atm"])
    a0 = np.zeros(U.shape[0]) if a_offset is None else a_offset

    def build(q_, c_):
        return LocalVarSurface(mk, baseline(mk, q_, cfg), cfg, a=a0 + U @ c_, U=U, c=c_) if a_offset is not None \
            else LocalVarSurface(mk, baseline(mk, q_, cfg), cfg, U=U, c=c_)

    cur = build(q, c)
    P = cur.P1 if prior_matrix is None else prior_matrix
    M = P / tau ** 2 + cur.Q                                       # smoothness / tau^2 + fixed wing anchor
    Pc = U.T @ M @ U; bvec = U.T @ M @ a0
    prior = lambda c_: 0.5 * (a0 + U @ c_) @ M @ (a0 + U @ c_)
    phi = lambda m_, q_, c_: lik(m_.c_ref_q) + penalty(q_, nT) + smoothness(q_, mk["T"], smooth) + prior(c_)
    f_cur = phi(cur, q, c); lam = 1e-2; small = 0; steps = 0
    Lc = np.linalg.cholesky(Pc)
    for _ in range(cfg.max_iter):
        Jn = jac_baseline(mk, q, cfg); gw, W, _ = lik.newton_weights(cur.c_ref_q)
        Am = cur.A                                                  # responses along U at the current point
        r_lin = -gw / W + Am @ c
        X = np.hstack([Jn, Am])
        dg = np.sqrt(lam * (np.sum(W[:, None] * X ** 2, axis=0) + 1e-12))
        Xs = np.vstack([np.sqrt(W)[:, None] * X, np.hstack([np.zeros((K, d)), Lc.T]), np.diag(dg)])
        ys = np.r_[np.sqrt(W) * r_lin, -np.linalg.solve(Lc, bvec), dg * np.r_[np.zeros(d), c]]
        th = lsq_linear(Xs, ys).x
        q_try = project_calendar(np.clip(q + th[:d] * widths(q, nT), lo, hi), nT); c_try = th[d:]
        trial = build(q_try, c_try); f_new = phi(trial, q_try, c_try)
        if not (np.isfinite(f_new) and f_new < f_cur):
            lam *= 5
            if lam > 1e7:
                break
            continue
        gain = (f_cur - f_new) / abs(f_cur); steps += 1
        cur, q, c, f_cur = trial, q_try, c_try, f_new; lam = max(lam / 3, 1e-6)
        if log:
            log(f"    step {steps}: -lnL = {lik(cur.c_ref_q):.1f}")
        small = small + 1 if gain < cfg.gain_tol else 0
        if small >= 3:
            break
    return cur, q, c, steps


def n_informative(kappa, cfg):
    """Number of modes to fit: n_modes if set, else every mode with kappa >= kappa_min (data information at
    least kappa_min times the prior's), at most n_modes_max."""
    if cfg.n_modes:
        return int(min(cfg.n_modes, len(kappa)))
    return int(np.clip(np.sum(np.asarray(kappa) >= cfg.kappa_min), 1, min(cfg.n_modes_max, len(kappa))))


def fit_modes(mk, q0, V, cfg, K=None, log=None, smooth=0.0, kappa=None):
    """Fit baseline parameters + the first K modes; tau chosen by free energy (a few rounds)."""
    if K is None:
        K = n_informative(kappa, cfg) if kappa is not None else (cfg.n_modes or V.shape[1])
    U = V[:, :K]; q = np.asarray(q0, float).copy(); c = np.zeros(K); tau = 1.0
    grid = tau_grid(cfg); lik = QuoteLikelihood(mk, cfg); total = 0
    for _ in range(cfg.tau_rounds):
        cur, q, c, steps = lm_fit(mk, cfg, q, U, c, tau, log=log, smooth=smooth); total += steps
        F, tau_new = free_energy(cur, lik, jac_baseline(mk, q, cfg), c, grid, U=U)
        if abs(np.log(tau_new / tau)) < 0.25:
            break
        tau = tau_new
    return dict(model=cur, q=q, c=c, U=U, a=U @ c, tau=tau, F=F, nll=lik(cur.c_ref_q), steps=total, K=K,
                smooth=smooth)
