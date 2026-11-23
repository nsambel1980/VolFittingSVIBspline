"""Information-ranked correction modes.

1. Linear responses of the exact surface to all D = n_x (1 + n_t) coefficients, around the SSVI baseline.
2. Constraints: a correction must not change, to first order, the ATMF level or the ATM skew at any expiry
   (2 nT rows), nor the mass and forward of the first slice (2 rows). Their null space is admissible.
3. Within it, the generalized eigenproblem  (A^T W A) v = kappa P1 v : data information (curvature of the
   bid/ask likelihood) against prior information. Modes = eigenvectors, largest kappa first; V^T P1 V = I.
"""
import numpy as np
from scipy.linalg import eigh, null_space
from .dupire import LocalVarSurface
from .likelihood import QuoteLikelihood


def atm_rows(model):
    """First-order change of the ATMF total variance (relative) and of the ATM skew at each expiry,
    per column of the model's responses (coefficients, or modes)."""
    h = 2 * model.dk; kk = np.array([-h, 0.0, h]); L, S = [], []
    for s in range(model.nT):
        dw = model.interp(model.Phi[s], kk) / model.base.vega_w(kk, model.Tn[s])[:, None]
        L.append(dw[1] / model.mk["theta_atm"][s]); S.append((dw[2] - dw[0]) / (2 * h))
    return np.array(L), np.array(S)


def constraint_rows(model):
    L, S = atm_rows(model)
    return np.vstack([L, S, model.cons_rows])


def rank_modes(model, rows, lik):
    """Null space of `rows`, ranked by data / prior information at the model's reference surface."""
    R = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    Z = null_space(R, rcond=1e-10)
    _, W, _ = lik.newton_weights(model.w_ref_q)
    Az = model.A @ Z
    kap, v = eigh(Az.T @ (W[:, None] * Az), Z.T @ model.P1 @ Z)
    order = np.argsort(kap)[::-1]
    return Z @ v[:, order], kap[order]


def compute_modes(mk, base, cfg):
    model = LocalVarSurface(mk, base, cfg, a=np.zeros(cfg.n_x * (1 + cfg.n_t)))
    lik = QuoteLikelihood(mk, cfg.s_noise)
    rows = constraint_rows(model)
    V, kappa = rank_modes(model, rows, lik)
    L, S = atm_rows(model)
    return dict(V=V, kappa=kappa, n_admissible=V.shape[1], n_coefficients=model.D,
                max_atm_leak=float(np.abs(L @ V).max()), max_skew_leak=float(np.abs(S @ V).max()),
                frac_floored=float(model.frac_floored))
