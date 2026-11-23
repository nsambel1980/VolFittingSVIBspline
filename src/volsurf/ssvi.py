"""Zeroth order: one SSVI slice per expiry, parametrized by ATMF total variance and ATM skew.

  w0(k) = 1/2 [theta + rho psi k + sqrt((psi k + theta rho)^2 + theta^2 (1 - rho^2))]
  ATM:   w0(0) = theta_i,  dw0/dk(0) = rho_i psi_i = s_i   (exactly)
  psi(theta) = eta theta^(1-gamma) (1 + theta)^(gamma-1)   (global curvature law),  rho_i = s_i / psi_i
  wings: w0 ~ psi (1 +- rho) |k| / 2;  butterfly (Gatheral-Jacquier) psi (1+|rho|) < 4, psi^2 (1+|rho|) <= 4 theta
Between expiries theta(T) is a monotone PCHIP spline through (0, 0) and the expiries, and the skew is a PCHIP
function of theta (smooth, so the local volatility has no jumps at the expiries).

Parameter vector q = (theta_1..n, s_1..n, eta, gamma).
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from .black import durrleman, vega_w as _vega_w


def psi_law(th, eta, gam):
    return eta * th ** (1 - gam) * (1 + th) ** (gam - 1)


def ssvi_w(k, th, psi, rho):
    return 0.5 * (th + rho * psi * k + np.sqrt((psi * k + th * rho) ** 2 + th ** 2 * (1 - rho ** 2)))


def unpack(q, nT):
    th, sk, eta, gam = q[:nT], q[nT:2 * nT], q[2 * nT], q[2 * nT + 1]
    psi = psi_law(th, eta, gam)
    return th, sk, psi, np.clip(sk / psi, -0.999, 0.999), eta, gam


def initial_q(theta_atm, eta=1.2, gam=0.45, rho=-0.6):
    return np.r_[theta_atm, rho * psi_law(theta_atm, eta, gam), eta, gam]


def bounds(theta_atm):
    nT = len(theta_atm)
    lo = np.r_[0.5 * theta_atm, np.full(nT, -1.0), 0.05, 0.05]
    hi = np.r_[2.0 * theta_atm, np.full(nT, 1.0), 5.0, 0.95]
    return lo, hi


def widths(q, nT):
    """Flat prior box widths: theta +-25%, skew over |rho| <= 1, eta over 4, gamma over 1."""
    th, sk, psi, rho, eta, gam = unpack(q, nT)
    return np.r_[0.5 * th, 2 * psi, 4.0, 1.0]


def penalty(q, nT):
    """Keeps |rho| < 1 and the Gatheral-Jacquier butterfly conditions (which include Lee's wing bound)."""
    th, sk, psi, rho, eta, gam = unpack(q, nT)
    r = np.abs(sk / psi)
    return (np.sum(np.maximum(r - 0.98, 0) ** 2) + np.sum(np.maximum(psi * (1 + r) - 3.9, 0) ** 2)
            + np.sum(np.maximum(psi ** 2 * (1 + r) - 3.9 * th, 0) ** 2 / th ** 2)) * 1e6


class SSVISlices:
    def __init__(self, T, q, skew_interp="pchip"):
        self.T = np.asarray(T, float); self.nT = len(self.T); self.q = np.asarray(q, float)
        th, sk, psi, rho, self.eta, self.gam = unpack(self.q, self.nT)
        self.th_i, self.sk_i, self.psi_i, self.rho_i = th, sk, psi, rho
        self._theta = PchipInterpolator(np.r_[0.0, self.T], np.r_[0.0, th])
        if skew_interp == "pchip" and np.all(np.diff(th) > 0):
            self._skew = PchipInterpolator(th, sk, extrapolate=True)
        else:
            self._skew = lambda t: np.interp(t, th, sk)

    # --- surface
    def theta(self, T):
        return float(self._theta(T))

    def slice_params(self, T):
        th = self.theta(T); psi = psi_law(th, self.eta, self.gam)
        return th, psi, float(np.clip(self._skew(th) / psi, -0.999, 0.999))

    def w0(self, k, T):
        th, psi, rho = self.slice_params(T)
        return ssvi_w(np.asarray(k, float), th, psi, rho)

    def dTw0(self, k, T):
        h = 1e-4 * T
        return (self.w0(k, T + h) - self.w0(k, T - h)) / (2 * h)

    def vega_w(self, k, T):
        return _vega_w(k, self.w0(k, T))

    def v0(self, k, T, floor=1e-6):
        """Baseline local variance by Dupire: dw/dT / g_Durrleman. Returns (v0, mask of floored nodes)."""
        g, _ = durrleman(lambda kk: self.w0(kk, T), k)
        v = self.dTw0(k, T) / np.where(g > 1e-8, g, np.nan)
        bad = ~np.isfinite(v) | (v < floor)
        return np.where(bad, floor, v), bad

    def w_at_quotes(self, mk):
        out = np.zeros(len(mk["k"]))
        for s in range(self.nT):
            m = mk["i"] == s
            out[m] = ssvi_w(mk["k"][m], self.th_i[s], self.psi_i[s], self.rho_i[s])
        return out

    # --- checks
    def checks(self):
        psi, rho, th = self.psi_i, self.rho_i, self.th_i
        return dict(butterfly_ok=(psi * (1 + np.abs(rho)) < 4) & (psi ** 2 * (1 + np.abs(rho)) <= 4 * th),
                    wing_slope_put=psi * (1 - rho) / 2, wing_slope_call=psi * (1 + rho) / 2)
