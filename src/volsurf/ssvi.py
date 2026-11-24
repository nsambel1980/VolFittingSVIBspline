"""Zeroth order: one SSVI slice per expiry, parametrized by ATMF total variance and ATM skew.

  w0(k) = 1/2 [theta + rho psi k + sqrt((psi k + theta rho)^2 + theta^2 (1 - rho^2))]
  ATM:   w0(0) = theta_i,  dw0/dk(0) = rho_i psi_i = s_i   (exactly)
  psi(theta) = eta theta^(1-gamma) (1 + theta)^(gamma-1)   (global curvature law),  rho_i = s_i / psi_i
  wings: w0 ~ psi (1 +- rho) |k| / 2;  butterfly (Gatheral-Jacquier) psi (1+|rho|) < 4, psi^2 (1+|rho|) <= 4 theta
No calendar arbitrage between slices (sufficient conditions, Hendriks & Martini 2019): theta and both wing slopes
psi (1 + rho) = psi + s and psi (1 - rho) = psi - s non-decreasing from one expiry to the next. They are
imposed as constraints in the fit (penalty), not just checked.

Between expiries theta(T) is a monotone PCHIP spline through (0, 0) and the expiries, and the two wing slopes
are monotone PCHIP functions of theta, so every interpolated slice also satisfies the conditions and the local
volatility has no jumps at the expiries. Before the first expiry and after the last, rho is held fixed and psi
follows the psi(theta) law, which is increasing in theta.

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


CAL_MARGIN = 1e-4      # calendar conditions are imposed with this margin, so a small penalized overshoot stays feasible


def calendar_gaps(q, nT):
    """Increments that must be >= 0 for no calendar arbitrage: theta, then the put and call wing slopes."""
    th, sk, psi, rho, eta, gam = unpack(q, nT)
    up, dn = psi * (1 + rho), psi * (1 - rho)
    return np.diff(th) / th[:-1], np.diff(dn), np.diff(up)


def penalty(q, nT):
    """Keeps |rho| < 1, the Gatheral-Jacquier butterfly conditions (which include Lee's wing bound) and the
    calendar conditions between slices."""
    th, sk, psi, rho, eta, gam = unpack(q, nT)
    r = np.abs(sk / psi)
    cal = sum(np.sum(np.maximum(CAL_MARGIN - g, 0) ** 2) for g in calendar_gaps(q, nT))
    return (np.sum(np.maximum(r - 0.98, 0) ** 2) + np.sum(np.maximum(psi * (1 + r) - 3.9, 0) ** 2)
            + np.sum(np.maximum(psi ** 2 * (1 + r) - 3.9 * th, 0) ** 2 / th ** 2) + cal) * 1e6


class SSVISlices:
    def __init__(self, T, q, skew_interp="pchip"):
        self.T = np.asarray(T, float); self.nT = len(self.T); self.q = np.asarray(q, float)
        th, sk, psi, rho, self.eta, self.gam = unpack(self.q, self.nT)
        self.th_i, self.sk_i, self.psi_i, self.rho_i = th, sk, psi, rho
        self._theta = PchipInterpolator(np.r_[0.0, self.T], np.r_[0.0, th])
        self.skew_interp = skew_interp
        up, dn = psi * (1 + rho), psi * (1 - rho)                    # wing slopes x 2
        if self.nT == 1:
            self._wings = lambda t: (up[0], dn[0])
        elif skew_interp == "pchip":
            if np.all(np.diff(th) > 0):
                fu, fd = PchipInterpolator(th, up), PchipInterpolator(th, dn)
            else:                                                     # infeasible theta: still defined
                o = np.argsort(th); fu = lambda t: np.interp(t, th[o], up[o]); fd = lambda t: np.interp(t, th[o], dn[o])
            self._wings = lambda t: (float(fu(t)), float(fd(t)))
        elif skew_interp == "linear":
            o = np.argsort(th)
            self._wings = lambda t: (float(np.interp(t, th[o], up[o])), float(np.interp(t, th[o], dn[o])))
        else:
            raise ValueError(f"skew_interp must be 'pchip' or 'linear', not {skew_interp!r}")

    # --- surface
    def theta(self, T):
        return float(self._theta(T))

    def slice_params(self, T):
        """(theta, psi, rho) at maturity T. At the expiries these are the fitted slices."""
        th = self.theta(T); lo, hi = self.th_i.min(), self.th_i.max()
        if th <= lo or th >= hi or self.nT == 1:                      # fixed rho, psi from the law
            j = int(np.argmin(self.th_i)) if th <= lo else int(np.argmax(self.th_i))
            return th, float(psi_law(th, self.eta, self.gam)), float(self.rho_i[j])
        up, dn = self._wings(th)
        psi = 0.5 * (up + dn)
        return th, psi, float(np.clip((up - dn) / (up + dn), -0.999, 0.999))

    def skew(self, T):
        th, psi, rho = self.slice_params(T)
        return psi * rho

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
        g = calendar_gaps(self.q, self.nT)
        return dict(butterfly_ok=(psi * (1 + np.abs(rho)) < 4) & (psi ** 2 * (1 + np.abs(rho)) <= 4 * th),
                    calendar_ok=bool(all(np.all(x >= 0) for x in g)),
                    wing_slope_put=psi * (1 - rho) / 2, wing_slope_call=psi * (1 + rho) / 2)

    def calendar_check(self, x=(-10.0, 10.0), n_k=401, n_t=400):
        """Smallest increment of total variance between neighbouring maturities on a fine grid
        (including between and beyond the expiries). Negative = calendar arbitrage in the baseline."""
        ts = np.geomspace(self.T[0] / 20, self.T[-1] * 1.5, n_t)
        k = np.linspace(x[0], x[1], n_k) * np.sqrt(self.th_i.max())
        W = np.array([self.w0(k, t) for t in ts])
        return float(np.min(np.diff(W, axis=0)))
