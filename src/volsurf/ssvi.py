"""Zeroth order: one SSVI slice per expiry, parametrized by ATMF total variance and ATM skew.

  w0(k) = 1/2 [theta + rho psi k + sqrt((psi k + theta rho)^2 + theta^2 (1 - rho^2))]
  ATM:   w0(0) = theta_i,  dw0/dk(0) = rho_i psi_i = s_i   (exactly)
  psi(theta) = eta theta^(1-gamma) (1 + theta)^(gamma-1)   (global curvature law),  rho_i = s_i / psi_i
  wings: w0 ~ psi (1 +- rho) |k| / 2;  butterfly (Gatheral-Jacquier) psi (1+|rho|) < 4, psi^2 (1+|rho|) <= 4 theta
No calendar arbitrage between slices (eSSVI conditions, Hendriks & Martini 2019): theta non-decreasing, both wing
slopes psi + s and psi - s non-decreasing, and phi = psi / theta non-increasing. The psi(theta) law gives the
last one at every maturity; the first two are imposed exactly (penalty in the fit, then `project_calendar`).
Rising wing slopes alone are NOT enough: a random test finds ~10% of such slice pairs crossing; adding phi
non-increasing, none.

Between expiries theta(T) is a monotone PCHIP spline through (0, 0) and the expiries; psi = psi(theta(T)) stays on
the law; the skew x = 2 s is a C1 cubic Hermite function of S = 2 psi whose slope stays in [-1, 1], so both wing
slopes (S +- x) / 2 rise with maturity. Every pair of maturities then satisfies all three conditions, and the
local volatility has no jumps at the expiries. Before the first expiry and after the last, rho is held fixed.

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


def project_calendar(q, nT, margin=CAL_MARGIN):
    """The nearest-in-order feasible parameters: theta increasing (relative margin) and both wing slopes
    psi +- s non-decreasing (|s_i - s_(i-1)| <= psi_i - psi_(i-1) - margin), by one forward pass. The penalty
    keeps fitted parameters close to feasible; this makes them exactly feasible, so the baseline has no
    calendar arbitrage whatever the optimizer did."""
    q = np.array(q, float); th = q[:nT].copy()
    for i in range(1, nT):
        th[i] = max(th[i], th[i - 1] * (1 + margin))
    eta, gam = q[2 * nT], q[2 * nT + 1]
    psi = psi_law(th, eta, gam)
    s = np.clip(q[nT:2 * nT], -0.999 * psi, 0.999 * psi)
    for i in range(1, nT):
        room = max(psi[i] - psi[i - 1] - margin, 0.0)
        s[i] = np.clip(s[i], s[i - 1] - room, s[i - 1] + room)
        s[i] = np.clip(s[i], -0.999 * psi[i], 0.999 * psi[i])
    q[:nT], q[nT:2 * nT] = th, s
    return q


def penalty(q, nT):
    """Keeps |rho| < 1, the Gatheral-Jacquier butterfly conditions (which include Lee's wing bound) and the
    calendar conditions between slices."""
    th, sk, psi, rho, eta, gam = unpack(q, nT)
    r = np.abs(sk / psi)
    cal = sum(np.sum(np.maximum(CAL_MARGIN - g, 0) ** 2) for g in calendar_gaps(q, nT))
    return (np.sum(np.maximum(r - 0.98, 0) ** 2) + np.sum(np.maximum(psi * (1 + r) - 3.9, 0) ** 2)
            + np.sum(np.maximum(psi ** 2 * (1 + r) - 3.9 * th, 0) ** 2 / th ** 2)) * 1e6 + cal * 1e9


def _lipschitz_hermite_slopes(S, x):
    """End derivatives (left, right) per interval of a cubic Hermite x(S) with |dx/dS| <= 1 everywhere, for data
    whose secants d_j lie in [-1, 1]. |x'| <= 1 means S + x and S - x both increase, a monotone-interpolation
    problem (Fritsch-Carlson); on interval j the box 1 + m <= 3 (1 + d_j), 1 - m <= 3 (1 - d_j), |m| <= 1 is
    sufficient. Each node takes the average of its neighbouring secants, clipped into each neighbouring
    interval's box: the same value on both sides (C1) unless the secants are extreme and of opposite sign,
    where the curve keeps |x'| <= 1 with a kink."""
    d = np.diff(x) / np.diff(S)
    want = np.r_[d[0], 0.5 * (d[:-1] + d[1:]), d[-1]] if len(d) > 1 else np.r_[d, d]
    box = lambda v: (max(-1.0, 3 * v - 2), min(1.0, 2 + 3 * v))
    left = np.array([np.clip(want[j], *box(d[j])) for j in range(len(d))])
    right = np.array([np.clip(want[j + 1], *box(d[j])) for j in range(len(d))])
    return left, right


def smoothness(q, T, scale):
    """-ln prior of a smooth term structure (proper, so `scale` can be chosen by evidence):
    Cauchy priors on the increments of log forward variance ln[(theta_i - theta_(i-1)) / (T_i - T_(i-1))] and of
    rho_i between neighbouring expiries, with scale `scale` and 0.5 `scale` per sqrt(ln T step). The heavy tails
    let one interval carry an earnings-type jump in variance without roughening the others."""
    nT = len(T)
    if nT < 2 or not scale:
        return 0.0
    th, sk, psi, rho, eta, gam = unpack(q, nT)
    Tp = np.r_[0.0, T]; thp = np.r_[0.0, th]
    fwd = np.maximum(np.diff(thp) / np.diff(Tp), 1e-12)
    tm = np.r_[T[0] / 2, np.sqrt(T[:-1] * T[1:])]                       # where each forward variance lives
    lf = np.log(fwd); dl = np.sqrt(np.diff(np.log(tm)))
    out = 0.0
    for d, sc in ((np.diff(lf), scale * dl), (np.diff(rho), 0.5 * scale * np.sqrt(np.diff(np.log(T))))):
        out += np.sum(np.log(np.pi * sc) + np.log1p((d / sc) ** 2))
    return float(out)


class SSVISlices:
    def __init__(self, T, q, skew_interp="pchip"):
        self.T = np.asarray(T, float); self.nT = len(self.T); self.q = np.asarray(q, float)
        th, sk, psi, rho, self.eta, self.gam = unpack(self.q, self.nT)
        self.th_i, self.sk_i, self.psi_i, self.rho_i = th, sk, psi, rho
        self._theta = PchipInterpolator(np.r_[0.0, self.T], np.r_[0.0, th])
        self._dtheta_end = max(float(self._theta.derivative()(self.T[-1])), 0.0)
        self.skew_interp = skew_interp
        self._S = 2 * psi; self._x = 2 * psi * rho                   # sum and difference of the wing slopes
        if skew_interp not in ("pchip", "linear"):
            raise ValueError(f"skew_interp must be 'pchip' or 'linear', not {skew_interp!r}")
        self._m = _lipschitz_hermite_slopes(self._S, self._x) if skew_interp == "pchip" and self.nT > 1 else None

    # --- surface
    def theta(self, T):
        """ATM total variance: monotone PCHIP through (0, 0) and the expiries; beyond the last expiry, linear
        with the end slope (a cubic extrapolation can turn down)."""
        Tn = self.T[-1]
        if T > Tn:
            return float(self.th_i[-1] + self._dtheta_end * (T - Tn))
        return float(self._theta(T))

    def slice_params(self, T):
        """(theta, psi, rho) at maturity T. At the expiries these are the fitted slices."""
        th = self.theta(T); lo, hi = self.th_i.min(), self.th_i.max()
        psi = float(psi_law(th, self.eta, self.gam))
        if th <= lo or th >= hi or self.nT == 1 or np.any(np.diff(self._S) <= 0):   # fixed rho
            j = int(np.argmin(self.th_i)) if th <= lo else int(np.argmax(self.th_i)) if th >= hi else \
                int(np.argmin(np.abs(self.th_i - th)))
            return th, psi, float(self.rho_i[j])
        S = 2 * psi; Si, xi = self._S, self._x
        j = int(np.clip(np.searchsorted(Si, S) - 1, 0, len(Si) - 2))
        h = Si[j + 1] - Si[j]; u = (S - Si[j]) / h
        if self._m is None:                                            # linear in S
            x = xi[j] + u * (xi[j + 1] - xi[j])
        else:                                                          # C1 cubic Hermite, |dx/dS| <= 1
            m0, m1 = self._m[0][j], self._m[1][j]
            x = ((2 * u ** 3 - 3 * u ** 2 + 1) * xi[j] + (u ** 3 - 2 * u ** 2 + u) * h * m0
                 + (-2 * u ** 3 + 3 * u ** 2) * xi[j + 1] + (u ** 3 - u ** 2) * h * m1)
        return th, psi, float(np.clip(x / S, -0.999, 0.999))

    def skew(self, T):
        """ATM skew of total variance, dw/dk at k = 0."""
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

    def c_at_quotes(self, mk):
        """Normalized call prices of the slices at the fitted quotes."""
        from .black import call
        return call(mk["k"], self.w_at_quotes(mk))

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
