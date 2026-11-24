"""The arbitrage-free surface: SSVI local variance times exp(spline correction), evolved by Dupire's equation.

  v(k, T)  = v0(k, T) exp(nu(u, t)),   nu = sum_mn a_mn B_m(u) B_n(t),   u = warp(k / sqrt(theta(T))),  t = ln T
  dc/dT    = 1/2 v (c_kk - c_k)        (normalized calls; Crank-Nicolson with a Rannacher start)
  start    first expiry, from the SSVI slice with a density correction
           q(k, T1) = q0(k, T1) [1 + sum_m a^1_m B_m(x)],  a^1_m >= -1  (butterfly-free), mass and forward kept
           (or, with start = "zero", from near t = 0 with the correction also acting before T1)

Any v > 0 evolved from an arbitrage-free slice gives a surface with no butterfly and no calendar arbitrage at
every maturity, so the local-variance coefficients need no constraint at all.

Everything is sized from the data:
  log-strike grid   non-uniform, k = A sinh(xi), xi uniform: fine near the money (A ~ the first expiry's ATM
                    standard deviation), wide enough for every quote and many standard deviations of the last expiry
  time steps        per expiry interval, proportional to the growth of ln theta (weeklies need few)
  strike splines    over the range of the two-sided quotes in x = k / sqrt(theta), at a fixed spacing
  maturity splines  clamped cubics with interior knots at the expiries (thinned to a minimum spacing in ln T)
  prior             squared-exponential correlations between spline centres in physical units (ell_x, ell_t)

Coefficient vector a = [a^1 (n_x first-slice) | a^nu (n_x * n_t)].  With modes U (D x K) the surface is
a = U c and responses are computed along the K modes only. Responses are in PRICE: A = d c(quote) / d coefficient.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve_banded
from .black import call, implied_w, durrleman, density_from_calls
from .splines import full_bspline_basis, uniform_knots, clamped_knots, clamped_basis, warp


# ---------------------------------------------------------------- finite differences on a non-uniform grid
def fd_coefficients(kg):
    """Three-point weights of d_kk - d_k at the interior nodes of a non-uniform grid: (lo, di, up)."""
    hm = np.r_[1.0, np.diff(kg)]; hp = np.r_[np.diff(kg), 1.0]
    L, C, U = 2 / (hm * (hm + hp)), -2 / (hm * hp), 2 / (hp * (hm + hp))
    L1, C1, U1 = -hp / (hm * (hm + hp)), (hp - hm) / (hm * hp), hm / (hp * (hm + hp))
    lo, di, up = L - L1, C - C1, U - U1
    lo[[0, -1]] = di[[0, -1]] = up[[0, -1]] = 0.0
    return lo, di, up


def op_bands(v, fd):
    """Tridiagonal operator 1/2 v (d_kk - d_k); boundary rows zero (Dirichlet)."""
    a = 0.5 * v
    return a * fd[0], a * fd[1], a * fd[2]


def apply_op(bands, C):
    lo, di, up = bands
    out = di[:, None] * C
    out[1:] += lo[1:, None] * C[:-1]
    out[:-1] += up[:-1, None] * C[1:]
    return out


def theta_step(C, bands0, bands1, dt, th=0.5, S0=None, S1=None, bc=None):
    """One theta-scheme step of dC/dT = L C + S: th = 1/2 Crank-Nicolson, th = 1 implicit Euler."""
    rhs = C + (1 - th) * dt * apply_op(bands0, C) if th < 1 else C.copy()
    if S0 is not None:
        rhs += dt * ((1 - th) * S0 + th * S1)
    lo, di, up = bands1
    ab = np.zeros((3, len(di)))
    ab[0, 1:] = -th * dt * up[:-1]; ab[1] = 1 - th * dt * di; ab[2, :-1] = -th * dt * lo[1:]
    rhs[0] = 0.0 if bc is None else bc[0]; rhs[-1] = 0.0 if bc is None else bc[1]
    return solve_banded((1, 1), ab, rhs)


def step(n, C, bands0, bands1, dt, n_implicit, S0=None, S1=None, bc=None):
    """Step n of the solve: the first n_implicit steps are two implicit half-steps each (Rannacher), which damp
    the high-frequency modes Crank-Nicolson keeps; the rest are Crank-Nicolson."""
    if n < n_implicit:
        C = theta_step(C, bands1, bands1, 0.5 * dt, 1.0, S1, S1, bc)
        return theta_step(C, bands1, bands1, 0.5 * dt, 1.0, S1, S1, bc)
    return theta_step(C, bands0, bands1, dt, 0.5, S0, S1, bc)


def _se(centres, ell):
    d = centres[:, None] - centres[None, :]
    return np.exp(-d ** 2 / (2 * ell ** 2))


def greville(t, k=3):
    return np.array([t[j + 1:j + k + 1].mean() for j in range(len(t) - k - 1)])


def log_strike_grid(mk, base, cfg):
    """Non-uniform grid k = A sinh(xi) covering every quote and grid_sd standard deviations of the last expiry."""
    sd1 = float(np.sqrt(min(mk["theta_atm"]))); sdn = float(np.sqrt(max(mk["theta_atm"])))
    kq = mk["k"]
    lo = min(kq.min() - 3 * sd1, -cfg.grid_sd * sdn, cfg.k_lo)
    hi = max(kq.max() + 3 * sd1, 0.75 * cfg.grid_sd * sdn, cfg.k_hi)
    lo, hi = max(lo, -9.0), min(hi, 6.0)
    A = cfg.grid_core * sd1
    xlo, xhi = np.arcsinh(lo / A), np.arcsinh(hi / A)
    dxi = (xhi - xlo) / (cfg.n_grid - 1)
    xi = dxi * np.arange(-int(round(-xlo / dxi)), int(round(xhi / dxi)) + 1)    # a node exactly at k = 0
    return A * np.sinh(xi)


def time_grid(knots, base, cfg):
    """Geometric steps in each interval, as many as the growth of ln theta needs (between min and max)."""
    parts = []
    for a_, b_ in zip(knots[:-1], knots[1:]):
        dl = abs(np.log(base.theta(b_) / max(base.theta(a_), 1e-300)))
        n = int(np.clip(np.ceil(cfg.steps_per_lntheta * dl) + 4, cfg.min_steps, cfg.steps_per_interval))
        parts.append(np.geomspace(a_, b_, n + 1))
    return np.unique(np.concatenate(parts))


def spline_range(mk, cfg):
    """Strike-spline range in x = k / sqrt(theta_ATM): the two-sided quotes, with a margin, within x_cap."""
    if cfg.x_lo is not None and cfg.x_hi is not None:
        return cfg.x_lo, cfg.x_hi
    x = mk["k"] / np.sqrt(mk["theta_atm"][mk["i"]])
    two = ~mk.get("one_sided", np.zeros(len(x), bool))
    xs = x[two] if two.any() else x
    lo = max(min(xs.min() - 0.6, -2.5), cfg.x_cap[0]); hi = min(max(xs.max() + 0.6, 1.5), cfg.x_cap[1])
    return float(lo), float(hi)


class LocalVarSurface:
    """Exact surface for given SSVI baseline and coefficients, with its linear responses.

    a=None        linearize around the baseline itself (no PDE solve of a reference surface)
    a given       solve the exact surface for a (cref, Href) and linearize around it
    U given       modes: a = U c (c=c), responses computed along the K columns of U
    """
    def __init__(self, mk, base, cfg, a=None, U=None, c=None):
        self.mk, self.base, self.cfg = mk, base, cfg
        self.nT, self.Tn = mk["nT"], mk["T"]
        self.U, self.c_ref = U, (None if c is None else np.asarray(c, float))
        if U is not None and a is None:
            a = U @ self.c_ref              # with an explicit a (offset + U c), keep it
        self.a_ref = None if a is None else np.asarray(a, float)
        if cfg.start not in ("zero", "first_slice"):
            raise ValueError(f"start must be 'zero' or 'first_slice', not {cfg.start!r}")
        self.zero_start = cfg.start == "zero"
        self.kg = log_strike_grid(mk, base, cfg); self.fd = fd_coefficients(self.kg)
        j0 = int(np.argmin(np.abs(self.kg)))
        self.dk = float(min(self.kg[j0 + 1] - self.kg[j0], self.kg[j0] - self.kg[j0 - 1]))   # spacing at the money
        self.t_start = self._start_time() if self.zero_start else float(self.Tn[0])
        knots = ([self.t_start] if self.zero_start else []) + list(self.Tn)
        self.tg = time_grid(knots, base, cfg) if len(knots) > 1 else np.array(knots, float)
        self.iexp = [int(np.argmin(np.abs(self.tg - t))) for t in self.Tn]
        # strike splines
        self.x_lo, self.x_hi = spline_range(mk, cfg)
        ulo, uhi = warp(self.x_lo, cfg.warp), warp(self.x_hi, cfg.warp)
        self.n_x = cfg.n_x if cfg.n_x else int(np.clip(round((uhi - ulo) / cfg.dx_spline) - 3, 9, cfg.n_x_max))
        self.tx = uniform_knots(ulo, uhi, self.n_x)
        # maturity splines: knots at the expiries
        t0, t1 = self.tcoord(self.Tn[0]), self.tcoord(self.Tn[-1])
        if cfg.n_t:
            self.n_t = cfg.n_t; self.tu = clamped_knots(t0, t1, cfg.n_t)
        else:
            inner, last = [], t0
            for T in self.Tn[1:-1]:
                tc = self.tcoord(T)
                if tc - last >= cfg.min_knot_gap and t1 - tc >= cfg.min_knot_gap:
                    inner.append(tc); last = tc
            self.tu = np.r_[[t0] * 4, inner, [t1] * 4] if t1 > t0 else clamped_knots(t0, t0 + 1.0, 4)
            self.n_t = len(self.tu) - 4
        self.Dnu = self.n_x * self.n_t; self.D = self.n_x + self.Dnu
        self.P1 = self._prior()
        self.Q = self._anchor()
        self._precompute()

    # ---------------------------------------------------------------- coordinates and bases
    def tcoord(self, T):
        return float(np.log(self.base.theta(T))) if self.cfg.t_clock == "lntheta" else float(np.log(T))

    def Bx(self, x):
        """Strike splines at standardized moneyness x = k / sqrt(theta)."""
        return full_bspline_basis(self.tx, warp(x, self.cfg.warp))

    def Bt(self, T):
        # before the first expiry (zero start) the correction keeps its first-expiry shape in x = k / sqrt(theta)
        T = max(float(T), float(self.Tn[0])) if self.zero_start else T
        tc = min(max(self.tcoord(T), self.tu[0]), self.tu[-1])
        return clamped_basis(self.tu, tc, self.n_t)[0]

    def _start_time(self):
        """Zero start: the solve begins at t_start, where the baseline's ATM standard deviation sqrt(theta) spans
        ~12 grid steps at the money, at most a quarter of the first expiry. Before t_start the local variance is
        the baseline's."""
        T1 = float(self.Tn[0]); target = (12 * self.dk) ** 2
        if self.base.theta(T1 / 4) <= target:
            return T1 / 4
        lo, hi = T1 / 4000, T1 / 4
        for _ in range(60):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if self.base.theta(mid) < target else (lo, mid)
        return hi

    def nu_basis(self, T, k=None):
        k = self.kg if k is None else k
        bx = self.Bx(k / np.sqrt(self.base.theta(T)))
        return (bx[:, :, None] * self.Bt(T)[None, None, :]).reshape(len(k), self.Dnu)

    def _prior(self):
        """Gaussian prior precision on the D coefficients (tau = 1): squared-exponential correlation between spline
        centres, ell_x in the strike coordinate and ell_t in the maturity coordinate."""
        cfg = self.cfg
        cx = greville(self.tx); ct = greville(self.tu)
        Sx = _se(cx, cfg.ell_x); St = _se(ct, cfg.ell_t)
        C1 = cfg.kappa_first_slice ** 2 * (Sx + 1e-4 * np.eye(self.n_x))
        Cnu = np.kron(Sx, St) + 1e-4 * np.eye(self.Dnu)
        P = np.zeros((self.D, self.D))
        P[:self.n_x, :self.n_x] = np.linalg.inv(C1); P[self.n_x:, self.n_x:] = np.linalg.inv(Cnu)
        return 0.5 * (P + P.T)

    def _anchor(self):
        """Wing anchor: a fixed Gaussian prior, NOT scaled by tau, on the coefficients of splines that lie where no
        quotes are. Q = diag(f_j) / sigma_w^2, f_j = share of spline j's weight outside the supported region
        (see `supported`). The smoothness prior P1 / tau^2 lets tau grow when the core needs freedom; without the
        anchor that freedom leaks into the unobserved wings, where exp(nu) is unbounded."""
        cfg = self.cfg; D = self.D
        sig = getattr(cfg, "wing_anchor", 0.0)
        if not sig or "full" not in self.mk:
            return np.zeros((D, D))
        xs = np.linspace(self.x_lo, self.x_hi, 173)
        ts = np.geomspace(self.Tn[0], self.Tn[-1], 41) if self.nT > 1 else self.Tn[:1]
        Bx = self.Bx(xs)
        f = np.zeros(D)
        u0 = 1.0 - self.supported(self.Tn[0], xs)
        f[:self.n_x] = (Bx * u0[:, None]).sum(0) / np.maximum(Bx.sum(0), 1e-12)
        num = np.zeros((self.n_x, self.n_t)); den = np.zeros_like(num)
        for t in ts:
            bt = self.Bt(t); u = 1.0 - self.supported(t, xs)
            num += np.outer((Bx * u[:, None]).sum(0), bt); den += np.outer(Bx.sum(0), bt)
        f[self.n_x:] = (num / np.maximum(den, 1e-12)).reshape(-1)
        self.unsupported_share = f
        return np.diag(f) / sig ** 2

    # ---------------------------------------------------------------- helpers
    def _Dc(self, c):
        """(d_kk - d_k) c on the grid (zero at the boundary nodes)."""
        lo, di, up = self.fd
        out = di[:, None] * c if c.ndim == 2 else di * c
        if c.ndim == 2:
            out[1:] += lo[1:, None] * c[:-1]; out[:-1] += up[:-1, None] * c[1:]
        else:
            out[1:] += lo[1:] * c[:-1]; out[:-1] += up[:-1] * c[1:]
        return out

    def interp(self, C, k):
        """Linear interpolation of grid values (vector or matrix with grid rows) at log-strikes k."""
        k = np.asarray(k, float)
        j = np.clip(np.searchsorted(self.kg, k) - 1, 0, len(self.kg) - 2)
        f = np.clip((k - self.kg[j]) / (self.kg[j + 1] - self.kg[j]), 0.0, 1.0)
        if C.ndim == 2:
            f = f[:, None]
        return (1 - f) * C[j] + f * C[j + 1]

    def _first_slice_modes(self):
        """Price modes of the first-slice density correction (delta_c = Q2 - e^k Q1) and its
        mass / forward rows (normalized)."""
        b, kg, T1 = self.base, self.kg, self.Tn[0]
        _, q0 = durrleman(lambda kk: b.w0(kk, T1), kg)
        Q = np.maximum(q0, 0.0)[:, None] * self.Bx(kg / np.sqrt(b.theta(T1)))
        Q1 = cumulative_trapezoid(Q[::-1], -kg[::-1], axis=0, initial=0)[::-1]
        Q2 = cumulative_trapezoid((np.exp(kg)[:, None] * Q)[::-1], -kg[::-1], axis=0, initial=0)[::-1]
        self.ICm = Q2 - np.exp(kg)[:, None] * Q1
        if self.zero_start:                       # no first-slice density correction: its coefficients are pinned
            self.ICm = np.zeros_like(self.ICm)
            return np.eye(self.n_x)
        E = np.vstack([Q1[0], Q2[0]])
        return E / np.linalg.norm(E, axis=1, keepdims=True)

    # ---------------------------------------------------------------- responses
    def _precompute(self):
        b, kg, tg, nx = self.base, self.kg, self.tg, self.n_x
        self.v0g, self.bands0, dTc0, nbad = [], [], [], 0
        for t in tg:
            v, bad = b.v0(kg, t); nbad += bad.sum()
            self.v0g.append(v); self.bands0.append(op_bands(v, self.fd))
            dTc0.append(b.vega_w(kg, t) * np.maximum(b.dTw0(kg, t), 0.0))
        self.frac_floored = nbad / (len(tg) * len(kg))
        E = self._first_slice_modes()
        k, idx = self.mk["k"], self.mk["i"]
        c0q = np.zeros(len(k))
        for s in range(self.nT):
            m = idx == s; c0q[m] = call(k[m], b.w0(k[m], self.Tn[s]))
        if self.a_ref is None:
            self.bands = self.bands0; src = dTc0; self.c_ref_q = c0q
        else:
            nu = [self.nu_basis(tg[n]) @ self.a_ref[nx:] for n in range(len(tg))]
            fac = [np.exp(np.clip(x, -30, 30)) for x in nu]
            self.bands = [op_bands(self.v0g[n] * fac[n], self.fd) for n in range(len(tg))]
            self.cref, self.Href = self.full_solve(self.a_ref, all_steps=True)
            src = [0.5 * self.v0g[n] * fac[n] * self._Dc(self.Href[n]) for n in range(len(tg))]
            self.c_ref_q = np.zeros(len(k))
            for s in range(self.nT):
                m = idx == s
                self.c_ref_q[m] = self.interp(self.cref[s], k[m])
        proj = (lambda M: M @ self.U) if self.U is not None else (lambda M: M)
        C = proj(np.hstack([self.ICm, np.zeros((len(kg), self.Dnu))]))
        zer = np.zeros((len(kg), nx))
        S_prev = proj(np.hstack([zer, self.nu_basis(tg[0]) * src[0][:, None]]))
        self.Phi = {0: C.copy()}
        bc_lin = (proj(np.r_[self.ICm[0], np.zeros(self.Dnu)][None, :])[0], np.zeros(C.shape[1]))
        for n in range(len(tg) - 1):
            S_next = proj(np.hstack([zer, self.nu_basis(tg[n + 1]) * src[n + 1][:, None]]))
            C = step(n, C, self.bands[n], self.bands[n + 1], tg[n + 1] - tg[n], self.cfg.rannacher, S_prev, S_next,
                     bc=bc_lin)
            S_prev = S_next
            if n + 1 in self.iexp:
                self.Phi[self.iexp.index(n + 1)] = C.copy()
        self.A = np.zeros((len(k), C.shape[1]))                    # d c(quote) / d coefficient, in price
        for s in range(self.nT):
            m = idx == s
            self.A[m] = self.interp(self.Phi[s], k[m])
        self.cons_rows = np.array([proj(np.r_[e, np.zeros(self.Dnu)][None, :])[0] for e in E])

    # ---------------------------------------------------------------- exact surface
    def full_solve(self, a=None, all_steps=False):
        b, kg, tg, nx = self.base, self.kg, self.tg, self.n_x
        off = 0.0 if a is None else float(self.ICm[0] @ a[:nx])        # conserved mass / forward mismatch
        bc = lambda t: (call(kg[:1], b.w0(kg[:1], t))[0] + off, call(kg[-1:], b.w0(kg[-1:], t))[0])

        def bands_at(n):
            if a is None:
                return self.bands0[n]
            return op_bands(self.v0g[n] * np.exp(np.clip(self.nu_basis(tg[n]) @ a[nx:], -30, 30)), self.fd)
        c = call(kg, b.w0(kg, tg[0]))
        if a is not None:
            c = c + self.ICm @ a[:nx]
        C = c[:, None]; out = {0: C[:, 0].copy()}; hist = [C[:, 0].copy()]
        bn = bands_at(0)
        for n in range(len(tg) - 1):
            bn1 = bands_at(n + 1)
            C = step(n, C, bn, bn1, tg[n + 1] - tg[n], self.cfg.rannacher, bc=bc(tg[n + 1])); bn = bn1
            if n + 1 in self.iexp:
                out[self.iexp.index(n + 1)] = C[:, 0].copy()
            if all_steps:
                hist.append(C[:, 0].copy())
        return (out, np.array(hist)) if all_steps else out

    # ---------------------------------------------------------------- outputs of the exact surface
    def calls(self, s, k):
        """Normalized call prices C/(D F) of the delivered surface at expiry s: the solver's prices, linearly
        interpolated in log-strike. The solver's prices are convex in strike and increasing in maturity on its
        grid, and linear interpolation keeps both, so what is delivered is what was checked."""
        return self.interp(self.cref[s], np.asarray(k, float))

    def w_exact(self, s, k):
        """Total implied variance of the exact surface at expiry s (nan where the price has no time value)."""
        k = np.asarray(k, float); c = self.calls(s, k)
        intr = np.maximum(1 - np.exp(k), 0.0); ok = c > intr + 1e-15
        w = np.full(len(k), np.nan)
        if ok.any():
            w[ok] = implied_w(k[ok], c[ok], hi=50.0)
        return w

    def iv_exact(self, s, k):
        return np.sqrt(self.w_exact(s, k) / self.Tn[s])

    def local_vol(self, T, x):
        """Local volatility at maturity T and standardized moneyness x = k / sqrt(theta(T)), evaluated at T itself
        (not snapped to the solver's time grid). Before the solve starts (the first expiry, or t_start with a zero
        start) it is the baseline's local volatility: see `supported` for where it is fitted to quotes."""
        T = float(T); k = np.asarray(x, float) * np.sqrt(self.base.theta(T))
        v0, _ = self.base.v0(self.kg, T)
        v = np.interp(k, self.kg, v0)
        if T >= self.t_start - 1e-12:
            v = v * np.exp(np.clip(self.nu_basis(T, k) @ self.a_ref[self.n_x:], -30, 30))
        return np.sqrt(v)

    def supported(self, T, x, max_gap=None):
        """True where local vol at (T, x = k / sqrt(theta(T))) is constrained by quotes: T between the first and
        last expiry, and at both neighbouring expiries x lies inside the quoted strikes with no gap between
        neighbouring two-sided quotes wider than max_gap (in x units; default cfg.support_gap). Elsewhere the
        local vol comes from the baseline extrapolation and the prior, not from data."""
        max_gap = self.cfg.support_gap if max_gap is None else max_gap
        x = np.atleast_1d(np.asarray(x, float)); T = float(T)
        if not (self.Tn[0] - 1e-12 <= T <= self.Tn[-1] + 1e-12) or "full" not in self.mk:
            return np.zeros(len(x), bool)
        j = int(np.clip(np.searchsorted(self.Tn, T) - 1, 0, self.nT - 2)) if self.nT > 1 else 0
        out = np.ones(len(x), bool)
        for s in sorted({j, min(j + 1, self.nT - 1)}):
            f = self.mk["full"][s]
            two = ~f.get("one_sided", np.zeros(len(f["k"]), bool))
            xs = np.sort(f["k"][two] / np.sqrt(self.base.th_i[s]))
            if len(xs) < 2:
                return np.zeros(len(x), bool)
            i = np.clip(np.searchsorted(xs, x), 1, len(xs) - 1)
            out &= (x >= xs[0]) & (x <= xs[-1]) & (xs[i] - xs[i - 1] <= max_gap)
        return out

    def densities(self):
        """Risk-neutral densities of the exact surface at each expiry (interior grid points)."""
        return [density_from_calls(self.cref[s], self.kg) for s in range(self.nT)]
