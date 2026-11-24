"""The arbitrage-free surface: SSVI local variance times exp(spline correction), evolved by Dupire's equation.

  v(k, T)  = v0(k, T) exp(nu(u, t)),   nu = sum_mn a_mn B_m(u) B_n(t),   u = warp(k / sqrt(theta(T))),  t = ln T
  dc/dT    = 1/2 v (c_kk - c_k)        (normalized calls, Crank-Nicolson on a uniform k grid)
  start    first expiry, from the SSVI slice with a density correction
           q(k, T1) = q0(k, T1) [1 + sum_m a^1_m B_m(x)],  a^1_m >= -1  (butterfly-free), mass and forward kept

Any v > 0 evolved from an arbitrage-free slice gives a surface with no butterfly and no calendar arbitrage at
every maturity, so the local-variance coefficients need no constraint at all.

Coefficient vector a = [a^1 (n_x first-slice) | a^nu (n_x * n_t)].  With modes U (D x K) the surface is
a = U c and responses are computed along the K modes only.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve_banded
from .black import call, implied_w, durrleman, density_from_calls
from .splines import full_bspline_basis, uniform_knots, clamped_knots, clamped_basis, warp


# ---------------------------------------------------------------- Crank-Nicolson pieces
def op_bands(v, dk):
    """Tridiagonal operator 1/2 v (d_kk - d_k); boundary rows zero (Dirichlet)."""
    a = 0.5 * v
    lo = a * (1 / dk ** 2 + 1 / (2 * dk)); di = -2 * a / dk ** 2; up = a * (1 / dk ** 2 - 1 / (2 * dk))
    lo[[0, -1]] = di[[0, -1]] = up[[0, -1]] = 0.0
    return lo, di, up


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


def cn_step(C, bands0, bands1, dt, S0=None, S1=None, bc=None):
    return theta_step(C, bands0, bands1, dt, 0.5, S0, S1, bc)


def step(n, C, bands0, bands1, dt, n_implicit, S0=None, S1=None, bc=None):
    """Step n of the solve: the first n_implicit steps are two implicit half-steps each (Rannacher), which damp
    the high-frequency modes Crank-Nicolson keeps; the rest are Crank-Nicolson."""
    if n < n_implicit:
        C = theta_step(C, bands1, bands1, 0.5 * dt, 1.0, S1, S1, bc)
        return theta_step(C, bands1, bands1, 0.5 * dt, 1.0, S1, S1, bc)
    return theta_step(C, bands0, bands1, dt, 0.5, S0, S1, bc)


def _se(n, ell):
    return np.exp(-(np.arange(n)[:, None] - np.arange(n)[None, :]) ** 2 / (2 * ell ** 2))


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
        self.kg = np.arange(cfg.k_lo, cfg.k_hi + cfg.dk / 2, cfg.dk); self.dk = cfg.dk
        self.zero_start = cfg.start == "zero"
        if cfg.start not in ("zero", "first_slice"):
            raise ValueError(f"start must be 'zero' or 'first_slice', not {cfg.start!r}")
        self.t_start = self._start_time() if self.zero_start else float(self.Tn[0])
        knots = ([self.t_start] if self.zero_start else []) + list(self.Tn)
        self.tg = np.unique(np.concatenate([np.geomspace(knots[i], knots[i + 1], cfg.steps_per_interval + 1)
                                            for i in range(len(knots) - 1)]))
        self.iexp = [int(np.argmin(np.abs(self.tg - t))) for t in self.Tn]
        self.n_x, self.n_t = cfg.n_x, cfg.n_t
        self.Dnu = cfg.n_x * cfg.n_t; self.D = cfg.n_x + self.Dnu
        self.tx = uniform_knots(warp(cfg.x_lo, cfg.warp), warp(cfg.x_hi, cfg.warp), cfg.n_x)
        t0, t1 = self.tcoord(self.Tn[0]), self.tcoord(self.Tn[-1])
        self.tu = clamped_knots(t0, t1, cfg.n_t)
        self.P1 = self._prior(t0, t1)
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
        return clamped_basis(self.tu, self.tcoord(T), self.n_t)[0]

    def _start_time(self):
        """Zero start: the solve begins at t_start, where the baseline's ATM standard deviation sqrt(theta) spans
        ~12 grid steps (so the smile is resolved), at most a quarter of the first expiry. Before t_start the local
        variance is the baseline's; the time is short and the payoff is close to its intrinsic value."""
        T1 = float(self.Tn[0]); target = (12 * self.cfg.dk) ** 2
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

    def _prior(self, t0, t1):
        """Gaussian prior precision on the D coefficients (tau = 1): smooth across neighbouring splines.
        Correlation lengths in physical units: ell_x in the strike coordinate, half the maturity range."""
        cfg = self.cfg
        spacing_x = (warp(cfg.x_hi, cfg.warp) - warp(cfg.x_lo, cfg.warp)) / (cfg.n_x + 3)
        ell_x = cfg.ell_x / spacing_x
        ell_t = (0.5 * (t1 - t0)) / ((t1 - t0) / max(cfg.n_t - 3, 1))
        C1 = cfg.kappa_first_slice ** 2 * (_se(cfg.n_x, ell_x) + 1e-4 * np.eye(cfg.n_x))
        Cnu = np.kron(_se(cfg.n_x, ell_x), _se(cfg.n_t, ell_t)) + 1e-4 * np.eye(self.Dnu)
        P = np.zeros((self.D, self.D))
        P[:cfg.n_x, :cfg.n_x] = np.linalg.inv(C1); P[cfg.n_x:, cfg.n_x:] = np.linalg.inv(Cnu)
        return P

    # ---------------------------------------------------------------- helpers
    def _Dc(self, c):
        out = np.zeros_like(c); dk = self.dk
        out[1:-1] = (c[2:] - 2 * c[1:-1] + c[:-2]) / dk ** 2 - (c[2:] - c[:-2]) / (2 * dk)
        return out

    def interp(self, C, k):
        j = np.clip(np.searchsorted(self.kg, k) - 1, 0, len(self.kg) - 2)
        f = ((k - self.kg[j]) / self.dk)[:, None] if C.ndim == 2 else (k - self.kg[j]) / self.dk
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
            self.v0g.append(v); self.bands0.append(op_bands(v, self.dk))
            dTc0.append(b.vega_w(kg, t) * np.maximum(b.dTw0(kg, t), 0.0))
        self.frac_floored = nbad / (len(tg) * len(kg))
        E = self._first_slice_modes()
        k, idx = self.mk["k"], self.mk["i"]
        w0q = np.zeros(len(k))
        for s in range(self.nT):
            m = idx == s; w0q[m] = b.w0(k[m], self.Tn[s])
        if self.a_ref is None:
            self.bands = self.bands0; src = dTc0; self.w_ref_q = w0q
        else:
            nu = [self.nu_basis(tg[n]) @ self.a_ref[nx:] for n in range(len(tg))]
            fac = [np.exp(np.clip(x, -30, 30)) for x in nu]
            self.bands = [op_bands(self.v0g[n] * fac[n], self.dk) for n in range(len(tg))]
            self.c0n = self.full_solve(None)
            self.cref, self.Href = self.full_solve(self.a_ref, all_steps=True)
            src = [0.5 * self.v0g[n] * fac[n] * self._Dc(self.Href[n]) for n in range(len(tg))]
            self.w_ref_q = np.zeros(len(k))
            for s in range(self.nT):
                m = idx == s
                self.w_ref_q[m] = implied_w(k[m], call(k[m], w0q[m]) + self.interp(self.cref[s] - self.c0n[s], k[m]))
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
        d1 = -k / np.sqrt(self.w_ref_q) + np.sqrt(self.w_ref_q) / 2
        vega_q = np.exp(-d1 ** 2 / 2) / np.sqrt(2 * np.pi) / (2 * np.sqrt(self.w_ref_q))
        self.A = np.zeros((len(k), C.shape[1]))                    # d w(quote) / d coefficient
        for s in range(self.nT):
            m = idx == s
            self.A[m] = self.interp(self.Phi[s], k[m]) / vega_q[m][:, None]
        self.cons_rows = np.array([proj(np.r_[e, np.zeros(self.Dnu)][None, :])[0] for e in E])
        if self.U is not None:
            aref = self.c_ref
        else:
            aref = self.a_ref
        self.r = self.mk["z"] - self.w_ref_q + (0 if aref is None else self.A @ aref)

    # ---------------------------------------------------------------- exact surface
    def full_solve(self, a=None, all_steps=False):
        b, kg, tg, nx = self.base, self.kg, self.tg, self.n_x
        off = 0.0 if a is None else float(self.ICm[0] @ a[:nx])        # conserved mass / forward mismatch
        bc = lambda t: (call(kg[:1], b.w0(kg[:1], t))[0] + off, call(kg[-1:], b.w0(kg[-1:], t))[0])

        def bands_at(n):
            if a is None:
                return self.bands0[n]
            return op_bands(self.v0g[n] * np.exp(np.clip(self.nu_basis(tg[n]) @ a[nx:], -30, 30)), self.dk)
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
    def w_exact(self, s, k):
        """Total implied variance of the exact surface at expiry s (requires a reference solve)."""
        w0 = self.base.w0(k, self.Tn[s])
        return implied_w(k, call(k, w0) + self.interp(self.cref[s] - self.c0n[s], k))

    def calls(self, s, k):
        """Normalized call prices C/(D F) of the delivered surface at expiry s: the analytic baseline plus the
        solver's correction (cref - c0n), which cancels most of the discretization error."""
        k = np.asarray(k, float)
        return call(k, self.base.w0(k, self.Tn[s])) + self.interp(self.cref[s] - self.c0n[s], k)

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
