"""All settings of the fit in one place.

Defaults are the final configuration chosen by free energy on SPX (13, 17 and 19 November 2026):
19 x 4 B-splines over x = ln(K/F) / (sigma_ATMF sqrt(T)) in [-5.55, 3.05] and ln T, prior correlation
length 1.6 ATM standard deviations, 24 information-ranked modes, one relinearization round.
"""
from dataclasses import dataclass, field, asdict, replace

# The nine SPX expiries used in the study (Cboe root, yymmdd). AM-settled standard monthlies / quarterlies.
DEFAULT_EXPIRIES = (("SPX", "261218"), ("SPX", "270115"), ("SPX", "270219"), ("SPX", "270319"),
                    ("SPX", "270617"), ("SPX", "270917"), ("SPX", "271217"), ("SPX", "280616"),
                    ("SPX", "281215"))


@dataclass(frozen=True)
class Config:
    # --- market data
    expiries: tuple = DEFAULT_EXPIRIES
    quote_x_lo: float = -3.5          # quoted range used, in ATM standard deviations
    quote_x_hi: float = 2.0
    n_per_expiry: int = 36            # strikes per expiry used in the fit (evenly spaced in x)
    one_sided: bool = True            # keep ask-only / bid-only quotes (as [0, ask] / [bid, inf) in vol)
    parity_band: float = 0.08         # strikes within 8% of spot for the put-call-parity regression

    # --- zeroth order: SSVI slices
    skew_interp: str = "pchip"        # skew between expiries as a function of theta: "pchip" (smooth) or "linear"

    # --- correction: v = v0 exp(sum a_mn B_m(u) B_n(t))
    n_x: int = 19                     # strike splines
    n_t: int = 4                      # maturity splines
    x_lo: float = -5.55               # spline range in x = k / sqrt(theta(T))
    x_hi: float = 3.05
    warp: float = 0.0                 # strike coordinate u = c asinh(x / c); 0 = no warp (u = x)
    t_clock: str = "lnT"              # maturity coordinate: "lnT" or "lntheta"
    ell_x: float = 1.6                # prior correlation length in the strike coordinate
    kappa_first_slice: float = 0.25   # prior scale of the first-slice density coefficients (relative)

    # --- modes and fit
    n_modes: int = 24
    s_noise: float = 0.0005           # extra noise in the bid/ask likelihood (0.05 vol pts)
    tau_grid: tuple = (0.05, 10.0, 18)  # prior-scale grid for the free energy: (min, max, points), log-spaced
    gain_tol: float = 1e-7            # Levenberg-Marquardt stop: relative objective gain below this, 3 times
    max_iter: int = 300
    tau_rounds: int = 4
    relin_rounds: int = 1

    # --- Dupire grid
    dk: float = 0.0015
    k_lo: float = -2.2
    k_hi: float = 1.6
    steps_per_interval: int = 40
    support_gap: float = 0.5          # local vol is "supported" where two-sided quotes are at most this far apart (x)
    rannacher: int = 2                # first solver steps taken as implicit half-steps (damps CN oscillations)
    start: str = "first_slice"        # "first_slice": density correction at T1, solve from T1;
                                      # "zero": solve from near t = 0, the correction also covers [0, T1]

    def with_(self, **kw):
        return replace(self, **kw)

    def to_dict(self):
        return asdict(self)
