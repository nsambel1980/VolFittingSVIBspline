"""All settings of the fit in one place.

Defaults are the final configuration chosen by free energy on SPX (13, 17 and 19 November 2026):
19 x 4 B-splines over x = ln(K/F) / (sigma_ATMF sqrt(T)) in [-5.55, 3.05] and ln T, prior correlation
length 1.6 ATM standard deviations, 24 information-ranked modes, one relinearization round.
"""
from dataclasses import dataclass, field, asdict, replace

# The nine SPX expiries of the SPX study (Cboe root, yymmdd): AM-settled standard monthlies / quarterlies.
# Config() itself fits every expiry in a chain; SPX_STUDY below reproduces the published SPX results.
DEFAULT_EXPIRIES = (("SPX", "261218"), ("SPX", "270115"), ("SPX", "270219"), ("SPX", "270319"),
                    ("SPX", "270617"), ("SPX", "270917"), ("SPX", "271217"), ("SPX", "280616"),
                    ("SPX", "281215"))


@dataclass(frozen=True)
class Config:
    # --- market data
    expiries: tuple = None            # (root, yymmdd) to fit; None = every expiry in the chain (>= min_days)
    min_days: float = 2.0             # expiries closer than this are skipped when expiries is None
    quote_x_lo: float = -float("inf") # quoted range used, in ATM standard deviations (default: all rows)
    quote_x_hi: float = float("inf")
    n_per_expiry: int = 0             # strikes per expiry used in the fit, evenly spaced in x; 0 = all
    one_sided: bool = True            # keep ask-only / bid-only quotes ([intrinsic, ask] / [bid, inf) in price)
    min_half_spread: float = 1e-6     # floor on the half-spread (normalized price) used to scale errors
    parity_band: float = 0.08         # strikes within 8% of spot for the put-call-parity regression

    # --- zeroth order: SSVI slices
    smooth_grid: tuple = (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)   # term-structure prior scales tried (evidence picks)
    skew_interp: str = "pchip"        # skew between expiries as a function of theta: "pchip" (smooth) or "linear"

    # --- correction: v = v0 exp(sum a_mn B_m(u) B_n(t))
    n_x: int = 0                      # strike splines; 0 = from the quoted range at spacing dx_spline
    dx_spline: float = 0.45           # strike-spline spacing in x (ATM standard deviations) when n_x = 0
    n_x_max: int = 41
    x_lo: float = None                # spline range in x = k / sqrt(theta); None = two-sided quotes +- 0.6
    x_hi: float = None
    x_cap: tuple = (-8.0, 5.0)        # the data-driven range never goes beyond this
    n_t: int = 0                      # maturity splines; 0 = knots at the expiries (min_knot_gap apart in ln T)
    min_knot_gap: float = 0.2
    warp: float = 0.0                 # strike coordinate u = c asinh(x / c); 0 = no warp (u = x)
    t_clock: str = "lnT"              # maturity coordinate: "lnT" or "lntheta"
    ell_x: float = 1.6                # prior correlation length in the strike coordinate (x units)
    ell_t: float = 0.7                # prior correlation length in the maturity coordinate (ln T units)
    kappa_first_slice: float = 0.25   # prior scale of the first-slice density coefficients (relative)

    # --- modes and fit
    n_modes: int = 0                  # modes fitted; 0 = every mode with information kappa >= kappa_min
    kappa_min: float = 1e-3
    n_modes_max: int = 400
    price_noise: float = 0.25         # noise in the bid/ask likelihood, in half-spreads of each quote
    mid_pull: float = 2.0             # mild pull to the mid: 1/2 ((c - mid) / (mid_pull h))^2; 0 = none
    tau_grid: tuple = (0.05, 1000.0, 30)  # prior-scale grid for the free energy: (min, max, points), log-spaced
    gain_tol: float = 1e-7            # Levenberg-Marquardt stop: relative objective gain below this, 3 times
    max_iter: int = 300
    tau_rounds: int = 4
    relin_rounds: int = 1
    relin_max_cost: float = 50.0      # reject a relinearization round that worsens -ln L by more than this

    # --- Dupire grid
    n_grid: int = 700                 # log-strike nodes, k = A sinh(xi) with A = grid_core * first ATM sd
    grid_core: float = 2.0
    grid_sd: float = 12.0             # the grid reaches grid_sd (puts) / 0.75 grid_sd (calls) last-expiry sds
    k_lo: float = -0.5                # ... and at least this range in k
    k_hi: float = 0.4
    steps_per_lntheta: float = 30.0   # time steps per unit growth of ln theta in each expiry interval
    min_steps: int = 6
    steps_per_interval: int = 40      # at most
    nu_amplitude: float = 1.5         # prior sd of every correction coefficient (fixed, not scaled by tau)
    wing_anchor: float = 0.25         # prior sd of nu (log local-variance factor) where no quotes are; 0 = off
    support_gap: float = 0.5          # local vol is "supported" where two-sided quotes are at most this far apart (x)
    rannacher: int = 2                # first solver steps taken as implicit half-steps (damps CN oscillations)
    start: str = "zero"               # "zero": solve from near t = 0, the correction also shapes [0, T1]
                                      # (positivity of local variance guarantees no arbitrage);
                                      # "first_slice": density correction q0 (1 + B a) at T1 - its factor
                                      # can turn negative on steep short-dated smiles (seen on SPX)

    def with_(self, **kw):
        return replace(self, **kw)

    def to_dict(self):
        return asdict(self)


# The configuration of the published SPX results: nine monthly expiries, every quote on them.
SPX_STUDY = Config(expiries=DEFAULT_EXPIRIES)
