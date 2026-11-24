"""Black (undiscounted, forward-normalized) call prices, implied total variance, Durrleman's condition.

Conventions used throughout the package:
  k = ln(K / F)             log-forward-moneyness
  w = sigma_implied^2 * T   total implied variance
  c = C / (D F)             call price normalized by the discounted forward
"""
import numpy as np
from scipy.stats import norm


def call(k, w):
    """Normalized Black call price c(k, w); w = 0 gives intrinsic value, w = inf gives 1."""
    k = np.asarray(k, float); w = np.asarray(w, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sw = np.sqrt(np.where(np.isfinite(w), np.maximum(w, 1e-300), 1.0))
        d1 = -k / sw + sw / 2
        c = norm.cdf(d1) - np.exp(k) * norm.cdf(d1 - sw)
    c = np.where(w <= 0, np.maximum(1 - np.exp(k), 0.0), c)
    return np.where(np.isinf(w), 1.0, c)


def vega_w(k, w):
    """dc/dw, the sensitivity of the normalized call to total variance."""
    d1 = -k / np.sqrt(w) + np.sqrt(w) / 2
    return norm.pdf(d1) / (2 * np.sqrt(w))


def implied_w(k, c, lo=1e-10, hi=6.0, iters=90):
    """Total implied variance from normalized call prices, by vectorized bisection."""
    c = np.asarray(c, float)
    lo = np.full_like(c, lo); hi = np.full_like(c, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = call(k, mid) < c
        lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def durrleman(wfun, k, h=1e-4):
    """Durrleman's g(k) and the risk-neutral density of ln(S_T / F) for a smile w(k) given as a function.
    No butterfly arbitrage  <=>  g >= 0  <=>  density >= 0."""
    w = wfun(k)
    w1 = (wfun(k + h) - wfun(k - h)) / (2 * h)
    w2 = (wfun(k + h) - 2 * w + wfun(k - h)) / h ** 2
    g = (1 - k * w1 / (2 * w)) ** 2 - w1 ** 2 / 4 * (1 / w + 0.25) + w2 / 2
    d2 = -k / np.sqrt(w) - np.sqrt(w) / 2
    return g, g * np.exp(-d2 ** 2 / 2) / np.sqrt(2 * np.pi * w)


def density_from_calls(c, k):
    """Density of ln(S_T / F) from normalized calls on a (possibly non-uniform) k grid: q = e^-k (c'' - c'),
    i.e. K d2C/dK2. Returns the interior points (length len(k) - 2)."""
    K = np.exp(k)
    slope = np.diff(c) / np.diff(K)
    return K[1:-1] * np.diff(slope) / (0.5 * (K[2:] - K[:-2]))
