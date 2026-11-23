"""Black (undiscounted, forward-normalized) call prices, implied total variance, Durrleman's condition.

Conventions used throughout the package:
  k = ln(K / F)             log-forward-moneyness
  w = sigma_implied^2 * T   total implied variance
  c = C / (D F)             call price normalized by the discounted forward
"""
import numpy as np
from scipy.stats import norm


def call(k, w):
    """Normalized Black call price c(k, w)."""
    sw = np.sqrt(w)
    d1 = -k / sw + sw / 2
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d1 - sw)


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
    """Density of ln(S_T / F) from normalized calls on a uniform k grid: q = e^-k (c'' - c').
    Returns the interior points (length len(k) - 2)."""
    dk = k[1] - k[0]
    return np.exp(-k[1:-1]) * ((c[2:] - 2 * c[1:-1] + c[:-2]) / dk ** 2 - (c[2:] - c[:-2]) / (2 * dk))
