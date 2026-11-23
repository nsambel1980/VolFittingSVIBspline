"""B-spline bases for the local-variance correction.

Strike direction: cubic B-splines on uniform knots in u = warp(x), x = k / sqrt(theta(T)), each evaluated on
its FULL support. (scipy's BSpline(t, eye, k)(x) only covers the base interval [t_k, t_{-k-1}] and silently
truncates the outer splines of a non-clamped knot vector; that bug cost the early fits their wings.)
Maturity direction: clamped cubic B-splines in ln T (or ln theta).
"""
import numpy as np
from scipy.interpolate import BSpline


def full_bspline_basis(t, x, k=3):
    """All len(t) - k - 1 B-splines on knots t, evaluated at x on their full supports (zero outside)."""
    x = np.asarray(x, float)
    n = len(t) - k - 1
    B = np.zeros((len(x), n))
    for j in range(n):
        B[:, j] = np.nan_to_num(BSpline.basis_element(t[j:j + k + 2], extrapolate=False)(x))
    return B


def uniform_knots(lo, hi, n_splines, k=3):
    """Uniform, non-clamped knot vector whose n_splines cubic splines all lie inside [lo, hi]."""
    return np.linspace(lo, hi, n_splines + k + 1)


def clamped_knots(lo, hi, n_splines, k=3):
    """Clamped knot vector on [lo, hi] (repeated end knots, uniform interior knots)."""
    interior = np.linspace(lo, hi, n_splines - k + 1)[1:-1]
    return np.r_[[lo] * (k + 1), interior, [hi] * (k + 1)]


def clamped_basis(t, x, n_splines, k=3):
    return np.nan_to_num(BSpline(t, np.eye(n_splines), k, extrapolate=False)(np.atleast_1d(x)))


def warp(x, c):
    """Strike coordinate u = c asinh(x / c); c = 0 means no warp."""
    x = np.asarray(x, float)
    return c * np.arcsinh(x / c) if c > 0 else x
