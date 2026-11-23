"""Fast unit tests: splines, Black formulas, SSVI identities, likelihood derivatives, put-call parity."""
import datetime as dt
import numpy as np
from scipy.interpolate import BSpline

from volsurf.splines import full_bspline_basis, uniform_knots
from volsurf.black import call, implied_w, durrleman, density_from_calls
from volsurf.ssvi import SSVISlices, ssvi_w, psi_law
from volsurf.likelihood import neg_log_lik
from volsurf.market import build_market_from_rows
from volsurf.config import Config
from conftest import true_q, T_SYN


# ---------------------------------------------------------------- splines
def test_full_basis_partition_of_unity_and_full_support():
    t = uniform_knots(-5.55, 3.05, 19)
    x = np.linspace(t[3], t[-4], 400)                       # where all overlapping splines exist
    B = full_bspline_basis(t, x)
    assert B.shape == (400, 19)
    np.testing.assert_allclose(B.sum(1), 1.0, atol=1e-12)
    # the outermost splines are non-zero near the ends; scipy's BSpline(t, eye) truncates them to zero
    x_edge = np.array([t[0] + 0.3 * (t[1] - t[0])])
    assert full_bspline_basis(t, x_edge)[0, 0] > 0
    truncated = np.nan_to_num(BSpline(t, np.eye(19), 3, extrapolate=False)(x_edge))
    assert truncated[0, 0] == 0


# ---------------------------------------------------------------- Black and Durrleman
def test_implied_variance_round_trip():
    k = np.linspace(-1.0, 0.5, 31); w = 0.04 + 0.05 * k ** 2
    np.testing.assert_allclose(implied_w(k, call(k, w)), w, rtol=1e-9)


def test_durrleman_density_is_a_martingale_density():
    k = np.linspace(-3, 2, 20001); dk = k[1] - k[0]
    wf = lambda kk: ssvi_w(kk, 0.02, psi_law(0.02, 1.2, 0.45), -0.7)
    g, q = durrleman(wf, k)
    assert g.min() > 0
    assert abs(q.sum() * dk - 1) < 1e-4                    # mass
    assert abs((np.exp(k) * q).sum() * dk - 1) < 1e-4      # forward
    q_fd = density_from_calls(call(k, wf(k)), k)
    big = q[1:-1] > 1e-3 * q.max()
    np.testing.assert_allclose(q_fd[big], q[1:-1][big], rtol=2e-3)


# ---------------------------------------------------------------- SSVI
def test_ssvi_atm_level_and_skew_are_the_parameters():
    b = SSVISlices(T_SYN, true_q())
    for s, T in enumerate(T_SYN):
        assert abs(b.w0(np.array([0.0]), T)[0] - b.th_i[s]) < 1e-14
        h = 1e-6; d = (b.w0(np.array([h]), T)[0] - b.w0(np.array([-h]), T)[0]) / (2 * h)
        assert abs(d - b.sk_i[s]) < 1e-7
    assert b.checks()["butterfly_ok"].all()


def test_smooth_skew_interpolation_removes_local_vol_jumps():
    q = true_q(); q[4:8] *= np.array([1.0, 1.3, 0.9, 1.2])   # uneven skew term structure
    xs = np.linspace(-2, 1.5, 51)
    def max_jump(kind):
        b = SSVISlices(T_SYN, q, kind); out = []
        for Ti in T_SYN[1:-1]:
            v = [np.sqrt(b.v0(xs * np.sqrt(b.theta(t)), t)[0]) for t in (Ti * (1 - 2e-3), Ti * (1 + 2e-3))]
            out.append(100 * np.abs(v[1] - v[0]).max())
        return max(out)
    assert max_jump("pchip") < 0.2
    assert max_jump("linear") > 5 * max_jump("pchip")


# ---------------------------------------------------------------- likelihood
def test_likelihood_derivatives_match_finite_differences():
    bid, ask, s = np.full(5, 0.20), np.full(5, 0.202), 0.0005
    m = np.array([0.195, 0.1995, 0.201, 0.2025, 0.21])
    f, g, h = neg_log_lik(m, bid, ask, s)
    e = 1e-7
    g_fd = (neg_log_lik(m + e, bid, ask, s)[0] - neg_log_lik(m - e, bid, ask, s)[0]) / (2 * e)
    np.testing.assert_allclose(g, g_fd, rtol=1e-4, atol=1e-2)
    e = 1e-5
    h_fd = (neg_log_lik(m + e, bid, ask, s)[1] - neg_log_lik(m - e, bid, ask, s)[1]) / (2 * e)
    np.testing.assert_allclose(h, np.maximum(h_fd, 0), rtol=1e-3, atol=1.0)
    assert abs(g[2]) < 1e-3 * abs(g[0])                    # flat in the middle of the spread


# ---------------------------------------------------------------- market: put-call parity
def test_parity_recovers_forward_and_discount():
    F, D, T = 7000.0, 0.98, 0.5
    expiry = dt.datetime(2027, 5, 19, 9, 30); quote_time = expiry - dt.timedelta(days=0.5 * 365)
    th = 0.2 ** 2 * T; wf = lambda k: ssvi_w(k, th, psi_law(th, 1.2, 0.45), -0.7)
    rows = []
    for K in np.arange(5000.0, 8400.0, 25.0):
        k = np.log(K / F); c = D * F * call(k, wf(k)); p = c - D * (F - K)
        for cp, price in (("C", c), ("P", p)):
            rows.append(dict(root="SPX", ymd="270519", cp=cp, K=K, bid=price - 0.05, ask=price + 0.05))
    cfg = Config(expiries=(("SPX", "270519"),), n_per_expiry=25)
    mk = build_market_from_rows(rows, 7050.0, quote_time, cfg)
    assert abs(mk["info"][0]["F"] / F - 1) < 1e-9 and abs(mk["info"][0]["D"] / D - 1) < 1e-9
    assert abs(mk["T"][0] - T) < 1e-12
    f = mk["full"][0]
    np.testing.assert_allclose(f["iv_mid"], np.sqrt(wf(f["k"]) / T), atol=1e-8)
