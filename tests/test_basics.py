"""Fast unit tests: splines, Black formulas, SSVI identities, likelihood derivatives, put-call parity."""
import datetime as dt
import numpy as np
from scipy.interpolate import BSpline

from volsurf.splines import full_bspline_basis, uniform_knots
from volsurf.black import call, implied_w, durrleman, density_from_calls
from volsurf.ssvi import SSVISlices, ssvi_w, psi_law
from volsurf.likelihood import neg_log_lik
from volsurf.market import build_market_from_rows, build_market_from_vols, load_vol_csv
from volsurf.config import Config
from conftest import true_q, T_SYN, synthetic_quotes, FAST
import pytest


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
            v = [np.sqrt(b.v0(xs * np.sqrt(b.theta(t)), t)[0]) for t in (Ti * (1 - 5e-4), Ti * (1 + 5e-4))]
            out.append(100 * np.abs(v[1] - v[0]).max())
        return max(out)
    assert max_jump("pchip") < 0.2                            # continuous: shrinks with the window
    assert max_jump("linear") > 20 * max_jump("pchip")


def test_baseline_has_no_calendar_arbitrage_between_or_beyond_expiries():
    from volsurf.ssvi import calendar_gaps
    q = true_q(); b = SSVISlices(T_SYN, q)
    assert b.checks()["calendar_ok"]
    assert b.calendar_check() > -1e-12                       # whole surface, x in [-10, 10], 0.005y to 1.5y
    # wing slopes that cross between expiries are flagged and penalized
    bad = q.copy(); bad[5] = -0.3 * bad[5]                   # second expiry: put wing flatter than the first
    assert not SSVISlices(T_SYN, bad).checks()["calendar_ok"]
    from volsurf.ssvi import penalty
    assert penalty(bad, 4) > 1e3 * max(penalty(q, 4), 1e-9)


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


# ---------------------------------------------------------------- market: from implied vols
def test_vols_give_the_same_market_as_prices():
    """Prices -> market, then that market's vols -> build_market_from_vols: the same fitted quotes."""
    F, D, T = 7000.0, 0.98, 0.5
    expiry = dt.datetime(2027, 5, 19, 9, 30); quote_time = expiry - dt.timedelta(days=0.5 * 365)
    th = 0.2 ** 2 * T; wf = lambda k: ssvi_w(k, th, psi_law(th, 1.2, 0.45), -0.7)
    rows = []
    for K in np.arange(5000.0, 8400.0, 25.0):
        k = np.log(K / F); c = D * F * call(k, wf(k)); p = c - D * (F - K)
        for cp, price in (("C", c), ("P", p)):
            rows.append(dict(root="SPX", ymd="270519", cp=cp, K=K, bid=price * 0.99 - 0.05, ask=price * 1.01 + 0.05))
    cfg = Config(expiries=(("SPX", "270519"),), n_per_expiry=25)
    mp = build_market_from_rows(rows, 7050.0, quote_time, cfg)
    f, q = mp["full"][0], mp["info"][0]
    mv = build_market_from_vols([dict(expiry="2027-05-19", root="SPX", strike=f["K"], forward=q["F"],
                                      iv_bid=f["iv_bid"], iv_ask=f["iv_ask"], iv_mid=f["iv_mid"])], cfg,
                                quote_time=quote_time)
    for key in ("T", "k", "i", "iv_bid", "iv_ask", "z", "theta_atm"):
        np.testing.assert_allclose(mv[key], mp[key], atol=1e-12, err_msg=key)


def test_vol_csv_round_trip(tmp_path):
    expiries = synthetic_quotes(FAST, n_all=30)[0]
    path = tmp_path / "vols.csv"
    with open(path, "w") as fh:
        fh.write("T,k,iv_bid,iv_ask\n")
        for e in expiries:
            for row in zip(e["k"], e["iv_bid"], e["iv_ask"]):
                fh.write(f"{float(e['T'])!r}," + ",".join(repr(float(v)) for v in row) + "\n")
    loaded = load_vol_csv(path)
    assert [e["T"] for e in loaded] == [e["T"] for e in expiries]
    a = build_market_from_vols(expiries, FAST.with_(n_per_expiry=12))
    b = build_market_from_vols(loaded, FAST.with_(n_per_expiry=12))
    for key in ("k", "i", "iv_bid", "iv_ask"):
        np.testing.assert_allclose(a[key], b[key], atol=1e-15)


def test_mid_vols_only_and_input_checks():
    e = synthetic_quotes(FAST, n_all=30)[0][1]
    mk = build_market_from_vols([dict(T=e["T"], k=e["k"], iv_mid=e["iv_mid"])], FAST)
    assert np.all(mk["iv_ask"] - mk["iv_bid"] > 0) and np.all(mk["iv_ask"] - mk["iv_bid"] < 1e-4)
    mk = build_market_from_vols([dict(T=e["T"], k=e["k"], iv_mid=e["iv_mid"], half_spread=0.002)], FAST)
    np.testing.assert_allclose(mk["iv_ask"] - mk["iv_bid"], 0.004)
    with pytest.raises(ValueError, match="percentages"):
        build_market_from_vols([dict(T=e["T"], k=e["k"], iv_mid=100 * e["iv_mid"])], FAST)
    with pytest.raises(ValueError, match="bracket"):
        build_market_from_vols([dict(T=e["T"], k=e["k"][e["k"] < 0], iv_mid=e["iv_mid"][e["k"] < 0])], FAST)
    with pytest.raises(ValueError, match="quote_time"):
        build_market_from_vols([dict(expiry="2027-01-15", k=e["k"], iv_mid=e["iv_mid"])], FAST)


# ---------------------------------------------------------------- one-sided quotes
def test_one_sided_likelihood_is_censored():
    m = np.array([0.15, 0.2, 0.25, 0.3])
    f_ask, g_ask, _ = neg_log_lik(m, np.zeros(4), np.full(4, 0.22), 0.0005)        # ask only: [0, ask]
    f_bid, g_bid, _ = neg_log_lik(m, np.full(4, 0.22), np.full(4, np.inf), 0.0005)  # bid only: [bid, inf)
    assert np.all(np.isfinite(f_ask)) and np.all(np.isfinite(f_bid))
    assert f_ask[0] == f_ask[1] and f_ask[3] > f_ask[2] > f_ask[1]                   # flat below the ask
    assert f_bid[2] == f_bid[3] == 0 and f_bid[0] > f_bid[1] > 0                     # flat above the bid


def test_one_sided_quotes_enter_from_prices_and_vols():
    e = synthetic_quotes(FAST, n_all=40)[0][2]
    b, a = e["iv_bid"].copy(), e["iv_ask"].copy()
    wing = np.arange(len(b)) < 6; b[wing] = np.nan                      # deep puts: ask only
    a[-3:] = np.nan                                                      # far calls: bid only
    cfg = FAST.with_(n_per_expiry=40)
    mk = build_market_from_vols([dict(T=e["T"], k=e["k"], iv_bid=b, iv_ask=a)], cfg)
    f = mk["full"][0]
    assert f["one_sided"].sum() == 9 and np.all(f["iv_bid"][:6] == 0) and np.all(np.isinf(f["iv_ask"][-3:]))
    assert np.isclose(mk["theta_atm"][0], np.interp(0, e["k"], e["iv_mid"] ** 2 * e["T"]))
    off = build_market_from_vols([dict(T=e["T"], k=e["k"], iv_bid=b, iv_ask=a)], cfg.with_(one_sided=False))
    assert off["full"][0]["one_sided"].sum() == 0 and len(off["full"][0]["k"]) == len(f["k"]) - 9
    # prices: an ask-only put far out of the money is kept
    F, D, T = 7000.0, 0.98, 0.5
    expiry = dt.datetime(2027, 5, 19, 9, 30); quote_time = expiry - dt.timedelta(days=0.5 * 365)
    th = 0.2 ** 2 * T; wf = lambda k: ssvi_w(k, th, psi_law(th, 1.2, 0.45), -0.7)
    rows = []
    for K in np.arange(4000.0, 8400.0, 25.0):
        k = np.log(K / F); c = D * F * call(k, wf(k)); p = c - D * (F - K)
        for cp, price in (("C", c), ("P", p)):
            bid = 0.0 if (cp == "P" and K < 4500) else price * 0.99
            rows.append(dict(root="SPX", ymd="270519", cp=cp, K=K, bid=bid, ask=price * 1.01 + 0.05))
    mk = build_market_from_rows(rows, 7050.0, quote_time,
                                Config(expiries=(("SPX", "270519"),), n_per_expiry=25, quote_x_lo=-6.0))
    f = mk["full"][0]
    assert f["one_sided"].sum() == np.sum(f["K"] < 4500) > 0 and np.all(f["iv_bid"][f["K"] < 4500] == 0)
