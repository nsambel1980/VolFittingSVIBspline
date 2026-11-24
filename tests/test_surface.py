"""The Dupire surface, the mode constraints, and an end-to-end recovery on the synthetic market."""
import numpy as np
import pytest

from volsurf.dupire import LocalVarSurface
from volsurf.ssvi import SSVISlices
from volsurf.modes import compute_modes, atm_rows
from volsurf.fit import fit_baseline, fit_modes, baseline
from volsurf.relinearize import relinearize, gap_summary
from volsurf.diagnostics import coverage, arbitrage, arbitrage_ok
from conftest import FAST, T_SYN, true_q, stub_market


def test_solver_reproduces_the_baseline():
    """Evolving the baseline's own local variance must give back the SSVI slices at the expiries."""
    b = SSVISlices(T_SYN, true_q())
    m = LocalVarSurface(stub_market(T_SYN, b.th_i), b, FAST, a=np.zeros(FAST.n_x * (1 + FAST.n_t)))
    for s, T in enumerate(T_SYN[1:], start=1):
        k = np.linspace(-2.5, 1.5, 41) * np.sqrt(b.th_i[s])
        err = 100 * np.abs(m.iv_exact(s, k) - np.sqrt(b.w0(k, T) / T)).max()
        assert err < 0.05, f"expiry {s}: {err:.3f} vol pts"


@pytest.mark.parametrize("start", ["first_slice", "zero"])
def test_any_coefficients_give_an_arbitrage_free_surface(start):
    """Solver prices AND the delivered surface (baseline + correction), for random coefficients."""
    cfg = FAST.with_(start=start)
    b = SSVISlices(T_SYN, true_q()); rng = np.random.default_rng(1)
    for _ in range(3):
        a = np.r_[np.zeros(cfg.n_x), rng.normal(0, 0.5, cfg.n_x * cfg.n_t)]
        m = LocalVarSurface(stub_market(T_SYN, b.th_i), b, cfg, a=a)
        arb = arbitrage(m)
        assert arbitrage_ok(arb), arb


def test_zero_start_reproduces_the_baseline_and_defines_early_local_vol():
    b = SSVISlices(T_SYN, true_q())
    cfg = FAST.with_(start="zero", steps_per_interval=40, min_steps=6)    # default time stepping
    m = LocalVarSurface(stub_market(T_SYN, b.th_i), b, cfg, a=np.zeros(cfg.n_x * (1 + cfg.n_t)))
    assert m.t_start < T_SYN[0]
    for s, T in enumerate(T_SYN):
        k = np.linspace(-2.5, 1.5, 41) * np.sqrt(b.th_i[s])
        assert 100 * np.abs(m.iv_exact(s, k) - np.sqrt(b.w0(k, T) / T)).max() < 0.05
    a = np.r_[np.zeros(cfg.n_x), np.full(cfg.n_x * cfg.n_t, 0.2)]          # +22% local variance everywhere
    m = LocalVarSurface(stub_market(T_SYN, b.th_i), b, cfg, a=a)
    x = np.array([0.0])
    ratio = m.local_vol(0.5 * T_SYN[0], x)[0] / LocalVarSurface(
        stub_market(T_SYN, b.th_i), b, cfg, a=np.zeros_like(a)).local_vol(0.5 * T_SYN[0], x)[0]
    assert abs(ratio - np.exp(0.1)) < 1e-6                                   # the correction acts before T1
    assert m.iv_exact(0, np.array([0.0]))[0] > np.sqrt(b.th_i[0] / T_SYN[0]) + 0.005


def test_modes_do_not_move_atmf_or_skew(synthetic):
    mk, base, _, _ = synthetic
    md = compute_modes(mk, base, FAST)
    assert md["max_atm_leak"] < 1e-9                                    # the skew is free since the redesign
    m = LocalVarSurface(mk, base, FAST, a=np.zeros(FAST.n_x * (1 + FAST.n_t)))
    np.testing.assert_allclose(md["V"].T @ m.P1 @ md["V"], np.eye(md["V"].shape[1]), atol=1e-8)
    assert np.all(np.diff(md["kappa"]) <= 1e-9)


@pytest.mark.slow
def test_recovers_the_synthetic_surface(synthetic):
    """End to end on the synthetic market: inside bid-ask, no arbitrage, ATMF / skew kept, truth recovered."""
    mk, base_true, a_true, truth = synthetic
    b = fit_baseline(mk, FAST)
    cover_base = np.mean([np.mean((np.sqrt(baseline(mk, b["q"], FAST).w0(f["k"], T) / T) >= f["iv_bid"]) &
                                  (np.sqrt(baseline(mk, b["q"], FAST).w0(f["k"], T) / T) <= f["iv_ask"]))
                          for f, T in zip(mk["full"], mk["T"])])
    md = compute_modes(mk, baseline(mk, b["q"], FAST), FAST)
    fit = fit_modes(mk, b["q"], md["V"], FAST)
    rl = relinearize(mk, FAST, fit)
    model = rl["model"]
    inside, by = coverage(mk, model)
    arb = arbitrage(model)
    gaps = gap_summary(model)
    err = max(100 * np.abs(model.iv_exact(s, f["k"]) - f["iv_true"]).max() for s, f in enumerate(mk["full"]))
    assert cover_base < 0.8, "the synthetic feature should be visible to the baseline"
    assert inside > 0.95, (inside, by)
    assert arbitrage_ok(arb), arb
    assert gaps[0] < 0.05
    assert err < 0.3, f"max error against the truth {err:.3f} vol pts"


def test_support_mask(synthetic):
    mk, base, _, _ = synthetic
    m = LocalVarSurface(mk, base, FAST, a=np.zeros(FAST.n_x * (1 + FAST.n_t)))
    x = np.array([-5.0, -3.0, 0.0, 1.5, 3.0])
    np.testing.assert_array_equal(m.supported(0.3, x), [False, True, True, True, False])
    assert not m.supported(0.5 * T_SYN[0], np.array([0.0]))[0]          # before the first expiry: modeled only
    assert not m.supported(0.3, np.array([0.0]), max_gap=0.01)[0]       # quotes further apart than max_gap
