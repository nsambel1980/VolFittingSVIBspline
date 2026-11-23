"""Shared test fixtures: a coarse configuration and a synthetic market with a known truth.

The synthetic market is an SSVI baseline whose local variance carries a known bump in the put wing, evolved
through the same Dupire solver, so its quotes are exactly arbitrage-free and the truth is known.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volsurf.config import Config                     # noqa: E402
from volsurf.ssvi import SSVISlices, psi_law          # noqa: E402
from volsurf.dupire import LocalVarSurface            # noqa: E402

FAST = Config(n_x=9, n_t=4, dk=0.003, steps_per_interval=12, n_modes=8, max_iter=80, tau_rounds=2,
              gain_tol=1e-6)
T_SYN = np.array([0.1, 0.25, 0.5, 1.0])


def true_q():
    th = np.array([0.18, 0.19, 0.195, 0.20]) ** 2 * T_SYN
    eta, gam = 1.2, 0.45
    return np.r_[th, -0.7 * psi_law(th, eta, gam), eta, gam]


def stub_market(T, theta):
    """The minimum a LocalVarSurface needs: expiries, ATM variances and one quote per expiry."""
    n = len(T)
    return dict(T=T, nT=n, k=np.zeros(n), i=np.arange(n), z=theta.copy(), theta_atm=theta.copy())


def true_coefficients(cfg, base):
    """A put-wing bump of the local variance: about +80% around x = -2, fading with maturity."""
    model = LocalVarSurface(stub_market(T_SYN, base.th_i), base, cfg, a=None)
    centres = 0.5 * (model.tx[:-4] + model.tx[4:])
    a = np.zeros(model.D)
    bump = np.exp(-0.5 * ((centres + 2.0) / 0.6) ** 2)
    nu = np.outer(bump, [0.6, 0.5, 0.35, 0.2])
    a[cfg.n_x:] = nu.reshape(-1)
    return a


def synthetic_market(cfg=FAST, n_fit=20, n_all=60, seed=0):
    base = SSVISlices(T_SYN, true_q(), cfg.skew_interp)
    a_true = true_coefficients(cfg, base)
    truth = LocalVarSurface(stub_market(T_SYN, base.th_i), base, cfg, a=a_true)
    rng = np.random.default_rng(seed)
    ks, idx, z, iv, ivb, iva, full, info = [], [], [], [], [], [], [], []
    for s, T in enumerate(T_SYN):
        x_all = np.linspace(-3.5, 2.0, n_all); k_all = x_all * np.sqrt(base.th_i[s])
        iv_true = truth.iv_exact(s, k_all)
        hs = 0.0005 + 0.0002 * x_all ** 2                              # half spread in vol
        mid = iv_true + 0.4 * hs * rng.uniform(-1, 1, n_all)            # true vol always inside bid-ask
        full.append(dict(k=k_all, x=x_all, K=np.exp(k_all), iv_bid=mid - hs, iv_ask=mid + hs, iv_mid=mid,
                         iv_true=iv_true))
        pick = np.unique(np.linspace(0, n_all - 1, n_fit).round().astype(int))
        full[-1]["used"] = np.isin(np.arange(n_all), pick)
        ks.append(k_all[pick]); idx.append(np.full(len(pick), s)); iv.append(mid[pick])
        ivb.append(mid[pick] - hs[pick]); iva.append(mid[pick] + hs[pick]); z.append(mid[pick] ** 2 * T)
        info.append(dict(root="SYN", ymd=f"syn{s}", T=T, F=1.0, D=1.0, rate=0.0, theta=base.th_i[s], n=len(pick)))
    mk = dict(T=T_SYN, k=np.concatenate(ks), i=np.concatenate(idx), z=np.concatenate(z), iv=np.concatenate(iv),
              iv_bid=np.concatenate(ivb), iv_ask=np.concatenate(iva), nT=len(T_SYN),
              theta_atm=np.array([np.interp(0.0, f["k"], f["iv_mid"] ** 2 * T) for f, T in zip(full, T_SYN)]),
              info=info, full=full, quote_time=None, spot=1.0)
    return mk, base, a_true, truth


@pytest.fixture(scope="session")
def synthetic():
    return synthetic_market()


@pytest.fixture(scope="session")
def fast_cfg():
    return FAST
