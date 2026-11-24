"""Synthetic single-name option books with a known, arbitrage-free truth.

The truth is a diffusion (an SSVI surface with a term structure of ATM vol and skew) plus an earnings jump
at t_event: ln S_T = X_T + J 1[T > t_event], with J independent of X, a two-point move blurred by a small Gaussian
and compensated so E[e^J] = 1. Expiries just after the event have W-shaped smiles. Prices are exact integrals
of the terminal density, so the book is free of butterfly and calendar arbitrage:
  - each slice is a convolution of an arbitrage-free density with a mean-one jump density,
  - E[(e^(X2 + J) - K)^+] >= E[(e^X2 - K)^+] >= E[(e^X1 - K)^+] across the event (Jensen, then SSVI).

Quotes mimic a listed single-name chain: dollar strikes (1 near the money, 2.5, then 5), wide strike ranges
for every expiry, tick-rounded bids and asks with spreads that grow with price, and ask-only quotes (bid 0)
far out of the money. Both calls and puts are listed, so the forward comes from put-call parity.
"""
import datetime as dt
import numpy as np
from .black import durrleman, implied_w
from .ssvi import ssvi_w, psi_law

QUOTE_TIME = dt.datetime(2026, 11, 20, 16, 0)


def expiry_dates(quote_time=QUOTE_TIME):
    """Six weeklies, then monthlies / quarterlies / LEAPs: 16 expiries from 1 week to 2 years."""
    days = [7, 14, 21, 28, 35, 42, 63, 91, 119, 154, 182, 273, 364, 455, 546, 728]
    return [(quote_time + dt.timedelta(days=d)).replace(hour=0, minute=0) for d in days]


def strikes(F, T, atm_vol):
    sd = atm_vol * np.sqrt(T)
    lo = F * np.exp(-max(0.35, 5 * sd)); hi = F * np.exp(max(0.30, 4 * sd))
    ks = set()
    for step, band in ((1.0, 0.15), (2.5, 0.40), (5.0, 10.0)):
        a = max(lo, F * (1 - band)); b = min(hi, F * (1 + band))
        ks |= set(np.arange(np.ceil(a / step) * step, b + 1e-9, step).round(2))
    return np.array(sorted(k for k in ks if lo <= k <= hi and k > 0))


class Truth:
    """Diffusion SSVI + earnings jump. All prices normalized: c = C / F (zero rates, F constant)."""

    def __init__(self, t_event=24.5 / 365, jumps=((0.14, 0.5), (-0.16, 0.5)), jump_blur=0.025,
                 vol_short=0.30, vol_long=0.31, rho_short=-0.25, rho_long=-0.55, eta=1.0, gamma=0.4):
        self.t_event, self.blur = t_event, jump_blur
        J = np.array([j for j, _ in jumps]); p = np.array([q for _, q in jumps]); p = p / p.sum()
        comp = np.log(np.sum(p * np.exp(J + 0.5 * jump_blur ** 2)))         # E[e^J] = 1 after shifting
        self.J, self.p = J - comp, p
        self.vs, self.vl, self.rs, self.rl, self.eta, self.gam = vol_short, vol_long, rho_short, rho_long, eta, gamma
        self.y = np.arange(-6.0, 4.0, 0.0005)                                 # log-return grid

    def atm_vol(self, T):
        return self.vl + (self.vs - self.vl) * np.exp(-T / 0.4)

    def diffusion_w(self, k, T):
        th = self.atm_vol(T) ** 2 * T
        rho = self.rl + (self.rs - self.rl) * np.exp(-T / 0.5)
        return ssvi_w(k, th, psi_law(th, self.eta, self.gam), rho)

    def density(self, T):
        _, q = durrleman(lambda k: self.diffusion_w(k, T), self.y)
        q = np.maximum(q, 0.0); dy = self.y[1] - self.y[0]
        if T > self.t_event:
            g = np.arange(-6 * self.blur, 6 * self.blur + dy / 2, dy)
            kern = np.exp(-0.5 * (g / self.blur) ** 2); kern /= kern.sum()
            out = np.zeros_like(q)
            for J, p in zip(self.J, self.p):
                out += p * np.interp(self.y - J, self.y, q, left=0.0, right=0.0)
            q = np.convolve(out, kern, mode="same")
        q /= q.sum() * dy
        q *= 1.0 / (np.sum(np.exp(self.y) * q) * dy)                           # exact martingale on the grid
        return q

    def calls(self, k, T):
        """Normalized calls c(k) = E[(e^Y - e^k)^+], by integration of the terminal density."""
        q = self.density(T); dy = self.y[1] - self.y[0]
        ey = np.exp(self.y)
        tail_q = np.cumsum((q * dy)[::-1])[::-1]                               # P(Y >= y)
        tail_e = np.cumsum((ey * q * dy)[::-1])[::-1]                          # E[e^Y; Y >= y]
        i = np.clip(np.searchsorted(self.y, k), 0, len(self.y) - 1)
        return np.maximum(tail_e[i] - np.exp(k) * tail_q[i], np.maximum(1 - np.exp(k), 0.0))

    def iv(self, k, T):
        return np.sqrt(implied_w(k, self.calls(k, T)) / T)


def book(truth=None, F=100.0, seed=0, shock=0.0, quote_time=QUOTE_TIME, root="SYN"):
    """Rows (root, ymd, cp, K, bid, ask) of a synthetic chain, the expiry list and per-row truth.
    shock: shift every two-sided mid by shock * U(-1, 1) half-spreads (the signed quote-shock test)."""
    truth = truth or Truth(); rng = np.random.default_rng(seed); srng = np.random.default_rng(5505)
    rows, expiries, true_rows = [], [], []
    for d in expiry_dates(quote_time):
        ymd = d.strftime("%y%m%d"); T = ((d + dt.timedelta(hours=16)) - quote_time).total_seconds() / (365 * 86400)
        expiries.append((root, ymd))
        Ks = strikes(F, T, truth.atm_vol(T)); k = np.log(Ks / F)
        c = truth.calls(k, T)
        for K, kk, cc in zip(Ks, k, c):
            otm_put = K < F
            p_otm = F * (cc - max(1 - np.exp(kk), 0.0))                         # dollars
            for cp in ("C", "P"):
                is_otm = (cp == "P") == otm_put
                price = p_otm + (0.0 if is_otm else abs(F - K))
                tick = 0.01 if price < 3 else 0.05
                spread = np.clip(0.04 + 0.035 * price, 0.05, 2.5) * (1.0 if is_otm else 1.6)
                mid = price + 0.3 * spread / 2 * rng.uniform(-1, 1) + shock * spread / 2 * srng.uniform(-1, 1)
                bid = np.floor((mid - spread / 2) / tick) * tick
                ask = np.ceil((mid + spread / 2) / tick) * tick
                if bid <= 0:
                    bid, ask = 0.0, max(ask, 0.05)
                rows.append(dict(root=root, ymd=ymd, cp=cp, K=float(K), bid=round(float(bid), 2),
                                 ask=round(float(ask), 2)))
                if is_otm:
                    true_rows.append(dict(ymd=ymd, K=float(K), T=T, c=float(cc)))
    return dict(rows=rows, expiries=tuple(expiries), spot=F, quote_time=quote_time, truth=truth,
                true_rows=true_rows)
