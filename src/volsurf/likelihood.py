"""Bid/ask likelihood in price space.

If all we know is that the fair price lies between bid and ask, the maximum-entropy distribution for it is
uniform on [bid, ask]. Blurred by Gaussian noise of size s (asynchronous marks, model error):

    L(c) = [Phi((ask - c)/s) - Phi((bid - c)/s)] / (ask - bid)

flat inside the spread, a smooth Gaussian penalty outside it, log-concave. Everything is in normalized call
prices c = C / (D F): no conversion to implied vol, so far out-of-the-money quotes (vega ~ 0) and mids below
intrinsic are handled like any other quote.

  noise         s_i = price_noise * h_i  (h_i = half-spread), so a miss is measured in half-spreads, as a
                trader would
  mid pull      optional and mild: + 1/2 ((c - mid) / (mid_pull * h))^2 on two-sided quotes; with the default
                mid_pull = 2 the pull at the edge of the spread is 1/8 nat
  one-sided     ask-only: [intrinsic, ask];  bid-only: [bid, inf), the censored Phi((c - bid)/s)
"""
import numpy as np
from scipy.special import log_ndtr


def neg_log_lik(m, bid, ask, s):
    """-ln L per quote, with its gradient and curvature with respect to the model value m."""
    finite = np.isfinite(ask)
    norm = np.log(np.where(finite, np.maximum(ask - bid, 1e-300), 1.0))

    def core(mm):
        x1, x2 = (ask - mm) / s, (bid - mm) / s
        flip = x2 > 0                                      # use the upper tail when both are positive
        a1 = np.where(flip, -x2, x1); a2 = np.where(flip, -x1, x2)
        l1, l2 = log_ndtr(a1), log_ndtr(a2)
        logD = l1 + np.log1p(-np.exp(np.minimum(l2 - l1, -1e-300)))
        lphi = lambda x: -0.5 * x * x - 0.5 * np.log(2 * np.pi)
        g = (np.exp(lphi(x1) - logD) - np.exp(lphi(x2) - logD)) / s
        return -logD + norm, g
    with np.errstate(invalid="ignore", over="ignore"):
        f, g = core(m)
        eps = 1e-6 * s
        h = (core(m + eps)[1] - core(m - eps)[1]) / (2 * eps)
    return f, g, np.maximum(h, 0.0)


class QuoteLikelihood:
    """The likelihood of the fitted quotes as a function of model normalized call prices at the quotes."""

    def __init__(self, mk, cfg):
        self.bid, self.ask, self.h = mk["c_bid"], mk["c_ask"], mk["h"]
        self.s = cfg.price_noise * self.h
        two = ~mk["one_sided"]
        self.mid = np.where(two, mk["c_mid"], 0.0)
        self.wm = np.where(two, 1.0 / (cfg.mid_pull * self.h) ** 2, 0.0) if cfg.mid_pull else np.zeros(len(self.h))

    def terms(self, c):
        f, g, h = neg_log_lik(c, self.bid, self.ask, self.s)
        d = c - self.mid
        return f + 0.5 * self.wm * d ** 2, g + self.wm * d, h + self.wm

    def __call__(self, c):
        return float(np.sum(self.terms(c)[0]))

    def inside(self, c):
        return (c >= self.bid) & (c <= self.ask)

    def newton_weights(self, c):
        """Gradient and curvature of -ln L in price: (g, W floored for steps, W raw for evidence)."""
        _, g, h = self.terms(c)
        return g, np.maximum(h, 0.01 / self.s ** 2), np.maximum(h, 1e-12 / self.s ** 2)
