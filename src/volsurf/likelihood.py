"""Bid/ask likelihood.

If all we know is that the fair implied vol lies between bid and ask, the maximum-entropy distribution for it is
uniform on [bid, ask]. Blurred by Gaussian noise of size s (asynchronous marks, model error):

    L(m) = [Phi((ask - m)/s) - Phi((bid - m)/s)] / (ask - bid)

flat inside the spread, a smooth Gaussian penalty outside it, log-concave.

One-sided quotes: an ask-only quote is the interval [0, ask] (a price is never below intrinsic, i.e. zero vol);
a bid-only quote is [bid, inf), whose likelihood is the censored Phi((m - bid)/s) with no width normalization.
"""
import numpy as np
from scipy.special import log_ndtr


def neg_log_lik(m, bid, ask, s):
    """-ln L per quote, with its gradient and curvature with respect to the model vol m."""
    finite = np.isfinite(ask)
    norm = np.log(np.where(finite, ask - bid, 1.0))

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
    eps = 1e-6
    with np.errstate(invalid="ignore", over="ignore"):
        h = (core(m + eps)[1] - core(m - eps)[1]) / (2 * eps)
    return f, g, np.maximum(h, 0.0)


class QuoteLikelihood:
    """The likelihood of the fitted quotes, as a function of model total variance at the quotes."""
    def __init__(self, mk, s_noise):
        self.bid, self.ask, self.s = mk["iv_bid"], mk["iv_ask"], s_noise
        self.Tq = mk["T"][mk["i"]]

    def vols(self, w):
        return np.sqrt(np.maximum(w, 1e-12) / self.Tq)

    def __call__(self, w):
        return float(np.sum(neg_log_lik(self.vols(w), self.bid, self.ask, self.s)[0]))

    def inside(self, w):
        m = self.vols(w)
        return (m >= self.bid) & (m <= self.ask)

    def newton_weights(self, w):
        """Gradient and curvature of -ln L in total variance: (g_w, W floored for steps, W raw for evidence)."""
        m = self.vols(w)
        _, g, h = neg_log_lik(m, self.bid, self.ask, self.s)
        dm = 1.0 / (2 * m * self.Tq)
        return g * dm, np.maximum(h, 0.01 / self.s ** 2) * dm ** 2, np.maximum(h * dm ** 2, 1e-12)
