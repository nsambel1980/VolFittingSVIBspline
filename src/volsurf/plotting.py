"""Figures: fit against bid-ask (portrait, one panel per expiry), local-vol slices and 3D surface."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq", ["#dce9f8", "#8fb8ea", "#2a78d6", "#14457f", "#0a2546"])
STYLE = {"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
         "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
         "axes.titlesize": 10.5, "axes.titlecolor": INK, "legend.frameon": False, "legend.labelcolor": INK,
         "lines.linewidth": 1.8}


def _label(q):
    return f"{q['label']} ({q['T']:.2f}y)"


def fit_vs_bidask(mk, model, out, title, baseline=None, extra=None):
    """Model minus mid vol against the bid-ask band, for every quoted strike, one panel per expiry.
    baseline: an SSVISlices to draw as well; extra: (label, model) for a second fitted surface."""
    plt.rcParams.update(STYLE)
    nT, T = mk["nT"], mk["T"]
    fig, axs = plt.subplots(nT, 1, figsize=(8.2, 2.2 * nT + 0.9), sharex=True)
    for s, ax in enumerate(np.atleast_1d(axs)):
        f = mk["full"][s]; iv = model.iv_exact(s, f["k"])
        one = f.get("one_sided", np.zeros(len(f["k"]), bool))
        # reference: mid for two-sided quotes, the quoted side for one-sided ones
        mid = np.where(np.isfinite(f["iv_mid"]), f["iv_mid"], np.where(np.isfinite(f["iv_ask"]), f["iv_ask"], f["iv_bid"]))
        ins = np.mean((iv >= f["iv_bid"]) & (iv <= f["iv_ask"]))
        lo = np.clip(100 * (f["iv_bid"] - mid), -5, 5); hi = np.clip(100 * (f["iv_ask"] - mid), -5, 5)
        two = ~one
        ax.fill_between(f["x"], np.where(two, lo, np.nan), np.where(two, hi, np.nan), color="#cfcec9", lw=0,
                        zorder=2, label="bid–ask range")
        if one.any():
            ax.scatter(f["x"][one], np.zeros(one.sum()), marker="v", s=14, color=INK2, zorder=3,
                       label="one-sided quote (ask ▼ / bid ▲)")
            bo = one & np.isfinite(f["iv_bid"]) & (f["iv_bid"] > 0)
            ax.scatter(f["x"][bo], np.zeros(bo.sum()), marker="^", s=14, color=INK2, zorder=3)
        ax.axhline(0, color=INK2, lw=0.6, zorder=3)
        if baseline is not None:
            ax.plot(f["x"], 100 * (np.sqrt(baseline.w0(f["k"], T[s]) / T[s]) - mid), color=ORANGE, zorder=4,
                    label="SSVI slices only")
        if extra is not None:
            ax.plot(f["x"], 100 * (extra[1].iv_exact(s, f["k"]) - mid), color=AQUA, lw=1.4, zorder=5, label=extra[0])
        ax.plot(f["x"], 100 * (iv - mid), color=BLUE, zorder=6, label="fitted surface")
        b = model.base
        ax.set_ylim(-0.9, 0.9); ax.set_ylabel("vol − mid (pts)")
        ax.set_title(f"{_label(mk['info'][s])}  ATMF {100*np.sqrt(b.th_i[s]/T[s]):.2f}%  skew {b.sk_i[s]:+.4f}  "
                     f"inside bid–ask {100*ins:.0f}%", loc="left")
    np.atleast_1d(axs)[-1].set_xlabel("k / √θ  (ATM std devs)")
    h, lab = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.987))
    fig.suptitle(title, x=0.02, ha="left", fontsize=12.5, color=INK, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.972), h_pad=0.7); fig.savefig(out, dpi=150); plt.close(fig)


def _lv_grid(model, xs):
    idx = np.unique(np.r_[np.arange(0, len(model.tg), 3), len(model.tg) - 1])
    Ts = model.tg[idx]
    return Ts, np.array([100 * model.local_vol(t, xs) for t in Ts])


def localvol_slices(models, labels, out, maturities=(0.1, 0.25, 0.5, 1.0, 2.0)):
    plt.rcParams.update(STYLE)
    xs = np.linspace(-3.5, 2.0, 221)
    cols = ["#a3c4ee", "#5f97dd", "#2a78d6", "#1a5aa8", "#0d3466"]
    names = {0.1: "5 weeks", 0.25: "3 months", 0.5: "6 months", 1.0: "1 year", 2.0: "2 years"}
    fig, axs = plt.subplots(len(models), 1, figsize=(8.2, 3.7 * len(models)), sharex=True, sharey=True)
    for ax, m, lab in zip(np.atleast_1d(axs), models, labels):
        for t, col in zip(maturities, cols):
            lv = 100 * m.local_vol(t, xs); sup = m.supported(t, xs)
            ax.plot(xs, np.where(sup, lv, np.nan), color=col, label=names.get(t, f"{t}y"))
            ax.plot(xs, np.where(sup, np.nan, lv), color=col, lw=1.0, ls=(0, (2, 2)))
        ax.set_title(lab, loc="left"); ax.set_ylabel("local vol (%)")
    np.atleast_1d(axs)[-1].set_xlabel("k / √θ  (ATM std devs)")
    np.atleast_1d(axs)[0].legend(fontsize=9.5, loc="upper right")
    fig.suptitle("Local volatility at fixed maturities (dashed: not supported by quotes)", x=0.02, ha="left",
                 fontsize=12.5, color=INK)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def localvol_3d(models, labels, out):
    plt.rcParams.update(STYLE)
    xs = np.linspace(-3.5, 2.0, 111)
    grids = [_lv_grid(m, xs) for m in models]
    zmin = min(g[1].min() for g in grids); zmax = max(g[1].max() for g in grids)
    fig = plt.figure(figsize=(8.2, 6.3 * len(models)))
    names = {0.1: "5W", 0.25: "3M", 0.5: "6M", 1: "1Y", 2: "2Y"}
    for j, ((Ts, S), lab) in enumerate(zip(grids, labels)):
        ax = fig.add_subplot(len(models), 1, j + 1, projection="3d", computed_zorder=False)
        X, Y = np.meshgrid(xs, np.log(Ts))
        ax.plot_surface(X, Y, S, cmap=SEQ, vmin=zmin, vmax=zmax, rstride=1, cstride=1, linewidth=0.12,
                        edgecolor=(1, 1, 1, 0.3))
        ax.set_box_aspect((1.55, 1.25, 0.85), zoom=1.12); ax.set_zlim(5, max(90, zmax * 1.05))
        ticks = [t for t in names if Ts[0] * 0.98 <= t <= Ts[-1] * 1.02]
        ax.set_yticks(np.log(ticks)); ax.set_yticklabels([names[t] for t in ticks])
        ax.set_xlabel("k / √θ  (ATM std devs)", labelpad=8); ax.set_ylabel("maturity", labelpad=6)
        ax.set_zlabel("local vol (%)", labelpad=4); ax.view_init(elev=26, azim=-60)
        ax.set_title(lab, loc="left", fontsize=12, pad=-4)
    fig.suptitle("Local volatility", x=0.03, ha="left", fontsize=13, color=INK, y=0.995)
    fig.subplots_adjust(left=0.02, right=0.93, top=1 - 0.1 / len(models), bottom=0.02, hspace=0.08)
    fig.savefig(out, dpi=150); plt.close(fig)
