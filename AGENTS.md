# Notes for agents and contributors

## What this is

An arbitrage-free implied-volatility surface for raw option chains (single names and indices) in two parts:

- **SSVI slices**: one ATMF total variance θᵢ and one ATM skew sᵢ per expiry, plus a global η and γ. The baseline
  is calendar-free by construction:
  - θ, ψ ± s and φ = ψ/θ are monotone (penalty, then `ssvi.project_calendar`);
  - between expiries ψ stays on its law and the skew is a Lipschitz Hermite in 2ψ.

  A Cauchy prior smooths the term structure; its scale is chosen by evidence.
- **Local-variance correction**: v = v₀·exp(ν), a tensor B-spline in x = k/√θ(T) and ln T, with knots at the
  expiries. The coefficients are combinations of information-ranked modes that keep every expiry's ATMF level.

Prices come from the Dupire forward equation on a non-uniform log-strike grid, and the delivered prices are
the solver's own. The surface is free of calendar and butterfly arbitrage for **any** coefficients. There is no
arbitrage penalty on the correction and no check grid in the fit.

## Layout

| path | contents |
|---|---|
| `src/volsurf/` | the library (module map in README.md) |
| `tests/` | `pytest -q`: fast unit tests plus one end-to-end recovery on a synthetic market (`-m "not slow"` skips it) |
| `exp/` | numbered scripts, each writing only to `results/<study>/`; `07_single_name.py` is the stress harness |
| `results/` | committed outputs from the reference snapshots and the synthetic single-name book |
| `data/` | `README.md` and `fetch.py` only; the chains are gitignored |

## Invariants: do not break these

1. **Never commit market data.** `data/*.json` stays out of git (Cboe redistribution terms). Check
   `git status` before every commit.
2. **Fit in price space.** The likelihood, the responses `A` and the baseline Jacobian are in normalized call
   prices. Never divide by Black vega in the fit: far out-of-the-money it is ~0, and that division caused
   every crash in the 30-book comparison. Implied vols are for display.
3. **The ATMF level belongs to θᵢ.** Modes live in the null space of `modes.constraint_rows` (`max_atm_leak`
   ~1e-10). The skew is free. `test_modes_do_not_move_atmf_or_skew` must pass.
4. **No arbitrage by construction:**
   - local variance stays `v0 * exp(nu)`;
   - the first-slice coefficients act through `P_E`, so mass and forward are exact;
   - baseline parameters pass through `project_calendar` whenever they change;
   - `calls()` returns the solver's prices.

   Do not add terms that can make local variance or a density negative, and do not deliver prices that are not
   the solver's.
5. **Rising wing slopes are not a calendar condition on their own.** SSVI slices also need φ = ψ/θ
   non-increasing: `test_baseline_has_no_calendar_arbitrage_between_or_beyond_expiries` and the random check
   in the commit history guard this.
6. **Full-support B-splines.** Use `splines.full_bspline_basis`, never `scipy.interpolate.BSpline(t, eye)` on
   non-clamped knots, which silently truncates the outer splines.
7. **Model selection uses the free energy** (Laplace evidence), not χ² or quote coverage. Compare F only on the
   same quotes.
8. **Quote time comes from the file** (`last_trade_time`); SPX AM expiries settle at 09:30 ET, others at 16:00.

## Conventions

- `k = ln(K/F)`; `w` = total implied variance; prices are normalized calls `c = C/(D F)`.
- Errors are reported in **half-spreads** (price) and **vol points** (×100).
- `Config()` fits every expiry and every row of a chain; `config.SPX_STUDY` is the published SPX setup.
  `Config` is frozen: change it with `cfg.with_(...)`, and save `config.json` next to every result.
- The coefficient layout depends on the data: the strike range comes from the quotes and the maturity knots
  sit at the expiries. Reusing modes across days needs a pinned layout (`x_lo`, `x_hi`, `n_x`; see
  `exp/02_stability.py`).
- Figures use `plotting.STYLE` and its palette. Keep the bid–ask band visible.
- Keep the library free of I/O except `market.load_chain`, `market.load_vol_csv` and `pipeline.save_day`.
- Both inputs (prices via `build_market_from_rows`, vols via `build_market_from_vols`) must produce the same
  market dict; `test_vols_give_the_same_market_as_prices` guards this.

## Running

```bash
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q                                   # ~15 s
python exp/07_single_name.py --shocks       # the single-name stress harness, ~8 min
python -m volsurf fit data/SPX_20261119_1522.json --out results/tmp
./reproduce.sh                              # quick; ./reproduce.sh full for every study
```

A 16-expiry single-name book of 1,270 rows fits in about 2.5 minutes on one core.
