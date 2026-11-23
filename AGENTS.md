# Notes for agents and contributors

## What this is

An arbitrage-free SPX implied-volatility surface in two parts:

- **SSVI slices**: one ATMF total variance θᵢ and one ATM skew sᵢ per expiry, plus a global η and γ. Between
  expiries, θ(T) is a monotone PCHIP and the skew is a PCHIP in θ.
- **Local-variance correction**: v = v₀·exp(ν), where ν is a tensor B-spline in x = k/√θ(T) and ln T. The
  coefficients are combinations of a few information-ranked modes, which by construction do not change ATMF
  or skew.

Prices come from the Dupire forward equation (Crank–Nicolson), so the surface is free of calendar and butterfly
arbitrage for **any** coefficients. There is no arbitrage penalty or check grid in the fit.

## Layout

| path | contents |
|---|---|
| `src/volsurf/` | the library (module map in README.md) |
| `tests/` | `pytest -q`: fast unit tests plus one end-to-end recovery on a synthetic market (`-m "not slow"` skips it) |
| `exp/` | numbered scripts, each writing only to `results/<study>/` |
| `results/` | committed outputs from the reference snapshots (JSON, Markdown, PNG) |
| `data/` | `README.md` and `fetch.py` only; the chains are gitignored |

## Invariants: do not break these

1. **Never commit market data.** `data/*.json` stays out of git (Cboe redistribution terms). Check
   `git status` before every commit.
2. **ATMF and skew belong to the SSVI parameters.** Modes live in the null space of `modes.constraint_rows`
   (`max_atm_leak`, `max_skew_leak` ~1e-10). After any change to the solver, basis or grid, the test
   `test_modes_do_not_move_atmf_or_skew` must still pass.
3. **No arbitrage by construction.** Local variance stays `v0 * exp(nu)` and the first-slice density stays
   `q0 * (1 + B a)` with `a >= -1`. Do not add terms that can make either negative.
4. **Full-support B-splines.** Use `splines.full_bspline_basis`, never `scipy.interpolate.BSpline(t, eye)`
   on non-clamped knots: that one silently truncates the outer splines (see the test).
5. **Model selection uses the free energy** (Laplace evidence) in `fit.free_energy`, not χ² or quote
   coverage. When comparing configurations, compare F on the same chains.
6. **Quote time comes from the file** (`last_trade_time`); expiries settle at 09:30 ET (AM SPX).

## Conventions

- `k = ln(K/F)`; `w` = total implied variance; prices are normalized calls `C/(D F)`.
- Vol errors in reports are in **vol points** (×100).
- `Config` is frozen. Change it with `cfg.with_(...)`, and save `config.json` next to every result.
- Figures use `plotting.STYLE` and its palette. Keep the bid–ask band visible.
- Keep the library free of I/O except `market.load_chain`, `market.load_vol_csv` and `pipeline.save_day`.
- Both inputs (prices via `build_market_from_rows`, vols via `build_market_from_vols`) must produce the same
  market dict through `market._slice` / `_assemble`; `test_vols_give_the_same_market_as_prices` guards this.

## Running

```bash
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q                                   # ~20 s
python -m volsurf fit data/SPX_20261119_1522.json --out results/tmp
./reproduce.sh                              # quick; ./reproduce.sh full for every study
```

One chain takes 10–25 minutes on one core with the default configuration.
