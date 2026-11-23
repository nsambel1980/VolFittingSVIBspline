# Data

The option chains are **not** in this repository: they are Cboe delayed-quote snapshots, and Cboe's terms
restrict redistribution. Put them here yourself; `.gitignore` keeps `data/*.json` out of git.

## Source

`https://cdn-api.cboe.com/api/global/delayed_quotes/options/_SPX.json`: every listed SPX and SPXW option,
with bid, ask, strike and expiry, 15-minute delayed.

```bash
python data/fetch.py            # saves today's snapshot as data/SPX_YYYYMMDD_HHMM.json
python data/fetch.py --verify   # checks the reference files below against their checksums
```

The file name uses the snapshot's `last_trade_time` (US Eastern). The fitting code takes the quote time from
inside the file, so the name is only for humans and for ordering.

## Reference snapshots used in `results/`

Cboe serves only the current snapshot, so these exact files cannot be downloaded again. Without them, any
snapshot you fetch runs through the same pipeline. The numbers will differ, but the checks (inside bid-ask, no
arbitrage, ATMF and skew kept) should hold.

| file | last_trade_time | sha256 |
|---|---|---|
| `SPX_20261113_1615.json` | 2026-11-13 16:14:59 | `4d4d350fca51acd2dcf15d1e6af7bb579844415d0a1cd1ddf81a47ca28a305cb` |
| `SPX_20261117_1615.json` | 2026-11-17 16:14:59 | `a7b0af492865b8684bca083a929c92d600da57d26f08af5ef46eefffea523973` |
| `SPX_20261119_1522.json` | 2026-11-19 15:22:35 | `5efc04754eb06d8ba4ae127382d62e910d336f4a15b83cb043fc92d9d223c61e` |

## Expiries

The default set (`volsurf.config.DEFAULT_EXPIRIES`) is the nine monthly AM-settled SPX expiries from Dec 2026
to Dec 2028. For a later snapshot, edit that list or pass your own `Config(expiries=...)`. An expiry missing
from the file is an error, not a silent skip.
