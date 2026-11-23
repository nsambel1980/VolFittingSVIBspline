"""Download the current Cboe SPX delayed-quote snapshot, or verify the reference snapshots.

usage:  python data/fetch.py [--out DIR]      save data/SPX_YYYYMMDD_HHMM.json
        python data/fetch.py --verify         check the sha256 of the reference files in data/
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

URL = "https://cdn-api.cboe.com/api/global/delayed_quotes/options/_SPX.json"
HERE = Path(__file__).resolve().parent
REFERENCE = {
    "SPX_20261113_1615.json": "4d4d350fca51acd2dcf15d1e6af7bb579844415d0a1cd1ddf81a47ca28a305cb",
    "SPX_20261117_1615.json": "a7b0af492865b8684bca083a929c92d600da57d26f08af5ef46eefffea523973",
    "SPX_20261119_1522.json": "5efc04754eb06d8ba4ae127382d62e910d336f4a15b83cb043fc92d9d223c61e",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(folder):
    ok = True
    for name, digest in REFERENCE.items():
        p = folder / name
        if not p.exists():
            print(f"missing   {name}"); ok = False
        elif sha256(p) != digest:
            print(f"MISMATCH  {name}"); ok = False
        else:
            print(f"ok        {name}")
    return ok


def fetch(folder):
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (volsurf research)"})
    raw = urllib.request.urlopen(req, timeout=120).read()
    snap = json.loads(raw)
    t = dt.datetime.fromisoformat(snap["data"]["last_trade_time"])
    out = folder / f"SPX_{t:%Y%m%d_%H%M}.json"
    out.write_bytes(raw)
    print(f"saved {out} ({len(snap['data']['options'])} options, last trade {t})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(HERE)); p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    folder = Path(a.out); folder.mkdir(parents=True, exist_ok=True)
    if a.verify:
        sys.exit(0 if verify(folder) else 1)
    fetch(folder)
