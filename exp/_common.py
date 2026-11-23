"""Paths and helpers shared by the experiment scripts.

DATA_DIR (default ./data) holds the Cboe chains named SPX_YYYYMMDD_HHMM.json; RESULTS_DIR (default ./results)
receives one sub-folder per study. Both can be set in .env or the environment.
"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))


def chains():
    files = sorted(DATA.glob("SPX_*.json"))
    if not files:
        sys.exit(f"no chains in {DATA}: run  python data/fetch.py  (see data/README.md)")
    return files


def day_of(path):
    return Path(path).stem.split("_")[1]


def out_dir(*parts):
    d = RESULTS.joinpath(*parts); d.mkdir(parents=True, exist_ok=True)
    return d


def dump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=float)
    print(f"  wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def markdown_table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"
