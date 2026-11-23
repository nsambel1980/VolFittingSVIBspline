#!/usr/bin/env bash
# Reproduce the results.
#   ./reproduce.sh          tests, then fit every chain in DATA_DIR and the cheap skew study    (~1 h)
#   ./reproduce.sh full     also the stability, number-of-modes and coordinate studies          (~10 h)
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q
python data/fetch.py --verify || echo "note: reference snapshots missing or different; results will differ from results/"

python exp/01_fit_days.py
python exp/05_skew_interp.py
if [ "${1:-quick}" = "full" ]; then
  python exp/02_stability.py
  python exp/03_n_modes.py
  python exp/04_coordinates.py
fi
