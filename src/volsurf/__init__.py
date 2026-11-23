"""volsurf: arbitrage-free SPX volatility surfaces.

SSVI slices per expiry (ATMF variance and ATM skew as parameters) times an exponential B-spline correction of
the local variance, evolved by Dupire's equation; corrections restricted to information-ranked modes that
leave ATMF and skew unchanged; bid/ask likelihood; every choice of size by free energy.
"""
from .config import Config
from .market import build_market
from .pipeline import run_day, save_day, parameters

__all__ = ["Config", "build_market", "run_day", "save_day", "parameters"]
__version__ = "0.1.0"
