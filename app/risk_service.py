"""Day 7: one shared risk engine for the whole API process.

The engine keeps each sender's recent history in memory. That history is lost
when the API restarts (a known limitation; a later step can rebuild it from the
risk_assessments table). Run a single API process, not several.

Environment variables:
  RISK_MODEL_PATH       where the trained model lives. Empty = rules only.
  RISK_SIMULATION_MODE  "1" lets a client set the clock with an X-Simulated-Time
                        header. DEVELOPMENT ONLY: in production a client could
                        fake timestamps to dodge the velocity rules.
"""
import logging
import os
from threading import Lock

from risk.engine import RiskEngine

log = logging.getLogger("risk")
_engine = None
_lock = Lock()


def simulation_mode():
    return os.getenv("RISK_SIMULATION_MODE") == "1"


def _build_engine():
    path = os.getenv("RISK_MODEL_PATH", "models/risk_model.joblib")
    if path:
        try:
            engine = RiskEngine.load(path)
            log.info("risk engine started with model %s", path)
            return engine
        except Exception as exc:  # a missing or incompatible model must not stop payments
            log.warning("risk model not loaded (%s); using rules only", exc)
    return RiskEngine()


def get_engine():
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = _build_engine()
    return _engine


def engine_mode():
    return "model+rules" if get_engine().model is not None else "rules_only"
