"""Settings applied before any test imports the app.

Tests use the rules only (no trained model), so a test result never depends on
the time of day or on which model happens to be saved. They also use
simulation mode, so each test sets the clock with an X-Simulated-Time header.
"""
import os

os.environ["RISK_MODEL_PATH"] = ""
os.environ["RISK_SIMULATION_MODE"] = "1"
