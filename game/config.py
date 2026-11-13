"""Liest config.toml (Physik-Werte, Regeln, Sensoren, Versionsnummern) – einmal beim Start."""
import os
import tomllib
from pathlib import Path

# KI_FAHRSCHULE_CONFIG: andere Datei verwenden (nur für Probeläufe, z.B. andere Sensor-Reichweite)
CONFIG_FILE = Path(os.environ.get("KI_FAHRSCHULE_CONFIG") or Path(__file__).resolve().parent.parent / "config.toml")

with open(CONFIG_FILE, "rb") as f:
    CONFIG = tomllib.load(f)

VERSIONS = CONFIG["versionen"]
