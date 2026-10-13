"""Liest config.toml (Physik-Werte, Regeln, Sensoren, Versionsnummern) – einmal beim Start."""
import tomllib
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.toml"

with open(CONFIG_FILE, "rb") as f:
    CONFIG = tomllib.load(f)

VERSIONS = CONFIG["versionen"]
