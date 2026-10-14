"""Ein Training = ein Ordner runs/<name>/ (Regel R9: alles nachvollziehbar).

runs/<name>/
  config.json        vollständige Konfiguration, Seed, alle Versionen, Git-Commit (beim Start geschrieben)
  summary.csv        eine Zeile pro Auswertung und pro Ereignis (Start, Pause, Fortsetzung, Profilwechsel, ...)
  checkpoint/        Fortsetzungspunkt: Modell, Replay-Buffer, Zähler, Zufallszustand
  best_model.zip     Modell der offiziellen Bestzeit (deterministische Auswertung, R6)
  ghosts/            Fahrt der offiziellen Bestzeit als Geist (für play.py --ghost / replay.py)
  snapshots/         Modell-Kopien alle N Schritte
  tb/                TensorBoard-Kurven
  code.tar.gz        Kopie des Codes, falls er vom Git-Commit abweicht (nicht eingecheckt oder neue Dateien)
"""
import csv
import json
import pickle
import platform
import random
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

SUMMARY_FIELDS = ["zeit", "laufzeit_s", "schritte", "versuche", "ereignis", "gueltig", "rundenzeit",
                  "fortschritt_pct", "ende", "wandberuehrungen", "bestzeit", "stufe", "serie", "profil", "bemerkung"]


def git_info():
    """Aktueller Git-Commit und ob es nicht eingecheckte Änderungen gibt."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD") or None, "geaendert": bool(status), "aenderungen": status.splitlines()}


def code_files():
    """Alle Code-Dateien des Projekts (eingecheckt oder neu, ohne .gitignore-Daten wie runs/ oder ghosts/)."""
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True).stdout.splitlines()
    return [f for f in out if (ROOT / f).is_file() and Path(f).suffix in (".py", ".toml", ".json", ".md", ".txt", ".ini")]


def package_versions():
    import gymnasium
    import stable_baselines3
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__, "gymnasium": gymnasium.__version__,
            "rechner": f"{platform.system()} {platform.machine()}"}


class Run:
    def __init__(self, name):
        self.name = name
        self.dir = RUNS_DIR / name
        self.checkpoint_dir = self.dir / "checkpoint"
        self.summary_file = self.dir / "summary.csv"
        self.config_file = self.dir / "config.json"
        self.tb_dir = self.dir / "tb"
        self.snapshot_dir = self.dir / "snapshots"
        self.ghost_dir = self.dir / "ghosts"
        self.best_model = self.dir / "best_model.zip"

    @property
    def exists(self):
        return self.config_file.exists()

    # ---------- Konfiguration (R9) ----------
    def create(self, config):
        self.dir.mkdir(parents=True, exist_ok=False)
        self.config_file.write_text(json.dumps(config, indent=2, ensure_ascii=False))
        if config.get("git", {}).get("geaendert"):     # R9: Code-Stand sichern, den der Commit nicht enthält
            with tarfile.open(self.dir / "code.tar.gz", "w:gz") as tar:
                for f in code_files():
                    tar.add(ROOT / f, arcname=f)

    def load_config(self):
        return json.loads(self.config_file.read_text())

    # ---------- summary.csv ----------
    def log(self, **row):
        new = not self.summary_file.exists()
        row.setdefault("zeit", time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(self.summary_file, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: _fmt(v) for k, v in row.items()})

    # ---------- Fortsetzungspunkt ----------
    def save_checkpoint(self, model, state):
        """Erst in einen Nebenordner schreiben, dann austauschen: bricht etwas mittendrin ab,
        bleibt der vorherige Fortsetzungspunkt heil."""
        tmp, old = self.dir / "checkpoint.neu", self.dir / "checkpoint.alt"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        model.save(tmp / "model.zip")
        model.save_replay_buffer(tmp / "replay_buffer.pkl")
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
        (tmp / "rng.pkl").write_bytes(pickle.dumps(rng))
        (tmp / "state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))
        shutil.rmtree(old, ignore_errors=True)
        if self.checkpoint_dir.exists():
            self.checkpoint_dir.rename(old)
        tmp.rename(self.checkpoint_dir)
        shutil.rmtree(old, ignore_errors=True)

    def find_checkpoint(self):
        """Ordner des letzten vollständigen Fortsetzungspunkts (oder None)."""
        for d in (self.checkpoint_dir, self.dir / "checkpoint.alt"):
            if (d / "state.json").exists():
                return d
        return None

    @staticmethod
    def load_state(cp_dir):
        return json.loads((cp_dir / "state.json").read_text())

    @staticmethod
    def restore_rng(cp_dir):
        rng = pickle.loads((cp_dir / "rng.pkl").read_bytes())
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "ja" if v else "nein"
    if isinstance(v, float):
        return f"{v:.2f}"
    return v
