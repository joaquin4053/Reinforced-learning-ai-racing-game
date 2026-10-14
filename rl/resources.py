"""Wie stark das Training den Mac belastet: Leistungsprofile, Akku- und Speicher-Wächter, wach bleiben."""
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import psutil
import torch

PROFILES = ("sparsam", "normal", "voll")


@dataclass
class Profile:
    name: str
    envs: int          # Autos, die gleichzeitig Erfahrung sammeln
    threads: int       # Rechen-Threads fürs Netz
    nice: int          # Prozess-Priorität (0 = normal, 19 = niedrigste)

    def __str__(self):
        return f"{self.name} ({self.envs} Umgebungen, {self.threads} Thread{'s' if self.threads > 1 else ''}, nice {self.nice})"


def make_profile(name, profiles_cfg):
    cfg = profiles_cfg[name]
    cores = os.cpu_count() or 2
    envs = cfg["umgebungen"]
    if envs == "halb":
        envs = cores // 2
    elif envs == "alle-1":
        envs = cores - 1
    return Profile(name, max(1, int(envs)), max(1, int(cfg["threads"])), int(cfg["prioritaet"]))


def apply_priority(profile):
    """Threads und Priorität setzen. Die Priorität kann ein normaler Prozess nur senken (nice erhöhen)."""
    torch.set_num_threads(profile.threads)
    current = os.nice(0)
    if profile.nice > current:
        os.nice(profile.nice - current)
    return os.nice(0)


def keep_awake():
    """Mac nicht einschlafen lassen, solange dieses Programm läuft (macOS: caffeinate -i).
    Zugeklappt schläft ein MacBook trotzdem ein."""
    if sys.platform != "darwin" or not shutil.which("caffeinate"):
        return None
    return subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@dataclass
class PowerState:
    on_battery: bool
    battery_pct: float | None
    free_gb: float

    @property
    def text(self):
        if self.battery_pct is None:
            return "Netz"
        return f"Akku {self.battery_pct:.0f} %" if self.on_battery else "Netzteil"


def power_state(path):
    bat = psutil.sensors_battery()
    free_gb = shutil.disk_usage(path).free / 1e9
    if bat is None:        # Rechner ohne Akku
        return PowerState(False, None, free_gb)
    return PowerState(not bat.power_plugged, bat.percent, free_gb)


class SpeedLimit:
    """Höchstens max_sps Schritte pro Sekunde (Wanduhr), indem zwischendurch kurz gewartet wird."""

    def __init__(self, max_sps):
        self.max_sps = max_sps
        self.reset(0)

    def reset(self, steps):
        self.t0, self.s0 = time.perf_counter(), steps

    def wait(self, steps):
        if not self.max_sps:
            return
        ahead = (steps - self.s0) / self.max_sps - (time.perf_counter() - self.t0)
        if ahead > 0:
            time.sleep(ahead)
