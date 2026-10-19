"""Aufgezeichnete Versuche abspielen (Plan-Schritt 6) – ohne Grafik, damit es testbar ist. viewer.py zeigt es an.

    index = EpisodeIndex(Path("runs/v2_lauf2"))
    player = Player(index)
    player.update(dt)               # jedes Bild: Zeit weiter, am Ende nächster Versuch
    player.pose()                   # (x, y, z, heading, pitch, roll, steer) zur aktuellen Zeit

viewer.py liest nur Dateien (summary.csv, episodes/*.npz). Er rechnet nichts nach und lädt kein Netz,
damit er das Training nicht bremst. Neue Versuche kommen mit refresh() dazu, während das Training läuft.

watch.py nutzt denselben Player mit VersionIndex: Statt einer Aufzeichnung fährt eine gespeicherte Version
des Netzes jetzt (LiveEpisode, deterministisch nach R6). Was schon gefahren ist, bleibt gespeichert,
deshalb kann man auch dort zurückspulen.
"""
import csv
import io
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from game.car import DT
from rl.recording import episode_name, load_episode

FILTERS = ("alle", "auswertungen", "gueltige")
FILTER_TEXT = {"alle": "Alle Versuche", "auswertungen": "Nur Auswertungen", "gueltige": "Nur gültige Runden"}
SPEEDS = (0.5, 1, 2, 5, 10, 50)


@dataclass
class Attempt:
    """Eine Zeile aus summary.csv."""
    versuch: int
    art: str
    schritte: int
    ergebnis: str
    ende: str
    reward: float
    rundenzeit: float | None
    fortschritt: float
    dauer: float
    snapshot: str = ""          # Snapshot, der bei diesem Versuch galt (models/, aus summary.csv)

    @property
    def gueltig(self):
        return self.ergebnis == "Ziel"

    @property
    def auswertung(self):
        return self.art == "auswertung"

    def passes(self, mode):
        return mode == "alle" or (mode == "auswertungen" and self.auswertung) or (mode == "gueltige" and self.gueltig)

    @staticmethod
    def from_row(r):
        num = lambda v: float(v) if v not in ("", None) else None   # noqa: E731
        return Attempt(int(r["versuch"]), r["art"], int(r["schritte"] or 0), r["ergebnis"], r["ende"],
                       num(r["reward"]) or 0.0, num(r["rundenzeit"]), num(r["fortschritt_pct"]) or 0.0,
                       num(r["dauer_s"]) or 0.0, r.get("snapshot") or "")


class EpisodeIndex:
    """Alle Versuche eines Laufs. refresh() liest nur das Neue in summary.csv (das Training hängt Zeilen an)."""

    CACHE = 220             # Platz für 100 Geister und die Gruppe davor/danach (je Versuch ca. 0.5 MB)

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.summary = run_dir / "summary.csv"
        self.episode_dir = run_dir / "episodes"
        self.attempts = []
        self._offset = 0
        self._header = None
        self._cache = OrderedDict()

    def refresh(self):
        """Neue Zeilen aus summary.csv übernehmen. -> Anzahl neuer Versuche."""
        try:
            size = self.summary.stat().st_size
        except FileNotFoundError:
            return 0
        if size < self._offset:                     # Datei neu angelegt -> von vorne
            self.attempts, self._offset, self._header = [], 0, None
        if size == self._offset:
            return 0
        with open(self.summary, "rb") as f:
            f.seek(self._offset)
            chunk = f.read(size - self._offset)
        end = chunk.rfind(b"\n")
        if end < 0:                                 # Zeile wird gerade geschrieben -> beim nächsten Mal
            return 0
        self._offset += end + 1
        lines = chunk[:end + 1].decode("utf-8").splitlines()
        if self._header is None:
            self._header = next(csv.reader([lines[0]]))
            lines = lines[1:]
            if "versuch" not in self._header:      # Lauf von vor der Aufzeichnung (alte summary.csv)
                return 0
        if "versuch" not in self._header:
            return 0
        before = len(self.attempts)
        for row in csv.DictReader(io.StringIO("\n".join(lines)), fieldnames=self._header):
            try:
                self.attempts.append(Attempt.from_row(row))
            except (KeyError, ValueError, TypeError):
                continue
        return len(self.attempts) - before

    def filtered(self, mode):
        return [a for a in self.attempts if a.passes(mode)]

    def path(self, versuch):
        return self.episode_dir / episode_name(versuch)

    def load(self, versuch):
        """-> Episode oder None (Datei fehlt, z.B. aufgeräumt oder Lauf von vor der Aufzeichnung)."""
        if versuch in self._cache:
            self._cache.move_to_end(versuch)
            return self._cache[versuch]
        try:
            ep = Episode(*load_episode(self.path(versuch)))
        except (OSError, ValueError, KeyError):
            return None
        self._cache[versuch] = ep
        if len(self._cache) > self.CACHE:
            self._cache.popitem(last=False)
        return ep


class Episode:
    """Ein aufgezeichneter Versuch. Zeit t in Sekunden Simulationszeit (R6), 0 .. dauer."""

    done = True              # Aufzeichnung ist vollständig (LiveEpisode: erst am Ende der Fahrt)
    expected = None          # bekannte Länge in s (LiveEpisode, für die Zeitleiste)

    def __init__(self, meta, arrays):
        self.meta = meta
        self.pos, self.rot, self.obs = arrays["pos"], arrays["rot"], arrays["obs"]
        self.action, self.reward = arrays["aktion"], arrays["reward"]
        self.frames = len(self.pos) - 1               # Anzahl Physik-Schritte

    @property
    def duration(self):
        return self.frames * DT

    @property
    def speed(self):
        """m/s vorwärts je Schritt (Sensor "tempo", siehe CLAUDE.md)."""
        return self.obs[:self.frames + 1, 17].astype(np.float32) * 110.0

    def extend(self, t):
        """Aufzeichnung: nichts zu tun (LiveEpisode fährt hier weiter)."""

    def frame_at(self, t):
        return max(0.0, min(float(self.frames), t / DT))

    def pose(self, t):
        """(x, y, z, heading, pitch, roll, steer) zwischen zwei Schritten überblendet."""
        f = self.frame_at(t)
        i = min(int(f), max(self.frames - 1, 0))
        k = f - i
        j = min(i + 1, self.frames)
        p = self.pos[i] + (self.pos[j] - self.pos[i]) * k
        r0, r1 = self.rot[i], self.rot[j]
        heading = r0[0] + ((r1[0] - r0[0] + math.pi) % (2 * math.pi) - math.pi) * k
        pitch, roll = r0[1] + (r1[1] - r0[1]) * k, r0[2] + (r1[2] - r0[2]) * k
        steer = self.obs[i, 20] + (self.obs[j, 20] - self.obs[i, 20]) * k
        return float(p[0]), float(p[1]), float(p[2]), float(heading), float(pitch), float(roll), float(steer)

    def obs_at(self, t):
        return self.obs[int(round(self.frame_at(t)))]

    def reward_until(self, t):
        return float(np.sum(self.reward[:int(self.frame_at(t))], dtype=np.float64))


class LiveEpisode(Episode):
    """Ein Netz fährt jetzt, deterministisch (R6), mit derselben Simulation wie der Spieler (R3).
    extend(t) fährt bis zur Zeit t weiter (höchstens BUDGET Sekunden Rechenzeit pro Aufruf, damit das Bild
    nicht hängt). Alles Gefahrene bleibt gespeichert -> zurückspulen geht ohne neu zu rechnen."""

    BUDGET = 0.02

    def __init__(self, policy, env, expected=None):
        n = int(env.sim.rules.time_limit / DT) + 2              # länger kann keine Fahrt dauern (R11)
        super().__init__({"versionen": dict(env.versions)},
                         {"pos": np.zeros((n, 3), np.float32), "rot": np.zeros((n, 3), np.float32),
                          "obs": np.zeros((n, 24), np.float32), "aktion": np.zeros((n, 4), np.float32),
                          "reward": np.zeros(n, np.float32)})
        self.policy, self.env, self.expected = policy, env, expected
        self.frames = 0
        self.done = False
        self.info = None
        self._obs, _ = env.reset()
        self._store(0, self._obs)

    def _store(self, i, obs):
        c = self.env.sim.car
        self.pos[i] = (c.x, c.y, c.z)
        self.rot[i] = (c.heading, c.pitch, c.roll)
        self.obs[i] = obs

    def extend(self, t):
        t0 = time.perf_counter()
        while not self.done and self.duration < t and time.perf_counter() - t0 < self.BUDGET:
            act, _ = self.policy.predict(self._obs, deterministic=True)
            self._obs, reward, terminated, truncated, info = self.env.step(act)
            self.action[self.frames] = np.asarray(act, np.float32).reshape(-1)[:4]
            self.reward[self.frames] = reward
            self.frames += 1
            self._store(self.frames, self._obs)
            if terminated or truncated:
                self.done, self.info = True, info


# ---------- Versionen des Netzes (watch.py) ----------
@dataclass
class Version(Attempt):
    """Eine gespeicherte Version des Netzes (alle 25'000 Schritte bei der Auswertung) mit dem Ergebnis
    dieser Auswertung. versuch = Schritte (eindeutiger Schlüssel), auswertung_nr = Versuchsnummer dort."""
    path: Path = None
    auswertung_nr: int | None = None
    bekannt: bool = False             # Ergebnis der Auswertung bekannt (sonst nur die Datei)


def _versions_from_csv(run_dir):
    """{Schritte: Zeile} aller Auswertungen: evals.csv (ab Schritt 5) oder alte summary.csv."""
    out, dauer = {}, {}
    try:
        with open(run_dir / "summary.csv") as f:
            for r in csv.DictReader(f):
                if r.get("art") == "auswertung" and r.get("dauer_s"):
                    dauer[int(r["schritte"])] = float(r["dauer_s"])
                elif r.get("ereignis") == "auswertung":
                    out[int(r["schritte"])] = r
    except (OSError, ValueError, KeyError):
        pass
    try:
        with open(run_dir / "evals.csv") as f:
            out.update({int(r["schritte"]): r for r in csv.DictReader(f)})
    except (OSError, ValueError, KeyError):
        pass
    return out, dauer


class VersionIndex:
    """Gleiche Schnittstelle wie EpisodeIndex, aber jeder Eintrag ist eine Version des Netzes.
    load() lädt das Netz und gibt eine LiveEpisode zurück, die beim Abspielen fährt."""

    def __init__(self, run_dir, make_env, load_policy, extra=()):
        self.run_dir = run_dir
        self.make_env, self.load_policy = make_env, load_policy
        self.extra = list(extra)            # zusätzliche Modelle: [(Schritte, Pfad)], z.B. Fortsetzungspunkt
        self.attempts = []
        self._known = None
        self._policies = OrderedDict()

    def refresh(self):
        files = {}
        for folder in (self.run_dir / "snapshots", self.run_dir / "versionen"):
            for f in folder.glob("model_*.zip"):
                try:
                    files[int(f.stem.split("_")[1])] = f
                except (IndexError, ValueError):
                    pass
        for steps, path in self.extra:
            files.setdefault(steps, Path(path))
        key = tuple(sorted(files))
        evals, dauer = _versions_from_csv(self.run_dir)
        known = (key, len(evals))
        if known == self._known:
            return 0
        before = len(self.attempts)
        self._known = known
        self.attempts = [self._item(s, files[s], evals.get(s), dauer.get(s)) for s in key]
        return len(self.attempts) - before

    @staticmethod
    def _item(steps, path, row, dauer):
        if row is None:
            return Version(steps, "auswertung", steps, "?", "unbekannt", 0.0, None, 0.0, dauer or 0.0, path=path)
        valid = row.get("gueltig") == "ja"
        ende = row.get("ende") or ""
        result = "Ziel" if valid else "Abbruch" if ende == "Zeitlimit" else "Crash"
        t = float(row["rundenzeit"]) if row.get("rundenzeit") else None
        nr = row.get("versuch") or row.get("versuche")
        return Version(steps, "auswertung", steps, result, ende, 0.0, t, float(row.get("fortschritt_pct") or 0),
                       dauer or t or 0.0, path=path, auswertung_nr=int(nr) if nr and nr.isdigit() else None,
                       bekannt=True)

    def filtered(self, mode):
        return [a for a in self.attempts if mode != "gueltige" or a.gueltig]

    def load(self, steps):
        item = next((a for a in self.attempts if a.versuch == steps), None)
        if item is None:
            return None
        if steps not in self._policies:
            try:
                self._policies[steps] = self.load_policy(item.path)
            except (OSError, ValueError, KeyError, RuntimeError):
                return None
            if len(self._policies) > 4:
                self._policies.popitem(last=False)
        return LiveEpisode(self._policies[steps], self.make_env(), expected=item.dauer or None)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


GHOST_COUNTS = (1, 5, 10, 20, 30, 50, 100)  # 1 = keine Geister, nur der gewählte Versuch
GROUP_MODES = ("bereich", "letzte", "meilensteine")
GROUP_TEXT = {"bereich": "Bereich", "letzte": "letzte N", "meilensteine": "Meilensteine"}
MILESTONES = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
DEFAULT_RANGE = 200                        # so breit ist ein neuer Bereich (in Versuchen)


class Player:
    """Zustand der Wiedergabe – die einzige Quelle für 3D-Ansicht, Zeitleisten und Info-Zeile.

    Geister (Plan-Schritt 7): Statt eines Versuchs läuft eine Gruppe gleichzeitig, alle mit derselben Zeit t.
      bereich       n Versuche gleichmässig verteilt von Versuch "von" bis "bis" (z.B. 30 aus 100–300, bis 100 Stück)
      letzte        die n neuesten Versuche (passt zu LIVE)
      meilensteine  Versuche 10, 20, 50, ... 10'000        (alle, die es gibt)
    Einer davon ist im Fokus (current/episode): Ihm folgt die Kamera, Info-Zeile und Dashboard zeigen ihn.
    Mit Geistern ist das immer der, der am weitesten kommt (im Ziel: der schnellste); ein Klick auf einen
    anderen Geist gilt, bis sich die Gruppe ändert.
    Die Gruppe ist zu Ende, wenn der längste Versuch fertig ist.
    """

    END_HOLD = 1.2          # s (echte Zeit): so lange bleibt das Ende einer Fahrt stehen

    def __init__(self, index, mode="alle", milestones=MILESTONES):
        self.index = index
        self.mode = mode
        self.list = []                  # gefilterte Versuche
        self.pos = -1                   # Fokus: Stelle in self.list
        self.t = 0.0
        self.playing = True
        self.speed_i = SPEEDS.index(1)
        self.live = True                # folgt dem Training: am Ende immer der neueste Versuch
        self.episode = None             # Episode des Fokus
        self.hold = None                # Restzeit am Ende einer Fahrt
        self.changed = True             # Fokus gewechselt (Anzeige zurücksetzen)
        self.group_changed = True       # andere Geister (Anzeige neu aufbauen)
        self.n_ghosts, self.group_mode = 1, "letzte"
        self.range = None               # (von, bis) als Versuchsnummern, für "bereich"
        self.milestones = tuple(milestones)
        self.group = []                 # [(Attempt, Episode oder None)], Fokus ist dabei
        self.refresh()

    # ---------- Daten ----------
    def refresh(self):
        new = self.index.refresh()
        cur = self.current
        self.list = self.index.filtered(self.mode)
        if cur is not None:
            self.pos = self._find(cur.versuch, self.pos)
        elif self.list:
            self.select(len(self.list) - 1, keep_live=True)
        return new

    def _find(self, versuch, default):
        return next((i for i, a in enumerate(self.list) if a.versuch == versuch), default)

    @property
    def current(self):
        return self.list[self.pos] if 0 <= self.pos < len(self.list) else None

    @property
    def speed(self):
        return SPEEDS[self.speed_i]

    @property
    def ghosts_on(self):
        return self.n_ghosts > 1 or self.group_mode == "meilensteine"

    @property
    def duration(self):
        """Länge der ganzen Gruppe (der längste Versuch)."""
        eps = [ep for _, ep in self.group if ep is not None] or ([self.episode] if self.episode else [])
        return max((ep.duration for ep in eps), default=0.0)

    @property
    def at_end(self):
        if self.episode is None:
            return False
        if not self.ghosts_on:
            return self.episode.done and self.t >= self.episode.duration
        return self.t >= self.duration

    # ---------- Gruppe ----------
    def _group_positions(self, pos):
        last = len(self.list) - 1
        if not self.ghosts_on:
            return [pos]
        if self.group_mode == "meilensteine":
            want = set(self.milestones)
            return [i for i, a in enumerate(self.list) if a.versuch in want]
        if self.group_mode == "letzte":
            return list(range(max(0, last - self.n_ghosts + 1), last + 1))
        von, bis = self.range or (self.list[pos].versuch, self.list[pos].versuch + DEFAULT_RANGE)
        inside = [i for i, a in enumerate(self.list) if von <= a.versuch <= bis]
        if len(inside) <= self.n_ghosts:
            return inside
        picks = np.linspace(0, len(inside) - 1, self.n_ghosts).round().astype(int)   # gleichmässig verteilt
        return [inside[k] for k in sorted(set(picks.tolist()))]

    @staticmethod
    def best_key(a):
        """Wer kommt am weitesten? Im Ziel vor allen anderen (schnellste Zeit zuerst), sonst nach Fortschritt."""
        return (0, a.rundenzeit) if a.gueltig else (1, -a.fortschritt)

    def _build_group(self, pos, focus=None):
        """Gruppe aufbauen. Fokus: mit Geistern der, der am weitesten kommt, sonst focus bzw. der neueste."""
        positions = self._group_positions(pos) or [pos]
        if self.ghosts_on and len(positions) > 1:
            focus = min(positions, key=lambda i: self.best_key(self.list[i]))
        elif focus not in positions:
            focus = positions[-1]
        old = {a.versuch: ep for a, ep in self.group}
        self.group = []
        for i in positions:
            a = self.list[i]
            ep = old[a.versuch] if a.versuch in old else self.index.load(a.versuch)
            self.group.append((a, ep))
        self.group_changed = True
        self._set_focus(focus)

    def _set_focus(self, pos):
        if pos != self.pos or self.episode is None:
            self.pos = pos
            a = self.list[pos]
            self.episode = next((ep for g, ep in self.group if g.versuch == a.versuch), None) or self.index.load(a.versuch)
            self.changed = True

    def focus_on(self, versuch):
        """Klick auf einen Geist: Kamera, Info-Zeile und Dashboard folgen ihm. Die Gruppe bleibt."""
        i = self._find(versuch, None)
        if i is not None and any(a.versuch == versuch for a, _ in self.group):
            self._set_focus(i)

    def set_range(self, von, bis):
        """Geister aus dem Bereich Versuch von..bis (Reihenfolge egal), gleichmässig verteilt."""
        von, bis = sorted((int(von), int(bis)))
        self.range = (von, bis)
        self.group_mode = "bereich"
        self.live = False
        if self.n_ghosts == 1:
            self.n_ghosts = 10
        if self.list:
            self._build_group(max(0, self._find_at_least(von)))
            self.t, self.hold = 0.0, None

    def _find_at_least(self, versuch):
        return next((i for i, a in enumerate(self.list) if a.versuch >= versuch), len(self.list) - 1)

    def set_ghosts(self, n=None, mode=None):
        """Anzahl oder Auswahl ändern; die Gruppe beginnt von vorne."""
        if n is not None:
            self.n_ghosts = n
        if mode is not None:
            self.group_mode = mode
            if mode == "letzte":
                self.live = True
            elif mode == "bereich":
                self.live = False
        if not self.list:
            return
        if self.group_mode == "bereich" and self.range is None and self.current is not None:
            v = self.current.versuch           # neuer Bereich: ab dem gezeigten Versuch
            self.range = (v, v + DEFAULT_RANGE)
        pos = len(self.list) - 1 if self.group_mode == "letzte" else max(self.pos, 0)
        self._build_group(pos, focus=self.pos)
        self.t, self.hold = 0.0, None

    # ---------- Bedienung ----------
    def select(self, pos, t=0.0, keep_live=False):
        """Einen Versuch wählen (Zeitleiste, B/N). Mit Geistern: der Bereich (gleiche Breite) beginnt dort."""
        if not self.list:
            return
        pos = max(0, min(len(self.list) - 1, pos))
        if not keep_live:
            self.live = False
        in_group = any(a.versuch == self.list[pos].versuch for a, _ in self.group)
        if self.ghosts_on and self.group_mode == "meilensteine" and in_group:
            self._set_focus(pos)                 # Meilenstein angeklickt: nur den Fokus wechseln
        elif self.ghosts_on and not keep_live:
            width = self.range[1] - self.range[0] if self.range else DEFAULT_RANGE
            v = self.list[pos].versuch
            self.group_mode, self.range = "bereich", (v, v + width)
            self._build_group(pos, focus=pos)
        else:
            self._build_group(pos)
        self.t, self.hold = t, None

    def select_versuch(self, versuch):
        self.select(self._find_at_least(versuch))

    def step_attempt(self, delta):
        """Ein Versuch zurück/weiter; mit Geistern im Bereich: der ganze Bereich (gleiche Breite)."""
        if self.ghosts_on and self.group_mode == "bereich" and self.range:
            von, bis = self.range
            width = bis - von + 1
            self.set_range(max(1, von + delta * width), max(1, von + delta * width) + width - 1)
        else:
            self.select(self.pos + delta)

    def go_live(self):
        self.live = True
        self.playing = True
        if self.ghosts_on and self.group_mode != "meilensteine":
            self.group_mode = "letzte"
        self.select(len(self.list) - 1, keep_live=True)

    def restart(self):
        """Fahrt von vorne (LiveEpisode: das Gefahrene bleibt, nichts wird neu gerechnet)."""
        self.seek(0.0)
        self.playing = True

    def toggle_play(self):
        if self.at_end and not self.playing:
            self.t = 0.0
        self.playing = not self.playing
        self.hold = None

    def seek(self, t):
        if self.episode:
            self.episode.extend(t)                   # LiveEpisode: bis dorthin vorausfahren
            self.t = max(0.0, min(self.duration, t))
            self.hold = None

    def faster(self, delta):
        self.speed_i = max(0, min(len(SPEEDS) - 1, self.speed_i + delta))

    def set_filter(self, mode):
        cur = self.current
        self.mode = mode
        self.list = self.index.filtered(mode)
        self.group = []
        if not self.list:
            self.pos, self.episode = -1, None
            self.group_changed = True
            return
        if self.ghosts_on and self.group_mode == "bereich" and self.range:
            self.pos = -1
            self.set_range(*self.range)          # gleicher Bereich, nur noch passende Versuche
        elif self.live or cur is None:
            self.select(len(self.list) - 1, keep_live=self.live)
        else:            # nächster passender Versuch ab dem aktuellen
            self.pos = -1
            self.select_versuch(cur.versuch)

    # ---------- jedes Bild ----------
    def update(self, dt):
        if not self.playing or self.current is None:
            return
        if self.episode is None and not self.ghosts_on:   # Datei fehlt -> kurz zeigen, dann weiter
            self.hold = self.END_HOLD if self.hold is None else self.hold - dt
            if self.hold <= 0:
                self._next()
            return
        if not self.at_end:
            target = self.t + dt * self.speed
            if self.episode is not None:
                self.episode.extend(target)
            self.t = min(self.duration, target)
            return
        self.hold = self._end_hold() if self.hold is None else self.hold - dt
        if self.hold <= 0:
            self._next()

    def _end_hold(self):
        return self.END_HOLD if self.speed <= 2 else 0.4

    def _next(self):
        """Ende der Fahrt bzw. der Gruppe: live -> die neuesten; ein Versuch -> der nächste;
        Bereich oder Meilensteine -> anhalten (mit N geht es zum nächsten Bereich)."""
        self.hold = None
        last = len(self.list) - 1
        if self.live:
            newest_in_group = any(a.versuch == self.list[last].versuch for a, _ in self.group)
            if not newest_in_group or (not self.ghosts_on and self.pos < last):
                self.select(last, keep_live=True)
            else:
                self.t = self.duration                 # warten auf den nächsten Versuch
                self.hold = 0.25
        elif self.ghosts_on:
            self.playing = False
        elif self.pos < last:
            self.select(self.pos + 1)
        else:
            self.playing = False                       # letzter Versuch, nicht live -> anhalten
