"""Das Training selbst: SAC aus Stable-Baselines3, regelmässige Auswertung, Erfolgsstufen, Pausieren.

train.py liest nur die Befehlszeile und ruft Trainer auf. Ablauf:
  - mehrere Umgebungen sammeln Erfahrung (Anzahl je nach Leistungsprofil)
  - alle N Schritte fährt das aktuelle Modell eine deterministische Runde (R6) -> offizielle Zeiten
  - Fortsetzungspunkt alle 30 min und bei jeder Pause (Ctrl+C, Akku, Festplatte)
"""
import os
import signal
import sys
import time

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from game.ghost import best_time, save_ghost
from game.sim import format_time
from rl.env import RaceEnv
from rl.resources import SpeedLimit, apply_priority, make_profile, power_state

STAGE_TEXT = {1: "Ankommen", 2: "Zuverlässig", 3: "Schnell (Plateau)", 4: "Schneller als ich"}
SUBPROCESS_FROM = 3       # ab so vielen Umgebungen bekommt jede ihren eigenen Prozess


# ---------- Umgebungen ----------
def _env_factory(reward_cfg, in_subprocess):
    def init():
        if in_subprocess:   # Ctrl+C / Fenster schliessen trifft nur das Hauptprogramm, das dann sauber pausiert
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        return RaceEnv(config=reward_cfg)
    return init


def make_vec_env(reward_cfg, n):
    sub = n >= SUBPROCESS_FROM
    cls = SubprocVecEnv if sub else DummyVecEnv
    return VecMonitor(cls([_env_factory(reward_cfg, sub) for _ in range(n)]))


def update_schedule(n_envs, ratio):
    """(train_freq, gradient_steps), sodass auf jeden gesammelten Schritt `ratio` Lern-Updates kommen,
    egal wie viele Umgebungen laufen. Beispiel 0.25: 2 Umgebungen -> alle 2 Runden 1 Update."""
    for freq in range(1, 9):
        g = n_envs * freq * ratio
        if g >= 1 and abs(g - round(g)) < 1e-9:
            return freq, round(g)
    freq = max(1, round(1 / (n_envs * ratio)))
    return freq, max(1, round(n_envs * freq * ratio))


def regroup_buffer(old, n_envs):
    """Replay-Buffer auf eine andere Anzahl Umgebungen umverteilen, ohne gesammelte Schritte wegzuwerfen
    (SB3 speichert sie als [Zeit, Umgebung]). Nötig, wenn das Profil wechselt."""
    total = old.buffer_size * old.n_envs
    new = ReplayBuffer(total, old.observation_space, old.action_space, device=old.device, n_envs=n_envs,
                       handle_timeout_termination=old.handle_timeout_termination)
    order = np.r_[old.pos:old.buffer_size, 0:old.pos] if old.full else np.arange(old.pos)
    m = (min(len(order) * old.n_envs, new.buffer_size * n_envs) // n_envs) * n_envs
    rows = m // n_envs
    for field in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        src = getattr(old, field)[order]
        flat = src.reshape((-1,) + src.shape[2:])[len(order) * old.n_envs - m:]
        getattr(new, field)[:rows] = flat.reshape((rows, n_envs) + src.shape[2:])
    new.full = rows == new.buffer_size
    new.pos = 0 if new.full else rows
    return new


# ---------- Auswertung (R6: deterministisch, ohne Zufall) ----------
def evaluate(model, reward_cfg):
    env = RaceEnv(config=reward_cfg, record=True)
    obs, info = env.reset()
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(act)
        done = terminated or truncated
    return info, env.sim


class _Hook(BaseCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def _on_step(self):
        return self.trainer.on_step(self)


class Trainer:
    def __init__(self, run, cfg, reward_cfg, resources_cfg, profiles_cfg, profile_name, max_sps=None,
                 stop_steps=None, stop_minutes=None):
        self.run, self.cfg, self.reward_cfg = run, cfg, reward_cfg
        self.res, self.profiles_cfg = resources_cfg, profiles_cfg
        self.chosen_profile = profile_name             # was der Benutzer gewählt hat
        self.power = power_state(run.dir.parent)
        self.profile_note = None
        if self.power.on_battery and profile_name != "sparsam":   # im Akkubetrieb gleich sparsam starten
            self.profile_note = f"{profile_name} -> sparsam: Akkubetrieb ({self.power.battery_pct:.0f} %)"
            profile_name = "sparsam"
        self.profile = make_profile(profile_name, profiles_cfg)
        self.limit = SpeedLimit(max_sps)
        self.stop_steps, self.stop_minutes = stop_steps, stop_minutes
        self.stop_reason = None
        self.model = self.env = None
        self.recent = []                                # letzte 100 Trainings-Fahrten: (gültig, Zeit)

    # ---------- Aufbau ----------
    def new(self, seed):
        s = self.cfg["sac"]
        apply_priority(self.profile)
        self.env = make_vec_env(self.reward_cfg, self.profile.envs)
        self.model = SAC("MlpPolicy", self.env, learning_rate=s["lernrate"], buffer_size=s["puffer"],
                         batch_size=s["batch"], gamma=s["gamma"], tau=s["tau"], learning_starts=s["lernstart"],
                         policy_kwargs={"net_arch": list(s["netz"])}, tensorboard_log=str(self.run.tb_dir),
                         seed=seed, device="cpu", verbose=0)
        self._set_schedule()
        self.state = {"versuche": 0, "laufzeit_s": 0.0, "auswertungen": 0, "serie": 0, "bestzeit": None,
                      "bestzeit_schritte": None, "bestzeit_versuch": None, "bestzeit_verlauf": [], "stufen": {},
                      "letzte_auswertung": 0, "letzter_snapshot": 0, "sitzungen": []}

    def resume(self, cp_dir):
        apply_priority(self.profile)
        self.env = make_vec_env(self.reward_cfg, self.profile.envs)
        self.model = SAC.load(cp_dir / "model.zip", env=self.env, device="cpu",
                              tensorboard_log=str(self.run.tb_dir))
        self.model.load_replay_buffer(cp_dir / "replay_buffer.pkl")
        if self.model.replay_buffer.n_envs != self.profile.envs:
            self.model.replay_buffer = regroup_buffer(self.model.replay_buffer, self.profile.envs)
        self._set_schedule()
        self.state = self.run.load_state(cp_dir)
        self.run.restore_rng(cp_dir)
        self._merge_records()

    def _records(self):
        st = self.state
        return {k: st.get(k) for k in ("bestzeit", "bestzeit_schritte", "bestzeit_versuch", "bestzeit_geist",
                                       "bestzeit_verlauf", "stufen")}

    def _merge_records(self):
        """Bestzeit und Stufen aus rekorde.json übernehmen, falls sie nach dem Fortsetzungspunkt entstanden sind
        (z.B. Programm ohne Pause beendet). Sonst würde best_model.zip später von einer schlechteren Zeit überschrieben."""
        rec, st = self.run.load_records(), self.state
        if rec.get("bestzeit") is not None and (st["bestzeit"] is None or rec["bestzeit"] < st["bestzeit"]):
            st.update({k: rec.get(k) for k in ("bestzeit", "bestzeit_schritte", "bestzeit_versuch", "bestzeit_geist",
                                               "bestzeit_verlauf")})
        for k, v in rec.get("stufen", {}).items():
            st["stufen"].setdefault(k, v)

    def _set_schedule(self):
        freq, grads = update_schedule(self.profile.envs, self.cfg["sac"]["updates_pro_schritt"])
        self.model.train_freq = TrainFreq(freq, TrainFrequencyUnit.STEP)
        self.model.gradient_steps = grads

    def switch_profile(self, name, why):
        """Während des Trainings auf ein anderes Profil wechseln (z.B. Akkubetrieb -> sparsam)."""
        old = self.profile
        self.profile = make_profile(name, self.profiles_cfg)
        nice = apply_priority(self.profile)
        if self.profile.envs != old.envs:
            self.env.close()
            self.env = make_vec_env(self.reward_cfg, self.profile.envs)
            self.model.replay_buffer = regroup_buffer(self.model.replay_buffer, self.profile.envs)
            self.model.n_envs = self.profile.envs
            self.model.set_env(self.env)        # angefangene Fahrten werden verworfen, Neustart am Start
        self._set_schedule()
        self.say(f"Profil {old.name} -> {self.profile} ({why}), Priorität jetzt nice {nice}")
        self._log("profilwechsel", bemerkung=f"{old.name} -> {self.profile.name}: {why}")

    # ---------- Hauptschleife ----------
    def train(self, session):
        self.state["sitzungen"].append(session)
        self.session_start = time.time()
        self.session_steps0 = self.steps
        self.laufzeit_base = self.state["laufzeit_s"]
        self.next_save = time.time() + self.res["sicherung_minuten"] * 60
        self.next_check = 0.0
        self.next_status = 0.0
        self.next_tb = self.steps + self.cfg["ablauf"]["tensorboard_alle_schritte"]
        self._install_signals()
        self.say(f"Profil: {self.profile}, {self._schedule_text()}")
        if self.profile_note:
            self.say(f"Profil {self.profile_note}")
            self._log("profilwechsel", bemerkung=self.profile_note)
        self.check_power(first=True)
        while self.stop_reason is None:
            self.limit.reset(self.steps)
            self.model.learn(total_timesteps=10**12, callback=_Hook(self), reset_num_timesteps=False,
                             log_interval=None, tb_log_name="sac", progress_bar=False)
            if self.stop_reason and self.stop_reason.startswith("profil:"):
                name, why = self.stop_reason[7:].split("|", 1)
                self.stop_reason = None
                self.switch_profile(name, why)
        self.save("pause", self.stop_reason)
        try:
            self.env.close()
        except (OSError, EOFError):
            pass
        return self.stop_reason

    def on_step(self, hook):
        loc = hook.locals
        for done, info in zip(loc["dones"], loc["infos"]):
            if done:
                self._episode_done(info)
        steps = self.steps
        if steps - self.state["letzte_auswertung"] >= self.cfg["ablauf"]["auswertung_alle_schritte"]:
            self.state["letzte_auswertung"] = steps
            self.run_evaluation()
        if steps - self.state["letzter_snapshot"] >= self.cfg["ablauf"]["snapshot_alle_schritte"]:
            self.state["letzter_snapshot"] = steps
            self.run.snapshot_dir.mkdir(exist_ok=True)
            self.model.save(self.run.snapshot_dir / f"model_{steps:010d}.zip")
        if steps >= self.next_tb:
            self.next_tb = steps + self.cfg["ablauf"]["tensorboard_alle_schritte"]
            self._tensorboard()
        now = time.time()
        if now >= self.next_save:
            self.next_save = now + self.res["sicherung_minuten"] * 60
            self.save("sicherung")
        if now >= self.next_check:
            self.next_check = now + self.res["pruefen_sekunden"]
            self.check_power()
        if now >= self.next_status:
            self.next_status = now + 1.0
            self.status_line()
        if self.stop_steps and steps >= self.stop_steps:
            self.stop_reason = f"{self.stop_steps} Schritte erreicht"
        if self.stop_minutes and now - self.session_start >= self.stop_minutes * 60:
            self.stop_reason = f"{self.stop_minutes:g} Minuten erreicht"
        self.limit.wait(steps)
        return self.stop_reason is None

    @property
    def steps(self):
        return self.model.num_timesteps

    @property
    def laufzeit(self):
        return self.laufzeit_base + time.time() - self.session_start

    # ---------- Fahrten ----------
    def _episode_done(self, info):
        self.state["versuche"] += 1
        valid = info["gueltige_runde"]
        self.recent = (self.recent + [(valid, info["rundenzeit"], info["fortschritt_prozent"])])[-100:]
        logger = self.model.logger
        logger.record_mean("fahrt/reward", info["episode"]["r"])
        logger.record_mean("fahrt/fortschritt_prozent", info["fortschritt_prozent"])
        logger.record_mean("fahrt/wandberuehrungen", info["wandberuehrungen"])
        if valid:
            logger.record_mean("fahrt/rundenzeit_gueltig", info["rundenzeit"])
            if "1" not in self.state["stufen"]:
                self.reach_stage(1, f"erste gültige Runde im Training: {format_time(info['rundenzeit'])} "
                                    f"(mit Zufall, keine offizielle Zeit)")

    def run_evaluation(self):
        info, sim = evaluate(self.model, self.reward_cfg)
        st = self.state
        st["auswertungen"] += 1
        valid, t = info["gueltige_runde"], info["rundenzeit"]
        st["serie"] = st["serie"] + 1 if valid else 0
        note = ""
        if valid and (st["bestzeit"] is None or t < st["bestzeit"]):
            old = st["bestzeit"]
            st.update(bestzeit=t, bestzeit_schritte=self.steps, bestzeit_versuch=st["versuche"])
            st["bestzeit_verlauf"].append([self.steps, t])
            self.model.save(self.run.best_model)
            ghost = save_ghost(sim, "ai", attempt=st["versuche"], driver=f"KI {self.run.name}",
                               folder=self.run.ghost_dir)
            st["bestzeit_geist"] = str(ghost)
            self.run.save_records(self._records())
            note = "neue offizielle Bestzeit" + (f" (vorher {format_time(old)})" if old else "")
            self.say(f"Neue offizielle Bestzeit: {format_time(t)}  (Schritt {self.steps:,}".replace(",", "'")
                     + f", Versuch {st['versuche']})")
        if valid:
            self.model.logger.record("auswertung/rundenzeit", t)
        self.model.logger.record("auswertung/gueltig", float(valid))
        self.model.logger.record("auswertung/fortschritt_prozent", info["fortschritt_prozent"])
        if st["bestzeit"] is not None:
            self.model.logger.record("auswertung/bestzeit", st["bestzeit"])
        self._log("auswertung", gueltig=valid, rundenzeit=t, fortschritt_pct=info["fortschritt_prozent"],
                  ende=info["ende"], wandberuehrungen=info["wandberuehrungen"], bemerkung=note)
        self.run.save_live(self.model, {"schritte": self.steps, "versuche": st["versuche"], "gueltig": valid,
                                        "rundenzeit": t, "fortschritt_pct": info["fortschritt_prozent"],
                                        "ende": info["ende"], "bestzeit": st["bestzeit"], "stufe": self.stage,
                                        "zeit": time.strftime("%Y-%m-%d %H:%M:%S")})
        self._check_stages()

    # ---------- Erfolgsstufen (CLAUDE.md) ----------
    @property
    def stage(self):
        return max((int(k) for k in self.state["stufen"]), default=0)

    def reach_stage(self, n, text):
        if str(n) in self.state["stufen"]:
            return
        self.state["stufen"][str(n)] = {"schritte": self.steps, "versuch": self.state["versuche"],
                                        "datum": time.strftime("%Y-%m-%d %H:%M:%S"), "text": text}
        self.run.save_records(self._records())
        self.say(f"*** Stufe {n} erreicht: {STAGE_TEXT[n]} – {text}")
        self._log("stufe", bemerkung=f"Stufe {n} ({STAGE_TEXT[n]}): {text}")

    def _check_stages(self):
        st, erfolg = self.state, self.cfg["erfolg"]
        if st["bestzeit"] is not None:
            self.reach_stage(1, f"gültige Runde in der Auswertung: {format_time(st['bestzeit'])}")
        if st["serie"] >= erfolg["zuverlaessig_serie"]:
            self.reach_stage(2, f"{st['serie']} Auswertungen in Folge im Ziel")
        if "2" in st["stufen"] and st["bestzeit"] is not None:
            window_start = self.steps - erfolg["plateau_schritte"]
            before = [t for s, t in st["bestzeit_verlauf"] if s <= window_start]
            if before and before[-1] - st["bestzeit"] < erfolg["plateau_sekunden"]:
                self.reach_stage(3, f"Bestzeit {format_time(st['bestzeit'])}, in {erfolg['plateau_schritte']:,} "
                                    f"Schritten um weniger als {erfolg['plateau_sekunden']} s besser".replace(",", "'"))
        mine = best_time(self.cfg["strecke_id"])   # nur gleiche Physik/Regeln/Strecke (R8)
        if mine and st["bestzeit"] is not None and st["bestzeit"] < mine["time"]:
            self.reach_stage(4, f"KI {format_time(st['bestzeit'])} gegen deine {format_time(mine['time'])}")

    # ---------- Akku, Festplatte ----------
    def check_power(self, first=False):
        p = power_state(self.run.dir)
        self.power = p
        if p.free_gb < self.res["speicher_minimum_gb"]:
            self.stop_reason = f"nur noch {p.free_gb:.1f} GB frei auf der Festplatte"
        elif p.on_battery and p.battery_pct < self.res["akku_minimum"]:
            self.stop_reason = f"Akku bei {p.battery_pct:.0f} %"
        elif p.on_battery and self.profile.name != "sparsam":
            self._request_profile("sparsam", f"Akkubetrieb ({p.battery_pct:.0f} %)", first)
        elif not p.on_battery and self.profile.name != self.chosen_profile:
            self._request_profile(self.chosen_profile, "wieder am Netzteil", first)

    def _request_profile(self, name, why, first):
        if self.stop_reason:           # Pause hat Vorrang
            return
        if first:                      # vor dem ersten Schritt: direkt wechseln
            self.switch_profile(name, why)
        else:                          # während learn(): erst anhalten, dann wechseln
            self.stop_reason = f"profil:{name}|{why}"

    # ---------- Speichern, Anzeige ----------
    def save(self, event, why=None):
        self.state["laufzeit_s"] = self.laufzeit
        self.state["schritte"] = self.steps
        self.state["profil"] = self.profile.name
        self.run.save_checkpoint(self.model, self.state)
        self._log(event, bemerkung=why or "")
        if event == "pause":
            self.status_line()
            print()
            self.say(f"Pausiert: {why}. Fortsetzungspunkt gespeichert.")
            self.say(f"Weiter mit:  .venv/bin/python train.py --name {self.run.name} --resume")

    def _log(self, event, **kw):
        st = self.state
        self.run.log(laufzeit_s=self.laufzeit if hasattr(self, "laufzeit_base") else st["laufzeit_s"],
                     schritte=self.steps, versuche=st["versuche"], ereignis=event, bestzeit=st["bestzeit"],
                     stufe=self.stage, serie=st["serie"], profil=self.profile.name, **kw)

    def _tensorboard(self):
        lg = self.model.logger
        if self.recent:
            lg.record("training/anteil_gueltig_letzte_100", sum(v for v, _, _ in self.recent) / len(self.recent))
        lg.record("training/stufe", self.stage)
        lg.record("training/versuche", self.state["versuche"])
        lg.record("training/schritte_pro_sekunde", self._sps())
        self.model.dump_logs()       # schreibt auch die SAC-eigenen Werte (Verluste, Entropie, Reward)

    def _sps(self):
        dt = time.time() - self.session_start
        return (self.steps - self.session_steps0) / dt if dt > 0 else 0.0

    def _schedule_text(self):
        f, g = self.model.train_freq.frequency, self.model.gradient_steps
        return f"{g} Lern-Update(s) alle {f * self.profile.envs} Schritte"

    def status_line(self):
        st = self.state
        best = format_time(st["bestzeit"]) if st["bestzeit"] is not None else "–"
        stage = f"Stufe {self.stage}" + (f" {STAGE_TEXT[self.stage]}" if self.stage else "")
        power = getattr(self, "power", None)
        text = (f"Versuch {st['versuche']:,} | Bestzeit {best} | {stage} | Schritte {self.steps:,} | "
                f"{self._sps():,.0f}/s | Laufzeit {_hms(self.laufzeit)} | {self.profile.name}"
                + (f" ({power.text})" if power else "")).replace(",", "'")
        _write("\r\033[K" + text)

    def say(self, text):
        _write("\r\033[K" + text + "\n")

    def _install_signals(self):

        reasons = {signal.SIGINT: "Ctrl+C", signal.SIGTERM: "beendet", signal.SIGHUP: "Terminal-Fenster geschlossen"}

        def handler(signum, frame):
            paused = self.stop_reason and not self.stop_reason.startswith("profil:")
            if paused and signum == signal.SIGINT:
                raise KeyboardInterrupt      # zweites Ctrl+C: sofort abbrechen (ohne Speichern)
            if not paused:
                self.stop_reason = reasons[signum]
                self.say("Pause angefordert – speichere den Fortsetzungspunkt ... "
                         "(nochmal Ctrl+C = ohne Speichern abbrechen)")
        for sig in reasons:
            signal.signal(sig, handler)


def _write(text):
    """Auf das Terminal schreiben. Ist das Fenster schon zu, ins Leere (das Speichern darf daran nicht scheitern)."""
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except OSError:
        sys.stdout = open(os.devnull, "w")


def _hms(seconds):
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
