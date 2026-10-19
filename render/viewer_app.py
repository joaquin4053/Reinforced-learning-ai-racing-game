"""Gemeinsame Zuschauer-Oberfläche für viewer.py (Aufzeichnungen) und watch.py (Netz fährt jetzt). Plan-Schritt 6.

Eine Bildschleife, ein Zustand (rl.playback.Player): 3D-Ansicht, beide Zeitleisten, Info-Zeile und Panel
lesen jedes Bild nur von dort und bleiben so synchron. Was sich zwischen den beiden Programmen unterscheidet
(Texte, Filter, Panel-Inhalt, langsames Laden), liefert eine Quelle (Source).

Ursina ruft update(), input() und text_input() im Hauptprogramm auf; die Programme leiten sie an die App weiter:
    app = ViewerApp(source, player)
    update, input, text_input = app.update, app.input, app.text_input
    app.run()
"""
import math
from time import perf_counter      # Wanduhr nur fürs Bremsen beim Ziehen (Ursinas "time" ist das Modul time)

from game.car import DT
from game.sensors import RAY_RANGE
from game.sim import format_time
from rl.playback import SPEEDS, read_json

FILTER_TEXT = {"alle": "Alle Versuche", "auswertungen": "Nur Auswertungen", "gueltige": "Nur gültige Runden"}
CAM_MODES = (("chase", "Verfolger"), ("top", "Von oben"), ("free", "Frei"))
TOP_H, BOTTOM_H, PANEL_W = 0.042, 0.17, 0.5


def nr(x):
    return f"{x:,}".replace(",", "'")


def result_text(a):
    if a.gueltig:
        return f"Ziel in {format_time(a.rundenzeit)}"
    if a.ergebnis == "Crash":
        return f"Crash: {a.ende} bei {a.fortschritt:.0f} %"
    if a.ergebnis == "?":
        return "Ergebnis unbekannt"
    return f"Abbruch ({a.ende}) bei {a.fortschritt:.0f} %"


def result_rgb(a):
    return (90, 255, 120) if a.gueltig else (255, 120, 100) if a.ergebnis == "Crash" else (255, 255, 255)


class Source:
    """Was die beiden Programme unterscheidet. Unterklassen überschreiben, was sie anders brauchen."""
    item = "Versuch"                    # ein Eintrag der oberen Zeitleiste
    items = "Versuche"
    filters = ("alle", "auswertungen", "gueltige")
    slow_load = False                   # Laden dauert (Netz) -> beim Ziehen erst beim Loslassen laden
    title = "Zuschauen"
    live_text = "neuester Versuch, folgt dem Training"
    filter_text = FILTER_TEXT

    def __init__(self, run_dir, cfg, track, ghosts=(), mine=None):
        self.run_dir, self.cfg, self.track = run_dir, cfg, track
        self.ghosts, self.mine = list(ghosts), mine      # Geist-Dateien, deine Bestzeit (für den Vergleich)

    def info(self, a):
        kind = "Auswertung (ohne Zufall)" if a.auswertung else "Training (mit Zufall)"
        return f"Versuch {nr(a.versuch)}  ·  {kind}  ·  Reward {a.reward:,.1f}  ·  {result_text(a)}".replace(",", "'")

    def hover(self, a):
        return f"Versuch {nr(a.versuch)} · {'Auswertung · ' if a.auswertung else ''}{result_text(a)}"

    def missing(self, a):
        return f"Versuch {a.versuch}: keine Aufzeichnung\n(Datei fehlt, z.B. aufgeräumt)"

    def end_text(self, a, ep):
        """Grosse Meldung am Ende einer Fahrt."""
        if a.gueltig:
            text = f"ZIEL  {format_time(a.rundenzeit)}"
            if self.mine:
                text += f"\nDu: {format_time(self.mine['time'])}  ({a.rundenzeit - self.mine['time']:+.2f} s)"
            return text
        return f"{a.ende.upper()}\n{a.fortschritt:.0f} % geschafft"

    def panel(self, index, player):
        recent = index.attempts[-100:]
        n_eval = sum(a.auswertung for a in index.attempts)
        lines = [f"Schritte (Training): {nr(index.attempts[-1].schritte)}" if index.attempts else "Schritte: –",
                 f"Versuche: {nr(len(index.attempts))}  (Auswertungen {nr(n_eval)})"]
        lines += self.records()
        lines.append(f"Im Ziel (letzte 100): {sum(a.gueltig for a in recent)} %" if len(recent) == 100 else
                     f"Im Ziel: {sum(a.gueltig for a in recent)} von {len(recent)}")
        a = player.current
        if a is not None:
            lines += ["", f"Dieser Versuch: {nr(a.versuch)}", f"   gefahren bei Schritt {nr(a.schritte)}"]
        return lines + self.footer(player)

    def records(self):
        rec = read_json(self.run_dir / "rekorde.json")
        best = rec.get("bestzeit")
        stage = max((int(k) for k in rec.get("stufen", {})), default=0)
        return [f"Offizielle Bestzeit: {format_time(best) if best else '–'}", f"Erfolgsstufe: {stage}",
                f"Deine Bestzeit: {format_time(self.mine['time'])}" if self.mine else "Deine Bestzeit: –"]

    def footer(self, player):
        v = self.cfg.get("versionen", {})
        lines = ["", f"Strecke: {self.track.name}", f"Physik v{v.get('physik')}, Regeln v{v.get('regeln')},",
                 f"Sensoren v{v.get('sensoren')}, Reward v{v.get('reward')}"]
        ep = player.episode
        if ep is not None:
            other = [k for k, val in ep.meta.get("versionen", {}).items() if v.get(k) != val]
            if other:
                lines += ["", f"Achtung: andere Version ({', '.join(other)})", "Zeiten nicht vergleichbar (R8)"]
        return lines


class ViewerApp:
    def __init__(self, source, player, test_keys="", photo=None):
        import render  # noqa: F401  GL-Setup für macOS, muss vor Ursina() geladen sein
        from ursina import Entity, Ursina, camera, window
        from ursina.shaders import unlit_shader

        from game.ghost import Ghost
        from render.cameras import CameraRig, FreeCamera
        from render.effects import DriftEffects, SpeedStreaks
        from render.hud import TextQuad
        from render.minimap import Minimap
        from render.viewer_ui import BAR, Button, Help, InfoBar, Panel, Strip
        from render.world import GhostCar, World

        self.src, self.player, self.index = source, player, player.index
        self.app = Ursina(title=f"KI-Fahrschule – {source.title} – {source.run_dir.name}", size=(1440, 860),
                          development_mode=False)
        render.fix_after_start()
        window.fps_counter.enabled = False
        window.exit_button.visible = False
        track = source.track
        self.world = World(track)
        self.rig = CameraRig(track, mode="chase")
        self.free_cam = FreeCamera()
        self.effects = DriftEffects()
        self.streaks = SpeedStreaks()
        self.ghosts = [GhostCar(Ghost(p), i) for i, p in enumerate(source.ghosts)]
        self.minimap = Minimap(track, [g.color for g in self.ghosts])
        self.car = ReplayCar()
        filt = " / ".join(source.filter_text[f].replace("Nur ", "nur ").replace("Alle ", "alle ")
                          for f in source.filters)
        keys = " / ".join(str(i + 1) for i in range(len(source.filters)))
        self.help_rows = [
            ("Leertaste", "Play / Pause"),
            ("← / →", "2 s zurück / vor (in der Fahrt)"),
            ("B / N", f"{'vorherige / nächste' if source.item == 'Version' else 'vorheriger / nächster'} "
                      f"{source.item}"),
            ("Pos1 / R", "Fahrt von vorne"),
            ("+ / −", "schneller / langsamer (0.5× bis 50×)"),
            ("L", f"Live: {source.live_text}"),
            (keys, f"Filter: {filt}"),
            ("C", "Kamera: Verfolger, von oben, frei"),
            ("W A S D  Q E", "freie Kamera fliegen (Shift = schneller, Mausrad = Tempo)"),
            ("Maus ziehen", "freie Kamera: umsehen (in der 3D-Ansicht)"),
            ("V", "Sensorstrahlen der KI ein/aus"),
            ("P", "Panel rechts ein/aus"),
            ("F", "Vollbild"),
            ("?  /  H", "diese Hilfe"),
            ("Esc", "Beenden"),
        ]
        self.ui = {
            "bottom": Entity(parent=camera.ui, model="quad", color=BAR, shader=unlit_shader, z=0.03),
            "info": InfoBar(),
            "attempts": Strip(source.items),
            "ride": Strip("Fahrt"),
            "panel": Panel(),
            "help": Help(self.help_rows),
            "big": TextQuad("", height=0.06, origin=(0, 0), heavy=True, color=(255, 215, 40), align="center"),
            "hover": TextQuad("", height=0.02, origin=(0, -.5), px=40, heavy=True),
            "msg": TextQuad("", height=0.022, origin=(-.5, .5), px=44),
        }
        p = player
        self.buttons = {
            "prev": Button(icon="prev", on_click=lambda: p.step_attempt(-1), tooltip=f"zurück: {source.item} (B)"),
            "play": Button(icon="pause", on_click=p.toggle_play, tooltip="Play / Pause (Leertaste)"),
            "next": Button(icon="next", on_click=lambda: p.step_attempt(1), tooltip=f"weiter: {source.item} (N)"),
            "speed": Button("1×", on_click=lambda: p.faster(1 if p.speed_i < len(SPEEDS) - 1 else -99),
                            tooltip="Tempo (+ / −)"),
            "live": Button("● LIVE", on_click=p.go_live, tooltip=f"{source.live_text} (L)"),
            "filter": Button("", on_click=self._cycle_filter, tooltip=f"Filter ({keys})"),
            "cam": Button("", on_click=self._cycle_cam, tooltip="Kamera (C)"),
            "panel": Button("Panel", on_click=self._toggle_panel, tooltip="Panel rechts (P)"),
            "help": Button("?", on_click=self.ui["help"].toggle, tooltip="Tastenkürzel (?)"),
        }
        self.st = {"cam": 0, "panel": True, "layout": None, "refresh": 0.0, "strip_key": None, "ride_key": None,
                   "drag_3d": None, "last_t": 0.0, "last_load": 0.0, "pending": None, "photo": None,
                   "mouse": None, "keys": test_keys.split(), "keys_wait": 0.5}
        self.photo = photo
        self.refresh_strips()

    def run(self):
        self.app.run()

    # ---------- Bedienung ----------
    def _cycle_cam(self):
        st = self.st
        st["cam"] = (st["cam"] + 1) % len(CAM_MODES)
        mode = CAM_MODES[st["cam"]][0]
        if mode == "free":
            self.free_cam.start()
        else:
            self.rig.mode, self.rig.first = mode, True

    def _cycle_filter(self):
        f = self.src.filters
        self.player.set_filter(f[(f.index(self.player.mode) + 1) % len(f)] if self.player.mode in f else f[0])

    def _toggle_panel(self):
        self.st["panel"] = not self.st["panel"]
        if self.st["panel"]:
            self.update_panel()

    # ---------- Layout: 3D-Ansicht in ihrer eigenen Display-Region, Leisten und Panel darum herum ----------
    def layout(self):
        from ursina import Vec3, camera, window
        st, ui = self.st, self.ui
        A = window.aspect_ratio
        panel = st["panel"]
        right_3d = A / 2 - (PANEL_W if panel else 0)
        # 3D-Bild nur in den freien Bereich rendern (nicht unter Panel und Leisten), Linse passend dazu
        frac_r = (right_3d + A / 2) / A
        camera.display_region.set_dimensions(0, frac_r, BOTTOM_H, 1 - TOP_H)
        w_px, h_px = window.size[0] * frac_r, window.size[1] * (1 - TOP_H - BOTTOM_H)
        camera.perspective_lens.set_aspect_ratio(w_px / max(h_px, 1))
        key = (A, panel, tuple(window.size))
        if key == st["layout"]:
            return
        st["layout"] = key
        ui["info"].layout(A, TOP_H)
        ui["bottom"].position = Vec3(0, -0.5 + BOTTOM_H / 2, 0.03)
        ui["bottom"].scale = Vec3(A, BOTTOM_H, 1)
        ui["panel"].enabled = panel
        if panel:
            ui["panel"].layout(right_3d, -0.5 + BOTTOM_H, A / 2, 0.5 - TOP_H)
        self.minimap.right, self.minimap.top = right_3d, 0.5 - TOP_H
        x_label, x0, x1 = -A / 2 + 0.015, -A / 2 + 0.11, A / 2 - 0.25
        yb = -0.5 + BOTTOM_H
        ui["attempts"].layout(x_label, x0, yb - 0.052, x1, yb - 0.014, x1 + 0.02)
        ui["ride"].layout(x_label, x0, yb - 0.088, x1, yb - 0.066, x1 + 0.02)
        y0, y1 = -0.5 + 0.012, -0.5 + 0.056
        x = -A / 2 + 0.015
        for name, w in (("prev", 0.05), ("play", 0.06), ("next", 0.05), ("speed", 0.08), ("live", 0.11),
                        ("filter", 0.25), ("cam", 0.24), ("panel", 0.09), ("help", 0.05)):
            self.buttons[name].layout(x, y0, x + w, y1)
            x += w + 0.01
        ui["msg"].position = Vec3(x + 0.01, y1 - 0.004, 0)
        cx = -A / 2 + (right_3d + A / 2) / 2
        ui["big"].position = Vec3(cx, 0.08, 0)
        ui["help"].layout(cx, 0.03)

    # ---------- Anzeige ----------
    def update_texts(self):
        p, ui, src = self.player, self.ui, self.src
        a, ep = p.current, p.episode
        n_all = len(self.index.attempts)
        if a is None:
            ui["info"].set(f"{src.run_dir.name}  ·  noch keine {src.items}", "warte auf das Training ...")
            ui["big"].text = f"Noch keine {src.items}" if not n_all else \
                f"Keine {src.items} für: {src.filter_text[p.mode]}"
            return
        right = f"{p.speed:g}×"
        if ep is not None:
            kmh = abs(float(ep.obs_at(p.t)[17])) * 110 * 3.6
            total = ep.duration if ep.done else max(ep.expected or 0.0, ep.duration)
            right = f"{format_time(p.t)} / {format_time(total)}  ·  {kmh:3.0f} km/h  ·  {p.speed:g}×"
        if p.live:
            right += "  ·  LIVE"
        ui["info"].set(f"{src.run_dir.name}  ·  {src.info(a)}", right, result_rgb(a))
        if ep is None:
            big = src.missing(a)
        elif p.at_end:
            big = src.end_text(a, ep)
            if p.live and p.pos == len(p.list) - 1:
                big += f"\n\nwarte auf {'die nächste Version' if src.item == 'Version' else 'den nächsten Versuch'} ..."
        else:
            big = ""
        ui["big"].text = big
        ui["attempts"].right.text = f"{nr(p.pos + 1)} / {nr(len(p.list))}"
        ui["ride"].right.text = f"Reward {nr(round(ep.reward_until(p.t)))}" if ep else ""
        b = self.buttons
        b["play"].set_icon("pause" if p.playing else "play")
        b["speed"].set_label(f"{p.speed:g}×")
        b["live"].active = p.live
        b["filter"].set_label(self.src.filter_text.get(p.mode, p.mode))
        b["cam"].set_label(f"Kamera: {CAM_MODES[self.st['cam']][1]}")
        b["panel"].active = self.st["panel"]

    def update_panel(self):
        self.ui["panel"].set("\n".join(self.src.panel(self.index, self.player)))

    def refresh_strips(self):
        from render.viewer_ui import attempts_image
        p = self.player
        key = (len(p.list), p.mode, len(self.index.attempts), id(p.list[-1]) if p.list else None)
        if key != self.st["strip_key"]:
            self.st["strip_key"] = key
            self.ui["attempts"].set_image(attempts_image(p.list))

    def _ride_scale(self, ep):
        """Länge der Fahrt-Zeitleiste in s. Fährt das Netz gerade, ist sie aus der Auswertung bekannt
        (deterministisch, R6) oder wächst mit."""
        return ep.duration if ep.done else max(ep.expected or 0.0, ep.duration, 1.0)

    def _ride_image(self, ep):
        from render.viewer_ui import speed_image
        scale = self._ride_scale(ep)
        key = (id(ep), ep.frames if not ep.done else -1, round(scale, 2))
        if key == self.st["ride_key"]:
            return
        if not ep.done and self.st["ride_key"] and self.st["ride_key"][0] == id(ep) and \
                perf_counter() - self.st.get("ride_time", 0) < 0.4:
            return                         # fährt noch: Bild höchstens 2-3 mal pro Sekunde neu
        self.st["ride_key"], self.st["ride_time"] = key, perf_counter()
        self.ui["ride"].set_image(speed_image(ep.speed, total=max(1, int(round(scale / DT))) + 1))

    def show_car(self, dt):
        from ursina import camera, held_keys
        p, st = self.player, self.st
        ep = p.episode
        if ep is None:
            return
        if p.changed or abs(p.t - st["last_t"]) > 1.0 + dt * p.speed:   # Sprung -> Spuren weg
            self.effects.reset()
            if p.changed:
                self.rig.first = True
                for g in self.ghosts:
                    g.reset()
                if st["panel"]:
                    self.update_panel()
            p.changed = False
        st["last_t"] = p.t
        self._ride_image(ep)
        pose = ep.pose(p.t)
        obs = ep.obs_at(p.t)
        self.car.set(pose, obs)
        x, y, z, heading, pitch, roll, steer = pose
        w = self.world
        w.car.set_state(x, y, z, heading, pitch, roll, steer)
        if w.rays.enabled:      # genau die Strahlen, die die KI gesehen hat (aus der Beobachtung)
            w.rays.set_distances(x, y, z, heading, [float(d) * RAY_RANGE for d in obs[:17]])
        play_dt = dt * p.speed if p.playing and not p.at_end else 0.0
        if CAM_MODES[st["cam"]][0] == "free":
            self.free_cam.update(dt, held_keys)
        else:
            self.rig.update(x, y, z, heading, self.car._v[0], dt, roll=roll,
                            move_heading=math.atan2(self.car.vx, self.car.vz))
        w.follow(x, y, z)
        self.effects.update(self.car, play_dt)
        self.streaks.update(self.car, camera.world_position, play_dt)
        for g in self.ghosts:
            g.show_frame(p.t / DT)
        self.minimap.update(x, z, heading, [(g.model.x, g.model.z) for g in self.ghosts])
        self.ui["ride"].set_frac(p.t / self._ride_scale(ep))

    # ---------- Maus ----------
    def _mouse(self):
        from ursina import mouse
        return self.st["mouse"] if self.st["mouse"] is not None else (mouse.x, mouse.y)

    def strips_and_buttons(self, mx, my):
        from ursina import Vec3
        p, st, ui = self.player, self.st, self.ui
        a, r = ui["attempts"], ui["ride"]
        hover = ""
        if a.show_hover(mx, my) and p.list:
            i = min(len(p.list) - 1, int(a.frac_at(mx) * len(p.list)))
            hover = self.src.hover(p.list[i])
            if a.dragging:
                st["pending"] = i
        if r.show_hover(mx, my) and p.episode:
            t = r.frac_at(mx) * self._ride_scale(p.episode)
            hover = hover or format_time(t)
            if r.dragging:
                p.seek(t)
        ui["hover"].text = hover
        if hover:
            ui["hover"].position = Vec3(mx, a.rect[3] + 0.012, 0)
        tip = ""
        for b in self.buttons.values():
            b.update(mx, my)
            if b.contains(mx, my):
                tip = b.tooltip
        ui["msg"].text = tip
        # beim Ziehen höchstens ~15 Ladevorgänge/s; ein Netz laden dauert -> erst beim Loslassen
        if st["pending"] is not None and not (self.src.slow_load and a.dragging) and \
                perf_counter() - st["last_load"] > 0.06:
            st["last_load"] = perf_counter()
            p.select(st["pending"])
            st["pending"] = None
        if p.current is not None and p.list:
            frac = (st["pending"] if st["pending"] is not None and a.dragging else p.pos) + 0.5
            a.set_frac(frac / len(p.list))

    def in_3d(self, mx, my):
        from ursina import window
        A = window.aspect_ratio
        right = A / 2 - (PANEL_W if self.st["panel"] else 0)
        return -A / 2 <= mx <= right and -0.5 + BOTTOM_H <= my <= 0.5 - TOP_H

    # ---------- von Ursina jedes Bild aufgerufen ----------
    def update(self):
        from ursina import time
        st = self.st
        dt = min(time.dt, 0.1)
        self.layout()
        self._test_input(dt)
        if self.photo:
            self._photo(dt)
        st["refresh"] -= dt
        if st["refresh"] <= 0:                       # neue Versuche/Versionen des laufenden Trainings
            st["refresh"] = 1.0
            self.player.refresh()
            self.refresh_strips()
            if st["panel"]:
                self.update_panel()
        mx, my = self._mouse()
        self.strips_and_buttons(mx, my)
        if st["drag_3d"] is not None and CAM_MODES[st["cam"]][0] == "free":
            px, py = st["drag_3d"]
            self.free_cam.look(mx - px, my - py)
            st["drag_3d"] = (mx, my)
        self.player.update(dt)
        self.show_car(dt)
        self.update_texts()

    def input(self, key):
        from ursina import application, window
        p, st, ui = self.player, self.st, self.ui
        mx, my = self._mouse()
        if key == "escape":
            if ui["help"].visible:
                ui["help"].toggle()
            else:
                application.quit()
        elif key == "left mouse down":
            for b in self.buttons.values():
                if b.contains(mx, my):
                    b.on_click()
                    return
            for s in (ui["attempts"], ui["ride"]):
                if s.contains(mx, my):
                    s.dragging = True
                    if s is ui["attempts"] and p.list:
                        st["pending"] = min(len(p.list) - 1, int(s.frac_at(mx) * len(p.list)))
                    return
            if self.in_3d(mx, my):
                st["drag_3d"] = (mx, my)
        elif key == "right mouse down" and self.in_3d(mx, my):
            st["drag_3d"] = (mx, my)
        elif key in ("left mouse up", "right mouse up"):
            a = ui["attempts"]
            if a.dragging and st["pending"] is not None:
                p.select(st["pending"])
                st["pending"] = None
            a.dragging = ui["ride"].dragging = False
            st["drag_3d"] = None
        elif key == "space":
            p.toggle_play()
        elif key in ("left arrow", "right arrow", "left arrow hold", "right arrow hold"):
            p.seek(p.t + (-2.0 if key.startswith("left") else 2.0))
        elif key in ("b", "n"):
            p.step_attempt(-1 if key == "b" else 1)
        elif key in ("home", "r"):
            p.restart()
        elif key == "l":
            p.go_live()
        elif key in ("1", "2", "3") and int(key) <= len(self.src.filters):
            p.set_filter(self.src.filters[int(key) - 1])
        elif key == "c":
            self._cycle_cam()
        elif key == "v":
            self.world.rays.enabled = not self.world.rays.enabled
        elif key == "p":
            self._toggle_panel()
        elif key == "f":
            window.fullscreen = not window.fullscreen
        elif key in ("h", "f1"):
            ui["help"].toggle()
        elif key in ("scroll up", "scroll down") and CAM_MODES[st["cam"]][0] == "free":
            self.free_cam.zoom(1 if key == "scroll up" else -1)

    def text_input(self, key):
        """Getippte Zeichen (unabhängig vom Tastatur-Layout): ?, +, −."""
        if key == "?":
            self.ui["help"].toggle()
        elif key == "+":
            self.player.faster(1)
        elif key == "-":
            self.player.faster(-1)

    # ---------- nur zum Prüfen (versteckte Optionen --foto / --tasten) ----------
    def _photo(self, dt):
        from pathlib import Path

        from panda3d.core import Filename
        from ursina import application
        self.st["photo"] = (self.st["photo"] or 0.0) + dt
        if self.st["photo"] >= float(self.photo[1]):
            application.base.win.saveScreenshot(Filename.fromOsSpecific(str(Path(self.photo[0]).resolve())))
            application.quit()

    def _test_input(self, dt):
        """Pro Bild ein Eintrag: maus:versuche:0.3 / maus:fahrt:0.5 / maus:knopf:play / maus:3d,
        klick, los (linke Maustaste), warte, sonst eine Taste ("left_arrow" = "left arrow")."""
        st = self.st
        st["keys_wait"] -= dt
        if not st["keys"] or st["keys_wait"] > 0:
            return
        k = st["keys"].pop(0).replace("_", " ")
        if k.startswith("maus:"):
            what = k.split(":")[1:]
            if what[0] in ("versuche", "fahrt"):
                x0, y0, x1, y1 = self.ui["attempts" if what[0] == "versuche" else "ride"].rect
                st["mouse"] = (x0 + (x1 - x0) * float(what[1]), (y0 + y1) / 2)
            elif what[0] == "knopf":
                x0, y0, x1, y1 = self.buttons[what[1]].rect
                st["mouse"] = ((x0 + x1) / 2, (y0 + y1) / 2)
            else:
                st["mouse"] = (0.0, 0.1)
        elif k in ("klick", "los"):
            self.input("left mouse down" if k == "klick" else "left mouse up")
        elif k == "warte":
            st["keys_wait"] = 0.5
        elif k in "?+-":
            self.text_input(k)
        else:
            self.input(k)


class ReplayCar:
    """So viel "Auto", wie Effekte und Kamera brauchen – aus Position, Rotation und Beobachtung."""

    def set(self, pose, obs):
        self.x, self.y, self.z, self.heading = pose[0], pose[1], pose[2], pose[3]
        vf, vr = float(obs[17]) * 110.0, float(obs[19]) * 110.0
        fx, fz = self.forward
        rx, rz = self.right
        self.vx, self.vz = vf * fx + vr * rx, vf * fz + vr * rz
        self._v = (vf, vr)
        self.grounded = obs[22] < 0.5

    @property
    def forward(self):
        return math.sin(self.heading), math.cos(self.heading)

    @property
    def right(self):
        return math.cos(self.heading), -math.sin(self.heading)

    @property
    def speed(self):
        return math.hypot(self.vx, self.vz)

    def local_velocity(self):
        return self._v
