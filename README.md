# KI-Fahrschule

Ziel, Regeln und Erfolgsstufen stehen in [CLAUDE.md](CLAUDE.md).

Einfacher Trackmania-Clone in Python (Ursina) als Spielwiese für Reinforcement Learning.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Benutzen
| Befehl | Was passiert |
|---|---|
| `.venv/bin/python play.py` | Selbst fahren (Map 1) |
| `.venv/bin/python play.py --ghost ghosts/map01_ai_000100.json` | Gegen einen Geist fahren (`--ghost` mehrfach möglich) |
| `.venv/bin/python play.py --best` | Gegen die eigene Bestzeit fahren |
| `.venv/bin/python replay.py ghosts/<datei>.json` | Geist als MP4 mit Kinokamera rendern (`videos/`) |
| `.venv/bin/python tools/list_ghosts.py` | Alle Geister mit Versuchszahl und Zeit auflisten |
| `.venv/bin/python tools/autopilot_run.py` | Testfahrt ohne Grafik, speichert einen Geist |
| `.venv/bin/python check_ready.py` | Prüft, ob die Simulation regelkonform und bereit fürs Training ist (OK/FEHLER je Prüfung) |
| `.venv/bin/python -m pytest` | Automatische Tests (u.a. Determinismus) |

**Steuerung:** W/S oder Pfeile = Gas/Bremse, A/D = Lenken (leicht geglättet), Leertaste = Drift, R/Backspace = Neustart,
C = Kamera, V = KI-Sensoren, Tab/F3 = Debug-Anzeige, Esc = Beenden.
**Gamepad:** linker Stick = Lenken (stufenlos), RT = Gas, LT = Bremse, X/RB = Drift, Y/Start = Neustart.
Bremse im Stand = rückwärts.

## Aufbau
Alle Physik-Werte, Regel-Grenzen, Sensor-Einstellungen und Versionsnummern stehen in **`config.toml`**.
Wer dort etwas ändert, erhöht die passende Versionsnummer (Regel R8).

```
game/            Simulation, OHNE Grafik (auch fürs KI-Training)
  car.py         Fahrphysik (fester Zeitschritt 1/60 s)
  sim.py         Sim: reset(), step(action), Regeln, Checkpoints
  sensors.py     Was die KI sieht: Strahlen + Fahrwerte, Reihenfolge der Beobachtung
  track.py       Map laden, progress_at(x, z) = Fortschritt entlang der Strecke
  ghost.py       Geister speichern/laden, Bestzeiten
  raycast.py     Abstandssensoren
render/          Darstellung (rechnet keine Physik)
  world.py       3D-Welt, Autos, Geister, HUD
  controls.py    Tastatur/Gamepad -> action
  cameras.py     Verfolger, Cockpit, Vogel, Kinokamera
  recorder.py    Geist -> MP4
tracks/          Maps (JSON)
ghosts/          gespeicherte Fahrten (Spieler und KI)
saves/           Bestzeiten
tests/           automatische Tests
```

Eine Aktion ist immer `{"steer": -1..1, "throttle": 0..1, "brake": 0..1, "drift": 0..1}`, egal ob vom Spieler oder von der KI.
`drift` ist optional: Solange er gedrückt ist, verliert das Heck Grip (zum Driften), beim Gegenlenken gibt es vollen Lenkeinschlag.

## Wann endet eine Fahrt?
Gleiche Regeln für Spieler und KI (Werte in `config.toml`, Abschnitt `[regeln]`):
- **Ziel** erreicht, nachdem alle Checkpoints in der richtigen Reihenfolge durchfahren wurden
- **Wand berühren beendet die Fahrt nicht.** Das Auto prallt ab, verliert Tempo und fährt weiter. Jede Berührung
  wird gemeldet (`events["wall_hit"]`, `sim.wall_hits`), damit die KI dafür eine Strafe bekommen kann.
  Mit `Rules(wall_is_crash=True)` ließe sich die Wand wieder tödlich machen.
- **Crash: kein Fortschritt.** In 2 s weniger als 1 m vorwärts entlang der Strecke.
- **Crash: falsche Richtung.** 15 m hinter den eigenen besten Fortschritt zurückgefallen.
- **Zeitlimit** 180 s

## Neue Map anlegen
Neue Datei `tracks/map02.json` nach dem Muster von `tracks/map01.json`:
- `points`: Mittellinie in Fahrtrichtung (Meter), wird automatisch zu weichen Kurven geglättet
- `width`: Streckenbreite, `loop`: Rundkurs ja/nein, `laps`: Runden
- `version`: bei jeder Änderung an einer bestehenden Map um 1 erhöhen
- `start`, `checkpoints`, `finish`: Indizes in `points`

Der Loader meldet, wenn Kurven zu eng sind oder sich die Strecke überlappt.
Danach `.venv/bin/python play.py tracks/map02.json`, und der Test `test_track_loads` prüft die Map automatisch mit.

## Geister
`ghosts/<map>_ai_<versuch>.json` bzw. `ghosts/<map>_player_<datum>.json`. Inhalt: Metadaten (Map, Quelle, Versuch,
Ergebnis, Endzeit oder Crash mit Fortschritt in %, Zwischenzeiten, Physik-Version) plus Position und Rotation pro Physik-Schritt.
Ändern sich Physik, Regeln oder Sensoren, erhöht man die Version in `config.toml` (`[versionen]`); ändert sich eine Map, erhöht man `"version"` in der Map-Datei.
Alte Geister werden dann als „alte Version“ markiert und alte Bestzeiten zählen nicht mehr (sie bleiben unter `alte_rekorde` aufgehoben).

## Für die KI
- **Gym-Umgebung:** `rl/env.py` (`RaceEnv`) mit `reset()` und `step(action)` nach Gymnasium-Standard.
  Aktion `[Lenkung, Gas, Bremse, Drift]`, Beobachtung siehe Tabelle in [CLAUDE.md](CLAUDE.md).
- **Reward** genau nach Regel R7, Gewichte in `reward_config.toml`. Beispiel mit echten Zahlen:
  `.venv/bin/python tools/reward_example.py`
- **Regel-Fahrer** `rl/rule_driver.py`: fährt nur mit den Sensoren der KI eine gültige Runde (Vergleichswert).
- `Sim.observation()`: 21 Werte, nur Fahrersicht (Regel R5), Code in `game/sensors.py`.
