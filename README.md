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
| `.venv/bin/python train.py --name test1` | KI trainieren (siehe [Training](#training)) |
| `.venv/bin/python watch.py test1 --live` | Der KI beim Fahren zuschauen (Netz fährt jetzt) |
| `.venv/bin/python viewer.py test1` | Aufgezeichnete Versuche ansehen wie im Videoplayer |

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
rl/              KI: Gym-Umgebung, Reward, Training
  env.py         RaceEnv (Gymnasium)
  trainer.py     SAC-Training, Auswertung, Erfolgsstufen, Pausieren
  run.py         Ordner runs/<name>/: Konfiguration, summary.csv, Fortsetzungspunkt
  resources.py   Leistungsprofile, Akku-/Speicher-Wächter
tracks/          Maps (JSON)
runs/            Trainings (je ein Ordner)
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
  wird gemeldet (`events["wall_hit"]`, `sim.wall_hits`). Eine eigene Reward-Strafe dafür gibt es nicht (R7),
  die Wand kostet aber Tempo und damit Zeit.
- **Crash: kein Fortschritt.** In 2 s weniger als 1 m vorwärts entlang der Strecke.
- **Crash: falsche Richtung.** 15 m hinter den eigenen besten Fortschritt zurückgefallen.
- **Zeitlimit** 180 s: kein Crash, sondern Abbruch ohne Wertung (im Training "truncated", R11)

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

## Training
Lernverfahren: **SAC** (Stable-Baselines3), kleines Netz mit 2 Schichten à 64 Neuronen für Actor und Critic.
Einstellungen in `train_config.toml`.

### Installieren und starten
```bash
.venv/bin/python -m pip install -r requirements.txt     # einmalig (torch, stable-baselines3, tensorboard, psutil)
.venv/bin/python train.py --name test1                  # neues Training, läuft bis Ctrl+C
```
Der Mac bleibt automatisch wach, solange das Training läuft (`train.py` startet dafür `caffeinate -i`).
**Deckel offen lassen**: Zugeklappt schläft ein MacBook trotzdem ein. Für eine Nacht am besten ans Netzteil.

### Während es läuft
Die Statuszeile zeigt Versuch, offizielle Bestzeit, Erfolgsstufe, Schritte, Schritte/s, Laufzeit und Profil.
Neue Bestzeiten und erreichte Stufen werden zusätzlich als eigene Zeile gemeldet.

TensorBoard (Kurven im Browser, in einem zweiten Terminal):
```bash
.venv/bin/tensorboard --logdir runs        # dann http://localhost:6006 öffnen
```
Wichtige Kurven: `auswertung/bestzeit`, `auswertung/gueltig`, `training/anteil_gueltig_letzte_100`,
`fahrt/reward`. Steigt `fahrt/reward`, ohne dass Runden gültiger oder schneller werden: Training stoppen (Regel R7).

### Pausieren und fortsetzen
- **Ctrl+C** pausiert sauber: Modell, Replay-Buffer, Zähler und Zufallszustand werden gespeichert.
  Ein zweites Ctrl+C bricht sofort ab, ohne zu speichern.
- **Terminal-Fenster schliessen** pausiert ebenfalls sauber (wie Ctrl+C).
- Bestzeit und Erfolgsstufen stehen sofort in `rekorde.json`. Endet das Training trotzdem einmal ohne Pause
  (Absturz, Strom weg), gehen nur die Schritte seit der letzten Sicherung verloren, Bestzeit und Stufen nicht.
  `ereignisse.csv` vermerkt das beim Fortsetzen als `absturz`.
- Weitermachen: `.venv/bin/python train.py --name test1 --resume`. Versuchszähler, Schritte, Snapshots, `summary.csv`
  und TensorBoard laufen nahtlos weiter (eine Versuchsnummer wird nie doppelt vergeben). Nur die gerade angefangenen Fahrten beginnen neu am Start.
- Alle 30 min wird automatisch ein Fortsetzungspunkt gespeichert. Bei einem Absturz oder leerem Akku
  sind also höchstens 30 min verloren.
- Haben sich seit dem Start Physik, Regeln, Sensoren, Strecke oder Reward geändert, verweigert `--resume`
  das Fortsetzen (Regel R8). Dann ein neues Training mit neuem Namen starten.

### Leistungsprofile
`--profil sparsam | normal | voll` (Standard: normal), beim Fortsetzen frei wechselbar.

| Profil | Umgebungen | Priorität | Gedacht für |
|---|---|---|---|
| sparsam | 2 | niedrig (nice 15) | nebenbei arbeiten, Akku |
| normal | Hälfte der Kerne | mittel (nice 10) | Mac wird nebenbei benutzt |
| voll | alle Kerne ausser einem | normal | über Nacht am Netzteil |

- `--max-sps 1000`: höchstens 1000 Schritte pro Sekunde (bremst zusätzlich).
- Die KI lernt in jedem Profil gleich (gleich viele Lern-Updates pro Schritt), sie ist nur unterschiedlich schnell.
- **Akkubetrieb:** automatisch sparsam. Am Netzteil wieder das gewählte Profil.
  Unter 30 % Akku pausiert das Training (Fortsetzungspunkt wird gespeichert).
- **Festplatte:** Wenn weniger als 10 GB frei sind, pausiert das Training.
- Weitere Optionen: `--minuten 30` (nach 30 min pausieren), `--schritte 500000`, `--seed 1` (nur bei neuem Training).

### Was gespeichert wird (`runs/<name>/`)
| Datei | Inhalt |
|---|---|
| `config.json` | vollständige Konfiguration (SAC, Reward, Physik, Strecke), Seed, alle Versionen, Git-Commit (R9) |
| `code.tar.gz` | Kopie des Codes, falls er vom Git-Commit abweicht |
| `summary.csv` | eine Zeile pro Versuch: Nummer, Art (training/auswertung), Reward, Ergebnis, Rundenzeit, Dauer, aktueller Snapshot |
| `evals.csv` | jede deterministische Auswertung (R6) mit Rundenzeit und Bestzeit |
| `ereignisse.csv` | Start, Pause, Fortsetzung, Profilwechsel, erreichte Stufen |
| `episodes/` | jeder Versuch als komprimierte Datei: Position, Rotation, Beobachtung, Aktion, Reward pro Schritt |
| `models/` | Snapshot alle 20 Versuche + Meilensteine (10, 20, 50, 100, ... 10'000), nur Actor und Critic |
| `best_model.zip` | Modell der offiziellen Bestzeit, komplett (zum Weitertrainieren und Gegen-die-KI-Fahren) |
| `ghosts/` | Fahrt der offiziellen Bestzeit als Geist (`play.py --ghost ...`) |
| `versionen/` | Modell jeder Auswertung (alle 25'000 Schritte), für die Zeitleiste in `watch.py` |
| `checkpoint/` | Fortsetzungspunkt (nur hier liegt der Replay-Buffer) |
| `tb/` | TensorBoard |

Format von `episodes/` und `models/`: siehe `rl/recording.py` (lesen mit `load_episode()` bzw. `load_snapshot()`).
Einstellungen unter `[aufzeichnung]` in `train_config.toml`. Läufe von vor der Aufzeichnung: Beim ersten
`--resume` wird die alte `summary.csv` in `evals.csv` und `ereignisse.csv` aufgeteilt (Original: `summary_alt.csv`).

### Speicher und Aufräumen
```bash
.venv/bin/python tools/speicher.py                       # belegt pro Lauf, Zuwachs pro 1 Mio. Schritte, Hochrechnung
.venv/bin/python tools/aufraeumen.py                     # Übersicht
.venv/bin/python tools/aufraeumen.py test1 --episoden    # nur episodes/ löschen (Snapshots, Bestzeit, summary.csv bleiben)
.venv/bin/python tools/aufraeumen.py test1 --alles       # ganzen Lauf löschen
```
`aufraeumen.py` zeigt vorher, was gelöscht wird, und löscht erst nach Eingabe von `ja`. Läufe, deren Training
gerade läuft, fasst es nicht an.

**Offizielle Zeiten** (R6) kommen nur aus der **Auswertung**: Alle 25'000 Schritte fährt das aktuelle Modell
eine Runde ohne Zufall. Runden während des Trainings (mit Zufall) zählen nur für Stufe 1.

### Zuschauen wie im Videoplayer (`viewer.py`)
```bash
.venv/bin/python viewer.py                        # neuester Lauf, live
.venv/bin/python viewer.py v2_lauf2               # bestimmter Lauf
.venv/bin/python viewer.py v2_lauf2 --versuch 500 --filter gueltige --tempo 5
```
Spielt die aufgezeichneten Versuche aus `episodes/` ab (genau das, was gefahren wurde). Läuft während oder
nach dem Training, liest nur Dateien und läuft mit niedriger Priorität (nice 15), damit das Training Vorrang hat.

- **Oben:** Info-Zeile mit Versuch, Art (Training mit Zufall / Auswertung ohne Zufall), Reward, Ergebnis.
- **Unten:** Zeitleiste aller Versuche (Höhe = Fortschritt; grün = Ziel, gelb = Ziel in der Auswertung,
  rot = Crash, grau = Zeitlimit). Ziehen = spulen, Klick = zu diesem Versuch. Darunter die aktuelle Fahrt
  (Tempo-Verlauf), darunter die Knöpfe: zurück, Play/Pause, weiter, Tempo (0.5× bis 50×), LIVE, Filter, Kamera, Panel.
- **Kameras** (C): Verfolger, Drohne (filmisch: schräg von oben, schwenkt langsam ums Auto), von oben, frei.
- **Rechts:** Panel (P), wird in Schritt 8 und 9 zum Dashboard.
- **LIVE:** springt zum neuesten Versuch und folgt dem Training. Neue Versuche erscheinen von selbst.
- Alle Tastenkürzel: `?` (oder H).

**Geister – viele Versuche gleichzeitig** (zweite Knopfreihe oder Tasten G / M / E / T):
```bash
.venv/bin/python viewer.py v2_lauf2 --geister 30 --von 100 --bis 300   # 30 Geister aus Versuch 100–300
.venv/bin/python viewer.py v2_lauf2 --geister 10 --von 500 --bis 1000  # 10 Geister aus Versuch 500–1000
.venv/bin/python viewer.py v2_lauf2 --geister 10                        # die letzten 10 Versuche, live
.venv/bin/python viewer.py v2_lauf2 --auswahl meilensteine              # Versuch 10, 20, 50, 100, ... 10'000
```
- **Anzahl** 5, 10, 20, 30, 50 oder 100 (G). **Auswahl** (M): *Bereich*, *letzte N* (live) oder *Meilensteine*.
- **Bereich** von–bis: Die Geister werden gleichmässig darin verteilt (30 aus 100–300 = etwa jeder 7. Versuch).
  Einstellen: Knopf „Bereich“ oder Taste E und z.B. `100-300` tippen, Enter. Oder mit der **rechten Maustaste**
  über die Versuchs-Zeitleiste ziehen. Der Bereich ist dort blau markiert. Linksklick auf die Zeitleiste
  verschiebt ihn (gleiche Breite), B / N springen einen ganzen Bereich zurück / weiter.
- Alle starten gleichzeitig, halb durchsichtig, ohne Kollision. Farbe nach der aktuellen Position im Rennen:
  vorderes Drittel grün, mittleres orange, hinteres rot (wechselt mit der Rangliste, auch auf der Minikarte).
  Über jedem Geist seine Nummer (T blendet sie aus). Wer crasht, bleibt grau an der Crash-Stelle stehen.
- **Klick auf einen Geist:** Kamera folgt ihm, Info-Zeile und Dashboard zeigen seine Daten.
- Zeitleiste, Play/Pause und Tempo gelten für alle gemeinsam. Oben links: Rangliste (im Ziel nach Zeit,
  dann wer noch fährt nach Fortschritt, dann die Gecrashten).
- Geister sind einfache Modelle (ein Mesh pro Geist). Gemessen mit laufendem Training: 30 Geister 56–60 Bilder/s,
  50 Geister ca. 56, 100 Geister ca. 40. Die Kamera folgt dem, der am weitesten kommt. Nummern stehen höchstens
  über den 30 nächsten Geistern.
  `watch.py` hat keine Geister (jeder bräuchte ein eigenes Netz, das jetzt fährt).

### Zuschauen, wie das Netz jetzt fährt (`watch.py`)
Gleiche Oberfläche und gleiche Tasten wie `viewer.py`. Statt eine Aufzeichnung abzuspielen, fährt eine gespeicherte
Version des Netzes jetzt, deterministisch (R6) – genau dieselbe Fahrt wie in ihrer Auswertung beim Training.
```bash
.venv/bin/python watch.py v2_lauf2 --live              # immer die neueste Version, folgt dem Training
.venv/bin/python watch.py v2_lauf2                     # Version der offiziellen Bestzeit
.venv/bin/python watch.py v2_lauf2 --modell neuester   # jüngste Version (ohne zu folgen)
.venv/bin/python watch.py v2_lauf2 --modell letzter    # letzter Fortsetzungspunkt
.venv/bin/python watch.py v2_lauf2 --gegen-mich        # gegen deine Bestzeit als Geist
```
Obere Zeitleiste = Versionen (eine pro Auswertung, alle 25'000 Schritte), untere = die Fahrt. Was schon gefahren
ist, lässt sich zurückspulen; vorspulen rechnet voraus. `--gegen-mich` gibt es auch bei `viewer.py`.

## Credits
Automodell: "2020 NASCAR Chevrolet Camaro Zl1 1LE" von Ddiaz Design, CC-BY-4.0. Details in [CREDITS.md](CREDITS.md).
