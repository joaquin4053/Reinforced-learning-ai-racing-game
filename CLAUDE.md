# KI-Fahrschule

## ZIEL
Die KI soll lernen, die Strecke regelkonform in möglichst kurzer Zeit vom Start ins Ziel zu fahren. Massstab ist allein die Rundenzeit, gemessen in Simulationszeit.

## REGELN (strikt)
R1 Gültige Runde: Eine Runde ist nur gültig, wenn das Auto stehend an der Startposition beginnt, alle Checkpoints in der festgelegten Reihenfolge durchfährt und danach die Ziellinie überquert. Nur gültige Runden erhalten eine Zeit.

R2 Auf der Strecke bleiben: Die Strecke schwebt frei, ohne Wände. Fällt das Auto von der Fahrbahn (mehr als 4 m unter die Strecke), ist das ein Crash: Die Fahrt endet, es gibt keine Rundenzeit und die Crash-Strafe nach R7. Prallt das Auto gegen die Seite eines Blocks, prallt es ab, verliert Tempo und die Fahrt geht weiter; jede Berührung wird von der Simulation gemeldet. Abkürzungen sind ungültig (Checkpoints nur in der festgelegten Reihenfolge, R1). Diese Regel gilt für KI und Spieler gleich.
*(Geändert am 2026-10-13 mit Zustimmung: vorher "Berührung = Crash, Fahrt endet sofort".)*
*(Geändert am 2026-10-18 mit Zustimmung: Spiel v2 in 3D ohne Wände, Runterfallen = Crash.)*

R3 Gleiche Physik, gleiches Auto: KI und Spieler nutzen exakt dasselbe Physik-Modul mit denselben Parametern und demselben festen Zeitschritt. Die KI hat keine Sonderrechte: kein höheres Tempo, keine andere Haftung, kein Zurücksetzen während der Fahrt.

R4 Gleiche Eingaben: Aktionen sind nur im erlaubten Bereich möglich: Lenkung -1 bis 1, Gas 0 bis 1, Bremse 0 bis 1, Drift 0 bis 1. Werte ausserhalb werden abgeschnitten. Höchstens eine Entscheidung pro Physik-Schritt.
*(Geändert am 2026-10-13 mit Zustimmung: Drift als vierte Eingabe für Spieler und KI ergänzt.)*

R5 Nur Fahrersicht: Die KI bekommt nur Informationen, die ein Fahrer im Auto wahrnehmen könnte: Abstände der Sensorstrahlen zur Streckenbegrenzung (Streckenkante), eigenes Tempo, Winkel zur Streckenrichtung, Neigung des Autos (Steigung), ob das Auto in der Luft ist und das senkrechte Tempo. Keine absoluten Koordinaten, keine Ideallinie, kein Wissen über die Strecke hinter der Sensorreichweite (200 m).
*(Ergänzt am 2026-10-18 mit Zustimmung: Steigung, in der Luft, senkrechtes Tempo für Spiel v2 in 3D. Reichweite bleibt 100 m.)*
*(Geändert am 2026-10-19 mit Zustimmung: Reichweite 100 m -> 200 m, Sensor-Version 6. Grund: Probelauf mit grösserem Netz und 200 m war deutlich schneller.)*

R6 Zeitmessung: Zeit = Anzahl Physik-Schritte × Zeitschritt, angezeigt auf 0.01 s. Nie mit der echten Uhr messen. Offizielle Bestzeiten der KI zählen nur aus deterministischen Fahrten (ohne Zufall).

R7 Belohnung dient nur dem Ziel: Der Reward besteht nur aus: Fortschritt entlang der Strecke (nur ein neuer Höchststand zählt), einer kleinen Zeitstrafe pro Schritt, einer Crash-Strafe und einem Ziel-Bonus. Keine Belohnung für Tempo an sich, für Hin-und-her-Fahren oder Rückwärtsfahren. Steigt der Reward, ohne dass Runden gültiger oder schneller werden, ist das ein Fehler oder eine Lücke: Training stoppen und melden.
*(Am 2026-10-19 mit Zustimmung kurz geändert – Ziel-Bonus nach Rundenzeit, Reward-Version 2 – und am selben Tag mit Zustimmung zurückgenommen: Das Training wurde damit instabil (v2_lauf3 bis v2_lauf5). Es gilt wieder der feste Ziel-Bonus, Reward-Version 1.)*

R8 Versionen: Jede Änderung an Physik, Strecke, Regeln, Sensoren oder Reward erhöht die passende Versionsnummer. Replays, Bestzeiten und Modelle speichern diese Versionen. Ergebnisse verschiedener Versionen werden nie direkt verglichen.

R9 Nachvollziehbarkeit: Jedes Training speichert seine vollständige Konfiguration, den Seed, alle Versionsnummern und den Git-Commit. Ein Training muss sich später nachvollziehen lassen.

R10 Regeln sind fest: Claude Code ändert keine Regel und keinen Bestandteil des Rewards ohne meine ausdrückliche Zustimmung. Wenn eine Regel ein Problem verursacht: zuerst erklären und fragen.

R11 Fahrt-Ende ohne Ziel: Eine Fahrt endet vorzeitig als Crash, wenn (a) das Auto 2 s lang weniger als 1 m Fortschritt entlang der Strecke macht (stecken geblieben) oder (b) es mehr als 15 m hinter seinen bisher besten Fortschritt zurückfällt (falsche Richtung). Ein Crash ergibt keine Rundenzeit und ist der "Crash" im Sinne von R7 (Crash-Strafe). Nach 180 s Simulationszeit wird die Fahrt abgebrochen (Zeitlimit): kein Crash, keine Crash-Strafe, keine Rundenzeit (im Training "truncated"). Die Regel gilt für KI und Spieler gleich. Die Grenzwerte stehen in config.toml unter [regeln]; jede Änderung erhöht die Regel-Version.
*(Ergänzt am 2026-10-14 mit Zustimmung. Geändert am 2026-10-14 mit Zustimmung: Zeitlimit ist kein Crash mehr, sondern ein Abbruch.)*

## ERFOLGSSTUFEN
- Stufe 1: Ankommen: Die KI schafft zum ersten Mal eine gültige Runde.
- Stufe 2: Zuverlässig: Die deterministische Auswertung des aktuellen Modells kommt bei 5 Auswertungen in Folge ins Ziel.
- Stufe 3: Schnell: Die Bestzeit sinkt, bis sie über längere Zeit kaum noch besser wird (Plateau).
- Stufe 4: Schneller als ich: Die Bestzeit der KI ist kürzer als meine eigene Bestzeit auf derselben Strecke und Version.

## PLAN (wird Schritt für Schritt gebaut, nicht alles jetzt)
1. Spiel regelkonform machen und prüfen
2. Sensoren (Fahrersicht)
3. Gym-Umgebung und Reward
4. Erstes Training
5. Alles aufzeichnen (jeder Versuch, Modell-Snapshots)
6. Zuschauer mit Zeitliste wie bei einem Videoplayer
7. Geister (viele Versuche gleichzeitig)

## BEOBACHTUNG DER KI (Sensor-Version 6)
Was `observation()` liefert, in genau dieser Reihenfolge. Erzeugt aus `game/sensors.py`; `check_ready.py` prüft, dass diese Tabelle zum Code passt.

| Index | Name | Bedeutung | Wertebereich |
|---|---|---|---|
| 0 | `strahl_0` | Abstand zur Streckenkante, Strahl 90.0° links (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 1 | `strahl_1` | Abstand zur Streckenkante, Strahl 78.8° links (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 2 | `strahl_2` | Abstand zur Streckenkante, Strahl 67.5° links (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 3 | `strahl_3` | Abstand zur Streckenkante, Strahl 56.2° links (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 4 | `strahl_4` | Abstand zur Streckenkante, Strahl 45.0° links (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 5 | `strahl_5` | Abstand zur Streckenkante, Strahl 33.8° links (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 6 | `strahl_6` | Abstand zur Streckenkante, Strahl 22.5° links (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 7 | `strahl_7` | Abstand zur Streckenkante, Strahl 11.2° links (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 8 | `strahl_8` | Abstand zur Streckenkante, Strahl 0.0° geradeaus (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 9 | `strahl_9` | Abstand zur Streckenkante, Strahl 11.2° rechts (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 10 | `strahl_10` | Abstand zur Streckenkante, Strahl 22.5° rechts (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 11 | `strahl_11` | Abstand zur Streckenkante, Strahl 33.8° rechts (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 12 | `strahl_12` | Abstand zur Streckenkante, Strahl 45.0° rechts (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 13 | `strahl_13` | Abstand zur Streckenkante, Strahl 56.2° rechts (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 14 | `strahl_14` | Abstand zur Streckenkante, Strahl 67.5° rechts (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 15 | `strahl_15` | Abstand zur Streckenkante, Strahl 78.8° rechts (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 16 | `strahl_16` | Abstand zur Streckenkante, Strahl 90.0° rechts (0 = direkt an der Kante, 1 = nichts in 200 m) | 0 bis 1 |
| 17 | `tempo` | Tempo vorwärts / 110 m/s (negativ = rückwärts) | -1 bis 1 |
| 18 | `winkel_strecke` | Winkel zwischen Fahrtrichtung und Streckenrichtung / 180° (positiv = Auto zeigt nach rechts) | -1 bis 1 |
| 19 | `rutschen` | Tempo seitwärts / 110 m/s (positiv = rutscht nach rechts) | -1 bis 1 |
| 20 | `lenkung` | aktuelle Lenkradstellung (-1 = voll links, +1 = voll rechts) | -1 bis 1 |
| 21 | `steigung` | Neigung des Autos nach vorne / 45° (positiv = bergauf) | -1 bis 1 |
| 22 | `in_der_luft` | 1 = Räder haben keinen Bodenkontakt, 0 = am Boden | 0 bis 1 |
| 23 | `tempo_vertikal` | Tempo nach oben / 30 m/s (negativ = fällt) | -1 bis 1 |