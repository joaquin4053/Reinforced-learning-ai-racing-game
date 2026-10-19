# Beobachtung: PolyTrack (CrazyGames)

Nur durch Zuschauen beim Spielen: Screenshots über REA (`capture_web_screenshot`) in einem eigenen
Chrome-Fenster, während der Benutzer spielt. Kein Code, keine Spieldateien, keine Modelle oder Strecken
übernommen. Ziel: Stil und Fahrgefühl als Vorlage für unser eigenes Spiel (Spiel v2), mit eigenen
Modellen, eigener Physik und eigenen Strecken.

- Sitzung 1 (2026-10-18, ~2 min, 6 Bilder): Summer 1 und Summer 2.
- Sitzung 2 (2026-10-18, 19:01–19:07, 16 Bilder): Summer 2, Summer 3, Summer 4 und eine Strecke mit 8 Checkpoints.

## 1. Fahrleistung (aus Tacho und Zeit abgelesen)
| Zeit nach Start | Tempo | Quelle |
|---|---|---|
| 1,23 s | 83 km/h | Summer 4, Neustart |
| 2,45 s | 155 km/h | Summer 4, Neustart |
| 4,07 s | 210 km/h | Summer 4 |
| 4,6 s | 197 km/h | Summer 3 (in einer Kurve, rutschend) |
| 14 s | 389 km/h | Strecke mit 8 Checkpoints |
| 26 s / Ziel | 380 km/h | Summer 4, lange Gerade |

- Anfahren extrem spritzig: anfangs ~19 m/s² (fast 2 g), bei 150–210 km/h noch ~9 m/s².
- Höchsttempo auf langen Geraden ~380–390 km/h. (Sitzung 1 sah nur 230–250 km/h: dort kurze Geraden.)
- Beschleunigung fällt mit dem Tempo deutlich ab (typisch "konstante Leistung").

## 2. Fahrverhalten und Physik
- **Grip:** sehr hoch, Arcade. Auch bei ~200 km/h werden enge Kurven gefahren.
- **Driften:** bei ~200 km/h in engen Kurven rutscht das Heck deutlich (Rutschwinkel ~25–30°).
  Es entstehen dunkle **Reifenspuren** auf der Fahrbahn und **Rauch** an den Hinterrädern.
  Auch die Geister-Fahrt (Rekord) driftet durch Kurven -> Driften gehört zur schnellen Linie.
- **Steilkurven:** Die Fahrbahn ist in Kurven stark überhöht (bis ~35–40°). Das Auto klebt sicher
  daran, auch bei 258 km/h. Die Kamera neigt sich mit.
- **Geschwungene Flächen:** Fahrbahn geht weich in die Wände über (wie eine Halfpipe/Rinne).
- **Wände:** Manche Strecken haben hohe Wände. Ein Einschlag kippt und dreht das Auto, es verliert stark
  Tempo; nach einem harten Treffer kann es sich komplett drehen (stand danach quer, 19 km/h, viel Rauch).
- **Runterfallen:** Das Auto fällt von der Kante und **überschlägt sich frei** in der Luft (dreht um alle
  Achsen). Unten liegt ein weites Tal.
- **Sprünge:** nicht direkt erwischt.

## 3. Kamera
- Verfolger hinter dem Auto, leicht erhöht. Auto füllt ~1/5 der Bildbreite, sitzt im unteren Drittel.
- Folgt beim Driften eher der **Fahrtrichtung** als der Nase -> man sieht das Auto schräg.
- **Neigt sich mit der Fahrbahn** (Steilkurven: Horizont schief).
- Im Ziel fährt die Kamera weiter/bleibt stehen, das Banner erscheint.

## 4. Grafik
- Low-Poly, flach schattiert, kräftige tiefe Sonne -> **lange, scharfe Schatten**.
- **Himmel:** kräftiges Blau mit weichen, hellen Schleierwolken.
- **Landschaft:** grüne, spitze Low-Poly-Berge am Horizont, flaches grünes Tal tief unter der Strecke.
- **Architektur:** grosse kantige Bauten, fast schwarz/dunkelgrau, mit **blauen Akzentstreifen** und
  blauen Dächern; einzelne **dunkelrote Blöcke**; Tunnel aus schrägen blauen Felsplatten; graue
  **L-Portale** in Reihe über/neben der Strecke; riesige Überführungen; Tribünen mit Dächern.
- **Fahrbahn:** mittelgrau, glatt, sehr breit. Rand: **schmaler rot-weisser Streifen**, dahinter eine
  **niedrige dunkle Mauer/Kante**. Strecken liegen auf mehreren Ebenen (Brücken, Teile darunter).
- **Start:** orange Linien auf dem Asphalt (Startbox), Portal mit gelb-schwarzen Pfeilschildern.
- **Checkpoint:** gelbe Linie quer über die Fahrbahn + gelber Balken auf zwei dünnen schwarzen Pfeilern
  (teilweise gelb-schwarzes Pfeilbanner).
- **Ziel:** roter Balken auf schwarzen Pfeilern, Schachbrettlinie.
- **Kurvenwarnung:** Wände mit gelb-schwarzen Pfeil-Schildern.

## 5. Bildschirm / Bedienung
- Oben links: "Exit", "Watch", "Next Track"; Streckenname, eigene Rekordzeit, Rang.
- Unten eine Leiste: links Checkpoints ("2/3"), Mitte "Record | Current | Difference", rechts km/h.
- **Beim Checkpoint:** über jedem Feld die Zwischenzeit (Rekord-Zwischenzeit, eigene, Differenz grün/rot),
  über dem Tempo der Tempo-Unterschied zum Rekord an dieser Stelle (z.B. "−5 km/h" rot, "+40 km/h" grün).
- **Geist** der Rekordfahrt fährt halbdurchsichtig mit.
- Zeit mit Tausendstel. Neustart mit T/Backspace. Ziel-Banner: Streckenname + Zeit auf dunklen Balken.
- Strecken haben 2–8 Checkpoints, Fahrzeiten ~27–60 s.

## 6. Umsetzung in unserem Spiel (eigene Gestaltung) – Version 2.1
| Beobachtung | Umsetzung bei uns |
|---|---|
| 0→155 km/h in 2,5 s, Spitze ~380 km/h | Antrieb/Luftwiderstand neu abgestimmt, gleiche Kurve angepeilt |
| Driften mit Spuren und Rauch | Reifenspuren + Rauch beim Rutschen (Anzeige) |
| Steilkurven, Kamera neigt sich | Baustein-Option "Überhöhung" (Grad), Physik nutzt Querneigung; Kamera rollt mit |
| Kamera folgt Fahrtrichtung | Kamera zwischen Nase und Fahrtrichtung gemischt |
| Überschlag beim Runterfallen | Auto dreht sich in der Luft (Anzeige) |
| scharfe, lange Schatten | Schatten-Bereich folgt dem Auto, tiefe Sonne |
| dunkle Architektur, blaue Akzente, rote Blöcke, L-Portale | eigene Kulisse dieser Art, Portale über Geraden |
| mittelgraue Fahrbahn, schmaler rot-weisser Rand, dunkle Kante | Belag/Rand neu |
| Checkpoint gelbe Linie + gelber Balken, Ziel rot | Tore neu |
| Zwischenzeiten über der Leiste, Tempo-Differenz, Geist | HUD erweitert |
| längere Strecken für hohes Tempo | drei neue, längere eigene Strecken |

## 7. Sitzung 3 (2026-10-18, 19:42–19:48, 12 Bilder): Kante, Streckenbreite, Sprünge
Strecken: Summer 5 (8 Checkpoints, 1:39.7 und dann 1:00.7), eine Eis-Strecke (5 Checkpoints, weiss/hellblau).

### Kante (Streckenrand)
- Aufbau von innen nach aussen: **aufgemalter rot-weisser Streifen** flach auf dem Belag (~eine halbe
  Autobreite), dann eine **niedrige, abgerundete Kante** (dunkel, auf der Eis-Strecke weiss), etwa
  **halb so hoch wie ein Rad**, schmal. Dahinter fällt es ab, oder es kommt Gras/Wand.
- Keine breite Rampe, sondern eine kleine Stufe. (Benutzer: langsam hält sie einen auf der Strecke,
  zu schnell spickt sie einen drüber.)
- Gibt es auch ganz ohne Kante (Eis-Strecke: Band ohne Rand, Teile mit Wänden).

### Streckenbreite
- Stark wechselnd: Start ~5–6 Autobreiten; **grosse offene Kurven gut 10 Autobreiten** breit
  (Ideallinie entscheidet); **schmale Schluchten** ~4 Autobreiten zwischen hohen Wänden, die man voll fährt.

### Physik
- Am Ende von Gefällstrecken **438–449 km/h** (über dem Geraden-Höchsttempo ~380).
- **Fliegen:** auf einer geschwungenen Bahn bei 344 km/h deutlich in der Luft (kein Schatten unter dem Auto).
  Steile Anstiege vor Kuppen -> Absprung.
- **Wand:** seitlich gegen eine Wand bei 199 km/h -> Auto kippt stark (~70°) und fährt weiter.
- **Gewölbte Wände (Halfpipe):** Auto fährt an fast senkrechten, gewölbten Flächen, rutscht dabei quer
  mit viel Rauch und dicken Spuren.
- **Langsam an eine Rundung:** Auto bäumt sich auf (Vorderräder in der Luft).

### Umsetzung (eigene Gestaltung)
- Kante neu: flacher rot-weisser Streifen + niedrige runde Stufe (~0.4 m). Langsam quer hinein -> prallt ab,
  schnell quer hinein -> hüpft drüber.
- Breite pro Baustein wählbar, mit weichen Übergängen (breite Kurven, enge Durchfahrten).
- Abheben über Kuppen auch ohne Rampe (bisher klebte das Auto zu stark am Boden).
