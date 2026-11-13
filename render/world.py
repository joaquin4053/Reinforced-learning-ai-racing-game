"""Ursina-Szene (Spiel v2): schwebende Fahrbahn mit Randwulst, Landschaft, Kulisse, Tore, Auto, Geister.

Rechnet KEINE Physik: bekommt nur Zustände (x, y, z, heading, pitch, roll) zum Anzeigen.
Sim-Koordinate (x, y, z) -> Ursina Vec3(x, y, z); heading -> rotation_y in Grad.

Eigener Low-Poly-Look (Notizen: docs/beobachtung_polytrack.md): flach schattiert, kräftige Sonne,
Wolkenhimmel, grüne Hügel unter der Strecke, grosse kantige Kulissenblöcke, rot-weisser Randwulst.
Alles hier ist selbst gezeichnet, nichts aus einem anderen Spiel übernommen.
"""
import math

import numpy as np
from PIL import Image, ImageFilter
from ursina import AmbientLight, DirectionalLight, Entity, Mesh, Texture, Vec3, camera, color, scene
from ursina.shaders import lit_with_shadows_shader, unlit_shader

from game.sensors import RAY_ANGLES, RAY_RANGE, ray_distances
from game.track import CURB_WIDTH, KANTE_WIDTH, curb_profile
from render.hud import TextQuad

THICK = 1.4                 # m Dicke der Fahrbahnblöcke
BLOCK = 8.0                 # m: Farbwechsel des Belags (Blockmuster)
STRIPE = 4.0                # m Länge eines Randstreifens (rot/weiss)
STEP = 5                    # jeder wievielte Abtastpunkt der Strecke wird gezeichnet (0.5 m)
JOINT = 4.0                 # m: Fugen quer über den Belag -> man sieht den Boden vorbeiziehen (Tempogefühl)
DASH_EVERY, DASH_LEN = 12.0, 4.0   # m: gestrichelte Fahrbahnlinien
JOINT_COLOR = color.rgb32(66, 70, 80)
DASH_COLOR = color.rgb32(170, 174, 182)
CURB_SEGMENTS = 4           # so viele Teilflächen hat der gerundete Randwulst

ROAD = [color.rgb32(84, 89, 100), color.rgb32(80, 85, 96)]          # mittelgrau, dezentes Blockmuster
CURB = [color.rgb32(200, 200, 206), color.rgb32(176, 30, 28)]               # Streckenrand: rot-weiss gestreift
LIP = color.rgb32(150, 154, 162)                                            # ohne Rand: helle Abbruchlinie
KANTE = color.rgb32(28, 30, 36)                                             # Kante: dunkel, rund
SIDE = color.rgb32(34, 36, 44)                                              # fast schwarze Blockseiten ...
ACCENT = color.rgb32(52, 78, 206)                                           # ... mit blauem Zierband
BOTTOM = color.rgb32(24, 26, 32)
ACCENT_BAND = 0.35          # m Höhe des blauen Bands an der Oberkante der Blockseiten
SKY_TOP, SKY_HORIZON = (48, 120, 226), (190, 222, 252)
GRASS = [(62, 128, 70), (54, 116, 62), (72, 138, 78), (48, 104, 58)]
VALLEY = [(70, 124, 70), (66, 118, 66)]
BUILDING = [(30, 32, 40), (40, 42, 52), (52, 54, 64)]                      # Kulisse: dunkle Blöcke
RED_BLOCK = (96, 30, 30)
PORTAL = (78, 82, 92)


# ---------- Texturen ----------
def _checker_texture(n=8, px=16):
    a = (np.indices((n * px, n * px)) // px).sum(axis=0) % 2
    img = np.where(a[..., None] == 1, 245, 20).astype(np.uint8).repeat(3, axis=2)
    return Texture(Image.fromarray(img, "RGB"))


def _hazard_texture(px=128):
    """Gelb-schwarze Warnstreifen (schräg) für Checkpoint-Tore."""
    y, x = np.indices((px // 4, px))
    stripe = ((x + y) // (px // 8)) % 2 == 0
    img = np.where(stripe[..., None], (250, 196, 30), (25, 25, 28)).astype(np.uint8)
    return Texture(Image.fromarray(img, "RGB"))


def _stripes_texture(px=128, colors=((236, 236, 240), (214, 46, 40))):
    x = np.indices((px // 4, px))[1]
    img = np.where(((x // (px // 8)) % 2 == 0)[..., None], colors[0], colors[1]).astype(np.uint8)
    return Texture(Image.fromarray(img, "RGB"))


def _sky_texture(w=1024, h=512, seed=3):
    """Himmel: Verlauf (Horizont hell, oben blau) + weiche Wolken aus geglättetem Rauschen."""
    t = np.linspace(0, 1, h)[:, None, None]                  # 0 = oben, 1 = unten (Kugel-UV)
    up = np.clip((0.5 - t) * 2, 0, 1)                         # 1 = Zenit, 0 = Horizont und darunter
    base = np.array(SKY_HORIZON) * (1 - up ** 0.6) + np.array(SKY_TOP) * up ** 0.6
    base = np.broadcast_to(base, (h, w, 3)).copy()
    rng = np.random.default_rng(seed)
    noise = Image.fromarray((rng.random((h // 8, w // 8)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    noise = np.asarray(noise.filter(ImageFilter.GaussianBlur(10)), dtype=np.float64) / 255
    band = np.exp(-((t[..., 0] - 0.36) / 0.12) ** 2)          # Wolken vor allem etwas über dem Horizont
    cloud = np.clip((noise - 0.5) * 4, 0, 1) * band
    img = base * (1 - cloud[..., None] * 0.85) + 255 * cloud[..., None] * 0.85
    return Texture(Image.fromarray(img.astype(np.uint8), "RGB"))


# ---------- Mesh-Hilfe ----------
class _MeshBuilder:
    def __init__(self):
        self.verts, self.cols, self.norms, self.tris = [], [], [], []

    def quad(self, p0, p1, p2, p3, n, col):
        k = len(self.verts)
        self.verts.extend([p0, p1, p2, p3])
        self.cols.extend([col] * 4)
        self.norms.extend([n] * 4)
        self.tris.extend([k, k + 2, k + 1, k, k + 3, k + 2])   # Reihenfolge = Vorderseite (Ursina)

    def tri(self, p0, p1, p2, col):
        k = len(self.verts)
        n = (p1 - p0).cross(p2 - p0).normalized()
        if n.y < 0:
            p1, p2, n = p2, p1, -n
        self.verts.extend([p0, p1, p2])
        self.cols.extend([col] * 3)
        self.norms.extend([n] * 3)
        self.tris.extend([k, k + 2, k + 1])

    def mesh(self):
        return Mesh(vertices=self.verts, triangles=self.tris, colors=self.cols, normals=self.norms)


def _runs(road):
    """Zusammenhängende Fahrbahnstücke als (Anfang, Ende) der Abtastindizes."""
    runs, start = [], None
    for i, on in enumerate(road):
        if on and start is None:
            start = i
        if not on and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(road) - 1))
    return runs


# ---------- Fahrbahn ----------
def _lift_pt(x, y, z, rx, rz, tl, k, lat, dy):
    """Punkt auf dem flachen Teil des Belags (für Fugen und Linien)."""
    return Vec3(x[k] + rx[k] * lat, y[k] + tl[k] * lat + dy, z[k] + rz[k] * lat)


def road_mesh(track):
    """Alle Fahrbahnblöcke: Belag (Blockmuster, Fugen, Linien), Rand (rot-weisser Streifen + dunkle runde
    Kante), Seiten mit blauem Band, Unterseite. Breite und Rand an jeder Stelle wie in der Strecke."""
    p = track.samples
    cum = track._sample_cum - track._dist_offset
    b = _MeshBuilder()
    # Querschnitt als Abstände von der Aussenkante: Kante (3 Teilflächen, rund), Streifen, dann Belag
    edge_off = np.array([0.0, 0.08, 0.18, KANTE_WIDTH, CURB_WIDTH])
    for a, e in _runs(p["road"]):
        idx = list(range(a, e + 1, STEP))
        if idx[-1] != e:
            idx.append(e)
        x, y, z, h, tl = p["x"][idx], p["y"][idx], p["z"][idx], p["heading"][idx], p["tilt"][idx]
        bd, half = p["border"][idx], p["width"][idx] / 2
        rx, rz = np.cos(h), -np.sin(h)
        lats = np.concatenate([-(half[:, None] - edge_off[None, :]),
                               (half[:, None] - edge_off[None, ::-1])], axis=1)          # (Punkt, Querschnitt)
        lift = curb_profile(lats, 2 * half[:, None], bd[:, None])
        n_edge = len(edge_off) - 1
        last = lats.shape[1] - 1

        def pt(k, j, dy=0.0):
            return Vec3(x[k] + rx[k] * lats[k, j], y[k] + tl[k] * lats[k, j] + lift[k, j] + dy,
                        z[k] + rz[k] * lats[k, j])

        def flat(k, lat, dy):
            return Vec3(x[k] + rx[k] * lat, y[k] + tl[k] * lat + dy, z[k] + rz[k] * lat)

        def bottom(k, side):
            lat = side * half[k]
            return Vec3(x[k] + rx[k] * lat, y[k] + tl[k] * lat - THICK, z[k] + rz[k] * lat)

        for k in range(len(idx) - 1):
            s = cum[idx[k]]
            for j in range(last):
                p0, p1, p2, p3 = pt(k, j), pt(k + 1, j), pt(k + 1, j + 1), pt(k, j + 1)
                n = (p1 - p0).cross(p3 - p0).normalized()
                if n.y < 0:
                    n = -n
                side_j = j if j < n_edge else last - 1 - j          # 0 = ganz aussen ... n_edge = Belag
                if side_j < n_edge - 1:                             # Kante (bzw. ohne Rand: helle Abbruchlinie)
                    col = KANTE if bd[k] else (LIP if side_j == 0 else ROAD[int(s // BLOCK) % 2])
                elif side_j == n_edge - 1:                          # rot-weisser Streifen
                    col = CURB[int(s // STRIPE) % 2] if bd[k] else ROAD[int(s // BLOCK) % 2]
                else:
                    col = ROAD[int(s // BLOCK) % 2]
                b.quad(p0, p1, p2, p3, n, col)
            s1 = cum[idx[k + 1]]
            inner_k, inner_k1 = half[k] - CURB_WIDTH, half[k + 1] - CURB_WIDTH
            if int(s // JOINT) != int(s1 // JOINT):                 # Fuge quer über den Belag (alle JOINT m)
                b.quad(flat(k, -inner_k, 0.02), flat(k + 1, -inner_k1, 0.02), flat(k + 1, inner_k1, 0.02),
                       flat(k, inner_k, 0.02), Vec3(0, 1, 0), JOINT_COLOR)
            if (s % DASH_EVERY) < DASH_LEN:                          # gestrichelte Fahrbahnlinien
                for f in (-1 / 3, 1 / 3):
                    la, lb = half[k] * f, half[k + 1] * f
                    b.quad(flat(k, la - 0.15, 0.03), flat(k + 1, lb - 0.15, 0.03), flat(k + 1, lb + 0.15, 0.03),
                           flat(k, la + 0.15, 0.03), Vec3(0, 1, 0), DASH_COLOR)
            nl = Vec3(-rx[k], 0, -rz[k])
            band = Vec3(0, -ACCENT_BAND, 0)
            b.quad(bottom(k, -1), bottom(k + 1, -1), pt(k + 1, 0) + band, pt(k, 0) + band, nl, SIDE)
            b.quad(pt(k, 0) + band, pt(k + 1, 0) + band, pt(k + 1, 0), pt(k, 0), nl, ACCENT)
            b.quad(pt(k, last) + band, pt(k + 1, last) + band, bottom(k + 1, 1), bottom(k, 1), -nl, SIDE)
            b.quad(pt(k, last), pt(k + 1, last), pt(k + 1, last) + band, pt(k, last) + band, -nl, ACCENT)
            b.quad(bottom(k, 1), bottom(k + 1, 1), bottom(k + 1, -1), bottom(k, -1), Vec3(0, -1, 0), BOTTOM)
        for k, sign in ((0, -1), (len(idx) - 1, 1)):           # Stirnseiten
            n = Vec3(math.sin(h[k]) * sign, 0, math.cos(h[k]) * sign)
            q = [pt(k, 0), pt(k, last), bottom(k, 1), bottom(k, -1)]
            b.quad(*(q if sign > 0 else q[::-1]), n, SIDE)
    return b.mesh()


# ---------- Landschaft ----------
def _distance_to_track(track, xs, zs):
    c = track.center[::4]
    d = np.full(xs.shape, np.inf)
    for i in range(0, len(c), 200):                            # in Blöcken, damit der Speicher klein bleibt
        part = c[i:i + 200]
        dd = np.hypot(xs[..., None] - part[:, 0], zs[..., None] - part[:, 1]).min(axis=-1)
        d = np.minimum(d, dd)
    return d


def terrain_mesh(track, seed=11, cell=40.0, margin=1400.0):
    """Landschaft: weites, fast flaches Tal tief unter der Strecke, aussen ein Ring aus spitzen Bergen."""
    lo = track.center.min(axis=0) - margin
    hi = track.center.max(axis=0) + margin
    nx, nz = int((hi[0] - lo[0]) / cell) + 1, int((hi[1] - lo[1]) / cell) + 1
    rng = np.random.default_rng(seed)
    gx, gz = np.meshgrid(lo[0] + np.arange(nx) * cell, lo[1] + np.arange(nz) * cell)
    gx = gx + rng.uniform(-0.35, 0.35, gx.shape) * cell
    gz = gz + rng.uniform(-0.35, 0.35, gz.shape) * cell
    floor = track.min_y - 45
    dist = _distance_to_track(track, gx, gz)
    ring = np.clip((dist - 500) / 500, 0, 1)                         # ab 500 m Abstand steigen die Berge an
    peaks = rng.random(gx.shape) ** 3                                # wenige hohe Spitzen, viele niedrige
    height = floor + rng.uniform(0, 3, gx.shape) + ring * (40 + peaks * 260)
    b = _MeshBuilder()
    for j in range(nz - 1):
        for i in range(nx - 1):
            v = [Vec3(gx[j + dj, i + di], height[j + dj, i + di], gz[j + dj, i + di])
                 for dj, di in ((0, 0), (0, 1), (1, 1), (1, 0))]
            mountain = ring[j, i] > 0.05
            pal = GRASS if mountain else VALLEY
            b.tri(v[0], v[1], v[2], color.rgb32(*pal[(i * 7 + j * 3) % len(pal)]))
            b.tri(v[0], v[2], v[3], color.rgb32(*pal[(i * 5 + j * 11 + 1) % len(pal)]))
    return b.mesh()


def _block(x, y0, y1, z, w, d, rot, col, accent=True):
    """Kantiger Bau: dunkler Block, oben ein blaues Band (eigene Gestaltung)."""
    Entity(model="cube", color=color.rgb32(*col), position=(x, (y0 + y1) / 2, z), scale=(w, y1 - y0, d),
           rotation_y=rot)
    if accent:
        Entity(model="cube", color=ACCENT, position=(x, y1 - 1.2, z), scale=(w + 0.3, 1.0, d + 0.3), rotation_y=rot)


def scenery(track, seed=5, count=45):
    """Kulisse neben der Strecke (keine Kollision): dunkle Bauten mit blauem Band, einzelne rote Blöcke."""
    rng = np.random.default_rng(seed)
    lo = track.center.min(axis=0) - 250
    hi = track.center.max(axis=0) + 250
    placed = 0
    for _ in range(count * 30):
        if placed >= count:
            break
        x, z = rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])
        w, dpt = rng.uniform(14, 50), rng.uniform(14, 50)
        d = float(_distance_to_track(track, np.array(x), np.array(z)))
        reach = math.hypot(w, dpt) / 2
        if d < track.width / 2 + reach + 14 or d > 220:
            continue
        top = track.min_y + rng.uniform(-5, 35)
        rot = float(rng.uniform(0, 90))
        if rng.random() < 0.18:
            _block(x, track.min_y - 45, top, z, w * 0.6, dpt * 0.6, rot, RED_BLOCK, accent=False)
        else:
            _block(x, track.min_y - 45, top, z, w, dpt, rot, BUILDING[rng.integers(len(BUILDING))])
        placed += 1


def portals(track, every=36.0, min_straight=150.0):
    """Rahmen-Portale über langen Geraden (eigene Gestaltung): zwei Stützen + Querträger, weit über der Fahrbahn."""
    p = track.samples
    cum = track._sample_cum - track._dist_offset
    straight = np.abs(np.diff(p["heading"], prepend=p["heading"][0])) < 1e-6
    runs, start = [], None
    for i, ok in enumerate(straight & p["road"] & (p["tilt"] == 0)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i - 1))
            start = None
    col = color.rgb32(*PORTAL)
    for a, e in runs:
        if cum[e] - cum[a] < min_straight:
            continue
        for s in np.arange(cum[a] + 30, cum[e] - 30, every)[:6]:   # höchstens 6 Portale pro Gerade
            i = int(np.searchsorted(cum, s))
            x, y, z, h = p["x"][i], p["y"][i], p["z"][i], p["heading"][i]
            root = Entity(position=(x, y, z), rotation_y=math.degrees(h))
            half = p["width"][i] / 2 + 2.5
            for side in (-1, 1):
                Entity(parent=root, model="cube", color=col, position=(side * half, 4.5, 0), scale=(1.6, 13, 1.6))
            Entity(parent=root, model="cube", color=col, position=(0, 11.5, 0), scale=(2 * half + 1.6, 1.6, 1.6))


# ---------- Tore ----------
def _gate(gate, width, bar_color, height=7.0, line_texture=None, line_color=None):
    """Tor: zwei dünne schwarze Pfeiler, farbiger Balken oben, Linie quer über die Fahrbahn."""
    t = gate.tangent
    root = Entity(position=Vec3(gate.center[0], gate.y, gate.center[1]),
                  rotation_y=math.degrees(math.atan2(t[0], t[1])))
    for side in (-1, 1):
        Entity(parent=root, model="cube", color=color.rgb32(20, 20, 24),
               position=(side * (width / 2 + 0.5), height / 2, 0), scale=(0.45, height, 0.45))
    Entity(parent=root, model="cube", color=bar_color, position=(0, height, 0), scale=(width + 1.6, 1.0, 0.5),
           shader=unlit_shader)
    if line_texture is not None or line_color is not None:
        Entity(parent=root, model="plane", texture=line_texture, color=line_color or color.white, shader=unlit_shader,
               position=(0, 0.05, 0), scale=(width - 2 * CURB_WIDTH, 1, 0.6 if line_texture is None else 3),
               texture_scale=(width / 3, 1))
    return root


def _start_box(gate, width):
    """Orange Startbox auf dem Asphalt."""
    t = gate.tangent
    root = Entity(position=Vec3(gate.center[0], gate.y + 0.05, gate.center[1]),
                  rotation_y=math.degrees(math.atan2(t[0], t[1])))
    orange = color.rgb32(255, 140, 20)
    for zz in (0.0, 9.0):
        Entity(parent=root, model="plane", color=orange, shader=unlit_shader, position=(0, 0, zz), scale=(5, 1, 0.3))
    for xx in (-2.5, 2.5):
        Entity(parent=root, model="plane", color=orange, shader=unlit_shader, position=(xx, 0, 4.5), scale=(0.3, 1, 9))


# ---------- Autos ----------
def _wheel_mesh(radius, width, n, tread, side, rim):
    """Rad als eigene Low-Poly-Walze entlang der x-Achse (Mitte = Radnabe): Lauffläche, Seitenwände,
    abgesetzte Felge. (Ursinas Cylinder verschiebt den Start immer entlang y -> schiefe Reifen.)"""
    b = _MeshBuilder()
    hw = width / 2
    ang = [2 * math.pi * k / n for k in range(n + 1)]
    ring = [(math.cos(a), math.sin(a)) for a in ang]
    for k in range(n):
        (c0, s0), (c1, s1) = ring[k], ring[k + 1]
        nrm = Vec3(0, (s0 + s1) / 2, (c0 + c1) / 2).normalized()
        b.quad(Vec3(-hw, s0 * radius, c0 * radius), Vec3(-hw, s1 * radius, c1 * radius),
               Vec3(hw, s1 * radius, c1 * radius), Vec3(hw, s0 * radius, c0 * radius), nrm, tread)
        for sx in (-1, 1):                                       # Seitenwand (Ring) und Felge (innen)
            x = sx * hw
            r_in = radius * 0.62
            q = [Vec3(x, s0 * radius, c0 * radius), Vec3(x, s0 * r_in, c0 * r_in),
                 Vec3(x, s1 * r_in, c1 * r_in), Vec3(x, s1 * radius, c1 * radius)]
            b.quad(*(q if sx > 0 else q[::-1]), Vec3(sx, 0, 0), side)
            xr = sx * (hw - 0.06)                                  # Felge leicht nach innen versetzt
            q = [Vec3(xr, 0, 0), Vec3(xr, s0 * r_in, c0 * r_in), Vec3(xr, s1 * r_in, c1 * r_in), Vec3(xr, 0, 0)]
            b.quad(*(q if sx > 0 else q[::-1]), Vec3(sx, 0, 0), rim)
            q = [Vec3(x, s0 * r_in, c0 * r_in), Vec3(xr, s0 * r_in, c0 * r_in),
                 Vec3(xr, s1 * r_in, c1 * r_in), Vec3(x, s1 * r_in, c1 * r_in)]
            b.quad(*(q if sx > 0 else q[::-1]), Vec3(0, -(s0 + s1) / 2, -(c0 + c1) / 2), side)
    return b.mesh()


class _Wheel(Entity):
    """Rundes Rad mit Felge und Speichen. Dreht sich um die Achse, Vorderräder lenken."""

    def __init__(self, parent, x, z, radius, width, alpha, steerable):
        super().__init__(parent=parent, position=(x, radius, z))
        self.steerable = steerable
        self.spin = Entity(parent=self)
        tread = color.rgba(0.08, 0.08, 0.09, alpha)
        side = color.rgba(0.14, 0.14, 0.16, alpha)
        rim = color.rgba(0.74, 0.76, 0.8, alpha)
        hub = color.rgba(0.3, 0.31, 0.35, alpha)
        Entity(parent=self.spin, model=_wheel_mesh(radius, width, 14, tread, side, rim), double_sided=True)
        out = 1 if x > 0 else -1
        for k in range(5):                                           # Speichen auf der Aussenseite
            Entity(parent=self.spin, model="cube", color=hub, x=out * (width / 2 - 0.05),
                   rotation_x=k * 36, scale=(0.04, radius * 1.2, 0.08))
        Entity(parent=self.spin, model="cube", color=hub, x=out * (width / 2 - 0.03), scale=(0.06, 0.16, 0.16))


class CarModel(Entity):
    """Eigener kompakter Low-Poly-Buggy (eigene Gestaltung, Proportionen nach docs/beobachtung_polytrack.md:
    kleines Auto, grosse Räder fast an den Ecken): kurze Wanne mit runder Front, offenes Cockpit mit Fahrer,
    Überrollbügel, kleiner Heckflügel, breite runde Reifen an sichtbaren Querlenkern.
    Räder drehen sich mit dem Tempo, Vorderräder lenken."""

    WHEEL_R = 0.56
    WHEEL_W = 0.58
    TRACK = 1.05          # halbe Spurweite (Radmitte)
    FRONT, REAR = 1.18, -1.22

    def __init__(self, body_color=color.rgb32(255, 122, 16), alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.body = Entity(parent=self)

        def c(col):
            return color.rgba(col.r, col.g, col.b, alpha)

        def part(col, **kw):
            return Entity(parent=self.body, model="cube", color=c(col), **kw)

        dark, black, white = color.rgb32(36, 38, 46), color.rgb32(16, 16, 18), color.rgb32(238, 238, 240)
        accent = body_color.tint(-.3)
        y0 = 0.5                                                    # Unterkante der Wanne (hoher Stand)
        part(dark, scale=(0.95, 0.12, 2.6), y=y0)                   # Bodenplatte
        part(body_color, scale=(1.3, 0.5, 2.0), y=y0 + 0.3)         # Wanne (kompakt)
        part(body_color, scale=(1.1, 0.38, 0.6), y=y0 + 0.24, z=1.25)            # Front
        part(accent, scale=(1.16, 0.12, 0.62), y=y0 + 0.06, z=1.27)              # Frontschürze
        part(white, scale=(0.3, 0.02, 2.62), y=y0 + 0.56)                        # Mittelstreifen
        for sx in (-1, 1):
            part(accent, scale=(0.08, 0.36, 1.6), x=sx * 0.67, y=y0 + 0.3, z=-0.1)   # Seitenleisten
        part(black, scale=(0.82, 0.12, 0.8), y=y0 + 0.57, z=-0.1)                 # Cockpit-Öffnung
        helmet = color.rgb32(40, 60, 150)
        Entity(parent=self.body, model="sphere", color=c(helmet), scale=(0.42, 0.42, 0.46), position=(0, y0 + 0.84, -0.15))
        part(white, scale=(0.07, 0.03, 0.48), y=y0 + 1.06, z=-0.15)
        part(color.rgb32(20, 24, 34), scale=(0.44, 0.1, 0.3), y=y0 + 0.86, z=-0.05)  # Visier
        for sx in (-0.45, 0.45):                                    # Überrollbügel
            part(dark, scale=(0.08, 0.66, 0.08), x=sx, y=y0 + 0.88, z=-0.6)
        part(dark, scale=(0.98, 0.08, 0.08), y=y0 + 1.2, z=-0.6)
        part(accent, scale=(1.0, 0.44, 0.55), y=y0 + 0.42, z=-1.05)               # Motorhaube hinten
        for sx in (-0.32, 0.32):                                    # kleiner Heckflügel
            part(dark, scale=(0.06, 0.4, 0.14), x=sx, y=y0 + 0.82, z=-1.25)
        part(black, scale=(1.6, 0.07, 0.42), y=y0 + 1.04, z=-1.3)
        for sx in (-1, 1):
            part(body_color, scale=(0.05, 0.28, 0.46), x=sx * 0.8, y=y0 + 0.98, z=-1.3)
        part(color.rgb32(255, 245, 210), scale=(0.7, 0.08, 0.04), y=y0 + 0.32, z=1.56, shader=unlit_shader)
        part(color.rgb32(255, 40, 40), scale=(0.7, 0.1, 0.04), y=y0 + 0.4, z=-1.34, shader=unlit_shader)

        self.wheels = []
        for z, steer in ((self.FRONT, True), (self.REAR, False)):
            for sx in (-1, 1):
                self.wheels.append(_Wheel(self.body, sx * self.TRACK, z, self.WHEEL_R, self.WHEEL_W, alpha, steer))
                for dy in (0.1, -0.1):                              # Querlenker oben/unten
                    part(dark, scale=(self.TRACK - 0.55, 0.05, 0.08), x=sx * (self.TRACK / 2 + 0.2),
                         y=self.WHEEL_R + dy, z=z)
        self.front_wheels = [w for w in self.wheels if w.steerable]
        self._spin = 0.0
        self._last = None

    def set_state(self, x, y, z, heading, pitch=0.0, roll=0.0, steer=0.0):
        if self._last is not None:                                  # Räder drehen: zurückgelegter Weg / Radius
            lx, lz = self._last
            d = (x - lx) * math.sin(heading) + (z - lz) * math.cos(heading)
            if abs(d) < 20:
                self._spin = (self._spin + math.degrees(d / self.WHEEL_R)) % 360
        self._last = (x, z)
        self.position = Vec3(x, y, z)
        self.rotation_y = math.degrees(heading)
        self.body.rotation_x = -math.degrees(pitch)
        self.body.rotation_z = math.degrees(roll)       # Ursina: + = rechte Seite tiefer (geprüft im Bild)
        for w in self.wheels:
            w.spin.rotation_x = self._spin
            if w.steerable:
                w.rotation_y = steer * 28


def make_car(**kwargs):
    """Auto aus der 3D-Datei (render/car_model.py), sonst der eingebaute Low-Poly-Buggy."""
    from render import car_model
    if car_model.available():
        return car_model.ModelCar(**kwargs)
    return CarModel(**kwargs)


def lerp_pose(a, b, t):
    """Zwischen zwei Posen (x, y, z, heading, pitch, roll, steer) überblenden, t = 0..1."""
    out = [a[i] + (b[i] - a[i]) * t for i in range(len(a))]
    out[3] = a[3] + ((b[3] - a[3] + math.pi) % (2 * math.pi) - math.pi) * t
    return tuple(out)


class GhostCar:
    """Halb durchsichtiges Auto eines Geists mit Namensschild. Keine Kollision (nur Anzeige)."""

    COLORS = [color.rgb32(60, 200, 255), color.rgb32(190, 110, 255), color.rgb32(120, 255, 140), color.rgb32(255, 200, 60)]

    def __init__(self, ghost, index=0):
        self.ghost = ghost
        col = self.color = self.COLORS[index % len(self.COLORS)]
        self.model = make_car(body_color=col, alpha=0.45)
        self.model.hide(0b0001)
        self.anchor = Entity(parent=self.model, y=3.0)
        self.label = TextQuad(parent=camera.ui, height=0.03, origin=(0, -.5), heavy=True)
        self.reset()

    def reset(self):
        self.crashed = False
        self.label.set(self.ghost.label + ("" if self.ghost.compatible else "  (alte Version)"), (255, 255, 255))

    def show_frame(self, frame):
        f0 = math.floor(frame)
        pose = lerp_pose((*self.ghost.state_at(f0), 0), (*self.ghost.state_at(f0 + 1), 0), frame - f0)
        self.model.set_state(*pose)
        if frame >= self.ghost.last_frame and self.ghost.meta["result"] == "crashed" and not self.crashed:
            self.crashed = True
            self.label.set(f"{self.ghost.label} · Crash {self.ghost.meta['progress_pct']:.0f} %", (255, 120, 100))
        to_ghost = self.anchor.world_position - camera.world_position
        self.label.enabled = to_ghost.dot(camera.forward) > 0 and to_ghost.length() > 6 and bool(self.label.text)
        if self.label.enabled:
            p = self.anchor.screen_position
            self.label.position = Vec3(p.x, p.y, 0)


class RayViz(Entity):
    """Sensorstrahlen der KI als Balken mit Trefferpunkt (rot = Kante nah, grün = frei)."""

    def __init__(self, track, **kwargs):
        super().__init__(**kwargs)
        self.track = track
        self.beams = [Entity(parent=self, model="cube", shader=unlit_shader) for _ in RAY_ANGLES]
        self.dots = [Entity(parent=self, model="cube", shader=unlit_shader, scale=0.45) for _ in RAY_ANGLES]

    def set_distances(self, x, y, z, heading, distances):
        y += 0.9
        for beam, dot, a, dist in zip(self.beams, self.dots, heading + RAY_ANGLES, distances):
            k = min(1.0, dist / 40.0)
            c = color.rgba(1 - k, 0.25 + 0.75 * k, 0.2, 0.8)
            sx, sz = math.sin(a), math.cos(a)
            beam.position = Vec3(x + sx * dist / 2, y, z + sz * dist / 2)
            beam.rotation_y = math.degrees(a)
            beam.scale = Vec3(0.08, 0.08, max(dist, 0.01))
            beam.color = c
            dot.position = Vec3(x + sx * dist, y, z + sz * dist)
            dot.color = c
            dot.visible = dist < RAY_RANGE

    def set_state(self, x, y, z, heading):
        """Strahlen aus einer Position neu berechnen (gleiche Funktion wie in der Sim)."""
        self.set_distances(x, y, z, heading, ray_distances(x, z, y, heading, self.track))


# ---------- Welt ----------
class World:
    SHADOW_SIZE = 140          # m: Schatten werden in diesem Bereich um das Auto berechnet (dafür scharf)

    def __init__(self, track, shadows=True):
        self.track = track
        mid = (track.center.min(axis=0) + track.center.max(axis=0)) / 2
        if shadows:
            Entity.default_shader = lit_with_shadows_shader

        Entity(model="sphere", texture=_sky_texture(), scale=9000, double_sided=True, shader=unlit_shader,
               position=(mid[0], track.min_y, mid[1]))
        Entity(model=terrain_mesh(track))
        scenery(track)
        portals(track)
        from render import deko
        deko.build(track)                                 # Kulisse aus der Strecken-Datei (falls vorhanden)
        Entity(model=road_mesh(track))

        _gate(track.start_gate, track.start_gate.width, color.rgb32(255, 140, 20), height=7.5)
        _start_box(track.start_gate, track.start_gate.width)
        _gate(track.finish_gate, track.finish_gate.width, color.rgb32(210, 34, 30), height=8.0,
              line_texture=_checker_texture())
        for g in track.checkpoints:
            _gate(g, g.width, color.rgb32(255, 220, 30), line_color=color.rgb32(255, 220, 30))

        self.sun = DirectionalLight(shadow_map_resolution=(4096, 4096), shadows=shadows)
        self.sun.look_at(Vec3(0.55, -0.62, 0.35))                  # tiefe Sonne -> lange Schatten
        AmbientLight(color=color.rgba(0.3, 0.32, 0.4, 1))
        self.shadows = shadows
        if shadows:                                                 # Schattenbereich hängt an der Sonne,
            size = self.SHADOW_SIZE                                 # die Sonne folgt dem Auto -> immer scharf
            self._shadow_box = Entity(parent=self.sun, model="cube", visible=False,
                                      scale=(size, size, size * 2))
            self.sun.update_bounds(self._shadow_box)

        scene.fog_color = color.rgb32(*SKY_HORIZON)
        scene.fog_density = (1500, 7000)
        self.car = make_car()
        self.rays = RayViz(track)
        self.rays.enabled = False
        camera.clip_plane_far = 9000

    def follow(self, x, y, z):
        """Sonne (und damit der scharfe Schattenbereich) wandert mit dem Auto."""
        if self.shadows:
            self.sun.position = Vec3(x, y, z)
