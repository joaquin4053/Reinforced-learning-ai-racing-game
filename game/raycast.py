"""Strahlen und Kreise gegen Wandsegmente (vektorisiert mit numpy)."""
import numpy as np


def cast_rays(ox, oz, angles, walls, max_range):
    """Distanz vom Punkt (ox, oz) entlang jedes Winkels bis zur nächsten Wand.

    angles: (R,) absolute Winkel (gleiche Konvention wie heading)
    walls:  (S, 4) Segmente x1 z1 x2 z2
    """
    dx, dz = np.sin(angles)[:, None], np.cos(angles)[:, None]  # (R,1)
    ax, az = walls[:, 0][None, :], walls[:, 1][None, :]          # (1,S)
    ex, ez = (walls[:, 2] - walls[:, 0])[None, :], (walls[:, 3] - walls[:, 1])[None, :]
    wx, wz = ax - ox, az - oz
    denom = dx * ez - dz * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx * ez - wz * ex) / denom
        u = (wx * dz - wz * dx) / denom
    hit = (np.abs(denom) > 1e-9) & (t >= 0) & (u >= 0) & (u <= 1)
    t = np.where(hit, t, max_range)
    return np.minimum(t.min(axis=1), max_range)


def closest_wall(px, pz, walls):
    """Nächster Punkt auf irgendeiner Wand -> (distanz, nx, nz) mit Normale Wand->Punkt."""
    ax, az = walls[:, 0], walls[:, 1]
    ex, ez = walls[:, 2] - ax, walls[:, 3] - az
    t = np.clip(((px - ax) * ex + (pz - az) * ez) / (ex * ex + ez * ez), 0.0, 1.0)
    cx, cz = px - (ax + ex * t), pz - (az + ez * t)
    d2 = cx * cx + cz * cz
    i = int(np.argmin(d2))
    d = float(np.sqrt(d2[i]))
    if d < 1e-9:
        return 0.0, 0.0, 0.0
    return d, float(cx[i]) / d, float(cz[i]) / d
