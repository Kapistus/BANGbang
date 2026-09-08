"""Line of sight: recursive shadowcasting plus banded vision cones.

Visibility and the cone are deliberately separate concerns:

  * What a cell can SEE depends only on position. It is symmetric, exact, and
    expensive, so it is computed once per cell occupied and cached.
  * What an actor is LOOKING AT depends on facing, and is a cheap angular
    classification over the cached mask, recomputed every frame.

That split matters because actors turn far more often than they change cell,
and turning is the thing that happens every frame.

Cones are banded rather than uniform, because human vision is not. Roughly:
a narrow central band that identifies at range, a wider band that recognises
movement and shape at medium range, and a broad short-range band that
registers motion only. Plus a small all-round radius, since you cannot creep
up to someone's elbow unseen.

Band values in the returned array:
    0  not visible
    1  identify    narrow, long range
    2  recognise   medium
    3  peripheral  wide, short range, motion only

`bands` / `band_at` are the discrete, gameplay-facing classification (they
decide how an enemy is drawn to the player - full readout, motion blip, or
stale ghost). `cone_intensity` is the continuous version used only for
rendering the player's own view as a single cone that feathers at its edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# (xx, xy, yx, yy) transforms for the eight octants
_OCTANTS = (
    (1, 0, 0, 1), (0, 1, 1, 0), (0, -1, 1, 0), (-1, 0, 0, 1),
    (-1, 0, 0, -1), (0, -1, -1, 0), (0, 1, -1, 0), (1, 0, 0, -1),
)


@dataclass(frozen=True)
class ConeSpec:
    """Half-angles in degrees and ranges in cells, widest band last."""

    identify_deg: float = 44.0
    recognise_deg: float = 100.0
    peripheral_deg: float = 180.0
    identify_range: float = 56.0
    recognise_range: float = 36.0
    peripheral_range: float = 22.0
    near_range: float = 6.0

    @property
    def max_range(self) -> float:
        return max(self.identify_range, self.recognise_range,
                   self.peripheral_range, self.near_range)


class VisibilityCache:
    """Shadowcast masks keyed by (cell, radius). Cleared when geometry changes."""

    def __init__(self, blocks_sight: np.ndarray, limit: int = 64):
        self.blocks = blocks_sight
        self.limit = limit
        self._store: dict[tuple[int, int, int], "VisionField"] = {}

    def invalidate(self) -> None:
        self._store.clear()

    def get(self, cx: int, cy: int, radius: float) -> "VisionField":
        key = (cx, cy, int(radius))
        hit = self._store.get(key)
        if hit is not None:
            return hit
        vf = shadowcast(self.blocks, cx, cy, int(radius))
        if len(self._store) >= self.limit:
            self._store.pop(next(iter(self._store)))
        self._store[key] = vf
        return vf


@dataclass
class VisionField:
    """Facing-independent visibility around one cell, on a cropped window."""

    visible: np.ndarray     # bool (h, w)
    dist: np.ndarray        # float32, cells from origin
    ang: np.ndarray         # float32, radians, atan2(dy, dx)
    x0: int
    y0: int
    origin: tuple[int, int]
    radius: int

    def sees(self, cx: int, cy: int) -> bool:
        lx, ly = cx - self.x0, cy - self.y0
        h, w = self.visible.shape
        if not (0 <= ly < h and 0 <= lx < w):
            return False
        return bool(self.visible[ly, lx])

    def bands(self, facing: float, cone: ConeSpec) -> np.ndarray:
        """Classify visible cells into cone bands for a given facing.

        Cheap: a few vectorised comparisons over the cached window.
        """
        d = np.abs((self.ang - facing + math.pi) % (2 * math.pi) - math.pi)
        out = np.zeros(self.visible.shape, dtype=np.uint8)
        v = self.visible

        half_p = math.radians(cone.peripheral_deg) * 0.5
        fwd = v & (d <= half_p)                       # forward hemisphere only
        per = fwd & (self.dist <= cone.peripheral_range)
        out[per] = 3
        rec = fwd & (d <= math.radians(cone.recognise_deg) * 0.5) & (self.dist <= cone.recognise_range)
        out[rec] = 2
        near = fwd & (self.dist <= cone.near_range)
        out[near] = np.maximum(out[near], 2)
        ide = v & (d <= math.radians(cone.identify_deg) * 0.5) & (self.dist <= cone.identify_range)
        out[ide] = 1
        return out

    def cone_intensity(self, facing: float, cone: ConeSpec) -> np.ndarray:
        """Smooth per-cell clarity in [0, 1] across the view cone.

        Bright toward `facing`, feathering to zero at the peripheral edge and
        at range. Optional `near_range` gives a short forward-only floor.
        There is a hard cut at the peripheral half-angle - nothing behind the
        player. Unlike `bands` this has no steps: it renders as one cone that
        fades at its edges rather than labelled slices.

        The effective range shrinks off-axis: full `identify_range` straight
        ahead, down to `peripheral_range` at the edge.
        """
        d = np.abs((self.ang - facing + math.pi) % (2 * math.pi) - math.pi)
        half_p = max(math.radians(cone.peripheral_deg) * 0.5, 1e-6)
        k = np.clip(d / half_p, 0.0, 1.0)
        ang_t = 0.5 + 0.5 * np.cos(np.pi * k)                # 1 on axis -> 0 at edge
        reach = cone.identify_range + k * (cone.peripheral_range
                                           - cone.identify_range)
        rad_t = np.clip(1.0 - self.dist / np.maximum(reach, 1e-6), 0.0, 1.0)
        rad_t = rad_t * rad_t * (3.0 - 2.0 * rad_t)          # smoothstep
        near_t = np.clip((cone.near_range - self.dist)
                         / max(cone.near_range, 1e-6), 0.0, 1.0)
        inten = np.maximum(near_t * 0.9, ang_t * rad_t)
        inten[d >= half_p] = 0.0                             # nothing behind, ever
        return (self.visible * inten).astype(np.float32)

    def band_at(self, cx: int, cy: int, facing: float, cone: ConeSpec) -> int:
        """Band for a single cell without building the whole array."""
        lx, ly = cx - self.x0, cy - self.y0
        h, w = self.visible.shape
        if not (0 <= ly < h and 0 <= lx < w) or not self.visible[ly, lx]:
            return 0
        dist = float(self.dist[ly, lx])
        d = abs((float(self.ang[ly, lx]) - facing + math.pi) % (2 * math.pi) - math.pi)
        if d <= math.radians(cone.identify_deg) * 0.5 and dist <= cone.identify_range:
            return 1
        if d > math.radians(cone.peripheral_deg) * 0.5:
            return 0                                   # behind the player - blind
        if dist <= cone.near_range:
            return 2
        if d <= math.radians(cone.recognise_deg) * 0.5 and dist <= cone.recognise_range:
            return 2
        if dist <= cone.peripheral_range:
            return 3
        return 0


def _scan(blocks, vis, cx, cy, x0, y0, row, start, end, radius,
          xx, xy, yx, yy, W, H, r2):
    """One recursive shadowcast sweep over a single octant."""
    if start < end:
        return
    for j in range(row, radius + 1):
        dx, dy = -j - 1, -j
        blocked = False
        new_start = start
        while dx <= 0:
            dx += 1
            X = cx + dx * xx + dy * xy
            Y = cy + dx * yx + dy * yy
            l_slope = (dx - 0.5) / (dy + 0.5)
            r_slope = (dx + 0.5) / (dy - 0.5)
            if start < r_slope:
                continue
            if end > l_slope:
                break

            inside = 0 <= X < W and 0 <= Y < H
            if dx * dx + dy * dy <= r2 and inside:
                vis[Y - y0, X - x0] = True
            solid = True if not inside else bool(blocks[Y, X])

            if blocked:
                if solid:
                    new_start = r_slope
                    continue
                blocked = False
                start = new_start
            elif solid and j < radius:
                blocked = True
                _scan(blocks, vis, cx, cy, x0, y0, j + 1, start, l_slope,
                      radius, xx, xy, yx, yy, W, H, r2)
                new_start = r_slope
        if blocked:
            break


def shadowcast(blocks_sight: np.ndarray, cx: int, cy: int,
               radius: int) -> VisionField:
    """Exact symmetric visibility from one cell, out to `radius` cells."""
    H, W = blocks_sight.shape
    x0 = max(0, cx - radius)
    x1 = min(W, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(H, cy + radius + 1)

    vis = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    vis[cy - y0, cx - x0] = True

    r2 = radius * radius
    for xx, xy, yx, yy in _OCTANTS:
        _scan(blocks_sight, vis, cx, cy, x0, y0, 1, 1.0, 0.0, radius,
              xx, xy, yx, yy, W, H, r2)

    ys = np.arange(y0, y1, dtype=np.float32) - cy
    xs = np.arange(x0, x1, dtype=np.float32) - cx
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    dist = np.hypot(gx, gy).astype(np.float32)
    ang = np.arctan2(gy, gx).astype(np.float32)

    return VisionField(visible=vis, dist=dist, ang=ang, x0=x0, y0=y0,
                       origin=(cx, cy), radius=radius)


def line_of_sight(blocks: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                  step: float = 0.5) -> bool:
    """Cheap point-to-point sight check over a blocking mask.

    Samples the open segment between two fine-grid cells and returns False
    if any sampled cell blocks sight. Endpoints are excluded, so a listener
    or target standing against a wall does not occlude itself. This is the
    single-ray fallback used by guard perception, where a full shadowcast
    per guard per frame would be wasteful.
    """
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return True
    n = max(1, int(dist / step))
    h, w = blocks.shape
    for k in range(1, n):
        f = k / n
        xi = int(x0 + dx * f)
        yi = int(y0 + dy * f)
        if 0 <= yi < h and 0 <= xi < w and blocks[yi, xi]:
            return False
    return True
