"""
sim/movement.py — one definition of how a body moves through a map.

Split out of main.py so the multiplayer server can move players exactly the way
the single-player renderer does. If these two ever disagree, clients rubber-band:
the server corrects a position the client predicted with different numbers.

No pygame, no rendering — safe to import from the networking layer.
"""
from __future__ import annotations

import math

from .tilemap import TileMap

BODY_R = 0.28                      # character collision radius, metres

# metres per second by stance. Mirrored in main.py's HUD text only; the values
# live here.
SPEED_CRAWL, SPEED_WALK, SPEED_RUN = 0.9, 2.2, 6.0

SPEEDS = {"crawl": SPEED_CRAWL, "walk": SPEED_WALK, "run": SPEED_RUN}


def speed_of(stance: str) -> float:
    return SPEEDS.get(stance, SPEED_WALK)


def try_move(m: TileMap, x: float, y: float, dx: float, dy: float,
             radius: float = BODY_R) -> tuple[float, float]:
    """Move by (dx, dy), testing each axis separately so a body slides along a
    wall instead of sticking to it. Returns the new position."""
    if m.can_stand(x + dx, y, radius):
        x += dx
    if m.can_stand(x, y + dy, radius):
        y += dy
    return x, y


def clamp_input(mx: float, my: float) -> tuple[float, float]:
    """Clamp a movement vector to the unit disc.

    Clients send a direction in [-1, 1] per axis. Taken literally, a diagonal
    would be 1.41x faster than a cardinal — and a modified client could simply
    send 5.0. The server runs every input through this."""
    try:
        mx = float(mx)
        my = float(my)
    except (TypeError, ValueError):
        return 0.0, 0.0
    if not (math.isfinite(mx) and math.isfinite(my)):
        return 0.0, 0.0
    mag = math.hypot(mx, my)
    if mag > 1.0:
        mx /= mag
        my /= mag
    return mx, my


def step(m: TileMap, x: float, y: float, mx: float, my: float,
         stance: str, dt: float, radius: float = BODY_R) -> tuple[float, float]:
    """One movement tick: clamp the input, scale by stance speed, collide."""
    mx, my = clamp_input(mx, my)
    if mx == 0.0 and my == 0.0:
        return x, y
    v = speed_of(stance) * dt
    return try_move(m, x, y, mx * v, my * v, radius)
