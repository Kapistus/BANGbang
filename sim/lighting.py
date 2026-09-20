"""
sim/lighting.py — the light model, as maths.

Light is not decoration in this game: an unlit room reveals nothing, so what
you can see is gated on what is lit. Single-player has always worked this way;
this module is the arithmetic behind it, pulled out so multiplayer runs the
same numbers rather than a lookalike.

Three pieces, all pure numpy over a shadowcast window:

  cone_beam   a directional beam — a flashlight, a guard's torch
  point_light a radial light — a muzzle flash, an explosion
  see_gate    illumination -> how much of the vision cone actually registers

Compositing the result onto a screen stays in the renderer (main._additive_blit),
because that part is pygame.
"""
from __future__ import annotations

import math

import numpy as np

# Illumination below LOW reveals nothing; at or above FULL you see normally.
# Between the two, perception fades — a dim room is half-seen.
SEE_MIN = 0.20
SEE_FULL = 0.50


def cone_beam(ang: np.ndarray, dist: np.ndarray, visible: np.ndarray,
              facing: float, half_deg: float, reach_cells: float,
              smooth: bool = True) -> np.ndarray:
    """A beam: everything visible, within `half_deg` of `facing`, out to
    `reach_cells`, fading with distance.

    `ang`, `dist` and `visible` come from a shadowcast field, so the beam stops
    at walls for free — the light does not round corners, and neither does what
    it lets you see."""
    d = np.abs((ang - facing + np.pi) % (2 * np.pi) - np.pi)
    lit = (d <= math.radians(half_deg)) & visible
    t = np.clip(1.0 - dist / max(reach_cells, 1.0), 0.0, 1.0)
    if smooth:
        t = t * t * (3.0 - 2.0 * t)          # smoothstep: a soft tip, not a wall
    return np.where(lit, t, 0.0).astype(np.float32)


def point_light(dist: np.ndarray, visible: np.ndarray, reach_cells: float,
                brightness: float, falloff: float = 1.6) -> np.ndarray:
    """A radial light: a muzzle flash, a detonation. Falloff above 1 keeps the
    pool tight around the source instead of washing the whole room."""
    return np.where(
        visible,
        np.clip(1.0 - dist / max(reach_cells, 1.0), 0.0, 1.0) ** falloff
        * brightness,
        0.0).astype(np.float32)


def pulse(age: float, attack: float = 0.32, power: float = 1.25) -> float:
    """Brightness of a flash over its life, `age` running 0..1.

    Fast up, slower down: a muzzle flash is a spike, not a fade-in. Returns 0
    outside its life so callers can cull cheaply."""
    if not 0.0 <= age < 1.0:
        return 0.0
    b = age / attack if age < attack else (1.0 - age) / (1.0 - attack)
    return max(0.0, b) ** power


def see_gate(illum: np.ndarray, lo: float = SEE_MIN,
             hi: float = SEE_FULL) -> np.ndarray:
    """How much of the vision cone survives the light available: 0 in the dark,
    1 in a lit room, ramped between."""
    return np.clip((illum - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def static_illumination(lightmap, y0: int, x0: int, h: int, w: int
                        ) -> np.ndarray:
    """The baked lightmap over a vision window, or darkness if the map has none.

    A map with no lights is dark, full stop. That makes an unlit map unplayable
    without a flashlight, which is correct: light is placed, not assumed."""
    if lightmap is None:
        return np.zeros((h, w), dtype=np.float32)
    return lightmap[y0:y0 + h, x0:x0 + w].astype(np.float32)
