"""Hitscan ballistics: wall penetration, body penetration, and blasts.

One trigger pull is one or more projectiles (``weapon.pellets``), each fired
with its own accuracy jitter by the caller. A projectile:

* marches the fine grid from the muzzle. The first solid face of each piece
  of cover it crosses spends ``weapon.pen`` budget equal to that tile's
  ``pen_cost``; thickness past the face is ignored. Budget negative -> the
  bullet stops in that cover. Each cover pierced multiplies surviving damage
  by ``weapons.WALL_DMG_RETAIN``.

* resolves damage against the nearest bodies before the wall stop, via
  ``Combatant.take`` so the shield/armor split lives in one place. A normal
  round stops on the first body. A piercing weapon (rail, laser) passes
  through a body that had no shields, losing ``weapons.BODY_DMG_RETAIN`` of
  its damage each time, and stops on the first body that did have shields.

``blast`` is the explosive path: full damage at the centre falling linearly
to zero at ``radius_m``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sim import weapons

MIN_PEN_COST = 0.05
STEP_C = 0.25          # march increment, fine-grid cells
BLAST_FALLOFF_EXP = 0.55   # < 1 keeps the outer half of a blast dangerous


@dataclass
class Hit:
    target: object
    damage: float
    dist_m: float


@dataclass
class Shot:
    segments: list          # (p0, p1, "air" | "wall") in world metres
    impact: tuple           # world metres where the projectile ended
    hits: list              # list[Hit]
    pierced: int = 0
    shattered: list = None  # fine (col, row) cells of glass the shot broke


def _ray_circle(ox, oy, dx, dy, cx, cy, r):
    mx, my = ox - cx, oy - cy
    b = mx * dx + my * dy
    c = mx * mx + my * my - r * r
    if c > 0.0 and b > 0.0:
        return None
    disc = b * b - c
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    t = -b - sq
    if t < 0.0:
        t = -b + sq
    return t if t >= 0.0 else None


def _trim(segs, ox, oy, dx, dy, t):
    out = []
    for p0, p1, kind in segs:
        d0 = (p0[0] - ox) * dx + (p0[1] - oy) * dy
        d1 = (p1[0] - ox) * dx + (p1[1] - oy) * dy
        if d0 >= t:
            continue
        if d1 > t:
            p1 = (ox + dx * t, oy + dy * t)
        out.append((p0, p1, kind))
    return out


def fire_shot(m, blocks_bullets, pen_cost, glass, origin, heading, weapon,
              targets, rng, now):
    """Resolve one projectile. Applies damage to whatever it hits.

    `glass` is a bool array the size of the fine grid: a glass cell lets the
    shot pass (for a token penetration cost, no damage penalty) and is added
    to the returned Shot.shattered list so the caller can break the pane.
    """
    cpm = m.cells_per_metre
    ox_m, oy_m = origin
    dx, dy = math.cos(heading), math.sin(heading)
    rows, cols = blocks_bullets.shape
    max_range = weapon.hard_range_m

    bodies = []
    for tgt in targets:
        if not getattr(tgt, "alive", True):
            continue
        t = _ray_circle(ox_m, oy_m, dx, dy, tgt.x, tgt.y,
                        getattr(tgt, "body_r", 0.28))
        if t is not None and t <= max_range:
            bodies.append((t, tgt))
    bodies.sort(key=lambda p: p[0])

    pos_x, pos_y = ox_m * cpm, oy_m * cpm
    prev_cell = (int(pos_x), int(pos_y))
    budget = weapon.pen
    retain = 1.0
    pierced = 0
    pierces = []            # (dist_m, retain_after)
    segments = []
    seg_start = (ox_m, oy_m)
    mode = "air"
    max_c = max_range * cpm
    travelled = 0.0
    stop_pt = None
    shattered = []

    while travelled < max_c:
        pos_x += dx * STEP_C
        pos_y += dy * STEP_C
        travelled += STEP_C
        ci, cj = int(pos_x), int(pos_y)
        cur_m = (pos_x / cpm, pos_y / cpm)
        if not (0 <= ci < cols and 0 <= cj < rows):
            stop_pt = cur_m
            break
        if (ci, cj) == prev_cell:
            continue
        prev_cell = (ci, cj)
        solid = bool(blocks_bullets[cj, ci])
        if solid and glass[cj, ci]:
            # glass: the shot goes straight through and the pane breaks
            if not shattered or shattered[-1] != (ci, cj):
                shattered.append((ci, cj))
            budget -= 0.1
            if weapon.pen <= 0.0 or budget < 0.0:
                stop_pt = cur_m       # explosives detonate on the pane
                break
            continue
        if solid and mode == "air":
            segments.append((seg_start, cur_m, "air"))
            seg_start = cur_m
            mode = "wall"
            if weapon.pen <= 0.0:
                stop_pt = cur_m
                break
            dist_m = math.hypot(cur_m[0] - ox_m, cur_m[1] - oy_m)
            budget -= max(float(pen_cost[cj, ci]), MIN_PEN_COST)
            retain *= weapons.WALL_DMG_RETAIN
            pierced += 1
            pierces.append((dist_m, retain))
            if budget < 0.0:
                stop_pt = cur_m
                break
        elif not solid and mode == "wall":
            segments.append((seg_start, cur_m, "wall"))
            seg_start = cur_m
            mode = "air"

    if stop_pt is None:
        stop_pt = (pos_x / cpm, pos_y / cpm)
    segments.append((seg_start, stop_pt, mode))
    stop_dist = math.hypot(stop_pt[0] - ox_m, stop_pt[1] - oy_m)

    hits = []
    for t, tgt in bodies:
        if t > stop_dist + 1e-6:
            break
        scale = 1.0
        for pd, pr in pierces:
            if pd <= t:
                scale = pr
        had_sh = tgt.had_shields()
        dmg = weapons.roll_damage(weapon, rng) * scale
        tgt.take(dmg, now, weapon.shield_mult, weapon.health_mult,
                 src=(ox_m, oy_m))
        hits.append(Hit(tgt, dmg, t))
        if weapon.pierce_bodies and not had_sh:
            for i, (pd, pr) in enumerate(pierces):
                pierces[i] = (pd, pr * weapons.BODY_DMG_RETAIN)
            if not pierces:
                pierces.append((t, weapons.BODY_DMG_RETAIN))
            continue
        stop_pt = (ox_m + dx * t, oy_m + dy * t)
        segments = _trim(segments, ox_m, oy_m, dx, dy, t)
        break

    return Shot(segments=segments, impact=stop_pt, hits=hits, pierced=pierced,
                shattered=shattered)


def blast(center, radius_m, weapon, targets, rng, now):
    """Area damage: full at the centre, tapering to zero at radius_m. The
    taper uses BLAST_FALLOFF_EXP (< 1) so the outer half of the blast still
    hurts rather than only the dead centre."""
    hits = []
    cx, cy = center
    for tgt in targets:
        if not getattr(tgt, "alive", True):
            continue
        d = math.hypot(tgt.x - cx, tgt.y - cy)
        if d > radius_m:
            continue
        dmg = weapons.roll_damage(weapon, rng) \
            * (1.0 - d / radius_m) ** BLAST_FALLOFF_EXP
        tgt.take(dmg, now, weapon.shield_mult, weapon.health_mult,
                 src=(cx, cy))
        hits.append(Hit(tgt, dmg, d))
    return hits
