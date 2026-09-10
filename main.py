"""Debug renderer: walk a map, emit sounds, watch the propagation field.

Movement
    WASD / arrows   move          shift  run          ctrl  crawl
    mouse           facing

Weapons
    left mouse      fire            r      reload
    mouse wheel /   change weapon   f      interact (doors)
    number keys 1-7
    b              cycle fire mode. The combat rifle and SMG have a full-auto
                   alternate mode - hold to fire, SMG fast, rifle slower, both
                   less accurate. The rocket launcher reloads after every shot.
                   Rail weapons HOLD to charge (0-3 s = +0-100% shot damage);
                   the spool-up pitch rises and peaks at full charge.
    Roster and the shield/armor/accuracy model are ported from mechanics/;
    see sim/weapons.py and sim/combat.py. Damage lands on shields first
    (energy weapons hard, rail weapons barely), then armor cuts the health
    portion, then health. Bullets punch through cover on a per-weapon
    penetration budget; rail and laser also pass through unshielded bodies.

Sound
    footsteps emit automatically while moving, louder when running and
    louder again on the grating; walking patters, running lands further
    apart, crawling makes no audible step at all
    space           knock (medium)
    c               clear active sounds
    m               mute audio. Placeholder SFX are synthesised in sim/audio.py;
                    enemy events are attenuated and stereo-panned by the same
                    propagation field that drives the on-screen cues.

Overlays
    F1 grid       F2 sound cost   F3 blocks_sight   F4 blocks_move
    F5 footstep   F6 arrival-time heatmap   F7 fog   F8 reset fog memory
    F9 frame profiler   F10 your own sounds (off by default)
    F11 enemy sound display: arcs (what you would actually perceive) /
        full ripples (debug) / off
    v  guard vision cones
    tab clear overlay      p  pause ripples      esc quit

Guards run a four-mode state machine (idle / patrol / search / combat).
Body colour is the mode; the ring around a spotted guard is its alertness.
They hear the player through the same sound field the player does and walk
a guess toward the loudest arrival. Bullets penetrate cover by spending a
per-weapon budget against each tile's pen_cost; glass (windows) lets a shot
straight through and shatters into a permanent, loud hole.

Damage lands shields -> armor -> health. Health is a small core; armor
(percent) and shields (regenerating buffer) are what let anything trade
fire. A naked target folds fast. Global damage is scaled up from the bible
figures by weapons.DMG_SCALE.

The probe arrow follows the mouse: it shows the bearing a listener standing
at that cell would perceive, which is the negative gradient of arrival time
and not the straight line back to the source.
"""

from __future__ import annotations

import dataclasses
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pygame

from sim import ai, audio, ballistics, combat, mapfile, sound, sprites, weapons
from sim.tilemap import (FLOOR_DARKEN, WALL_LIGHTEN, TileMap, compute_roof,
                         load_map, validate_patrols, validate_spawns)


def load_any_map(path):
    """Dispatch on extension: .map is the JSON editor format, anything else is
    the legacy char-grid sidecar. Falls back to a .map sibling and gives a
    readable error if nothing matches."""
    p = Path(path)
    if p.suffix == ".map":
        return mapfile.load_map(p)
    if p.exists():
        return load_map(p)
    alt = p.with_suffix(".map")
    if alt.exists():
        return mapfile.load_map(alt)
    here = sorted(q.name for q in p.parent.glob("*.map")) \
        + sorted(q.name for q in p.parent.glob("*.toml"))
    raise SystemExit(f"map not found: {path}\n"
                     f"available in {p.parent}/: {', '.join(here) or '(none)'}")
from sim.vision import ConeSpec, VisibilityCache, line_of_sight, shadowcast

PX_PER_M = 48          # pixels per world metre (fixed; the camera scrolls)
VIEW_PPM = 48          # 1920x1080 sweet spot: readable characters (~49 px), the
                       # whole vision cone visible when aiming sideways; aiming
                       # straight up/down clips the outer identify/recognise cone
MAX_VIEW = (1860, 776)    # largest on-screen viewport: near-full width on a
                          # 1920x1080 display; 776 + 224 HUD = 1000 clears the
                          # taskbar. The world is usually bigger -> camera pans.
HUD_H = 224

# A sound field depends only on origin and geometry, so consecutive
# footsteps close together can share one solve. This is the distance the
# player must travel before the cached field is rebuilt.
FIELD_REUSE_M = 1.0

# Cells finalised per frame when a solve is spread across frames. Dijkstra
# emits cells in arrival-time order, so a partial field is correct out to a
# radius - and that radius races ahead of the audible wavefront far faster
# than the sound travels. Lower this if firing still hitches.
SOLVE_BUDGET = 1200

# Ripples are not drawn in cells whose sound cost is at or above this, so the
# wavefront vanishes into walls and reappears on the far side rather than
# visibly crawling through them.
SOUND_RENDER_MAX_COST = 5.0
FPS_CAP = 60          # 0 = uncapped

SPEED_CRAWL, SPEED_WALK, SPEED_RUN = 0.9, 2.2, 6.0   # run +30% (was 4.6)
SLING_SPEED_MULT = 1.20   # moving with the gun slung (not ready) is 20% faster
RAISE_TIME = 0.35         # seconds to bring a slung weapon back up before it can fire

STAMINA_DRAIN = 0.16      # per second while sprinting (~6 s to empty)
STAMINA_REGEN_MOVE = 0.07  # per second while walking / sneaking (~14 s to full)
STAMINA_REGEN_IDLE = 0.20  # per second while stationary (~5 s to full)
STAMINA_UNLOCK = 0.25     # sprint re-enables once stamina climbs back to this

# Metres travelled per footstep - this sets both the sound-propagation
# cadence and the audible step rhythm. Walking patters (short spacing),
# running lands heavier and further apart, crawling is slow and, for the
# player, produces no step SFX at all (see the sfx call below).
STRIDE = {"crawl": 1.50, "walk": 0.70, "run": 1.10}

# Sound energies are the METRES a sound carries in open air. Converted to
# cell units at load, so changing subdiv no longer rescales how far
# everything is audible.
FOOTSTEP_REACH_M = {"crawl": 2.0, "walk": 5.5, "run": 11.0}
KNOCK_REACH_M = 22.0
DRYFIRE_REACH_M = 3.5
MAGDROP_REACH_M = 7.5
GLASS_BREAK_REACH_M = 15.0
MAX_ACTIVE = 14

# Ammo is unlimited in reserve but finite per magazine, so reloading is the
# real cost. Sound energy is what separates them tactically.
# reach_m is the SOUND energy; dmg/pen/falloff_m/spread_deg/range_m are the
# Weapons, damage (shields/armor/health) and the accuracy model all live in
# sim/weapons.py and sim/combat.py, ported from mechanics/. The player carries
# weapons.DEFAULT_LOADOUT; cycle with the wheel or keys 1-7.
RESPAWN_DELAY = 1.2
TRACER_FADE = 0.18
BLAST_FADE = 0.32
BACKBLAST_FADE = 0.30       # rocket-launcher exhaust cone behind the shooter
MUZZLE_LIGHT_TIME = 0.11    # muzzle-flash light: expand+brighten then dim, seconds
MUZZLE_LIGHT_GAIN = 110     # additive world brighten from a muzzle flash you can see
MFLASH_BLUR = 3             # fine-cell blur that softens the muzzle-flash light
                           # (small: it must not spill across a wall it is cast behind)
MFLASH_POP_GAIN = 180      # brightness of the coloured muzzle-flash glow over the fog


def muzzle_light_spec(w) -> tuple:
    """(rgb, brightness, reach_m) for a weapon's muzzle-flash light, by kind:
    orange/white for slugthrowers, dim blue for rail, blue-green for plasma,
    cyan for energy/laser, dim wide orange for the rocket."""
    n = w.name.lower()
    cat = w.category
    if "rail" in n:
        return (95, 140, 255), 0.55, 1.8
    if cat == "plasma":
        return (120, 255, 180), 0.85, 2.6
    if cat in ("laser", "energy"):
        return (150, 235, 255), 0.70, 2.2
    if "rocket" in n:
        return (255, 195, 140), 0.45, 3.8
    if "flame" in n:
        return (255, 150, 65), 0.40, 2.0
    return (255, 224, 170), 0.9, 2.4            # ballistic + heavy


def blast_flash_spec(w) -> tuple:
    """(rgb, brightness, reach_m, life_s) for the light of an explosion - a
    bigger, slightly longer muzzle_lights entry at the detonation point."""
    if w.category == "plasma":
        return (140, 255, 200), 1.1, max(w.blast_r * 2.2, 2.5), 0.20
    if "rocket" in w.name.lower():
        return (255, 210, 150), 1.35, max(w.blast_r * 1.8, 4.0), 0.24
    return (255, 190, 120), 0.95, max(w.blast_r * 1.6, 2.0), 0.18   # flak / heavy
NEAR_MISS_M = 1.3          # a shot passing this close to a guard = "shot at"
INTERACT_RANGE = 1.6

VIS_SPEED = 140.0    # fine-cells/sec: sound-propagation speed (gates the ripple
                      # AND when actors hear a sound). ~17.5 m/s at cpm 8, about
                      # 2x the old pace, still slow enough to watch.
BODY_R = 0.28

# Vision. Angles are full widths in degrees; ranges here are METRES and get
# converted to cells at load, so changing subdiv in tiles.toml no longer
# silently rescales them.
CONE_DEG = (44.0, 100.0, 180.0)      # identify, recognise, peripheral
# identify, recognise, peripheral, near. There is no vision behind the player
# at all - the peripheral band is a hard forward hemisphere and there is no
# all-round near radius.
CONE_RANGE_M = (18.0, 14.0, 8.0, 0.0)   # identify shortened so it fits the viewport


# Black overlay alpha per visibility state. 0 = fully lit, 255 = hidden.
A_UNKNOWN = 255       # never seen: fully hidden - no bleed from lit rooms you
                      # have no line of sight into
A_REMEMBERED = 138    # seen before: geometry readable but clearly stale
CONE_REVEAL = 0.35    # how far the vision cone lifts the veil (1 = clears it to
                      # a flashlight beam, lower = a faint lightening of the wedge)
FLASHLIGHT_GAIN = 135    # additive brightness at the beam core (player `l` toggle)
FLASHLIGHT_HALF_DEG = 16.0  # beam half-angle (identify band is ~22deg)
FLASHLIGHT_BLUR = 1     # tiny - just anti-alias the side edge, keep it sharp
LIGHT_SEE_MIN = 0.20    # illumination below this reveals nothing (dark = blind)
LIGHT_SEE_FULL = 0.50   # illumination at/above this gives full perception
FLASHLIGHT_SELF = 0.55  # how much the flashlight lights the player holding it
FLASHLIGHT_SELF_SEEN = 0.40  # how visible the player's own flashlight makes them
GUARD_FLASH_GAIN = 105  # additive world brighten from a guard's flashlight beam
GUARD_FLASH_POP = 70    # dimmer glow of that beam drawn over the fog (a tell)
SHOT_GLOW = 0.75        # brief player-sprite brightness lift on each shot fired
SHOT_GLOW_TIME = 0.13
RAIL_IDS = ("rail_rifle", "rail_pistol")   # hold-to-charge weapons
RAIL_CHARGE_MAX = 3.0   # seconds of hold for a full charge
RAIL_CHARGE_BOOST = 1.0  # +100% released-shot damage at full charge
FOG_BLUR_R = 3        # box-blur radius (fine cells) that feathers every fog edge
ROOF_BLUR_R = 3      # same feather for the roof reveal edge (window / door peek)
ROOF_REVEAL = 0.02   # cone intensity above which a roofed cell is shown through

GHOST = (150, 96, 72)
BLIP = (200, 140, 90)
RIPPLE_ENEMY = (231, 88, 60)   # enemy sound wavefront - same ripple style as the
                               # player's, warm red instead of the player's amber

# An arriving sound is drawn as an arc at the player's own position, in the
# direction the wavefront came from. Width encodes confidence: a strong
# arrival gives a tight bearing, a faint one a vague smear. Below
# CUE_MIN_ENERGY there is no usable direction at all and the cue becomes a
# full ring - you know something happened, not where.
CUE_RADIUS_M = 2.3
CUE_FADE = 1.7
CUE_NARROW_DEG = 14.0
CUE_WIDE_DEG = 75.0
CUE_MIN_ENERGY = 0.10

BG = (28, 28, 26)
HUD_BG = (20, 20, 19)
TEXT = (222, 220, 212)
DIM = (140, 138, 132)
PLAYER = (29, 158, 117)
FACING = (239, 159, 39)
RIPPLE = (239, 159, 39)
PROBE = (216, 90, 48)
LIGHT = (250, 220, 150)
GUARD = (216, 90, 48)
MODE_COL = {
    ai.Mode.IDLE: (120, 120, 128),
    ai.Mode.PATROL: (90, 140, 180),
    ai.Mode.SEARCH: (232, 168, 60),
    ai.Mode.COMBAT: (222, 70, 50),
}
TRACER_AIR = (255, 240, 180)
TRACER_WALL = (150, 60, 45)
DEAD = (72, 70, 66)
DOOR_WOOD = (150, 118, 78)      # the door leaf
DOOR_FRAME = (86, 66, 40)       # frame / seam on a closed door
DOOR_OPEN_EDGE = (110, 175, 110)  # green outline marks a passable doorway
DOOR_FLOOR = (206, 202, 190)    # what shows through an opened doorway
GUARD_DOOR_REACH_M = 1.4        # a blocked guard shoves a door within this


class ActiveSound:
    """One audible event. `energy` is this event's own loudness budget, which
    may be smaller than the shared field's, so a quiet footstep can reuse a
    field solved for a loud one."""

    __slots__ = ("field", "t0", "label", "solve_ms", "energy", "enemy", "cued")

    def __init__(self, field, t0, label, solve_ms, energy=None, enemy=False):
        self.cued = False
        self.field = field
        self.t0 = t0
        self.label = label
        self.solve_ms = solve_ms
        self.energy = field.energy if energy is None else energy
        self.enemy = enemy

    def elapsed_cells(self, now: float) -> float:
        return (now - self.t0) * VIS_SPEED

    def done(self, now: float) -> bool:
        return self.elapsed_cells(now) > self.field.max_travel + 20


def apply_lightmap(base: pygame.Surface, lm: np.ndarray) -> None:
    """Multiply the static light map into the base surface (in place). Lights
    are static, so this is a one-time bake - no per-frame cost."""
    g = np.clip(lm * 255.0, 0, 255).astype(np.uint8)          # (h, w)
    tex = pygame.surfarray.make_surface(np.repeat(g.T[:, :, None], 3, axis=2))
    base.blit(pygame.transform.smoothscale(tex, base.get_size()),
              (0, 0), special_flags=pygame.BLEND_RGB_MULT)


def _flat_base(m: TileMap, world_w: int, world_h: int) -> pygame.Surface:
    """Legacy .grid maps: one flat colour per cell, no tile art."""
    rows, cols = m.chars.shape
    arr = np.zeros((rows, cols, 3), dtype=np.uint8)
    for ch, t in m.tiles.items():
        mask = m.chars == ch
        if mask.any():
            arr[mask] = t.colour
    small = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
    return pygame.transform.scale(small, (world_w, world_h))


def build_base_surface(m: TileMap) -> pygame.Surface:
    """The static map layer. For a .map (which carries floor/object id grids +
    the tileset) this blits the real tile PNGs, tinted by role - floor cells
    darker, wall cells lighter. Door cells are left to the dynamic door draw;
    bush (overlay) is not on the base."""
    world_w = round(m.width_m * PX_PER_M)
    world_h = round(m.height_m * PX_PER_M)
    ts = getattr(m, "tileset", None)
    fids = getattr(m, "floor_ids", None)
    if ts is None or fids is None:
        return _flat_base(m, world_w, world_h)

    oids = m.object_ids
    rows, cols = fids.shape
    cell = max(1, round(PX_PER_M * m.metres_per_char))
    root = ts.path.parent
    dk = int(round(255 * FLOOR_DARKEN))
    lt = int(round(255 * WALL_LIGHTEN))
    cache: dict = {}

    def tile(tid: str, role: str) -> pygame.Surface:
        got = cache.get((tid, role))
        if got is not None:
            return got
        td = ts.tiles.get(tid)
        try:
            img = pygame.image.load(str(root / td.png)).convert_alpha()
            s = pygame.transform.smoothscale(img, (cell, cell))
        except Exception:
            s = pygame.Surface((cell, cell))
            s.fill(td.colour if td is not None else (150, 150, 150))
        s = s.copy()
        if role == "floor":
            s.fill((dk, dk, dk, 255), special_flags=pygame.BLEND_RGB_MULT)
        elif role == "wall":
            s.fill((lt, lt, lt, 0), special_flags=pygame.BLEND_RGB_ADD)
        cache[(tid, role)] = s
        return s

    surf = pygame.Surface((world_w, world_h))
    for r in range(rows):
        for c in range(cols):
            x, y = c * cell, r * cell
            surf.blit(tile(fids[r, c] or ts.default_floor, "floor"), (x, y))
            oid = oids[r, c]
            if not oid:
                continue
            otd = ts.tiles.get(oid)
            special = otd is not None and (otd.door or otd.glass or otd.bush)
            if otd is not None and otd.overlay:
                continue                      # bush canopy: not on the base
            if otd is not None and otd.door:
                continue                      # door: the dynamic door draw owns it
            surf.blit(tile(oid, "obj" if special else "wall"), (x, y))
    return surf


def field_surface(fine: np.ndarray, rgb, alpha: np.ndarray) -> pygame.Surface:
    h, w = fine.shape
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    px = pygame.surfarray.pixels3d(surf)
    al = pygame.surfarray.pixels_alpha(surf)
    px[:, :, 0], px[:, :, 1], px[:, :, 2] = rgb
    al[:, :] = np.transpose(np.clip(alpha, 0, 255)).astype(np.uint8)
    del px, al
    return surf


def rgba_surface(rgb: np.ndarray, alpha: np.ndarray) -> pygame.Surface:
    """Surface from per-cell colour, so several layers can be composited
    into one array and drawn with a single scale and blit. Scaling is by far
    the most expensive part of drawing an overlay, so doing it once instead
    of once per layer is the whole optimisation."""
    h, w = alpha.shape
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    px = pygame.surfarray.pixels3d(surf)
    al = pygame.surfarray.pixels_alpha(surf)
    px[:, :, :] = np.transpose(rgb, (1, 0, 2))
    al[:, :] = np.transpose(np.clip(alpha, 0, 255)).astype(np.uint8)
    del px, al
    return surf


def box_blur(a: np.ndarray, radius: int = 2) -> np.ndarray:
    """Separable box blur on a 2D float array - used to feather the hard
    shadowcast edges of the vision field so the fog reads soft, not stencilled.
    Edge cells get slightly less blur, which is fine here."""
    out = a.astype(np.float32).copy()
    src = out.copy()
    for d in range(1, radius + 1):
        out[:, d:] += src[:, :-d]
        out[:, :-d] += src[:, d:]
    out /= (2 * radius + 1)
    src = out.copy()
    for d in range(1, radius + 1):
        out[d:, :] += src[:-d, :]
        out[:-d, :] += src[d:, :]
    out /= (2 * radius + 1)
    return out


def scale_to_view(surf, m: TileMap, cells_w: int, cells_h: int):
    ppc = PX_PER_M / m.cells_per_metre
    return pygame.transform.scale(surf, (int(cells_w * ppc), int(cells_h * ppc)))


def static_overlay(field: np.ndarray, m: TileMap, lo, hi, rgb):
    norm = np.clip((field.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    s = field_surface(field, rgb, norm * 190)
    return scale_to_view(s, m, field.shape[1], field.shape[0])


def seg_point_dist(px, py, ax, ay, bx, by):
    """Shortest distance from point (px,py) to the segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def find_doors(m: TileMap) -> dict[tuple[int, int], bool]:
    """Map coarse (row, col) of every door tile to its open state."""
    out = {}
    for r in range(m.chars.shape[0]):
        for c in range(m.chars.shape[1]):
            if m.tiles[m.chars[r, c]].door:
                out[(r, c)] = False
    return out


def set_door(m: TileMap, cost, r: int, c: int, is_open: bool) -> None:
    """Rewrite the fine arrays for one door tile. Callers must invalidate
    the visibility cache and any held sound fields afterwards."""
    s = m.subdiv
    y0, y1 = r * s, (r + 1) * s
    x0, x1 = c * s, (c + 1) * s
    t = m.tiles[m.chars[r, c]]
    m.blocks_move[y0:y1, x0:x1] = False if is_open else t.blocks_move
    m.blocks_sight[y0:y1, x0:x1] = False if is_open else t.blocks_sight
    m.blocks_bullets[y0:y1, x0:x1] = False if is_open else t.blocks_bullets
    val = 1.0 if is_open else t.sound_cost
    m.sound_cost[y0:y1, x0:x1] = val
    cost[y0:y1, x0:x1] = val


def draw_cue(screen, cx, cy, ang, half_deg, alpha, ppm):
    """An arc centred on the listener, facing where the sound came from."""
    r_out = CUE_RADIUS_M * ppm
    r_in = r_out - max(4.0, ppm * 0.16)
    pad = int(r_out) + 4
    surf = pygame.Surface((pad * 2, pad * 2), pygame.SRCALPHA)
    half = math.radians(half_deg)
    steps = max(6, int(half_deg / 3))
    outer, inner = [], []
    for i in range(steps + 1):
        a = ang - half + 2 * half * i / steps
        ca, sa = math.cos(a), math.sin(a)
        outer.append((pad + ca * r_out, pad + sa * r_out))
        inner.append((pad + ca * r_in, pad + sa * r_in))
    pts = outer + inner[::-1]
    if len(pts) >= 3:
        pygame.draw.polygon(surf, (*RIPPLE_ENEMY, int(alpha)), pts)
    screen.blit(surf, (int(cx) - pad, int(cy) - pad))


def draw_guard_cone(screen, g, ppm):
    """Soft debug wedges for a guard's identify and peripheral vision (key v).

    Drawn as a stack of concentric wedge layers with the alpha fading out
    toward the rim, so the cone reads as a gradient rather than a flat slab.
    """
    surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    cx, cy = g.x * ppm, g.y * ppm
    for fov, rng_m, col in (
        (ai.FOV_PERIPH, ai.PERIPH_RANGE_M, (60, 70, 95)),
        (ai.FOV_IDENT, ai.IDENT_RANGE_M, (70, 110, 150)),
    ):
        for i in range(7, 0, -1):
            frac = i / 7.0
            r = rng_m * ppm * frac
            alpha = int(24 * (1.0 - frac) ** 1.3) + 3
            pts = [(cx, cy)]
            for k in range(17):
                a = g.facing - fov + 2.0 * fov * k / 16.0
                pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
            pygame.draw.polygon(surf, (*col, alpha), pts)
    screen.blit(surf, (0, 0))


def try_move(m: TileMap, x, y, dx, dy):
    if m.can_stand(x + dx, y, BODY_R):
        x += dx
    if m.can_stand(x, y + dy, BODY_R):
        y += dy
    return x, y


def emit(m: TileMap, cost, x, y, energy, label, sounds, now, cache=None,
         jobs=None, enemy=False):
    """Emit a sound. If `cache` is given it is a one-slot reusable field for
    the player: a fresh solve happens only once the player has moved
    FIELD_REUSE_M, and nearer footsteps reuse it with their own energy."""
    cx, cy = m.cell_of(x, y)
    if not np.isfinite(cost[cy, cx]):
        return
    if cache is not None:
        f = cache.get("field")
        ox, oy = cache.get("pos", (1e9, 1e9))
        reuse = (f is not None
                 and f.energy >= energy
                 and math.hypot(x - ox, y - oy) <= FIELD_REUSE_M)
        if reuse:
            sounds.append(ActiveSound(f, now, label, 0.0, energy, enemy))
            while len(sounds) > MAX_ACTIVE:
                sounds.pop(0)
            return
    t0 = time.perf_counter()
    f, job = sound.begin(cost, (cx, cy), energy)
    if job is not None and jobs is not None:
        # queue only. Stepping here would spend a full budget per emit, so
        # four guards stepping on one frame would cost four budgets.
        jobs.append(job)
    elif job is not None:
        while not job.step(SOLVE_BUDGET):
            pass
    ms = (time.perf_counter() - t0) * 1000.0
    if cache is not None:
        cache["field"] = f
        cache["pos"] = (x, y)
    sounds.append(ActiveSound(f, now, label, ms, energy, enemy))
    while len(sounds) > MAX_ACTIVE:
        sounds.pop(0)


def main(map_path: str = "maps/arena.toml") -> None:
    global PX_PER_M
    pygame.init()
    audio_on = audio.init()
    print("audio:", "on (procedural placeholders)" if audio_on else "off (no device)")
    m = load_any_map(map_path)
    rows, cols = m.chars.shape
    # Fixed zoom (pixels per world metre); the viewport is a camera window
    # onto a world that can be far larger than the screen.
    PX_PER_M = VIEW_PPM
    cost = m.sound_cost.astype(np.float64)

    print(sound.backend_report())
    warm = sound.warmup()
    if warm > 0.05:
        print(f"JIT compile: {warm:.1f}s (cached to disk, instant next run)")

    issues = validate_spawns(m) + validate_patrols(m)
    for msg in issues:
        print("MAP WARNING:", msg)
    if not issues:
        print("map checks passed")

    world_w = round(m.width_m * PX_PER_M)
    world_h = round(m.height_m * PX_PER_M)
    view_w = min(world_w, MAX_VIEW[0])
    view_h = min(world_h, MAX_VIEW[1])
    window = pygame.display.set_mode((view_w, view_h + HUD_H))
    world = pygame.Surface((world_w, world_h))   # the full map is drawn here,
    screen = window                              # then a camera rect is blitted
    cam_x = cam_y = 0
    pygame.display.set_caption(f"BANGbang \u2014 {m.name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,monospace", 13)
    font_mid = pygame.font.SysFont("consolas,monospace", 18, bold=True)
    font_big = pygame.font.SysFont("consolas,monospace", 34, bold=True)
    pygame.mouse.set_visible(False)          # we draw our own crosshair
    show_debug = False                       # \u00a7 toggles the raw debug overlay

    bank = sprites.SpriteBank()
    bank.load()
    bank.set_scale(PX_PER_M / sprites.ART_PPM)
    print("sprites:", "on" if bank.ok else "off (circle fallback)")

    base = build_base_surface(m)
    if m.lightmap is not None:
        apply_lightmap(base, m.lightmap)
    overlays = {
        pygame.K_F2: ("sound_cost", static_overlay(m.sound_cost, m, 1, 9, (239, 159, 39))),
        pygame.K_F3: ("blocks_sight", static_overlay(m.blocks_sight.astype(np.float32), m, 0, 1, (216, 90, 48))),
        pygame.K_F4: ("blocks_move", static_overlay(m.blocks_move.astype(np.float32), m, 0, 1, (55, 138, 221))),
        pygame.K_F5: ("footstep_mult", static_overlay(m.footstep_mult, m, 1, 2, (99, 153, 34))),
    }
    active_ov = None
    show_grid = False
    show_heat = False
    paused = False
    show_fog = True

    vis_cache = VisibilityCache(m.blocks_sight)
    known = np.zeros(m.blocks_sight.shape, dtype=bool)

    px_, py_ = m.player_spawn
    prev_px, prev_py = px_, py_
    in_bush = False
    facing = -math.pi / 2
    since_step = 0.0
    sounds: list[ActiveSound] = []
    now = 0.0
    last_solve = "none yet"

    cpm = m.cells_per_metre
    cone = ConeSpec(
        identify_deg=CONE_DEG[0], recognise_deg=CONE_DEG[1],
        peripheral_deg=CONE_DEG[2],
        identify_range=CONE_RANGE_M[0] * cpm,
        recognise_range=CONE_RANGE_M[1] * cpm,
        peripheral_range=CONE_RANGE_M[2] * cpm,
        near_range=CONE_RANGE_M[3] * cpm,
    )
    e_step = {k: v * cpm for k, v in FOOTSTEP_REACH_M.items()}
    e_knock = KNOCK_REACH_M * cpm
    e_dry = DRYFIRE_REACH_M * cpm
    e_mag = MAGDROP_REACH_M * cpm
    e_glass = GLASS_BREAK_REACH_M * cpm
    e_step_max = max(e_step.values()) * float(m.footstep_mult.max())
    step_cache: dict = {}
    jobs: list = []

    rng = random.Random()
    pen = m.pen_cost
    bbul = m.blocks_bullets
    player = combat.player_commando(px_, py_)
    loadout = list(weapons.DEFAULT_LOADOUT)
    tracers: list = []          # (segments, t0)
    blasts: list = []           # ((x, y), radius_m, t0) for explosion rings
    rockets: list = []          # travelling explosives: dicts, see spawn below
    backblasts: list = []       # {x,y,ang,t0} exhaust cones behind a backblast weapon
    muzzle_lights: list = []    # {x,y,t0,col,gain,reach} - brief flash of light at a muzzle
    respawn_t = 0.0
    show_cones = False

    A_UNKNOWN_F = A_UNKNOWN / 255.0
    A_REMEMBER_F = A_REMEMBERED / 255.0
    ripple_buf = np.zeros(m.blocks_sight.shape, dtype=np.float32)
    ripple_enemy = np.zeros(m.blocks_sight.shape, dtype=np.float32)

    # Sound crossing a wall advances ~40 arrival-time units per cell, so the
    # ring's narrow band sits inside the masonry for seconds before emerging.
    # The late emergence is correct - it is muffled transmission - but the
    # wave should not be drawn inside solid material.
    audible_open = m.sound_cost < SOUND_RENDER_MAX_COST
    prof = {}
    show_prof = False
    prof_smooth = {}
    prof_peak = {}

    _ov_h, _ov_w = m.blocks_sight.shape
    # scratch surface for the fog/ripple overlay - only the on-screen slice of
    # the fine grid, recreated when the visible slice changes size (map edges)
    ov_surf = pygame.Surface((_ov_w, _ov_h), pygame.SRCALPHA)
    roof_surf = pygame.Surface((_ov_w, _ov_h), pygame.SRCALPHA)   # roof pass slice

    guards = [ai.Guard(g, cpm) for g in m.guards]
    memory: dict[str, tuple[float, float, float, float]] = {}
    show_own_sound = False   # player's own sound world is hidden until F10
    enemy_mode = 2           # 0 off, 1 arcs, 2 full ripples (F11 cycles)
    cues: list = []

    doors = find_doors(m)
    wi = 0
    mags = [weapons.ROSTER[n].mag for n in loadout]
    reserves = [weapons.ROSTER[n].reserve for n in loadout]   # -1 = unlimited
    # index into each weapon's fire_modes(); the SMG starts on full-auto
    fmode = [(weapons.ROSTER[n].fire_modes().index("auto")
              if n == "smg" and "auto" in weapons.ROSTER[n].fire_modes() else 0)
             for n in loadout]
    reload_t = 0.0
    fire_cd = 0.0
    swap_t = 0.0                   # weapon-change animation timer
    recoil_t = 0.0                 # gun-kick timer (player)
    flash_t = 0.0                  # muzzle-flash timer (player)
    shot_glow_t = -9.0             # last-shot time - briefly lifts the player sprite
    rail_charging = False          # rail weapon: trigger held, spooling up
    rail_charge = 0.0              # seconds held so far (0..RAIL_CHARGE_MAX)
    rail_charge_ch = None          # the spool-up sound channel, so it can be cut
    slung = False                  # gun lowered (h) - faster, cannot fire
    raise_t = 0.0                  # bringing a slung gun back up
    flashlight = False             # l - a beam the shape of the identify cone
    stamina = 1.0                  # 0..1, drained by sprinting
    sprint_locked = False          # true when stamina bottomed out, until STAMINA_UNLOCK
    flash_roll = 0.0               # per-shot flash spin, degrees
    flash_scale = 1.0              # per-shot flash size jitter
    msg = ""
    msg_t = 0.0

    muted = False

    def sfx(name, gain=1.0, pan=0.0):
        if not muted:
            audio.play(name, gain, pan)

    def sfx_fire(w, gain=1.0, pan=0.0):
        if not muted:
            audio.play_fire(w, gain, pan)

    def draw_muzzle(art, cx, cy, facing, ft, roll, sc):
        """Two-layer muzzle flash at a weapon's muzzle: big + small core early,
        fading core late. `cx, cy` is the weapon sprite's blit centre."""
        if ft <= 0.0 or not bank.ok or art not in sprites.MUZZLE_PX:
            return
        half = sprites.FLASH_TIME * 0.5
        if ft > half:
            bank.flash(screen, art, "big", cx, cy, facing, roll, sc * 1.15, 255)
            bank.flash(screen, art, "small", cx, cy, facing, -roll * 1.7, sc, 255)
        else:
            bank.flash(screen, art, "small", cx, cy, facing, roll, sc * 0.8,
                       int(210 * ft / half))

    def cur_weapon():
        return weapons.ROSTER[loadout[wi]]

    def cur_fire_mode():
        w = cur_weapon()
        modes = w.fire_modes()
        return modes[fmode[wi] % len(modes)]

    def held_moving():
        k = pygame.key.get_pressed()
        return bool(k[pygame.K_w] or k[pygame.K_a] or k[pygame.K_s]
                    or k[pygame.K_d] or k[pygame.K_UP] or k[pygame.K_DOWN]
                    or k[pygame.K_LEFT] or k[pygame.K_RIGHT])

    broken_glass: set = set()      # coarse (row, col) of shattered panes

    def break_glass(fine_cells):
        """Turn every glass tile named by a fired shot into an open hole:
        bullets and sight pass, the frame still blocks movement, and the
        break is loud."""
        nonlocal audible_open
        s = m.subdiv
        changed = False
        for ci, cj in fine_cells:
            r, c = cj // s, ci // s
            if (r, c) in broken_glass or not m.tiles[m.chars[r, c]].glass:
                continue
            broken_glass.add((r, c))
            changed = True
            y0, y1, x0, x1 = r * s, (r + 1) * s, c * s, (c + 1) * s
            m.blocks_bullets[y0:y1, x0:x1] = False
            m.blocks_sight[y0:y1, x0:x1] = False
            m.glass[y0:y1, x0:x1] = False
            m.sound_cost[y0:y1, x0:x1] = 1.5
            cost[y0:y1, x0:x1] = 1.5
            emit(m, cost, c + 0.5, r + 0.5, e_glass, "glass", sounds, now, jobs=jobs)
        if changed:
            audible_open = m.sound_cost < SOUND_RENDER_MAX_COST
            vis_cache.invalidate()
            sfx("glass")

    def player_fire(w, moving, acc=None, rounds=None, charge=0.0):
        """One trigger pull: `rounds` (default the weapon's burst) x pellets,
        each with its own accuracy jitter; damage resolved inside fire_shot,
        plus a blast at impact for explosives. `acc` overrides the weapon's
        accuracy rating (full-auto passes the lower auto_accuracy)."""
        mxp, myp = pygame.mouse.get_pos()
        ax, ay = (mxp + cam_x) / PX_PER_M, (myp + cam_y) / PX_PER_M
        base = math.atan2(ay - py_, ax - px_)
        aim_d = math.hypot(ax - px_, ay - py_)
        live = [g for g in guards if g.alive]
        # fire from the gun's actual muzzle point (per-art, matches the flash) -
        # unless that would put the origin inside cover the player is hugging
        if bank.ok:
            _mdx, _mdy = bank.muzzle_world(sprites.weapon_art(loadout[wi]), facing)
        else:
            _mdx = math.cos(base) * weapons.MUZZLE_M
            _mdy = math.sin(base) * weapons.MUZZLE_M
        ox, oy = px_ + _mdx, py_ + _mdy
        _mcx, _mcy = m.cell_of(ox, oy)
        if bbul[_mcy, _mcx]:
            ox, oy = px_, py_
        travels = w.blast_r > 0.0 and w.projectile_speed > 0.0
        _fired = False
        for _ in range(rounds if rounds else w.burst):
            if mags[wi] <= 0:
                break
            mags[wi] -= 1
            _fired = True
            for _p in range(w.pellets):
                hd = (base + weapons.pellet_offset(w, rng)
                      + weapons.jitter(w, aim_d, moving, rng, acc))
                sh = ballistics.fire_shot(m, bbul, pen, m.glass, (ox, oy),
                                          hd, w, live, rng, now,
                                          apply_damage=not travels)
                if sh.segments and not travels:
                    tracers.append((sh.segments, now))
                if sh.shattered:
                    break_glass(sh.shattered)
                # a shot that cracks past a guard still counts as "being shot
                # at" - so a distant back-shot makes them whirl, not just a hit
                _hit_ids = {id(h.target) for h in sh.hits}
                for _g in live:
                    if id(_g) in _hit_ids:
                        continue
                    if seg_point_dist(_g.x, _g.y, ox, oy, *sh.impact) < NEAR_MISS_M:
                        _g.last_hit_t = now
                        _g.hit_from = (ox, oy)
                if travels:
                    ix, iy = sh.impact
                    d = math.hypot(ix - ox, iy - oy)
                    rockets.append({"x": ox, "y": oy, "dx": math.cos(hd),
                                    "dy": math.sin(hd), "w": w, "ix": ix,
                                    "iy": iy, "dist": d, "flown": 0.0})
                elif w.blast_r > 0.0:
                    # the shooter is not immune to their own blast
                    hit = live + [player] if player.alive else live
                    ballistics.blast(sh.impact, w.blast_r, w, hit, rng, now)
                    blasts.append((sh.impact, w.blast_r, now))
                    _bc, _bg, _br, _bl = blast_flash_spec(w)
                    muzzle_lights.append({"x": sh.impact[0], "y": sh.impact[1],
                                          "t0": now, "col": _bc, "gain": _bg,
                                          "reach": _br, "life": _bl})
        if _fired:
            _mcol, _mgain, _mreach = muzzle_light_spec(w)
            muzzle_lights.append({"x": ox, "y": oy, "t0": now, "col": _mcol,
                                  "gain": _mgain * (1.0 + 0.9 * charge),
                                  "reach": _mreach * (1.0 + 0.6 * charge)})
        if w.backblast:
            backblasts.append({"x": px_ - math.cos(base) * 0.55,
                               "y": py_ - math.sin(base) * 0.55,
                               "ang": base + math.pi, "t0": now})

    running = True
    while running:
        dt = (clock.tick(FPS_CAP) if FPS_CAP else clock.tick()) / 1000.0
        if not paused:
            now += dt

        # camera: keep the player centred, clamped to the world edges
        cam_x = int(min(max(px_ * PX_PER_M - view_w * 0.5, 0), max(0, world_w - view_w)))
        cam_y = int(min(max(py_ * PX_PER_M - view_h * 0.5, 0), max(0, world_h - view_h)))

        _t = time.perf_counter()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_F1:
                    show_grid = not show_grid
                elif ev.key == pygame.K_F6:
                    show_heat = not show_heat
                elif ev.key == pygame.K_TAB:
                    active_ov = None
                elif ev.key == pygame.K_F7:
                    show_fog = not show_fog
                elif ev.key == pygame.K_F8:
                    known[:] = False
                elif ev.key == pygame.K_F9:
                    show_prof = not show_prof
                elif ev.unicode == "§" or ev.key == pygame.K_BACKQUOTE:
                    show_debug = not show_debug
                elif ev.key == pygame.K_F10:
                    show_own_sound = not show_own_sound
                elif ev.key == pygame.K_F11:
                    enemy_mode = (enemy_mode + 1) % 3
                elif ev.key == pygame.K_v:
                    show_cones = not show_cones
                elif ev.key == pygame.K_m:
                    muted = not muted
                    msg, msg_t = "audio muted" if muted else "audio on", now
                elif ev.key == pygame.K_b:
                    w = weapons.ROSTER[loadout[wi]]
                    modes = w.fire_modes()
                    if len(modes) > 1:
                        fmode[wi] = (fmode[wi] + 1) % len(modes)
                        msg, msg_t = f"{w.name}: {modes[fmode[wi]]}", now
                    else:
                        msg, msg_t = f"{w.name}: no alt fire", now
                elif ev.key == pygame.K_r:
                    w = weapons.ROSTER[loadout[wi]]
                    if reload_t <= 0.0 and swap_t <= 0.0 and mags[wi] < w.mag:
                        if reserves[wi] == 0:
                            msg, msg_t = f"{w.name}: out of ammo", now
                        else:
                            reload_t = w.reload_step
                            msg, msg_t = f"reloading {w.name}", now
                            sfx("reload")
                elif ev.key == pygame.K_h and swap_t <= 0.0:
                    slung = not slung
                    raise_t = RAISE_TIME if not slung else 0.0
                    msg, msg_t = ("weapon slung" if slung else "weapon ready"), now
                elif ev.key == pygame.K_l:
                    flashlight = not flashlight
                    msg, msg_t = ("flashlight on" if flashlight
                                  else "flashlight off"), now
                elif pygame.K_1 <= ev.key <= pygame.K_9:
                    idx = ev.key - pygame.K_1
                    if idx < len(loadout) and idx != wi and swap_t <= 0.0:
                        wi = idx
                        reload_t = 0.0
                        swap_t = sprites.SWAP_TIME
                        slung, raise_t = False, 0.0   # a new weapon comes up ready
                        msg, msg_t = f"switched to {weapons.ROSTER[loadout[wi]].name}", now
                elif ev.key == pygame.K_f:
                    best, bd = None, INTERACT_RANGE
                    for (r_, c_) in doors:
                        d = math.hypot(c_ + 0.5 - px_, r_ + 0.5 - py_)
                        if d < bd:
                            best, bd = (r_, c_), d
                    if best is None:
                        msg, msg_t = "nothing to interact with", now
                    else:
                        want_open = not doors[best]
                        _br, _bc = best
                        _nx = min(max(px_, _bc), _bc + 1.0)
                        _ny = min(max(py_, _br), _br + 1.0)
                        in_leaf = math.hypot(px_ - _nx, py_ - _ny) < BODY_R + 0.05
                        if not want_open and in_leaf:
                            # closing the leaf on your own body traps you
                            msg, msg_t = "stand clear to close the door", now
                        else:
                            doors[best] = want_open
                            set_door(m, cost, best[0], best[1], want_open)
                            compute_roof(m, doors)
                            vis_cache.invalidate()
                            sounds.clear()
                            emit(m, cost, best[1] + 0.5, best[0] + 0.5,
                                 e_knock * 0.7, "door", sounds, now, jobs=jobs)
                            sfx("door")
                            msg = f"door {'opened' if want_open else 'closed'}"
                            msg_t = now
                elif ev.key == pygame.K_p:
                    paused = not paused
                elif ev.key == pygame.K_c:
                    sounds.clear()
                elif ev.key == pygame.K_SPACE:
                    emit(m, cost, px_, py_, e_knock, "knock", sounds, now, jobs=jobs)
                    sfx("knock")
                elif ev.key in overlays:
                    active_ov = None if active_ov == ev.key else ev.key
            elif ev.type == pygame.MOUSEWHEEL:
                if ev.y and swap_t <= 0.0:
                    wi = (wi - (1 if ev.y > 0 else -1)) % len(loadout)
                    reload_t = 0.0
                    swap_t = sprites.SWAP_TIME
                    slung, raise_t = False, 0.0
                    msg, msg_t = f"switched to {weapons.ROSTER[loadout[wi]].name}", now
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if pygame.mouse.get_pos()[1] < view_h:
                    w = weapons.ROSTER[loadout[wi]]
                    auto = cur_fire_mode() == "auto"
                    # firing cancels a shell-by-shell reload in progress
                    if reload_t > 0.0 and w.shell_reload and mags[wi] > 0:
                        reload_t = 0.0
                        msg, msg_t = "reload interrupted", now
                    if slung or raise_t > 0.0:
                        if slung:                       # a slung gun raises first
                            slung = False
                            raise_t = RAISE_TIME
                            msg, msg_t = "weapon ready", now
                    elif reload_t > 0.0 or fire_cd > 0.0 or swap_t > 0.0 or player.hp <= 0.0:
                        pass
                    elif loadout[wi] in RAIL_IDS:
                        # rail weapons charge while the trigger is held; the
                        # shot goes out on release (MOUSEBUTTONUP below)
                        if mags[wi] > 0:
                            rail_charging = True
                            rail_charge = 0.0
                            rail_charge_ch = (None if muted else
                                              audio.play_channel("railcharge", 0.26))
                        else:
                            emit(m, cost, px_, py_, e_dry, "dryfire", sounds, now,
                                 jobs=jobs)
                            msg, msg_t = f"{w.name}: empty - press R", now
                    elif mags[wi] > 0:
                        # a tap always fires one action; holding streams via the
                        # auto poll below
                        fire_cd = w.auto_refire if auto else w.burst_time
                        emit(m, cost, px_, py_, w.sound_reach_m * cpm, w.name,
                             sounds, now, jobs=jobs)
                        player_fire(w, held_moving(),
                                    acc=w.auto_accuracy if auto else None,
                                    rounds=1 if auto else None)
                        sfx_fire(w)
                        shot_glow_t = now
                        if w.blast_r <= 0.0:
                            recoil_t = sprites.RECOIL_TIME
                            flash_t = sprites.FLASH_TIME
                            flash_roll = rng.uniform(-180.0, 180.0)
                            flash_scale = rng.uniform(0.85, 1.25)
                    else:
                        # empty: the gun never reloads on its own - press R
                        emit(m, cost, px_, py_, e_dry, "dryfire", sounds, now,
                             jobs=jobs)
                        msg, msg_t = f"{w.name}: empty - press R", now
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and rail_charging:
                rail_charging = False
                if rail_charge_ch is not None:
                    rail_charge_ch.fadeout(60)
                    rail_charge_ch = None
                frac = min(rail_charge / RAIL_CHARGE_MAX, 1.0)
                rail_charge = 0.0
                w = weapons.ROSTER[loadout[wi]]
                if (mags[wi] > 0 and player.hp > 0.0 and fire_cd <= 0.0
                        and swap_t <= 0.0 and not slung):
                    mult = 1.0 + RAIL_CHARGE_BOOST * frac
                    w_fired = dataclasses.replace(
                        w, dmg_lo=w.dmg_lo * mult, dmg_hi=w.dmg_hi * mult)
                    fire_cd = w.burst_time
                    emit(m, cost, px_, py_, w.sound_reach_m * cpm, w.name,
                         sounds, now, jobs=jobs)
                    player_fire(w_fired, held_moving(), charge=frac)
                    sfx_fire(w_fired, gain=min(1.0, 0.8 + 0.4 * frac))
                    shot_glow_t = now
                    recoil_t = sprites.RECOIL_TIME
                    flash_t = sprites.FLASH_TIME
                    flash_roll = rng.uniform(-180.0, 180.0)
                    flash_scale = rng.uniform(0.85, 1.25) * (1.0 + 0.5 * frac)
                    if frac >= 0.999:
                        msg, msg_t = "rail: full charge", now

        if not paused:
            fire_cd = max(0.0, fire_cd - dt)
            swap_t = max(0.0, swap_t - dt)
            recoil_t = max(0.0, recoil_t - dt)
            flash_t = max(0.0, flash_t - dt)
            raise_t = max(0.0, raise_t - dt)
            if rail_charging:
                if (loadout[wi] not in RAIL_IDS or player.hp <= 0.0 or slung
                        or swap_t > 0.0 or reload_t > 0.0
                        or not pygame.mouse.get_pressed()[0]):
                    rail_charging = False        # abandoned - no shot
                    rail_charge = 0.0
                    if rail_charge_ch is not None:
                        rail_charge_ch.stop()
                        rail_charge_ch = None
                else:
                    rail_charge = min(RAIL_CHARGE_MAX, rail_charge + dt)
            if reload_t > 0.0:
                reload_t -= dt
                if reload_t <= 0.0:
                    reload_t = 0.0
                    _rw = weapons.ROSTER[loadout[wi]]
                    emit(m, cost, px_, py_, e_mag, "magazine", sounds, now, jobs=jobs)
                    sfx("magazine")
                    if _rw.shell_reload:
                        mags[wi] += 1                            # one shell
                        if reserves[wi] > 0:
                            reserves[wi] -= 1
                        if mags[wi] < _rw.mag and reserves[wi] != 0:
                            reload_t = _rw.shell_reload_s        # keep loading
                        else:
                            msg, msg_t = "reloaded", now
                    else:
                        _need = _rw.mag - mags[wi]
                        if reserves[wi] < 0:
                            mags[wi] = _rw.mag
                        else:
                            _take = min(_need, reserves[wi])
                            mags[wi] += _take
                            reserves[wi] -= _take
                        msg, msg_t = "reloaded", now
            # full-auto alternate fire: fire while the button is held down
            if (player.hp > 0.0 and cur_fire_mode() == "auto"
                    and fire_cd <= 0.0 and reload_t <= 0.0 and swap_t <= 0.0
                    and not slung and raise_t <= 0.0
                    and mags[wi] > 0
                    and pygame.mouse.get_pressed()[0]
                    and pygame.mouse.get_pos()[1] < view_h):
                w = cur_weapon()
                fire_cd = w.auto_refire
                emit(m, cost, px_, py_, w.sound_reach_m * cpm, w.name,
                     sounds, now, jobs=jobs)
                player_fire(w, held_moving(), acc=w.auto_accuracy, rounds=1)
                sfx_fire(w)
                shot_glow_t = now
                if w.blast_r <= 0.0:
                    recoil_t = sprites.RECOIL_TIME
                    flash_t = sprites.FLASH_TIME
                    flash_roll = rng.uniform(-180.0, 180.0)
                    flash_scale = rng.uniform(0.85, 1.25)

        keys = pygame.key.get_pressed()
        vx = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        vy = (keys[pygame.K_s] or keys[pygame.K_DOWN]) - (keys[pygame.K_w] or keys[pygame.K_UP])

        want_run = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        moving_now = bool(vx or vy) and not paused
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            gait = "crawl"
        elif want_run and not sprint_locked and stamina > 0.0:
            gait = "run"
        else:
            gait = "walk"
        speed = {"crawl": SPEED_CRAWL, "walk": SPEED_WALK, "run": SPEED_RUN}[gait]
        if slung:
            speed *= SLING_SPEED_MULT

        # stamina: sprinting burns it, standing still refills it fastest,
        # walking/sneaking refills it slowly
        if not paused:
            if gait == "run" and moving_now:
                stamina = max(0.0, stamina - STAMINA_DRAIN * dt)
                if stamina <= 0.0:
                    sprint_locked = True
            else:
                rate = STAMINA_REGEN_MOVE if moving_now else STAMINA_REGEN_IDLE
                stamina = min(1.0, stamina + rate * dt)
                if sprint_locked and stamina >= STAMINA_UNLOCK:
                    sprint_locked = False

        if (vx or vy) and not paused:
            n = math.hypot(vx, vy)
            ox, oy = px_, py_
            px_, py_ = try_move(m, px_, py_, vx / n * speed * dt, vy / n * speed * dt)
            since_step += math.hypot(px_ - ox, py_ - oy)
            if since_step >= STRIDE[gait]:
                since_step = 0.0
                cx, cy = m.cell_of(px_, py_)
                e = e_step[gait] * float(m.footstep_mult[cy, cx])
                if step_cache.get("field") is None or \
                        step_cache["field"].energy < e_step_max:
                    step_cache.clear()
                emit(m, cost, px_, py_, e, f"step/{gait}", sounds, now,
                     cache=step_cache, jobs=jobs)
                if gait != "crawl":          # sneaking makes no audible step
                    sfx("footstep", 0.09 if gait == "walk" else 0.15)

        player.x, player.y = px_, py_

        # bush concealment: standing still inside foliage hides you from the AI;
        # the moment you move you rustle it and become visible again
        _bcx, _bcy = m.cell_of(px_, py_)
        in_bush = bool(m.bush[_bcy, _bcx])
        moved_now = (px_ != prev_px or py_ != prev_py)
        prev_px, prev_py = px_, py_
        player.concealed = in_bush and not moved_now and player.hp > 0.0

        # `player.see_light`: how lit the player is right now (0..1). Guards use
        # the same "you only see what is lit" rule the player does - a dim
        # player is hard to spot. Sources: the baked lightmap, the player's own
        # flashlight (a giveaway), any muzzle flash on them, any guard torch.
        def _point_lit(sx, sy, sfac, half_deg, reach_m, tx, ty, gain):
            dx, dy = tx - sx, ty - sy
            d = math.hypot(dx, dy)
            if d >= reach_m:
                return 0.0
            if half_deg < 179.0:
                off = abs((math.atan2(dy, dx) - sfac + math.pi)
                          % (2 * math.pi) - math.pi)
                if off > math.radians(half_deg):
                    return 0.0
            scx, scy = m.cell_of(sx, sy)
            tcx, tcy = m.cell_of(tx, ty)
            if not line_of_sight(m.blocks_sight, scx, scy, tcx, tcy):
                return 0.0
            return gain * (1.0 - d / reach_m)

        _il = float(m.lightmap[_bcy, _bcx]) if m.lightmap is not None else 0.0
        if flashlight:
            _il = max(_il, FLASHLIGHT_SELF_SEEN)
        for _ml in muzzle_lights:
            _il = max(_il, _point_lit(_ml["x"], _ml["y"], 0.0, 180.0,
                                      _ml["reach"] * 1.2, px_, py_,
                                      0.9 * _ml["gain"]))
        for _g in guards:
            if _g.alive and getattr(_g, "flashlight", False):
                _il = max(_il, _point_lit(
                    _g.x, _g.y, _g.facing, ai.GUARD_FLASH_HALF_DEG,
                    ai.GUARD_FLASH_RANGE_M, px_, py_, 0.85))
        _st = max(0.0, min(1.0, (_il - LIGHT_SEE_MIN)
                           / (LIGHT_SEE_FULL - LIGHT_SEE_MIN)))
        player.see_light = _st * _st * (3.0 - 2.0 * _st)
        player_illum_now = _il          # also lifts the player sprite tint below

        if not paused and player.hp > 0.0:
            def _gemit(gx, gy, ge, glabel, cache=None):
                emit(m, cost, gx, gy, ge, glabel, sounds, now,
                     cache=cache, jobs=jobs, enemy=True)
            for g in guards:
                g.tick(dt, now)
                if not g.alive:
                    continue
                for sh in g.update(dt, now, m, cost, pen, m.blocks_sight,
                                   bbul, sounds, player, rng, _gemit):
                    if sh.segments:
                        tracers.append((sh.segments, now))
                        _o = sh.segments[0][0]
                        _mc, _mg, _mr = muzzle_light_spec(g.weapon)
                        muzzle_lights.append({"x": _o[0], "y": _o[1], "t0": now,
                                              "col": _mc, "gain": _mg, "reach": _mr})
                    if sh.shattered:
                        break_glass(sh.shattered)
                    if g.weapon.blast_r > 0.0:
                        if g.weapon.projectile_speed > 0.0 and sh.segments:
                            o = sh.segments[0][0]
                            ix, iy = sh.impact
                            d = math.hypot(ix - o[0], iy - o[1])
                            if d > 1e-6:
                                dxn, dyn = (ix - o[0]) / d, (iy - o[1]) / d
                                rockets.append({
                                    "x": o[0], "y": o[1],
                                    "dx": dxn, "dy": dyn,
                                    "w": g.weapon, "ix": ix, "iy": iy,
                                    "dist": d, "flown": 0.0})
                                if g.weapon.backblast:
                                    backblasts.append({
                                        "x": g.x - dxn * 0.55,
                                        "y": g.y - dyn * 0.55,
                                        "ang": math.atan2(-dyn, -dxn),
                                        "t0": now})
                        else:
                            blasts.append((sh.impact, g.weapon.blast_r, now))
                            _bc, _bg, _br, _bl = blast_flash_spec(g.weapon)
                            muzzle_lights.append(
                                {"x": sh.impact[0], "y": sh.impact[1], "t0": now,
                                 "col": _bc, "gain": _bg, "reach": _br, "life": _bl})
                # a guard stuck against a closed door ahead of it shoves it open
                if getattr(g, "_blocked", False):
                    _ft = getattr(g, "_face_target", g.facing)
                    fx, fy = math.cos(_ft), math.sin(_ft)
                    for (r_, c_), is_open in doors.items():
                        if is_open:
                            continue
                        vx_, vy_ = c_ + 0.5 - g.x, r_ + 0.5 - g.y
                        if (math.hypot(vx_, vy_) < GUARD_DOOR_REACH_M
                                and vx_ * fx + vy_ * fy > 0.0):
                            doors[(r_, c_)] = True
                            set_door(m, cost, r_, c_, True)
                            compute_roof(m, doors)
                            vis_cache.invalidate()
                            sounds.clear()
                            emit(m, cost, c_ + 0.5, r_ + 0.5, e_knock * 0.7,
                                 f"{g.id}/door", sounds, now, jobs=jobs, enemy=True)
                            sfx("door")
                            break
            player.tick(dt, now)

        if not paused and rockets:
            still = []
            for rk in rockets:
                v = rk["w"].projectile_speed * dt
                rk["x"] += rk["dx"] * v
                rk["y"] += rk["dy"] * v
                rk["flown"] += v
                if rk["flown"] >= rk["dist"]:
                    ctr = (rk["ix"], rk["iy"])
                    tgts = [g for g in guards if g.alive]
                    if player.alive:
                        tgts.append(player)
                    ballistics.blast(ctr, rk["w"].blast_r, rk["w"], tgts, rng, now)
                    blasts.append((ctr, rk["w"].blast_r, now))
                    _bc, _bg, _br, _bl = blast_flash_spec(rk["w"])
                    muzzle_lights.append({"x": ctr[0], "y": ctr[1], "t0": now,
                                          "col": _bc, "gain": _bg, "reach": _br,
                                          "life": _bl})
                    emit(m, cost, ctr[0], ctr[1], rk["w"].sound_reach_m * cpm,
                         "explosion", sounds, now, jobs=jobs)
                    sfx("boom")
                else:
                    still.append(rk)
            rockets = still

        if not paused:
            if player.hp <= 0.0 and respawn_t <= 0.0:
                respawn_t = RESPAWN_DELAY
                msg, msg_t = "you are down", now
            elif respawn_t > 0.0:
                respawn_t = max(0.0, respawn_t - dt)
                if respawn_t == 0.0:
                    px_, py_ = m.player_spawn
                    player.x, player.y = px_, py_
                    player.heal_full()
                    slung, raise_t = False, 0.0
                    stamina, sprint_locked = 1.0, False
                    for g in guards:
                        g.calm()
                    sounds.clear()
                    cues.clear()
                    memory.clear()
                    msg, msg_t = "respawned", now

        prof["sim"] = (time.perf_counter() - _t) * 1000

        mx, my = pygame.mouse.get_pos()
        wmx, wmy = (mx + cam_x) / PX_PER_M, (my + cam_y) / PX_PER_M   # world metres
        if my < view_h:
            facing = math.atan2(wmy - py_, wmx - px_)
        probe = m.cell_of((min(mx, view_w - 1) + cam_x) / PX_PER_M,
                          (min(my, view_h - 1) + cam_y) / PX_PER_M)

        if jobs:
            budget = max(200, SOLVE_BUDGET // len(jobs))
            jobs = [j for j in jobs if not j.step(budget)]

        pc = m.cell_of(px_, py_)
        for snd in sounds:
            if not snd.enemy or snd.cued:
                continue
            f = snd.field
            tv = f.arrival_time(*pc)
            if not math.isfinite(tv):
                if snd.done(now):
                    snd.cued = True
                continue
            if f.arrival(*pc) > snd.energy:
                snd.cued = True
                continue
            if snd.elapsed_cells(now) >= tv:
                snd.cued = True
                rem = f.remaining(*pc)
                b = f.bearing(*pc)
                cues.append({
                    "x": px_, "y": py_, "t": now,
                    "ang": math.atan2(b[1], b[0]) if b else 0.0,
                    "rem": rem,
                    "dir": b is not None and rem >= CUE_MIN_ENERGY,
                })
                _cg = min(1.0, rem * 1.3)
                if snd.label.endswith("/step"):
                    _cg *= 0.4                # footsteps are a faint cue, not loud
                sfx(audio.enemy_clip(snd.label), gain=_cg,
                    pan=(float(b[0]) if b else 0.0))
        cues = [c for c in cues if now - c["t"] < CUE_FADE]

        sounds = [s for s in sounds if not s.done(now)]

        _t = time.perf_counter()
        screen = world                  # world-space draws target the full map
        screen.blit(base, (0, 0))       # opaque + full-size, so it clears too
        if active_ov is not None:
            screen.blit(overlays[active_ov][1], (0, 0))

        # door state, drawn under the fog: solid leaf = shut, hollow green =
        # open (with the swung leaf tucked against a jamb)
        for (r_, c_), is_open in doors.items():
            drc = pygame.Rect(int(c_ * PX_PER_M), int(r_ * PX_PER_M),
                              int(PX_PER_M), int(PX_PER_M))
            if is_open:
                pygame.draw.rect(screen, DOOR_FLOOR, drc)
                lw = max(3, int(PX_PER_M * 0.22))
                pygame.draw.rect(screen, DOOR_WOOD, (drc.x, drc.y, lw, drc.h))
                pygame.draw.rect(screen, DOOR_OPEN_EDGE, drc, 2)
            else:
                pygame.draw.rect(screen, DOOR_WOOD, drc)
                pygame.draw.rect(screen, DOOR_FRAME, drc, 2)
                pygame.draw.line(screen, DOOR_FRAME, drc.midtop, drc.midbottom, 1)
        prof["base"] = (time.perf_counter() - _t) * 1000

        ppc = PX_PER_M / m.cells_per_metre

        if show_heat and sounds:
            f = sounds[-1].field
            t = f.t
            norm = np.where(np.isfinite(t), 1.0 - np.clip(t / f.energy, 0, 1), 0.0)
            s = field_surface(t, (120, 90, 200), norm * 150)
            screen.blit(scale_to_view(s, m, t.shape[1], t.shape[0]),
                        (f.x0 * ppc, f.y0 * ppc))

        _t = time.perf_counter()
        ripple_buf[:] = 0.0
        ripple_enemy[:] = 0.0
        any_ripple = False
        # fine-cell rect currently on screen
        vx0, vy0 = cam_x / ppc, cam_y / ppc
        vx1, vy1 = (cam_x + view_w) / ppc, (cam_y + view_h) / ppc
        band = 2.5
        # only the last few sounds can still have a visible wavefront; cap the
        # per-frame array work so a burst of gunfire cannot stall the frame.
        # collapse a burst of enemy fire: one ring per emitter (the newest), not
        # a stack of concentric arcs sweeping the whole screen.
        _ripple_src = []
        _seen_org = set()
        for _s in reversed(sounds[-8:]):
            if _s.enemy:
                _o = _s.field.origin
                _ok = (round(_o[0] / 2.0), round(_o[1] / 2.0))
                if _ok in _seen_org:
                    continue
                _seen_org.add(_ok)
            _ripple_src.append(_s)
        for snd in _ripple_src:
            # bail before any array work if this sound will not be drawn at all
            if snd.enemy and enemy_mode != 2:
                continue
            if not snd.enemy and not show_own_sound:
                continue
            f = snd.field
            ox, oy = f.origin
            # an enemy ripple is a DIRECTION cue: only show it when the player
            # has no line of sight to the source (if you can see it, you don't
            # need a wave pointing at it), and only when the sound actually
            # reaches the player - then dim the ring by how loud it is there
            if snd.enemy:
                if line_of_sight(m.blocks_sight, pc[0], pc[1], ox, oy):
                    continue
                ploss = f.arrival(*pc)
                if not math.isfinite(ploss) or ploss > snd.energy:
                    continue
                p_rem = 1.0 - ploss / snd.energy
            else:
                p_rem = 1.0
            t, tv = f.t, f.travel
            el = snd.elapsed_cells(now)
            if el - band > f.max_travel:
                continue
            # cull if the expanding ring's radius does not currently bracket the
            # visible rectangle (it has already swept past, or not reached it)
            near_x = 0.0 if vx0 <= ox <= vx1 else min(abs(ox - vx0), abs(ox - vx1))
            near_y = 0.0 if vy0 <= oy <= vy1 else min(abs(oy - vy0), abs(oy - vy1))
            far = math.hypot(max(abs(ox - vx0), abs(ox - vx1)),
                             max(abs(oy - vy0), abs(oy - vy1)))
            if el - band > far or el + band < math.hypot(near_x, near_y):
                continue
            # the ring follows geometric travel, so it expands at a constant
            # speed; brightness follows attenuation, so it dims where the
            # route was lossy. Sound through a wall is quiet, not late.
            d = np.abs(tv - el)
            open_sub = audible_open[f.y0:f.y0 + t.shape[0],
                                    f.x0:f.x0 + t.shape[1]]
            ring = np.where(np.isfinite(tv) & (d < band) & (t <= snd.energy)
                            & open_sub, (1.0 - d / band) ** 2, 0.0)
            if ring.max() <= 0.0:
                continue
            any_ripple = True
            atten = np.where(np.isfinite(t), 1.0 - np.clip(t / snd.energy, 0, 1), 0.0)
            dest = ripple_enemy if snd.enemy else ripple_buf
            sub = dest[f.y0:f.y0 + t.shape[0], f.x0:f.x0 + t.shape[1]]
            np.maximum(sub, ring * atten * (235.0 * (0.3 + 0.7 * p_rem)), out=sub)
        prof["ripple"] = (time.perf_counter() - _t) * 1000

        _t = time.perf_counter()
        pcx, pcy = m.cell_of(px_, py_)
        vf = vis_cache.get(pcx, pcy, cone.max_range)
        inten_raw = vf.cone_intensity(facing, cone)
        prof["vision"] = (time.perf_counter() - _t) * 1000
        h_, w_ = inten_raw.shape

        # player flashlight (l): a tight beam - hard side edges, no outer spill,
        # a smooth range gradient that fades out before the LOS cone edge.
        fl = None
        if flashlight and not in_bush:
            _fd = np.abs((vf.ang - facing + np.pi) % (2 * np.pi) - np.pi)
            ang_ok = (_fd <= math.radians(FLASHLIGHT_HALF_DEG)) & vf.visible
            reach = max(cone.identify_range * 0.9, 1.0)
            t = np.clip(1.0 - vf.dist / reach, 0.0, 1.0)
            t = t * t * (3.0 - 2.0 * t)                 # smooth tip gradient
            fl = np.where(ang_ok, t, 0.0).astype(np.float32)
            fl = box_blur(fl, FLASHLIGHT_BLUR)          # 1 = just anti-alias the edge

        # active muzzle flashes as a brief radial light, shadow-cast from each
        # flash so a wall stops it, then masked to what the player can see.
        # `mflash` = mono field (perception + world brighten), `mflash_rgb` =
        # the per-flash coloured glow drawn over the fog.
        mflash = None
        mflash_rgb = None
        if muzzle_lights:
            _mf = np.zeros((h_, w_), np.float32)
            _mrgb = np.zeros((h_, w_, 3), np.float32)
            for ml in muzzle_lights:
                _age = (now - ml["t0"]) / ml.get("life", MUZZLE_LIGHT_TIME)
                if _age >= 1.0:
                    continue
                _b = _age / 0.32 if _age < 0.32 else (1.0 - _age) / 0.68
                _b = max(0.0, _b) ** 1.25 * ml["gain"]
                if _b <= 0.02:
                    continue
                _fcx, _fcy = m.cell_of(ml["x"], ml["y"])
                _rc = max(2, int(ml["reach"] * cpm * 1.3))
                _sf = shadowcast(m.blocks_sight, _fcx, _fcy, _rc)
                _lit = np.where(
                    _sf.visible,
                    np.clip(1.0 - _sf.dist / _rc, 0.0, 1.0) ** 1.6 * _b,
                    0.0).astype(np.float32)
                _oy, _ox = _sf.y0 - vf.y0, _sf.x0 - vf.x0
                _y0, _x0 = max(0, _oy), max(0, _ox)
                _y1 = min(h_, _oy + _lit.shape[0])
                _x1 = min(w_, _ox + _lit.shape[1])
                if _y1 > _y0 and _x1 > _x0:
                    src = _lit[_y0 - _oy:_y1 - _oy, _x0 - _ox:_x1 - _ox]
                    np.maximum(_mf[_y0:_y1, _x0:_x1], src,
                               out=_mf[_y0:_y1, _x0:_x1])
                    _col = np.array(ml["col"], np.float32) / 255.0
                    _mrgb[_y0:_y1, _x0:_x1] += src[:, :, None] * _col
            if _mf.max() > 0.0:
                _vis = vf.visible.astype(np.float32)
                _mf = box_blur(_mf, MFLASH_BLUR) * _vis
                if _mf.max() > 0.0:
                    mflash = _mf
                _mrgb *= _vis[:, :, None]
                if _mrgb.max() > 0.0:
                    mflash_rgb = _mrgb

        # guard flashlights: each lit guard's beam shadow-cast from that guard,
        # masked to what the PLAYER can see - so you see the room a guard's
        # torch lights, and the sweeping cone itself
        gbeam = None
        _lg = [g for g in guards
               if g.alive and getattr(g, "flashlight", False)
               and cam_x - 200 < g.x * PX_PER_M < cam_x + view_w + 200
               and cam_y - 200 < g.y * PX_PER_M < cam_y + view_h + 200]
        if _lg:
            _gb = np.zeros((h_, w_), np.float32)
            _grc = max(2, int(ai.GUARD_FLASH_RANGE_M * cpm))
            _gha = math.radians(ai.GUARD_FLASH_HALF_DEG)
            for g in _lg:
                gcx, gcy = m.cell_of(g.x, g.y)
                _gsf = shadowcast(m.blocks_sight, gcx, gcy, _grc)
                _gd = np.abs((_gsf.ang - g.facing + np.pi) % (2 * np.pi) - np.pi)
                _gt = np.clip(1.0 - _gsf.dist / _grc, 0.0, 1.0)
                _gt = _gt * _gt * (3.0 - 2.0 * _gt)
                _gl = np.where((_gd <= _gha) & _gsf.visible, _gt, 0.0).astype(np.float32)
                _oy, _ox = _gsf.y0 - vf.y0, _gsf.x0 - vf.x0
                _y0, _x0 = max(0, _oy), max(0, _ox)
                _y1 = min(h_, _oy + _gl.shape[0])
                _x1 = min(w_, _ox + _gl.shape[1])
                if _y1 > _y0 and _x1 > _x0:
                    np.maximum(_gb[_y0:_y1, _x0:_x1],
                               _gl[_y0 - _oy:_y1 - _oy, _x0 - _ox:_x1 - _ox],
                               out=_gb[_y0:_y1, _x0:_x1])
            _gb *= vf.visible
            if _gb.max() > 0.0:
                gbeam = box_blur(_gb, 1)

        # STATIC illumination (baked lightmap + flashlight) gates what enters
        # fog memory: a room with no lamp and the flashlight off reveals nothing.
        if m.lightmap is not None:
            illum = m.lightmap[vf.y0:vf.y0 + h_, vf.x0:vf.x0 + w_].astype(np.float32)
        else:
            illum = np.zeros((h_, w_), dtype=np.float32)
        if fl is not None:
            illum = np.maximum(illum, fl)
        if gbeam is not None:
            illum = np.maximum(illum, gbeam)
        see = np.clip((illum - LIGHT_SEE_MIN) / (LIGHT_SEE_FULL - LIGHT_SEE_MIN),
                      0.0, 1.0)
        gated = inten_raw * see
        if in_bush:
            # hunkered in foliage: you can't see out, only the leaves around you
            bsub = m.bush[vf.y0:vf.y0 + h_, vf.x0:vf.x0 + w_]
            gated = np.where(bsub, gated, 0.0)
        known[vf.y0:vf.y0 + h_, vf.x0:vf.x0 + w_] |= gated > 0.03
        inten_raw = gated                        # fog cone-carve + roof peek use this

        # additive brighten of the world (base surface + characters) under the
        # flashlight and any muzzle flash - BEFORE the fog, like a real light
        add = None
        if fl is not None:
            add = fl * FLASHLIGHT_GAIN
        if mflash is not None:
            _m2 = mflash * MUZZLE_LIGHT_GAIN
            add = _m2 if add is None else np.maximum(add, _m2)
        if gbeam is not None:
            _g2 = gbeam * GUARD_FLASH_GAIN
            add = _g2 if add is None else np.maximum(add, _g2)
        if add is not None and add.max() > 1.0:
            g = np.clip(add, 0, 255).astype(np.uint8)
            warm = np.stack([g, (g * 0.94).astype(np.uint8),
                             (g * 0.82).astype(np.uint8)], axis=2)
            tex = pygame.surfarray.make_surface(np.transpose(warm, (1, 0, 2)))
            sc = pygame.transform.smoothscale(
                tex, (max(1, round(w_ * ppc)), max(1, round(h_ * ppc))))
            screen.blit(sc, (round(vf.x0 * ppc), round(vf.y0 * ppc)),
                        special_flags=pygame.BLEND_RGB_ADD)

        _t = time.perf_counter()
        if show_fog:
            ov_a = np.where(known, A_REMEMBER_F, A_UNKNOWN_F).astype(np.float32)
            ov_rgb = np.zeros(known.shape + (3,), dtype=np.float32)
            # carve the cone into the veil from the crisp field: cells in view
            # are lightened (never below their remembered level)
            sub_a = ov_a[vf.y0:vf.y0 + h_, vf.x0:vf.x0 + w_]
            lit = inten_raw > 0.004
            sub_a[lit] = np.minimum(
                sub_a[lit],
                A_REMEMBER_F * (1.0 - CONE_REVEAL * inten_raw[lit]))
            if mflash is not None:
                # a muzzle flash briefly parts the veil ONLY where it lights - a
                # transient lift that never writes `known`
                _mm = mflash > 0.004
                mlift = np.clip(mflash[_mm] * 2.2, 0.0, 1.0)
                sub_a[_mm] = np.minimum(sub_a[_mm], A_REMEMBER_F * (1.0 - mlift))
            # then blur the whole alpha field once: this is what feathers the
            # cone rim, the shadow edges AND the explored/fog boundary - every
            # hard line between "seen now", "remembered" and "unknown" softens
            ov_a = box_blur(ov_a, FOG_BLUR_R)
        else:
            ov_rgb = np.zeros(known.shape + (3,), dtype=np.float32)
            ov_a = np.zeros(known.shape, dtype=np.float32)

        if any_ripple:
            own = ripple_buf
            enemy = ripple_enemy
            strength = np.maximum(own, enemy)
            ra = strength / 255.0
            mine = own >= enemy
            rip_rgb = np.where(mine[..., None],
                               np.array(RIPPLE, dtype=np.float32),
                               np.array(RIPPLE_ENEMY, dtype=np.float32))
            out_a = ra + ov_a * (1.0 - ra)
            safe = np.maximum(out_a, 1e-6)[..., None]
            ov_rgb = (rip_rgb * ra[..., None]
                      + ov_rgb * (ov_a * (1.0 - ra))[..., None]) / safe
            ov_a = out_a

        if any_ripple or show_fog:
            # composite only the fine-grid slice that is on screen, then scale
            # that to viewport pixels - not the whole world every frame
            foh, fow = ov_a.shape
            mgn = FOG_BLUR_R + 2
            fx0 = max(0, int(cam_x / ppc) - mgn)
            fy0 = max(0, int(cam_y / ppc) - mgn)
            fx1 = min(fow, int((cam_x + view_w) / ppc) + mgn + 1)
            fy1 = min(foh, int((cam_y + view_h) / ppc) + mgn + 1)
            fw, fh = fx1 - fx0, fy1 - fy0
            if ov_surf.get_size() != (fw, fh):
                ov_surf = pygame.Surface((fw, fh), pygame.SRCALPHA)
            px3 = pygame.surfarray.pixels3d(ov_surf)
            pxa = pygame.surfarray.pixels_alpha(ov_surf)
            px3[:, :, :] = np.transpose(
                ov_rgb[fy0:fy1, fx0:fx1].astype(np.uint8), (1, 0, 2))
            pxa[:, :] = np.transpose(np.clip(
                ov_a[fy0:fy1, fx0:fx1] * 255.0, 0, 255)).astype(np.uint8)
            del px3, pxa
            scaled = pygame.transform.scale(
                ov_surf, (round(fw * ppc), round(fh * ppc)))
            screen.blit(scaled, (round(fx0 * ppc), round(fy0 * ppc)))
        prof["overlay"] = (time.perf_counter() - _t) * 1000

        if enemy_mode == 1:
            for c in cues:
                age = (now - c["t"]) / CUE_FADE
                alpha = (1.0 - age) ** 1.5 * 210
                if alpha < 2:
                    continue
                if c["dir"]:
                    half = CUE_NARROW_DEG + (CUE_WIDE_DEG - CUE_NARROW_DEG) * \
                        (1.0 - min(1.0, c["rem"] / 0.55))
                else:
                    half = 180.0
                draw_cue(screen, c["x"] * PX_PER_M, c["y"] * PX_PER_M,
                         c["ang"], half, alpha, PX_PER_M)

        _t = time.perf_counter()
        if show_grid:
            gpx = m.metres_per_char * PX_PER_M
            for c in range(m.chars.shape[1] + 1):
                pygame.draw.line(screen, (60, 60, 58), (c * gpx, 0), (c * gpx, world_h))
            for r in range(m.chars.shape[0] + 1):
                pygame.draw.line(screen, (60, 60, 58), (0, r * gpx), (world_w, r * gpx))

        if show_debug:
            for lt in m.lights:
                c = (int(lt.pos[0] * PX_PER_M), int(lt.pos[1] * PX_PER_M))
                pygame.draw.circle(screen, LIGHT, c, 4)
                pygame.draw.circle(screen, LIGHT, c, int(lt.radius * PX_PER_M), 1)

        for s_ in m.idle_spots:
            sx, sy = int(s_.pos[0] * PX_PER_M), int(s_.pos[1] * PX_PER_M)
            pygame.draw.circle(screen, DIM, (sx, sy), 5, 1)
            a = math.radians(s_.facing_deg)
            pygame.draw.line(screen, DIM, (sx, sy),
                             (sx + math.cos(a) * 16, sy + math.sin(a) * 16), 1)

        if show_cones:
            for g in guards:
                if g.alive:
                    draw_guard_cone(screen, g, PX_PER_M)

        live_n = 0
        for g in guards:
            gcx, gcy = m.cell_of(g.x, g.y)
            band = vf.band_at(gcx, gcy, facing, cone)
            sx, sy = g.x * PX_PER_M, g.y * PX_PER_M
            if not g.alive:
                if not (bank.ok and bank.blit(screen, "soldier_idle", sx, sy,
                                              g.facing, tint=(70, 70, 70), alpha=200)):
                    pygame.draw.circle(screen, DEAD, (int(sx), int(sy)), 7)
                pygame.draw.line(screen, DEAD, (sx - 6, sy - 6), (sx + 6, sy + 6), 2)
                pygame.draw.line(screen, DEAD, (sx - 6, sy + 6), (sx + 6, sy - 6), 2)
                continue
            if band in (1, 2):
                live_n += 1
                memory[g.id] = (g.x, g.y, g.facing, now)
                col = MODE_COL[g.mode]
                tense = g.mode in (ai.Mode.COMBAT, ai.Mode.SEARCH)
                g_slung = g.reload_t > 0.0 or not tense   # reloading -> gun down
                if bank.ok:
                    bank.blit(screen, "soldier_idle" if g_slung else "soldier_ready",
                              sx, sy, g.facing)
                    gkick = (g.recoil_t / ai.RECOIL_TIME
                             * sprites.RECOIL_M * PX_PER_M)
                    gwx = sx - math.cos(g.facing) * gkick
                    gwy = sy - math.sin(g.facing) * gkick
                    bank.blit(screen,
                              "weapon_sling" if g_slung else sprites.weapon_art_for(g.weapon),
                              gwx, gwy, g.facing)
                    if not g_slung:
                        draw_muzzle(sprites.weapon_art_for(g.weapon), gwx, gwy,
                                    g.facing, g.flash_t, g.flash_roll, g.flash_scale)
                    pygame.draw.circle(screen, col, (int(sx), int(sy)), 12, 1)
                else:
                    pygame.draw.circle(screen, col, (int(sx), int(sy)), 7)
                    pygame.draw.line(screen, (28, 28, 26), (sx, sy),
                                     (sx + math.cos(g.facing) * 17,
                                      sy + math.sin(g.facing) * 17), 3)
                if g.alert > 0.02:
                    pygame.draw.circle(screen, col, (int(sx), int(sy)),
                                       int(14 + 6 * g.alert), 1)
                pygame.draw.rect(screen, (20, 20, 20), (sx - 9, sy - 14, 18, 3))
                pygame.draw.rect(screen, (90, 200, 120),
                                 (sx - 9, sy - 14, int(18 * g.health / g.max_health), 3))
                if g.max_shields > 0 and g.shields > 0:
                    pygame.draw.rect(screen, (90, 160, 230),
                                     (sx - 9, sy - 17, int(18 * g.shields / g.max_shields), 2))
                if g.stamina < 0.98:            # sprint fuel (only when spent)
                    pygame.draw.rect(screen, (20, 20, 20), (sx - 9, sy - 20, 18, 2))
                    pygame.draw.rect(
                        screen,
                        (120, 120, 120) if g.sprint_locked else (210, 185, 110),
                        (sx - 9, sy - 20, int(18 * g.stamina), 2))
            elif band == 3:
                pygame.draw.circle(screen, BLIP, (int(sx), int(sy)), 5, 2)
            elif g.id in memory:
                mx_, my_, mf, mt = memory[g.id]
                gx, gy = mx_ * PX_PER_M, my_ * PX_PER_M
                pygame.draw.circle(screen, GHOST, (int(gx), int(gy)), 7, 2)
                pygame.draw.line(screen, GHOST, (gx, gy),
                                 (gx + math.cos(mf) * 15, gy + math.sin(mf) * 15), 1)
                age = font.render(f"{now - mt:.0f}s", True, GHOST)
                screen.blit(age, (gx + 9, gy - 18))

        # the sound-field probe at the cursor is a debug visualisation only
        # (it draws a line straight to the emitter) - hidden unless § is on
        probe_txt = "probe: nothing heard at cursor"
        if show_debug and sounds:
            f = sounds[-1].field
            arr = f.arrival(*probe)
            if math.isfinite(arr):
                b = f.bearing(*probe)
                rem = f.remaining(*probe)
                pxp = probe[0] * ppc + ppc / 2
                pyp = probe[1] * ppc + ppc / 2
                pygame.draw.line(screen, (90, 88, 84), (pxp, pyp),
                                 (f.origin[0] * ppc, f.origin[1] * ppc), 1)
                if b:
                    bx, by = float(b[0]), float(b[1])
                    pygame.draw.line(screen, PROBE, (pxp, pyp),
                                     (pxp + bx * 46, pyp + by * 46), 3)
                    heard = math.degrees(math.atan2(by, bx)) % 360
                    sx, sy = f.origin
                    true = math.degrees(math.atan2(sy - probe[1], sx - probe[0])) % 360
                    err = (heard - true + 180) % 360 - 180
                    probe_txt = (f"probe: heard {heard:6.1f}  true {true:6.1f}  "
                                 f"err {err:+6.1f}deg  travel {f.arrival_time(*probe):6.1f}"
                                 f"  loss {arr:6.1f}  energy {rem:.2f}")
                pygame.draw.circle(screen, PROBE, (int(pxp), int(pyp)), 4, 1)

        prof["entities"] = (time.perf_counter() - _t) * 1000

        for segs, t0 in tracers:
            fade = 1.0 - (now - t0) / TRACER_FADE
            if fade <= 0.0:
                continue
            for p0, p1, kind in segs:
                air = kind == "air"
                pygame.draw.line(
                    screen, TRACER_AIR if air else TRACER_WALL,
                    (p0[0] * PX_PER_M, p0[1] * PX_PER_M),
                    (p1[0] * PX_PER_M, p1[1] * PX_PER_M), 2 if air else 1)
        tracers = [tp for tp in tracers if now - tp[1] < TRACER_FADE]

        for rk in rockets:                       # in-flight projectiles
            rx, ry = rk["x"] * PX_PER_M, rk["y"] * PX_PER_M
            if rk["w"].category in ("plasma", "energy", "laser"):
                c_trail, c_core, tlen = (130, 220, 255), (235, 250, 255), 44
            else:
                c_trail, c_core, tlen = (250, 190, 120), (255, 235, 190), 22
            tx = rx - rk["dx"] * tlen
            ty = ry - rk["dy"] * tlen
            pygame.draw.line(screen, c_trail, (tx, ty), (rx, ry), 3)
            pygame.draw.circle(screen, c_core, (int(rx), int(ry)), 4)

        for bb in backblasts:                    # rocket exhaust cone (placeholder)
            age = (now - bb["t0"]) / BACKBLAST_FADE
            if age >= 1.0:
                continue
            cxp, cyp = bb["x"] * PX_PER_M, bb["y"] * PX_PER_M
            rr = int((0.5 + 1.7 * age) * PX_PER_M)
            cs = pygame.Surface((rr * 2 + 4, rr * 2 + 4), pygame.SRCALPHA)
            # TODO: blit one of the 3 backblast sprites by `age` once art lands
            pts = [(rr + 2, rr + 2)]
            for k in range(-3, 4):
                aa = bb["ang"] + k * 0.17
                pts.append((rr + 2 + math.cos(aa) * rr, rr + 2 + math.sin(aa) * rr))
            pygame.draw.polygon(cs, (245, 175, 105, int(130 * (1.0 - age) ** 2)), pts)
            screen.blit(cs, (cxp - rr - 2, cyp - rr - 2))
        backblasts = [b for b in backblasts if now - b["t0"] < BACKBLAST_FADE]

        if mflash_rgb is not None:               # coloured muzzle-flash glow,
            # already shadow-cast per flash + masked to player LOS - no bleed
            g = np.clip(mflash_rgb * MFLASH_POP_GAIN, 0, 255).astype(np.uint8)
            tex = pygame.surfarray.make_surface(np.transpose(g, (1, 0, 2)))
            sc = pygame.transform.smoothscale(
                tex, (max(1, round(w_ * ppc)), max(1, round(h_ * ppc))))
            screen.blit(sc, (round(vf.x0 * ppc), round(vf.y0 * ppc)),
                        special_flags=pygame.BLEND_RGB_ADD)
        muzzle_lights = [ml for ml in muzzle_lights
                         if now - ml["t0"] < ml.get("life", MUZZLE_LIGHT_TIME)]

        if gbeam is not None:                    # a guard torch beam over the fog
            g = np.clip(gbeam * GUARD_FLASH_POP, 0, 255).astype(np.uint8)
            cool = np.stack([g, g, (g * 1.0).astype(np.uint8)], axis=2)
            tex = pygame.surfarray.make_surface(np.transpose(cool, (1, 0, 2)))
            sc = pygame.transform.smoothscale(
                tex, (max(1, round(w_ * ppc)), max(1, round(h_ * ppc))))
            screen.blit(sc, (round(vf.x0 * ppc), round(vf.y0 * ppc)),
                        special_flags=pygame.BLEND_RGB_ADD)

        for (bx, by), br, t0 in blasts:
            age = (now - t0) / BLAST_FADE
            if age >= 1.0:
                continue
            cxp, cyp = int(bx * PX_PER_M), int(by * PX_PER_M)
            rpx = int(br * PX_PER_M * (0.45 + 0.55 * age))   # expands to full radius
            bs = pygame.Surface((rpx * 2 + 4, rpx * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(bs, (255, 170, 70, int(120 * (1.0 - age) ** 2)),
                               (rpx + 2, rpx + 2), rpx)
            pygame.draw.circle(bs, (255, 230, 160, int(200 * (1.0 - age))),
                               (rpx + 2, rpx + 2), rpx, 2)
            screen.blit(bs, (cxp - rpx - 2, cyp - rpx - 2))
        blasts = [b for b in blasts if now - b[2] < BLAST_FADE]

        _t = time.perf_counter()
        ppx, ppy = px_ * PX_PER_M, py_ * PX_PER_M
        down = player.hp <= 0.0

        # the player sprite's brightness + contrast track the light on their
        # cell (crushed toward a dark silhouette in shadow), lifted by the
        # flashlight when it is on
        _pl = float(m.lightmap[pcy, pcx]) if m.lightmap is not None else 0.0
        _pl = max(_pl, player_illum_now)         # lamps + a guard's torch on you
        if flashlight:
            _pl = max(_pl, FLASHLIGHT_SELF)
        if mflash is not None:                   # your own / a nearby muzzle flash
            _ly, _lx = pcy - vf.y0, pcx - vf.x0
            if 0 <= _ly < h_ and 0 <= _lx < w_:
                _pl = max(_pl, float(mflash[_ly, _lx]) * 1.4)
        _sg = now - shot_glow_t                  # brief pop of light on each shot
        if 0.0 <= _sg < SHOT_GLOW_TIME:
            _pl = max(_pl, SHOT_GLOW * (1.0 - _sg / SHOT_GLOW_TIME) ** 0.6)
        _k = min(1.0, max(0.0, (_pl - 0.14) / 0.78))
        _k = _k * _k * (3.0 - 2.0 * _k)
        _v = int(3 + (255 - 3) * _k)
        ptint = (_v, int(_v * 0.97), int(_v * 0.87)) if flashlight else (_v, _v, _v)
        if down:
            ptint = tuple(int(c * 0.55) for c in ptint)

        drew = False
        if bank.ok:
            if down:
                drew = bank.blit(screen, "soldier_idle", ppx, ppy, facing,
                                 tint=ptint)
            elif swap_t > 0.0:
                pose, wsprite = sprites.swap_frame(1.0 - swap_t / sprites.SWAP_TIME,
                                                   loadout[wi])
                drew = bank.blit(screen, pose, ppx, ppy, facing, tint=ptint)
                if wsprite:
                    bank.blit(screen, wsprite, ppx, ppy, facing, tint=ptint)
            elif reload_t > 0.0:
                drew = bank.blit(screen, "soldier_idle", ppx, ppy, facing, tint=ptint)
                bank.blit(screen, "weapon_sling", ppx, ppy, facing, tint=ptint)
            elif slung or raise_t > 0.0:
                drew = bank.blit(screen, "soldier_idle", ppx, ppy, facing, tint=ptint)
                bank.blit(screen, "weapon_sling", ppx, ppy, facing, tint=ptint)
            else:
                drew = bank.blit(screen, "soldier_ready", ppx, ppy, facing, tint=ptint)
                kick = recoil_t / sprites.RECOIL_TIME * sprites.RECOIL_M * PX_PER_M
                gwx = ppx - math.cos(facing) * kick
                gwy = ppy - math.sin(facing) * kick
                bank.blit(screen, sprites.weapon_art(loadout[wi]), gwx, gwy, facing,
                          tint=ptint)
                draw_muzzle(sprites.weapon_art(loadout[wi]), gwx, gwy, facing,
                            flash_t, flash_roll, flash_scale)
            if drew and not down:
                # flashlight kit overlay, centred on the body like a weapon.
                # off = an unlit attachment (follows body light); on = the lamp
                # itself, drawn full-bright.
                if flashlight:
                    bank.blit(screen, "light_on", ppx, ppy, facing)
                else:
                    bank.blit(screen, "light_off", ppx, ppy, facing, tint=ptint)
        if not drew:
            pygame.draw.line(screen, FACING, (ppx, ppy),
                             (ppx + math.cos(facing) * 26,
                              ppy + math.sin(facing) * 26), 2)
            pygame.draw.circle(screen, DEAD if down else PLAYER,
                               (int(ppx), int(ppy)), int(BODY_R * PX_PER_M))

        # roof pass: a building's roof draws OVER everything - floor, walls,
        # characters, fx - so its interior is hidden until the vision cone
        # reaches inside (through a window or an open door) or the player is
        # standing inside that same building.
        if m.roof is not None:
            roof_a = np.where(m.roof, 1.0, 0.0).astype(np.float32)
            rslc = roof_a[vf.y0:vf.y0 + h_, vf.x0:vf.x0 + w_]
            rslc[inten_raw > ROOF_REVEAL] = 0.0
            pbid = int(m.roof_bid[pcy, pcx])
            if pbid:
                roof_a[m.roof_bid == pbid] = 0.0
            roof_a = box_blur(roof_a, ROOF_BLUR_R)
            # snap the plateau back to fully opaque - the blur is only meant to
            # feather the reveal edge, not thin the whole cover
            roof_a = np.minimum(roof_a * 1.8, 1.0)
            roh, rcol = roof_a.shape
            rmgn = ROOF_BLUR_R + 2
            gx0 = max(0, int(cam_x / ppc) - rmgn)
            gy0 = max(0, int(cam_y / ppc) - rmgn)
            gx1 = min(rcol, int((cam_x + view_w) / ppc) + rmgn + 1)
            gy1 = min(roh, int((cam_y + view_h) / ppc) + rmgn + 1)
            gw, gh = gx1 - gx0, gy1 - gy0
            if gw > 0 and gh > 0:
                if roof_surf.get_size() != (gw, gh):
                    roof_surf = pygame.Surface((gw, gh), pygame.SRCALPHA)
                rp3 = pygame.surfarray.pixels3d(roof_surf)
                rpa = pygame.surfarray.pixels_alpha(roof_surf)
                rp3[:, :, :] = np.transpose(
                    m.roof_rgb[gy0:gy1, gx0:gx1], (1, 0, 2))
                rpa[:, :] = np.transpose(np.clip(
                    roof_a[gy0:gy1, gx0:gx1] * 255.0, 0, 255)).astype(np.uint8)
                del rp3, rpa
                rsc = pygame.transform.scale(
                    roof_surf, (round(gw * ppc), round(gh * ppc)))
                screen.blit(rsc, (round(gx0 * ppc), round(gy0 * ppc)))

        # lift the camera window out of the world and back to the real screen
        # (world is opaque and always covers the viewport, so no fill needed)
        window.blit(world, (-cam_x, -cam_y))
        screen = window

        # crosshair (own cursor; system cursor is hidden) - fixed, tells nothing
        mxp, myp = pygame.mouse.get_pos()
        if myp < view_h:
            for a, b in (((-11, 0), (-4, 0)), ((4, 0), (11, 0)),
                         ((0, -11), (0, -4)), ((0, 4), (0, 11))):
                pygame.draw.line(screen, (235, 235, 235), (mxp + a[0], myp + a[1]),
                                 (mxp + b[0], myp + b[1]), 2)
            if rail_charging:                     # rail spool-up ring, own state
                _cf = min(rail_charge / RAIL_CHARGE_MAX, 1.0)
                pygame.draw.circle(screen, (60, 90, 150), (mxp, myp), 16, 1)
                if _cf > 0.0:
                    pygame.draw.arc(screen, (120, 190, 255),
                                    (mxp - 16, myp - 16, 32, 32),
                                    -math.pi / 2,
                                    -math.pi / 2 + _cf * 2 * math.pi,
                                    3 if _cf < 0.999 else 4)

        pygame.draw.rect(screen, HUD_BG, (0, view_h, view_w, HUD_H))
        cx, cy = m.cell_of(px_, py_)
        ch = m.chars[int(py_), int(px_)]
        y0 = view_h
        _w = weapons.ROSTER[loadout[wi]]
        _fm = cur_fire_mode()

        if not show_debug:
            def _bar(x, y, wd, ht, frac, fg, bg=(44, 44, 48)):
                pygame.draw.rect(screen, bg, (x, y, wd, ht))
                fw = int(wd * max(0.0, min(1.0, frac)))
                if fw > 0:
                    pygame.draw.rect(screen, fg, (x, y, fw, ht))
                pygame.draw.rect(screen, (12, 12, 14), (x, y, wd, ht), 1)

            bx, bw = 22, 300
            if player.max_shields > 0:
                _bar(bx, y0 + 20, bw, 15,
                     player.shields / player.max_shields, (90, 160, 235))
                screen.blit(font_mid.render(
                    f"SH {int(player.shields):3d}/{int(player.max_shields):<3d}",
                    True, TEXT), (bx + bw + 12, y0 + 17))
            hy = y0 + 42
            hpf = player.health / max(1.0, player.max_health)
            hpc = ((90, 200, 120) if hpf > 0.5 else
                   (232, 200, 70) if hpf > 0.25 else (225, 70, 55))
            _bar(bx, hy, bw, 22, hpf, hpc)
            screen.blit(font_big.render(f"{max(0, int(player.health))}", True, TEXT),
                        (bx + bw + 12, hy - 9))
            screen.blit(font.render(
                f"/ {int(player.max_health)}    armor {player.armor_lo:.0f}%",
                True, DIM), (bx + bw + 74, hy + 7))
            _bar(bx, y0 + 70, bw, 7, stamina,
                 (120, 120, 130) if sprint_locked else (230, 205, 110))
            if player.hp <= 0.0:
                screen.blit(font_big.render("DOWN", True, (225, 70, 55)),
                            (bx, y0 + 80))

            wx = view_w // 2 - 130
            screen.blit(font_mid.render(_w.name.upper(), True, TEXT), (wx, y0 + 16))
            _st = ("SLUNG" if slung else "RAISING" if raise_t > 0 else
                   "RELOADING" if reload_t > 0 else
                   "SWITCHING" if swap_t > 0 else _fm.upper())
            screen.blit(font.render(_st, True,
                        (232, 200, 70) if (reload_t > 0 or swap_t > 0 or slung
                                           or raise_t > 0) else DIM),
                        (wx, y0 + 40))
            _am = font_big.render(f"{mags[wi]:2d}", True,
                                  (225, 70, 55) if mags[wi] == 0 else TEXT)
            screen.blit(_am, (wx, y0 + 58))
            _res = "∞" if reserves[wi] < 0 else str(reserves[wi])
            screen.blit(font.render(f"/ {_w.mag}   res {_res}", True, DIM),
                        (wx + _am.get_width() + 8, y0 + 74))

            # no detection readout by design: the player reads the guards
            # themselves and the sound world, nothing is spelled out

            if msg and now - msg_t < 2.5:
                _mm = font_mid.render(msg, True, (245, 220, 140))
                screen.blit(_mm, (view_w // 2 - _mm.get_width() // 2,
                                  y0 + HUD_H - 32))
            screen.blit(font.render(f"{clock.get_fps():4.0f} fps    § = debug",
                        True, DIM), (view_w - 170, y0 + HUD_H - 18))
            pygame.display.flip()
            continue

        last_solve = "-"
        if sounds:
            s0 = sounds[-1]
            last_solve = (f"{s0.label} energy {s0.field.energy:.0f} "
                          f"window {s0.field.t.shape[1]}x{s0.field.t.shape[0]} "
                          f"solve {s0.solve_ms:.1f}ms")
        lines = [
            f"pos {px_:6.2f},{py_:6.2f}m  cell {cx:3d},{cy:3d}  tile '{ch}' "
            f"{m.tiles[ch].name}  gait {gait}"
            f"{'  [CONCEALED]' if player.concealed else ('  [in bush - moving]' if in_bush else '')}",
            f"sounds {len(sounds)} (enemy {sum(1 for s_ in sounds if s_.enemy)})"
            f"  solving {len(jobs)}  own {'on' if show_own_sound else 'OFF'}"
            f"  enemy {('off', 'arcs', 'ripples')[enemy_mode]}"
            f"  cues {len(cues)}  last: {last_solve}",
            probe_txt,
            (lambda w, fm: (
                f"weapon {w.name:<15s} {mags[wi]:2d}/{w.mag:<2d}"
                f"{'' if reserves[wi] < 0 else ' res' + str(reserves[wi])}"
                f"  {w.dmg_lo:.0f}-{w.dmg_hi:.0f}dmg  "
                f"acc {(w.auto_accuracy if fm == 'auto' else w.accuracy):.2f}"
                f"  rng {w.range_m:.0f}m  pen {w.pen:.1f}"
                f"{'  x' + str(w.pellets) if w.pellets > 1 else ''}"
                f"  mode {fm.upper()}"
                f"{'  AoE' if w.blast_r > 0 else ''}"
                f"{'  RELOADING' if reload_t > 0 else ''}"
                f"   {msg if now - msg_t < 2.5 else ''}"))(
                    weapons.ROSTER[loadout[wi]], cur_fire_mode()),
            f"health {int(player.health):3d}/{int(player.max_health):3d}"
            f"   shields {int(player.shields):3d}/{int(player.max_shields):3d}"
            f"   armor {player.armor_lo:.0f}-{player.armor_hi:.0f}%"
            f"{'   DOWN - respawning' if player.hp <= 0.0 else ''}",
            "guards  " + "   ".join(
                f"{g.id}:{'DOWN' if not g.alive else g.mode.value}"
                + ("" if not g.alive
                   else f" a{g.alert:.2f} h{int(g.health)}/s{int(g.shields)}")
                for g in guards),
            f"cone {CONE_DEG[0]:.0f}/{CONE_DEG[1]:.0f}/{CONE_DEG[2]:.0f}deg  "
            f"{CONE_RANGE_M[0]:.0f}/{CONE_RANGE_M[1]:.0f}/{CONE_RANGE_M[2]:.0f}m   "
            f"guards live {live_n} ghosts {len(memory)}   "
            f"known {100.0*known.mean():4.1f}%   fog {'on' if show_fog else 'OFF'}   "
            f"{clock.get_fps():5.1f} fps{'  PAUSED' if paused else ''}",
            "LMB fire  b fire-mode  r reload  h sling(+20% move)  wheel/1-7 weapon  f door  space knock  "
            "c clear  m mute  F1-F8 overlays  F9 prof  F10 own snd  F11 enemy  v cones  p pause  § game HUD",
        ]
        for i, ln in enumerate(lines):
            screen.blit(font.render(ln, True, TEXT if i < 6 else DIM),
                        (10, view_h + 8 + i * 21))

        if show_prof:
            for k, v in prof.items():
                prof_smooth[k] = prof_smooth.get(k, v) * 0.9 + v * 0.1
                prof_peak[k] = max(v, prof_peak.get(k, 0.0) * 0.97)
            order = ("sim", "base", "vision", "ripple", "overlay",
                     "entities", "hud", "flip")
            parts = "  ".join(f"{k} {prof_smooth[k]:4.1f}" for k in order
                              if k in prof_smooth)
            worst = max(prof_peak, key=lambda k: prof_peak[k])
            screen.blit(font.render(
                f"[ms] {parts}   sum {sum(prof_smooth.values()):5.1f}"
                f"   peak {worst} {prof_peak[worst]:.1f}"
                f"   frame {dt * 1000:5.1f}",
                True, (250, 200, 120)), (10, view_h + 8 + 8 * 21))
        prof["hud"] = (time.perf_counter() - _t) * 1000

        _t = time.perf_counter()
        pygame.display.flip()
        prof["flip"] = (time.perf_counter() - _t) * 1000

    pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "maps/arena.toml")
