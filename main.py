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

import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pygame

from sim import ai, audio, ballistics, combat, mapfile, sound, sprites, weapons
from sim.tilemap import TileMap, load_map, validate_patrols, validate_spawns


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
from sim.vision import ConeSpec, VisibilityCache

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

SPEED_CRAWL, SPEED_WALK, SPEED_RUN = 0.9, 2.2, 4.6

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
A_UNKNOWN = 248       # never seen: almost nothing shows
A_REMEMBERED = 138    # seen before: geometry readable but clearly stale
FOG_BLUR_R = 3        # box-blur radius (fine cells) that feathers every fog edge

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


def build_base_surface(m: TileMap) -> pygame.Surface:
    rows, cols = m.chars.shape
    arr = np.zeros((rows, cols, 3), dtype=np.uint8)
    for ch, t in m.tiles.items():
        mask = m.chars == ch
        if mask.any():
            arr[mask] = t.colour
    small = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
    return pygame.transform.scale(
        small, (round(m.width_m * PX_PER_M), round(m.height_m * PX_PER_M)))


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
    pygame.display.set_caption(f"stealth debug \u2014 {m.name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,monospace", 13)

    bank = sprites.SpriteBank()
    bank.load()
    bank.set_scale(PX_PER_M / sprites.ART_PPM)
    print("sprites:", "on" if bank.ok else "off (circle fallback)")

    base = build_base_surface(m)
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
    ov_surf = pygame.Surface((_ov_w, _ov_h), pygame.SRCALPHA)
    ov_scaled = pygame.Surface((world_w, world_h), pygame.SRCALPHA)

    guards = [ai.Guard(g, cpm) for g in m.guards]
    memory: dict[str, tuple[float, float, float, float]] = {}
    show_own_sound = False   # player's own sound world is hidden until F10
    enemy_mode = 2           # 0 off, 1 arcs, 2 full ripples (F11 cycles)
    cues: list = []

    doors = find_doors(m)
    wi = 0
    mags = [weapons.ROSTER[n].mag for n in loadout]
    fmode = [0] * len(loadout)     # index into each weapon's fire_modes()
    reload_t = 0.0
    fire_cd = 0.0
    swap_t = 0.0                   # weapon-change animation timer
    recoil_t = 0.0                 # gun-kick timer (player)
    flash_t = 0.0                  # muzzle-flash timer (player)
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

    def player_fire(w, moving, acc=None, rounds=None):
        """One trigger pull: `rounds` (default the weapon's burst) x pellets,
        each with its own accuracy jitter; damage resolved inside fire_shot,
        plus a blast at impact for explosives. `acc` overrides the weapon's
        accuracy rating (full-auto passes the lower auto_accuracy)."""
        mxp, myp = pygame.mouse.get_pos()
        ax, ay = (mxp + cam_x) / PX_PER_M, (myp + cam_y) / PX_PER_M
        base = math.atan2(ay - py_, ax - px_)
        aim_d = math.hypot(ax - px_, ay - py_)
        live = [g for g in guards if g.alive]
        for _ in range(rounds if rounds else w.burst):
            if mags[wi] <= 0:
                break
            mags[wi] -= 1
            for _p in range(w.pellets):
                hd = base + weapons.jitter(w, aim_d, moving, rng, acc)
                sh = ballistics.fire_shot(m, bbul, pen, m.glass, (px_, py_),
                                          hd, w, live, rng, now)
                if sh.segments:
                    tracers.append((sh.segments, now))
                if sh.shattered:
                    break_glass(sh.shattered)
                if w.blast_r > 0.0:
                    ballistics.blast(sh.impact, w.blast_r, w, live, rng, now)
                    blasts.append((sh.impact, w.blast_r, now))

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
                        reload_t = w.reload_s
                        msg, msg_t = f"reloading {w.name}", now
                        sfx("reload")
                elif pygame.K_1 <= ev.key <= pygame.K_9:
                    idx = ev.key - pygame.K_1
                    if idx < len(loadout) and idx != wi and swap_t <= 0.0:
                        wi = idx
                        reload_t = 0.0
                        swap_t = sprites.SWAP_TIME
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
                        doors[best] = not doors[best]
                        set_door(m, cost, best[0], best[1], doors[best])
                        vis_cache.invalidate()
                        sounds.clear()
                        emit(m, cost, best[1] + 0.5, best[0] + 0.5,
                             e_knock * 0.7, "door", sounds, now, jobs=jobs)
                        sfx("door")
                        msg = f"door {'opened' if doors[best] else 'closed'}"
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
                    msg, msg_t = f"switched to {weapons.ROSTER[loadout[wi]].name}", now
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if pygame.mouse.get_pos()[1] < view_h:
                    w = weapons.ROSTER[loadout[wi]]
                    auto = cur_fire_mode() == "auto"
                    if reload_t > 0.0 or fire_cd > 0.0 or swap_t > 0.0 or player.hp <= 0.0:
                        pass
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
                        if w.blast_r <= 0.0:
                            recoil_t = sprites.RECOIL_TIME
                            flash_t = sprites.FLASH_TIME
                            flash_roll = rng.uniform(-180.0, 180.0)
                            flash_scale = rng.uniform(0.85, 1.25)
                    elif reload_t <= 0.0 and mags[wi] < w.mag:
                        reload_t = w.reload_s        # click on empty -> reload
                        msg, msg_t = f"reloading {w.name}", now
                        sfx("reload")

        if not paused:
            fire_cd = max(0.0, fire_cd - dt)
            swap_t = max(0.0, swap_t - dt)
            recoil_t = max(0.0, recoil_t - dt)
            flash_t = max(0.0, flash_t - dt)
            if reload_t > 0.0:
                reload_t -= dt
                if reload_t <= 0.0:
                    reload_t = 0.0
                    mags[wi] = weapons.ROSTER[loadout[wi]].mag
                    emit(m, cost, px_, py_, e_mag, "magazine", sounds, now, jobs=jobs)
                    sfx("magazine")
                    msg, msg_t = "reloaded", now
            # full-auto alternate fire: fire while the button is held down
            if (player.hp > 0.0 and cur_fire_mode() == "auto"
                    and fire_cd <= 0.0 and reload_t <= 0.0 and swap_t <= 0.0
                    and mags[wi] > 0
                    and pygame.mouse.get_pressed()[0]
                    and pygame.mouse.get_pos()[1] < view_h):
                w = cur_weapon()
                fire_cd = w.auto_refire
                emit(m, cost, px_, py_, w.sound_reach_m * cpm, w.name,
                     sounds, now, jobs=jobs)
                player_fire(w, held_moving(), acc=w.auto_accuracy, rounds=1)
                sfx_fire(w)
                if w.blast_r <= 0.0:
                    recoil_t = sprites.RECOIL_TIME
                    flash_t = sprites.FLASH_TIME
                    flash_roll = rng.uniform(-180.0, 180.0)
                    flash_scale = rng.uniform(0.85, 1.25)

        keys = pygame.key.get_pressed()
        vx = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        vy = (keys[pygame.K_s] or keys[pygame.K_DOWN]) - (keys[pygame.K_w] or keys[pygame.K_UP])

        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            speed, gait = SPEED_CRAWL, "crawl"
        elif keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            speed, gait = SPEED_RUN, "run"
        else:
            speed, gait = SPEED_WALK, "walk"

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
                    sfx("footstep", 0.15 if gait == "walk" else 0.24)

        player.x, player.y = px_, py_
        if not paused and player.hp > 0.0:
            def _gemit(gx, gy, ge, glabel):
                emit(m, cost, gx, gy, ge, glabel, sounds, now,
                     jobs=jobs, enemy=True)
            for g in guards:
                g.tick(dt, now)
                if not g.alive:
                    continue
                for sh in g.update(dt, now, m, cost, pen, m.blocks_sight,
                                   bbul, sounds, player, rng, _gemit):
                    if sh.segments:
                        tracers.append((sh.segments, now))
                    if sh.shattered:
                        break_glass(sh.shattered)
                    if g.weapon.blast_r > 0.0:
                        blasts.append((sh.impact, g.weapon.blast_r, now))
            player.tick(dt, now)

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
                sfx(audio.enemy_clip(snd.label),
                    gain=min(1.0, rem * 1.3),
                    pan=(float(b[0]) if b else 0.0))
        cues = [c for c in cues if now - c["t"] < CUE_FADE]

        sounds = [s for s in sounds if not s.done(now)]

        _t = time.perf_counter()
        screen = world                  # world-space draws target the full map
        screen.fill(BG)
        screen.blit(base, (0, 0))
        if active_ov is not None:
            screen.blit(overlays[active_ov][1], (0, 0))
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
        for snd in sounds:
            f = snd.field
            el = snd.elapsed_cells(now)
            t, tv = f.t, f.travel
            band = 2.5
            if el - band > f.max_travel:
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
            if snd.enemy and enemy_mode != 2:
                continue
            if not snd.enemy and not show_own_sound:
                continue
            any_ripple = True
            atten = np.where(np.isfinite(t), 1.0 - np.clip(t / snd.energy, 0, 1), 0.0)
            dest = ripple_enemy if snd.enemy else ripple_buf
            sub = dest[f.y0:f.y0 + t.shape[0], f.x0:f.x0 + t.shape[1]]
            np.maximum(sub, ring * atten * 235, out=sub)
        prof["ripple"] = (time.perf_counter() - _t) * 1000

        _t = time.perf_counter()
        pcx, pcy = m.cell_of(px_, py_)
        vf = vis_cache.get(pcx, pcy, cone.max_range)
        inten_raw = vf.cone_intensity(facing, cone)
        prof["vision"] = (time.perf_counter() - _t) * 1000
        h_, w_ = inten_raw.shape
        known[vf.y0:vf.y0 + h_, vf.x0:vf.x0 + w_] |= inten_raw > 0.03

        _t = time.perf_counter()
        if show_fog:
            ov_a = np.where(known, A_REMEMBER_F, A_UNKNOWN_F).astype(np.float32)
            ov_rgb = np.zeros(known.shape + (3,), dtype=np.float32)
            # carve the cone into the veil from the crisp field: cells in view
            # are lightened (never below their remembered level)
            sub_a = ov_a[vf.y0:vf.y0 + h_, vf.x0:vf.x0 + w_]
            lit = inten_raw > 0.004
            sub_a[lit] = np.minimum(sub_a[lit],
                                    A_REMEMBER_F * (1.0 - inten_raw[lit]))
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
            px3 = pygame.surfarray.pixels3d(ov_surf)
            pxa = pygame.surfarray.pixels_alpha(ov_surf)
            px3[:, :, :] = np.transpose(ov_rgb.astype(np.uint8), (1, 0, 2))
            pxa[:, :] = np.transpose(
                np.clip(ov_a * 255.0, 0, 255)).astype(np.uint8)
            del px3, pxa
            pygame.transform.scale(ov_surf, (world_w, world_h), ov_scaled)
            screen.blit(ov_scaled, (0, 0))
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
                slung = g.reload_t > 0.0 or not tense   # reloading -> gun down
                if bank.ok:
                    bank.blit(screen, "soldier_idle" if slung else "soldier_ready",
                              sx, sy, g.facing)
                    gkick = (g.recoil_t / ai.RECOIL_TIME
                             * sprites.RECOIL_M * PX_PER_M)
                    gwx = sx - math.cos(g.facing) * gkick
                    gwy = sy - math.sin(g.facing) * gkick
                    bank.blit(screen,
                              "weapon_sling" if slung else sprites.weapon_art_for(g.weapon),
                              gwx, gwy, g.facing)
                    if not slung:
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

        probe_txt = "probe: nothing heard at cursor"
        if sounds:
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
        drew = False
        if bank.ok:
            if down:
                drew = bank.blit(screen, "soldier_idle", ppx, ppy, facing,
                                 tint=(90, 90, 90))
            elif swap_t > 0.0:
                pose, wsprite = sprites.swap_frame(1.0 - swap_t / sprites.SWAP_TIME,
                                                   loadout[wi])
                drew = bank.blit(screen, pose, ppx, ppy, facing)
                if wsprite:
                    bank.blit(screen, wsprite, ppx, ppy, facing)
            elif reload_t > 0.0:
                drew = bank.blit(screen, "soldier_idle", ppx, ppy, facing)
                bank.blit(screen, "weapon_sling", ppx, ppy, facing)
            else:
                drew = bank.blit(screen, "soldier_ready", ppx, ppy, facing)
                kick = recoil_t / sprites.RECOIL_TIME * sprites.RECOIL_M * PX_PER_M
                gwx = ppx - math.cos(facing) * kick
                gwy = ppy - math.sin(facing) * kick
                bank.blit(screen, sprites.weapon_art(loadout[wi]), gwx, gwy, facing)
                draw_muzzle(sprites.weapon_art(loadout[wi]), gwx, gwy, facing,
                            flash_t, flash_roll, flash_scale)
        if not drew:
            pygame.draw.line(screen, FACING, (ppx, ppy),
                             (ppx + math.cos(facing) * 26,
                              ppy + math.sin(facing) * 26), 2)
            pygame.draw.circle(screen, DEAD if down else PLAYER,
                               (int(ppx), int(ppy)), int(BODY_R * PX_PER_M))

        # lift the camera window out of the world and back to the real screen
        window.fill(BG)
        window.blit(world, (-cam_x, -cam_y))
        screen = window

        pygame.draw.rect(screen, HUD_BG, (0, view_h, view_w, HUD_H))
        cx, cy = m.cell_of(px_, py_)
        ch = m.chars[int(py_), int(px_)]
        if sounds:
            s0 = sounds[-1]
            last_solve = (f"{s0.label} energy {s0.field.energy:.0f} "
                          f"window {s0.field.t.shape[1]}x{s0.field.t.shape[0]} "
                          f"solve {s0.solve_ms:.1f}ms")
        lines = [
            f"pos {px_:6.2f},{py_:6.2f}m  cell {cx:3d},{cy:3d}  tile '{ch}' "
            f"{m.tiles[ch].name}  gait {gait}",
            f"sounds {len(sounds)} (enemy {sum(1 for s_ in sounds if s_.enemy)})"
            f"  solving {len(jobs)}  own {'on' if show_own_sound else 'OFF'}"
            f"  enemy {('off', 'arcs', 'ripples')[enemy_mode]}"
            f"  cues {len(cues)}  last: {last_solve}",
            probe_txt,
            (lambda w, fm: (
                f"weapon {w.name:<15s} {mags[wi]:2d}/{w.mag:<2d}"
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
            "LMB fire  b fire-mode  r reload  wheel/1-7 weapon  f door  space knock  "
            "c clear  m mute  F1-F8 overlays  F9 prof  F10 own snd  F11 enemy  v cones  p pause",
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
