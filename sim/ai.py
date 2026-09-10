"""Guard behaviour: a small state machine driven by a continuous alert level.

There are four modes - IDLE, PATROL, SEARCH, COMBAT - but the guard does
not snap between them. A single ``alert`` scalar in [0, 1] rises on stimulus
(sight fast, sound slow) and decays otherwise, and it is what actually
selects the mode, and also scales movement speed, turn rate and how far the
head sweeps. Because ``alert`` moves continuously and the mode thresholds
have a dead band, a guard winding down from a chase eases through SEARCH
and back into PATROL instead of cutting straight there.

The guard is never told where a sound truly came from. It hears a bearing
and a loudness and walks a guess along that bearing: louder means closer.
Frequently wrong, which is the intent - the player can bait it.

PATROL also rolls the odd standing pause: the guard drops into IDLE for a
few seconds, sweeps its head around, then carries on the route.
"""

from __future__ import annotations

import math
import random
from enum import Enum

from sim import ballistics, weapons
from sim.combat import ENEMY_PROFILES, Combatant, enemy as make_enemy
from sim.vision import line_of_sight


class Mode(str, Enum):
    IDLE = "idle"
    PATROL = "patrol"
    SEARCH = "search"
    COMBAT = "combat"


# Alert thresholds. The gap between A_SEARCH_ON and A_SEARCH_OFF is the
# hysteresis dead band that keeps mode changes from chattering.
A_COMBAT = 0.90
A_SEARCH_ON = 0.55
COMBAT_HOLD = 0.7    # keep fighting this long after a brief loss of contact, so a
                    # guard doesn't flip COMBAT<->SEARCH (and head-scan) each frame
                    # when the player hovers on a perception-range boundary
A_SEARCH_OFF = 0.06

SIGHT_GAIN = 3.5      # alert per second with the player identified in view
PERIPH_GAIN = 1.5     # alert per second with the player in peripheral vision
SOUND_GAIN = 1.2      # alert per second at full sound loudness
DECAY = 0.15          # alert lost per second when idle suspicion never escalated
DECAY_ENGAGED = 0.03  # alert lost per second while actively hunting (in the timer)
ENGAGED_FLOOR = 0.30  # alert cannot fall below this while the hunt timer runs
SEARCH_TIME = 26.0    # seconds a guard keeps hunting after last real contact
LOSE_SIGHT_ALERT = 0.85  # alert a guard drops to the instant it loses the player
SHOT_MEMORY = 1.8     # seconds after being hit that the guard stays "under fire"
BLIND_ENGAGE_M = 40.0  # a recent attacker within this is fought (cover-seek / return
                       # fire) with no cone sight needed - being shot IS the trigger,
                       # no waiting for the gunshot's sound to propagate
HEAR_MIN = 0.10       # remaining sound energy that still registers

ALERT_SPEEDUP = 0.9   # extra fraction of base speed at full alert
TURN_BASE = 2.4       # rad/s (base head/aim turn rate; scaled up by alert)
TURN_ALERT_MULT = 1.0  # extra turn rate at full alert (was 1.5 - felt "on a dime")
AIM_ON = math.radians(7.0)   # a guard holds fire until `facing` is this close to
                             # `_face_target` - it must swing the gun onto you
STRIDE = 0.85         # metres between footstep sounds
STUCK_S = 0.7         # seconds of no-progress before a guard tries a sidestep
# beat between spotting the player and the first shot, by skill
REACTION_S = {"veteran": 0.25, "seasoned": 0.40, "rookie": 0.60}

VIEW_RANGE_M = 17.0
FOV_IDENT = math.radians(40.0)    # half-angle of the identifying cone
FOV_PERIPH = math.radians(95.0)   # half-angle of peripheral vision
IDENT_RANGE_M = 15.0
PERIPH_RANGE_M = 7.0

# guards obey the same "you only see what is lit" rule as the player: a sighting
# is dropped in the dark and its alert gain scales with the light on the player
# (`player.see_light`, 0..1, set by main.py). A hunting guard in a dark spot
# switches its own flashlight on.
GUARD_SEE_DARK = 0.04            # below this light on the player -> not seen at all
GUARD_FLASH_DARK = 0.30          # guard's own cell darker than this -> want light
GUARD_FLASH_MIN_ON = 2.5        # seconds it stays on once lit (no strobing)
GUARD_FLASH_HALF_DEG = 15.0     # beam half-angle
GUARD_FLASH_RANGE_M = 13.0      # beam reach
LOST_CHASE_S = 4.0             # after a COMBAT sighting is lost, push toward the
LOST_CHASE_LEAD = 4.0          # player's flight line (this far past their last
                              # spot) before falling back to a random search

KEEP_MIN_M = 3.5      # back off if the player gets closer than this
KEEP_MAX_M = 8.0      # advance if the player is further than this
AUTO_RANGE_M = 7.0    # within this, an auto-capable guard hoses full-auto
FIRE_SND_EVERY = 0.3  # min seconds between a guard's shot-noise propagation emits
GUARD_FOOTSTEP_M = 5.0

# skill tier -> (chance a burst is aimed off, max bearing error in radians).
# This shifts the whole burst off the player (an aim error), it does NOT widen
# the per-shot spread. "veteran" is the baseline (dead-on).
SKILL_AIM_ERR = {
    "veteran":  (0.0,  0.0),
    "seasoned": (0.45, math.radians(4.5)),
    "rookie":   (0.75, math.radians(10.0)),
}

# guards can sprint toward a distant target while alarmed, on their own stamina
GUARD_SPRINT_MULT = 1.7
SPRINT_MIN_M = 4.0       # only sprint when the move target is at least this far
GUARD_STAM_DRAIN = 0.18  # per second sprinting (~5.5 s to empty)
GUARD_STAM_REGEN = 0.09  # per second otherwise (~11 s to full)
GUARD_STAM_UNLOCK = 0.30 # sprint re-enables once stamina climbs back to this
RECOIL_TIME = 0.09       # gun-kick timer, seconds (visual only, read by main.py)
FLASH_TIME = 0.05        # muzzle-flash timer, seconds (visual only)
DEFAULT_PROFILE = "imperium_soldier"

# Cover-seeking in COMBAT
COVER_PICK_EVERY = 1.1    # min seconds between cover re-evaluations
COVER_HUG_M = 1.7         # a shielding blocker must be within this of the spot
COVER_MAX_ENGAGE_M = 14.0  # do not pick cover further than this from the player
PEEK_OFFSET_M = 0.8      # lateral step out from cover to take a shot
HUNKER_MIN, HUNKER_MAX = 0.7, 1.6   # seconds tucked behind cover, not firing
PEEK_MIN, PEEK_MAX = 0.9, 1.8        # seconds leaning out to fire


class Guard(Combatant):
    """One patrolling guard. Construct from a tilemap.GuardSpec.

    Health, shields, armor, speed and the weapon all come from a
    combat.ENEMY_PROFILES entry (default: an Imperium soldier with a combat
    rifle).
    """

    def __init__(self, spec, cells_per_metre: float,
                 profile: str = DEFAULT_PROFILE, difficulty: float = 1.0):
        pts = [tuple(p) for p in (spec.patrol or [(1.5, 1.5)])]
        base = make_enemy(profile, pts[0][0], pts[0][1], difficulty)
        super().__init__(
            x=base.x, y=base.y,
            max_health=base.max_health, max_shields=base.max_shields,
            armor_lo=base.armor_lo, armor_hi=base.armor_hi,
            shield_regen=base.shield_regen, shield_delay=base.shield_delay,
            faction=base.faction, speed=base.speed)
        self.profile = profile
        # the map/editor may override the profile's default gun per guard
        wid = getattr(spec, "weapon", None) or ENEMY_PROFILES[profile]["weapon"]
        self.weapon = weapons.ROSTER.get(
            wid, weapons.ROSTER[ENEMY_PROFILES[profile]["weapon"]])
        self.skill = getattr(spec, "skill", "veteran")
        if self.skill not in SKILL_AIM_ERR:
            self.skill = "veteran"
        self._aim_err = 0.0        # bearing offset for the current burst
        self.stamina = 1.0        # 0..1, sprint fuel
        self.sprint_locked = False
        self._want_sprint = False
        self.id = spec.id
        self.cpm = cells_per_metre
        self.pts = pts
        self.i = 0
        self.mode = Mode.PATROL
        self.alert = 0.0
        self.facing = 0.0
        self._face_target = 0.0
        self.since_step = 0.0
        self._blocked = False       # last move attempt made ~no progress
        self._stuck_t = 0.0         # accumulated no-progress time (sidestep trigger)
        self._combat_hold_until = -1e9
        self._sb_hold = -1e9        # "shot was blocked recently" hysteresis
        self._cover = None          # (x, y) cover anchor being used, or None
        self._cover_next = 0.0      # earliest time to re-evaluate cover
        self._peeking = False       # currently leaning out of cover to fire
        self._peek_end = 0.0
        self._cur_side = 1.0        # which way this peek leans
        self._peek_last = 1.0       # last side that had a clean shot
        self.recoil_t = 0.0        # gun-kick timer, decays each update (visual)
        self.snd_cache = {}        # one reusable sound field (like the player's step_cache)
        self._fire_snd_t = -1e9    # last shot-noise emit; throttled by FIRE_SND_EVERY
        self.flash_t = 0.0         # muzzle-flash timer (visual)
        self.flash_roll = 0.0      # per-shot flash spin, degrees
        self.flash_scale = 1.0     # per-shot flash size jitter
        self._peek_dist = PEEK_OFFSET_M   # how far to lean (wide cover needs more)
        self._flank_dir = 1.0      # committed strafe direction when flanking
        self._flank_until = 0.0
        self.wait = 0.0
        self.idle_until = 0.0
        self._idle_base = 0.0
        self.scan_t = random.random() * 6.0
        self.investigate = None        # (x, y) best guess of where to look
        self.search_until = 0.0
        self._poke_next = 0.0
        self.mag = self.weapon.mag
        self.fire_cd = 0.0
        self.reload_t = 0.0
        self._next_idle_in = self._roll_idle()
        self.flashlight = False        # guard torch - on while hunting in the dark
        self._flash_since = -1e9
        self._pl_prev = None           # player pos last update, for a flight bearing
        self._pl_head = None           # player's recent movement heading
        self._chase_until = -1e9       # pushing along the flight line after losing sight

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _roll_idle() -> float:
        """Metres of patrol travel until the next standing pause, or never."""
        return random.uniform(6.0, 16.0) if random.random() < 0.6 else 1e9

    def _cell(self):
        return int(self.x * self.cpm), int(self.y * self.cpm)

    def _speed(self) -> float:
        base = self.speed * (1.0 + ALERT_SPEEDUP * self.alert)
        if self._want_sprint and self.stamina > 0.0 and not self.sprint_locked:
            return base * GUARD_SPRINT_MULT
        return base

    def _route_facing(self) -> float:
        ax, ay = self.pts[self.i]
        bx, by = self.pts[(self.i + 1) % len(self.pts)]
        return math.atan2(by - ay, bx - ax)

    def calm(self) -> None:
        """Reset to a quiet patrol - used when the player respawns."""
        self.mode = Mode.PATROL
        self.alert = 0.0
        self.investigate = None
        self.fire_cd = 0.0
        self.reload_t = 0.0
        self.mag = self.weapon.mag
        self.wait = 0.0
        self._cover = None
        self._peeking = False
        self._aim_err = 0.0
        self._stuck_t = 0.0
        self._combat_hold_until = -1e9
        self._sb_hold = -1e9
        self.stamina = 1.0
        self.sprint_locked = False
        self._want_sprint = False
        self.flashlight = False
        self._flash_since = -1e9
        self._chase_until = -1e9

    # -- perception ---------------------------------------------------

    def sees(self, blocks_sight, player) -> int:
        """0 nothing, 1 identified (in the narrow cone), 3 peripheral only."""
        if not player.alive or getattr(player, "concealed", False):
            return 0                              # e.g. holding still in a bush
        dx, dy = player.x - self.x, player.y - self.y
        dist = math.hypot(dx, dy)
        if dist > VIEW_RANGE_M:
            return 0
        off = abs((math.atan2(dy, dx) - self.facing + math.pi)
                  % (2 * math.pi) - math.pi)
        gx, gy = self._cell()
        pcx, pcy = int(player.x * self.cpm), int(player.y * self.cpm)
        if off <= FOV_IDENT and dist <= IDENT_RANGE_M:
            if line_of_sight(blocks_sight, gx, gy, pcx, pcy):
                return 1
        if off <= FOV_PERIPH and dist <= PERIPH_RANGE_M:
            if line_of_sight(blocks_sight, gx, gy, pcx, pcy):
                return 3
        return 0

    def _hear(self, sounds, now):
        """Loudest already-arrived player sound: (strength, bearing) or None."""
        best = None
        gx, gy = self._cell()
        for snd in sounds:
            if snd.enemy:
                continue
            f = snd.field
            tv = f.arrival_time(gx, gy)
            if not math.isfinite(tv) or snd.elapsed_cells(now) < tv:
                continue
            rem = f.remaining(gx, gy)
            if rem <= HEAR_MIN:
                continue
            if best is None or rem > best[0]:
                b = f.bearing(gx, gy)
                ang = math.atan2(b[1], b[0]) if b else None
                best = (rem, ang)
        return best

    def _dark_ahead(self, m) -> bool:
        """True if the guard stands in the dark, is hunting toward a dark spot,
        or (idle) has clear line of sight to a dark patch a few metres ahead."""
        lm = getattr(m, "lightmap", None)
        if lm is None:
            return True                          # a map with no lights = all dark
        hh, ww = lm.shape

        def _dk(cx, cy):
            return 0 <= cy < hh and 0 <= cx < ww and lm[cy, cx] < GUARD_FLASH_DARK

        gcx, gcy = self._cell()
        if _dk(gcx, gcy):
            return True
        hunting = self.mode in (Mode.SEARCH, Mode.COMBAT)
        if hunting and self.investigate is not None:
            # keep it on toward where the guard is heading, corner or no corner
            if _dk(int(self.investigate[0] * self.cpm),
                   int(self.investigate[1] * self.cpm)):
                return True
        cx, cy = math.cos(self.facing), math.sin(self.facing)
        for dm in (2.0, 4.0, 6.5):
            scx = int((self.x + cx * dm) * self.cpm)
            scy = int((self.y + cy * dm) * self.cpm)
            if not (0 <= scy < hh and 0 <= scx < ww):
                break
            if not hunting and not line_of_sight(m.blocks_sight, gcx, gcy, scx, scy):
                break                            # idle: don't light past a wall
            if _dk(scx, scy):
                return True
        return False

    def _guess_source(self, heard):
        rem, ang = heard
        if ang is None:
            return self.investigate
        reach = 11.0 * (1.0 - rem) + 3.0        # louder -> closer guess
        return (self.x + math.cos(ang) * reach,
                self.y + math.sin(ang) * reach)

    # -- motion primitives ------------------------------------------

    def _turn(self, dt: float) -> None:
        rate = TURN_BASE * (1.0 + TURN_ALERT_MULT * self.alert)
        d = (self._face_target - self.facing + math.pi) % (2 * math.pi) - math.pi
        self.facing += max(-rate * dt, min(rate * dt, d))

    def _step_toward(self, m, tx, ty, dt, emit_cb, gait_energy,
                     face: bool = True) -> float:
        """Move toward (tx, ty), sliding on walls. Returns distance still to go.

        Sets `self._blocked` when the attempted move made almost no progress
        (walked into geometry), so callers can stop grinding a wall. With
        `face=False` the caller keeps control of `_face_target` - a guard
        backing away in a fight still points its gun at you.
        """
        dx, dy = tx - self.x, ty - self.y
        d = math.hypot(dx, dy)
        if d < 1e-4:
            self._blocked = False
            return 0.0
        if face:
            self._face_target = math.atan2(dy, dx)
        if self.alert > 0.4 and d > SPRINT_MIN_M:
            self._want_sprint = True          # alarmed + far to go -> sprint
        step = min(d, self._speed() * dt)
        bx, by = self.x, self.y
        fx, fy = dx / d * step, dy / d * step      # straight toward the target
        sidestep = self._stuck_t > STUCK_S
        if sidestep:
            # been grinding a wall - slip along it instead of into it, committing
            # ~2 s to one direction before trying the other (no pathfinding, so
            # this just keeps the guard moving until the way opens or a caller
            # re-targets it)
            side = 1.0 if int((self._stuck_t - STUCK_S) / 2.0) % 2 == 0 else -1.0
            a = math.atan2(dy, dx) + side * (math.pi / 2.0)
            ux, uy = math.cos(a) * step, math.sin(a) * step
        else:
            ux, uy = fx, fy
        if m.can_stand(self.x + ux, self.y, self.body_r):
            self.x += ux
        if m.can_stand(self.x, self.y + uy, self.body_r):
            self.y += uy
        moved = math.hypot(self.x - bx, self.y - by)
        self._blocked = moved < step * 0.35
        if not sidestep and moved >= step * 0.35:
            self._stuck_t = 0.0                    # normal progress
        elif sidestep and (m.can_stand(self.x + fx, self.y, self.body_r)
                           and m.can_stand(self.x, self.y + fy, self.body_r)):
            self._stuck_t = 0.0                    # rounded it - path is open again
        else:
            self._stuck_t += dt                   # still stuck / still slipping
        self.since_step += moved
        if self.since_step >= STRIDE:
            self.since_step = 0.0
            emit_cb(self.x, self.y,
                    gait_energy * (0.7 + 0.6 * self.alert), f"{self.id}/step",
                    self.snd_cache)
        return d - moved

    # -- per-mode behaviour ---------------------------------------

    def _do_patrol(self, dt, now, m, rng, emit_cb):
        if self.wait > 0.0:
            self.wait -= dt
            self.scan_t += dt
            self._face_target = (self._route_facing()
                                 + math.sin(self.scan_t * 1.1) * 0.5)
            return
        tx, ty = self.pts[(self.i + 1) % len(self.pts)]
        before = math.hypot(tx - self.x, ty - self.y)
        rem = self._step_toward(m, tx, ty, dt, emit_cb,
                                GUARD_FOOTSTEP_M * self.cpm)
        self._next_idle_in -= max(0.0, before - rem)
        if self._next_idle_in <= 0.0:
            self._next_idle_in = self._roll_idle()
            self._idle_base = self._route_facing()
            self.idle_until = now + rng.uniform(2.5, 6.0)
            self.mode = Mode.IDLE
            return
        if rem < 0.10 or self._blocked or self._stuck_t > STUCK_S:
            # reached it, or the leg is obstructed (a door shut, say) - either
            # way move on to the next waypoint rather than grind the wall
            self.i = (self.i + 1) % len(self.pts)
            self.wait = rng.uniform(0.6, 1.8)
            self._blocked = False
            self._stuck_t = 0.0

    def _do_idle(self, dt, now):
        self.scan_t += dt
        self._face_target = self._idle_base + math.sin(self.scan_t * 0.9) * 0.8
        if now >= self.idle_until:
            self.mode = Mode.PATROL

    def _do_search(self, dt, now, m, rng, emit_cb):
        self.scan_t += dt
        if now - self.last_hit_t < SHOT_MEMORY and self.hit_from is not None:
            # under fire from range: march straight at the shot origin with the
            # gun on it - no idle head-scan, no wandering, and NOT toward the
            # jittery sound-bearing guess. this is what stops the "spazzing".
            hx, hy = self.hit_from
            self._step_toward(m, hx, hy, dt, emit_cb,
                              GUARD_FOOTSTEP_M * self.cpm, face=False)
            self._face_target = math.atan2(hy - self.y, hx - self.x)
            return
        if self.investigate is None:
            self.investigate = (self.x, self.y)
        tx, ty = self.investigate
        rem = self._step_toward(m, tx, ty, dt, emit_cb,
                                GUARD_FOOTSTEP_M * self.cpm)
        base = math.atan2(ty - self.y, tx - self.x)
        # look firmly toward the source when alarmed; only idly scan when the
        # lead is vague (low alert). stops the "hesitant, won't-turn" wobble.
        wob = 0.9 * max(0.0, 1.0 - self.alert * 1.4)
        self._face_target = base + math.sin(self.scan_t * 2.0) * wob
        if (rem < 0.5 or now >= self._poke_next or self._blocked
                or self._stuck_t > STUCK_S):
            self._poke_next = now + rng.uniform(1.5, 3.0)
            _was_blocked = self._blocked or self._stuck_t > STUCK_S
            self._blocked = False
            if (now < self._chase_until and self._pl_head is not None
                    and not _was_blocked):
                # still chasing the lost target: keep pushing along its flight
                # line, do not scatter to a random point
                _adv = None
                for step in (3.0, 5.5):
                    nx = self.x + math.cos(self._pl_head) * step
                    ny = self.y + math.sin(self._pl_head) * step
                    if m.can_stand(nx, ny, self.body_r):
                        _adv = (nx, ny)
                if _adv is not None:
                    self.investigate = _adv
                else:
                    self._chase_until = 0.0        # flight line walled - wander
            else:
                if _was_blocked:
                    self._chase_until = 0.0        # bumped the corner - now scan
                for _ in range(6):
                    nx = self.x + rng.uniform(-3.5, 3.5)
                    ny = self.y + rng.uniform(-3.5, 3.5)
                    if m.can_stand(nx, ny, self.body_r):
                        self.investigate = (nx, ny)
                        break

    def _bullet_block_dist(self, bbul, ax, ay, bx, by):
        """Metres from (ax,ay) to the first bullet-blocking cell toward (bx,by),
        or None if the path is clear. Both points in world metres."""
        cpm = self.cpm
        x0, y0, x1, y1 = ax * cpm, ay * cpm, bx * cpm, by * cpm
        dx, dy = x1 - x0, y1 - y0
        dcell = math.hypot(dx, dy)
        if dcell < 1e-6:
            return None
        h, w = bbul.shape
        n = max(1, int(dcell / 0.5))
        for k in range(1, n + 1):
            f = k / n
            xi, yi = int(x0 + dx * f), int(y0 + dy * f)
            if 0 <= yi < h and 0 <= xi < w and bbul[yi, xi]:
                return f * dcell / cpm
        return None

    def _cover_valid(self, bbul, player) -> bool:
        if self._cover is None:
            return False
        cx, cy = self._cover
        d = math.hypot(player.x - cx, player.y - cy)
        if d < KEEP_MIN_M * 0.8 or d > self.weapon.range_m:
            return False
        return self._bullet_block_dist(bbul, cx, cy, player.x, player.y) is not None

    def _find_cover(self, m, bbul, player, max_near=7.0):
        """A standable spot with a bullet-blocker between it and the player,
        from which a small side-step opens a clean shot. Candidates include
        the guard's own position, so a guard already beside good cover just
        settles in. `max_near` caps how far the guard will detour for it.
        None if nothing usable."""
        px, py = player.x, player.y
        lo = KEEP_MIN_M * 0.7
        hi = min(self.weapon.range_m, COVER_MAX_ENGAGE_M)
        cpm = self.cpm

        cands = [(self.x, self.y),
                 (self.x + 1.0, self.y), (self.x - 1.0, self.y),
                 (self.x, self.y + 1.0), (self.x, self.y - 1.0)]
        for k in range(24):
            a = 2.0 * math.pi * k / 24.0
            ca, sa = math.cos(a), math.sin(a)
            for r in (2.0, 3.5, 5.0, 6.5):
                cands.append((self.x + ca * r, self.y + sa * r))

        best, best_s = None, 0.0
        for cx, cy in cands:
            if not m.can_stand(cx, cy, self.body_r):
                continue
            d = math.hypot(px - cx, py - cy)
            if d < lo or d > hi:
                continue
            near = math.hypot(cx - self.x, cy - self.y)
            if near > max_near:
                continue
            if near > 1.6 and self._bullet_block_dist(
                    m.blocks_move, self.x, self.y, cx, cy) is not None:
                continue                          # wall-slide can't reach it
            hb = self._bullet_block_dist(bbul, cx, cy, px, py)
            if hb is None or not (0.15 <= hb <= COVER_HUG_M):
                continue
            ux, uy = (px - cx) / d, (py - cy) / d
            if not any(
                    m.can_stand(cx - uy * PEEK_OFFSET_M * s,
                                cy + ux * PEEK_OFFSET_M * s, self.body_r)
                    and self._bullet_block_dist(
                        bbul, cx - uy * PEEK_OFFSET_M * s,
                        cy + ux * PEEK_OFFSET_M * s, px, py) is None
                    for s in (1.0, -1.0)):
                continue
            score = (2.0 - hb) - 0.25 * near - 0.05 * abs(d - 7.0)
            if score > best_s:
                best_s, best = score, (cx, cy)
        return best

    def _peekable_from(self, pt, m, bbul, player) -> bool:
        """True if some small lateral lean from `pt` opens a clean shot."""
        cx, cy = pt
        d = math.hypot(player.x - cx, player.y - cy) or 1.0
        ux, uy = (player.x - cx) / d, (player.y - cy) / d
        for off in (PEEK_OFFSET_M, 1.4, 2.0):
            for s in (1.0, -1.0):
                ox, oy = cx - uy * off * s, cy + ux * off * s
                if (m.can_stand(ox, oy, self.body_r)
                        and self._bullet_block_dist(
                            bbul, ox, oy, player.x, player.y) is None):
                    return True
        return False

    def _peek_side(self, m, bbul, player) -> float:
        """Pick a side and lean distance that opens a clean shot on the player
        from behind the current cover. Returns the side (-1/0/1) and stores the
        distance in self._peek_dist. 0 = no lean clears it (fully walled off)."""
        cx, cy = self._cover
        d = math.hypot(player.x - cx, player.y - cy) or 1.0
        ux, uy = (player.x - cx) / d, (player.y - cy) / d
        for off in (PEEK_OFFSET_M, 1.4, 2.0):
            for s in (self._peek_last, -self._peek_last):
                ox, oy = cx - uy * off * s, cy + ux * off * s
                if (m.can_stand(ox, oy, self.body_r)
                        and self._bullet_block_dist(
                            bbul, ox, oy, player.x, player.y) is None):
                    self._peek_last = s
                    self._peek_dist = off
                    return s
        return 0.0

    def _do_combat(self, dt, now, m, pen, blocks_sight, blocks_bullets,
                   player, rng, emit_cb):
        """Get behind solid cover if any is within reach, then alternate
        hunkering (tucked, not firing - the sim stops the player's bullets on
        the cover) with peeking a step to the side to shoot. With no usable
        cover, fall back to holding a firing distance in the open."""
        dx, dy = player.x - self.x, player.y - self.y
        dist = math.hypot(dx, dy)
        # in a fight the guard always faces the player - repositioning never
        # steers the aim, so backing away does not turn its back on you
        self._face_target = math.atan2(dy, dx)
        w = self.weapon
        ge = GUARD_FOOTSTEP_M * self.cpm
        cpm = self.cpm
        moving = True

        gx, gy = self._cell()
        pcx, pcy = int(player.x * cpm), int(player.y * cpm)
        have_shot = (dist <= w.range_m
                     and line_of_sight(blocks_sight, gx, gy, pcx, pcy)
                     and self._bullet_block_dist(
                         blocks_bullets, self.x, self.y, player.x, player.y) is None)

        if now >= self._cover_next:
            self._cover_next = now + COVER_PICK_EVERY
            if not self._cover_valid(blocks_bullets, player):
                self._cover = None
            # with a clean shot already, only settle into cover that is right
            # here; with no angle (or under fire) it is worth moving for
            under_fire = now - self.last_hit_t < 2.5
            max_near = 3.0 if (have_shot and not under_fire) else 7.0
            cand = self._find_cover(m, blocks_bullets, player, max_near)
            if cand is not None:
                self._cover = cand

        at_cover = (self._cover is not None
                    and math.hypot(self._cover[0] - self.x,
                                   self._cover[1] - self.y) < 0.5)

        if self._cover is not None and not at_cover:
            self._step_toward(m, self._cover[0], self._cover[1], dt,
                              emit_cb, ge, face=False)
            if self._blocked:                     # can't reach it straight-line;
                self._cover = None               # give up on cover for a while
                self._cover_next = now + 4.0
        elif self._cover is not None:             # at cover: hunker / peek cycle
            if now >= self._peek_end:
                if self._peeking:
                    self._peeking = False
                    self._peek_end = now + rng.uniform(HUNKER_MIN, HUNKER_MAX)
                else:
                    self._cur_side = self._peek_side(m, blocks_bullets, player)
                    if self._cur_side == 0.0:
                        # cover we cannot shoot from - abandon it and flank
                        self._cover = None
                        self._cover_next = now + 3.0
                    else:
                        self._peeking = True
                        self._peek_end = now + rng.uniform(PEEK_MIN, PEEK_MAX)
            if self._cover is not None:
                cx, cy = self._cover
                if self._peeking:
                    u = dist or 1.0
                    tx = cx - (dy / u) * self._peek_dist * self._cur_side
                    ty = cy + (dx / u) * self._peek_dist * self._cur_side
                    self._step_toward(m, tx, ty, dt, emit_cb, ge, face=False)
                else:
                    self._step_toward(m, cx, cy, dt, emit_cb, ge, face=False)
                    moving = False

        if self._cover is None:                   # open fight / flanking
            sb = self._bullet_block_dist(blocks_bullets, self.x, self.y,
                                         player.x, player.y)
            if sb is not None:
                self._sb_hold = now + 0.35        # a far/edge blocker flickers in
                                                 # and out; treat it as blocked
                                                 # for a beat so we don't jitter
            sb_blocked = sb is not None or now < self._sb_hold
            u = dist or 1.0
            if sb is not None and sb <= 2.0 and now >= self._cover_next \
                    and self._peekable_from((self.x, self.y), m, blocks_bullets, player):
                # a blocker is right here and we CAN lean out of it - use it
                self._cover = (self.x, self.y)
                self._peek_end = now
                moving = False
            elif sb_blocked or now < self._flank_until:
                # blocked shot (or still mid-commit) - flank one way for a full
                # beat rather than snapping advance<->sidestep frame to frame
                if now >= self._flank_until:
                    self._flank_dir = self._peek_last
                    self._flank_until = now + 1.8
                tgt = None
                for s in (self._flank_dir, -self._flank_dir):
                    tx = self.x - (dy / u) * 2.5 * s + (dx / u) * 1.2
                    ty = self.y + (dx / u) * 2.5 * s + (dy / u) * 1.2
                    if m.can_stand(tx, ty, self.body_r):
                        tgt = (tx, ty)
                        break
                if tgt is not None:
                    self._step_toward(m, tgt[0], tgt[1], dt, emit_cb, ge,
                                      face=False)
                else:
                    moving = False        # boxed in - hold and face the player,
                                          # do NOT thrash _flank_dir every frame
            elif dist > KEEP_MAX_M:
                self._step_toward(m, player.x, player.y, dt, emit_cb, ge, face=False)
            elif dist < KEEP_MIN_M and dist > 1e-3:
                self._step_toward(m, self.x - dx, self.y - dy, dt, emit_cb, ge, face=False)
            else:
                moving = False

        if self.reload_t > 0.0:
            return None
        if self.mag <= 0:
            self.reload_t = w.reload_s
            emit_cb(self.x, self.y, 7.0 * self.cpm, f"{self.id}/reload")
            return None
        if self.fire_cd > 0.0 or dist > w.range_m:
            return None
        gx, gy = self._cell()
        pcx, pcy = int(player.x * self.cpm), int(player.y * self.cpm)
        if not line_of_sight(blocks_sight, gx, gy, pcx, pcy):
            return None
        if self._bullet_block_dist(blocks_bullets, self.x, self.y,
                                   player.x, player.y) is not None:
            return None                          # own cover in the way - hold fire
        aim_off = abs((self._face_target - self.facing + math.pi)
                      % (2 * math.pi) - math.pi)
        if aim_off > AIM_ON:
            return None                          # gun not on target yet - keep turning

        # fire mode: hose full-auto when close and committed, else the
        # weapon's primary burst/semi pattern
        auto = (w.has_auto and dist <= AUTO_RANGE_M
                and self.mode == Mode.COMBAT and self.alert >= A_COMBAT)
        self.fire_cd = w.auto_refire if auto else w.burst_time
        rounds = 1 if auto else w.burst
        acc = w.auto_accuracy if auto and w.auto_accuracy > 0.0 else None
        # skill: roll a per-burst aim error (offsets the whole burst, not spread)
        _p_off, _e_off = SKILL_AIM_ERR.get(self.skill, (0.0, 0.0))
        self._aim_err = (rng.uniform(-_e_off, _e_off)
                         if _p_off and rng.random() < _p_off else 0.0)
        if w.blast_r <= 0.0:
            self.recoil_t = RECOIL_TIME
            self.flash_t = FLASH_TIME
            self.flash_roll = random.uniform(-180.0, 180.0)
            self.flash_scale = random.uniform(0.85, 1.25)
        # fire from the muzzle, ahead of the body, unless it lands in cover
        ox = self.x + math.cos(self.facing) * weapons.MUZZLE_M
        oy = self.y + math.sin(self.facing) * weapons.MUZZLE_M
        _mx, _my = int(ox * self.cpm), int(oy * self.cpm)
        h_, wd_ = blocks_bullets.shape
        if not (0 <= _my < h_ and 0 <= _mx < wd_) or blocks_bullets[_my, _mx]:
            ox, oy = self.x, self.y
        shots = []
        for _ in range(rounds):
            if self.mag <= 0:
                break
            self.mag -= 1
            travels = w.blast_r > 0.0 and w.projectile_speed > 0.0
            for _p in range(w.pellets):
                hd = (self.facing + self._aim_err
                      + weapons.pellet_offset(w, rng)
                      + weapons.jitter(w, dist, moving, rng, acc))
                sh = ballistics.fire_shot(m, blocks_bullets, pen, m.glass,
                                          (ox, oy), hd, w,
                                          [player], rng, now,
                                          apply_damage=not travels)
                shots.append(sh)
                if w.blast_r > 0.0 and not travels:
                    # instant blast; a travelling one is spawned by the caller
                    # from the returned Shot and detonates on arrival
                    ballistics.blast(sh.impact, w.blast_r, w, [player], rng, now)
        if now - self._fire_snd_t >= FIRE_SND_EVERY:
            self._fire_snd_t = now
            emit_cb(self.x, self.y, w.sound_reach_m * self.cpm,
                    f"{self.id}/fire", self.snd_cache)
        return shots

    # -- top level ------------------------------------------------

    def update(self, dt, now, m, cost, pen, blocks_sight, blocks_bullets,
               sounds, player, rng, emit_cb):
        """Advance one guard. Returns the list of ballistics.Shot it fired."""
        self.fire_cd = max(0.0, self.fire_cd - dt)
        self.recoil_t = max(0.0, self.recoil_t - dt)
        self.flash_t = max(0.0, self.flash_t - dt)
        if self.reload_t > 0.0:
            self.reload_t -= dt
            if self.reload_t <= 0.0:
                self.mag = self.weapon.mag
        elif self.mode != Mode.COMBAT and self.mag < self.weapon.mag:
            self.reload_t = self.weapon.reload_s      # top up between fights

        if self._pl_prev is not None:
            _vx = player.x - self._pl_prev[0]
            _vy = player.y - self._pl_prev[1]
            if _vx * _vx + _vy * _vy > 2.5e-4:
                self._pl_head = math.atan2(_vy, _vx)
        self._pl_prev = (player.x, player.y)

        band = self.sees(blocks_sight, player)
        # same rule as the player: you only see what is lit. `see_light` (0..1)
        # is the illumination on the player, set by main.py from the lightmap +
        # every flashlight + muzzle flash.
        see_light = getattr(player, "see_light", 1.0)
        if band and see_light <= GUARD_SEE_DARK:
            band = 0
        heard = self._hear(sounds, now)

        # taking fire is its own stimulus: a guard shot from outside its
        # vision cone should whirl toward the hit, not amble off searching
        shot_recently = (now - self.last_hit_t < SHOT_MEMORY
                         and self.hit_from is not None)
        if shot_recently:
            self.alert = 1.0
            self.search_until = now + SEARCH_TIME
            self.investigate = self.hit_from        # head for the shooter

        if band == 1:
            self.alert = min(1.0, self.alert + SIGHT_GAIN * see_light * dt)
            self.investigate = (player.x, player.y)
            self.search_until = now + SEARCH_TIME
        elif band == 3:
            self.alert = min(1.0, self.alert + PERIPH_GAIN * see_light * dt)
            self.investigate = (player.x, player.y)
            self.search_until = now + SEARCH_TIME
        elif heard is not None:
            self.alert = min(1.0, self.alert + SOUND_GAIN * heard[0] * dt)
            if not shot_recently:
                # a noisy long-range bearing guess must not fight the crisp
                # hit_from we already set - that was the "spazzing" source
                self.investigate = self._guess_source(heard)
            self.search_until = now + SEARCH_TIME
        else:
            hunting = (self.mode in (Mode.SEARCH, Mode.COMBAT)
                       and now < self.search_until)
            rate = DECAY_ENGAGED if hunting else DECAY
            floor = ENGAGED_FLOOR if hunting else 0.0
            self.alert = max(floor, self.alert - rate * dt)

        prev = self.mode
        pdist = math.hypot(player.x - self.x, player.y - self.y)
        # any live visual (identify OR peripheral) plus full alert is enough to
        # fight - peripheral range is point-blank, and staying in SEARCH there
        # just makes the guard scan past a target it can plainly see
        if band in (1, 3):
            self._combat_hold_until = now + COMBAT_HOLD
        if band in (1, 3) and player.alive and self.alert >= A_COMBAT:
            self.mode = Mode.COMBAT
        elif shot_recently and player.alive and pdist <= BLIND_ENGAGE_M:
            self.mode = Mode.COMBAT                 # point-blank attacker
        elif self.mode == Mode.COMBAT and band in (1, 3) and player.alive:
            pass                                   # hold the fight
        elif (self.mode == Mode.COMBAT and player.alive
              and now < self._combat_hold_until):
            pass                                   # brief contact loss - hold on
        elif self.alert >= A_SEARCH_ON:
            self.mode = Mode.SEARCH
        elif self.alert <= A_SEARCH_OFF:
            if self.mode in (Mode.SEARCH, Mode.COMBAT):
                self.mode = Mode.PATROL
        # otherwise keep the current mode: the dead band is the wind-down

        if self.mode == Mode.COMBAT:
            # keep the hunt timer topped up so losing sight always grants the
            # full SEARCH_TIME from that moment
            self.search_until = now + SEARCH_TIME
            self.investigate = (player.x, player.y)
            if prev != Mode.COMBAT:              # first-shot reaction beat
                self.fire_cd = max(self.fire_cd,
                                   REACTION_S.get(self.skill, 0.25))
        if prev == Mode.COMBAT and self.mode != Mode.COMBAT:
            self.alert = max(self.alert, LOSE_SIGHT_ALERT)
            self.search_until = now + SEARCH_TIME
            # break contact -> chase the flight line past their last spot (round
            # the corner) before settling into a scattered search
            self._chase_until = now + LOST_CHASE_S
            if self._pl_head is not None:
                _lx = player.x + math.cos(self._pl_head) * LOST_CHASE_LEAD
                _ly = player.y + math.sin(self._pl_head) * LOST_CHASE_LEAD
                self.investigate = ((_lx, _ly) if m.can_stand(_lx, _ly, self.body_r)
                                    else (player.x, player.y))
            elif self.investigate is None:
                self.investigate = (player.x, player.y)
        if self.mode == Mode.SEARCH and prev != Mode.SEARCH:
            self.search_until = max(self.search_until, now + SEARCH_TIME)
            self._poke_next = now + 2.0

        # flashlight: on when the guard is in the dark OR is looking into a dark
        # patch it has clear LOS to (a lit guard peering down a black corridor
        # lights it up). Sticky for GUARD_FLASH_MIN_ON so a head-sweep across a
        # dark doorway doesn't strobe it.
        want_light = self._dark_ahead(m)
        if want_light and not self.flashlight:
            self.flashlight = True
            self._flash_since = now
        elif (self.flashlight and not want_light
              and now - self._flash_since > GUARD_FLASH_MIN_ON):
            self.flashlight = False

        shots = []
        if self.mode == Mode.COMBAT:
            shots = self._do_combat(dt, now, m, pen, blocks_sight,
                                    blocks_bullets, player, rng, emit_cb) or []
        elif self.mode == Mode.SEARCH:
            self._do_search(dt, now, m, rng, emit_cb)
        elif self.mode == Mode.IDLE:
            self._do_idle(dt, now)
        else:
            self._do_patrol(dt, now, m, rng, emit_cb)

        sprinting = (self._want_sprint and self.stamina > 0.0
                     and not self.sprint_locked)
        if sprinting:
            self.stamina = max(0.0, self.stamina - GUARD_STAM_DRAIN * dt)
            if self.stamina <= 0.0:
                self.sprint_locked = True
        else:
            self.stamina = min(1.0, self.stamina + GUARD_STAM_REGEN * dt)
            if self.sprint_locked and self.stamina >= GUARD_STAM_UNLOCK:
                self.sprint_locked = False
        self._want_sprint = False

        self._turn(dt)
        return shots
