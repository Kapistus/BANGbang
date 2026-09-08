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
BLIND_ENGAGE_M = 7.0  # a recent attacker this close is fought even without a cone sight
HEAR_MIN = 0.10       # remaining sound energy that still registers

ALERT_SPEEDUP = 0.9   # extra fraction of base speed at full alert
TURN_BASE = 3.2       # rad/s
STRIDE = 0.85         # metres between footstep sounds

VIEW_RANGE_M = 17.0
FOV_IDENT = math.radians(40.0)    # half-angle of the identifying cone
FOV_PERIPH = math.radians(95.0)   # half-angle of peripheral vision
IDENT_RANGE_M = 15.0
PERIPH_RANGE_M = 7.0

KEEP_MIN_M = 3.5      # back off if the player gets closer than this
KEEP_MAX_M = 8.0      # advance if the player is further than this
GUARD_FOOTSTEP_M = 5.0
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
        self.weapon = weapons.ROSTER[ENEMY_PROFILES[profile]["weapon"]]
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
        self._cover = None          # (x, y) cover anchor being used, or None
        self._cover_next = 0.0      # earliest time to re-evaluate cover
        self._peeking = False       # currently leaning out of cover to fire
        self._peek_end = 0.0
        self._cur_side = 1.0        # which way this peek leans
        self._peek_last = 1.0       # last side that had a clean shot
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

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _roll_idle() -> float:
        """Metres of patrol travel until the next standing pause, or never."""
        return random.uniform(6.0, 16.0) if random.random() < 0.6 else 1e9

    def _cell(self):
        return int(self.x * self.cpm), int(self.y * self.cpm)

    def _speed(self) -> float:
        return self.speed * (1.0 + ALERT_SPEEDUP * self.alert)

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

    # -- perception ---------------------------------------------------

    def sees(self, blocks_sight, player) -> int:
        """0 nothing, 1 identified (in the narrow cone), 3 peripheral only."""
        if not player.alive:
            return 0
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

    def _guess_source(self, heard):
        rem, ang = heard
        if ang is None:
            return self.investigate
        reach = 11.0 * (1.0 - rem) + 3.0        # louder -> closer guess
        return (self.x + math.cos(ang) * reach,
                self.y + math.sin(ang) * reach)

    # -- motion primitives ------------------------------------------

    def _turn(self, dt: float) -> None:
        rate = TURN_BASE * (1.0 + 1.5 * self.alert)
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
        step = min(d, self._speed() * dt)
        bx, by = self.x, self.y
        ux, uy = dx / d * step, dy / d * step
        if m.can_stand(self.x + ux, self.y, self.body_r):
            self.x += ux
        if m.can_stand(self.x, self.y + uy, self.body_r):
            self.y += uy
        moved = math.hypot(self.x - bx, self.y - by)
        self._blocked = moved < step * 0.35
        self.since_step += moved
        if self.since_step >= STRIDE:
            self.since_step = 0.0
            emit_cb(self.x, self.y,
                    gait_energy * (0.7 + 0.6 * self.alert), f"{self.id}/step")
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
        if rem < 0.10 or self._blocked:
            # reached it, or the leg is obstructed (a door shut, say) - either
            # way move on to the next waypoint rather than grind the wall
            self.i = (self.i + 1) % len(self.pts)
            self.wait = rng.uniform(0.6, 1.8)
            self._blocked = False

    def _do_idle(self, dt, now):
        self.scan_t += dt
        self._face_target = self._idle_base + math.sin(self.scan_t * 0.9) * 0.8
        if now >= self.idle_until:
            self.mode = Mode.PATROL

    def _do_search(self, dt, now, m, rng, emit_cb):
        self.scan_t += dt
        if self.investigate is None:
            self.investigate = (self.x, self.y)
        tx, ty = self.investigate
        rem = self._step_toward(m, tx, ty, dt, emit_cb,
                                GUARD_FOOTSTEP_M * self.cpm)
        base = math.atan2(ty - self.y, tx - self.x)
        self._face_target = base + math.sin(self.scan_t * 2.0) * 0.9
        if rem < 0.5 or now >= self._poke_next or self._blocked:
            self._poke_next = now + rng.uniform(1.5, 3.0)
            self._blocked = False
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
            if near > 1.8 and not line_of_sight(
                    m.blocks_sight, self.x * cpm, self.y * cpm,
                    cx * cpm, cy * cpm):
                continue                          # can't straight-line to it
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
            u = dist or 1.0
            if sb is not None and sb <= 2.0 and now >= self._cover_next \
                    and self._peekable_from((self.x, self.y), m, blocks_bullets, player):
                # a blocker is right here and we CAN lean out of it - use it
                self._cover = (self.x, self.y)
                self._peek_end = now
                moving = False
            elif sb is not None:
                # blocked shot, no peekable cover - commit to flanking one way
                # around the obstacle for a beat rather than dithering
                if now >= self._flank_until:
                    self._flank_dir = self._peek_last
                    self._flank_until = now + 1.8
                s = self._flank_dir
                tx = self.x - (dy / u) * 2.5 * s + (dx / u) * 1.2
                ty = self.y + (dx / u) * 2.5 * s + (dy / u) * 1.2
                if not m.can_stand(tx, ty, self.body_r):
                    self._flank_dir = s = -s
                    self._flank_until = now + 1.8
                    tx = self.x - (dy / u) * 2.5 * s + (dx / u) * 1.2
                    ty = self.y + (dx / u) * 2.5 * s + (dy / u) * 1.2
                self._step_toward(m, tx, ty, dt, emit_cb, ge, face=False)
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

        self.fire_cd = w.burst_time
        shots = []
        for _ in range(w.burst):
            if self.mag <= 0:
                break
            self.mag -= 1
            for _p in range(w.pellets):
                hd = self._face_target + weapons.jitter(w, dist, moving, rng)
                sh = ballistics.fire_shot(m, blocks_bullets, pen, m.glass,
                                          (self.x, self.y), hd, w,
                                          [player], rng, now)
                shots.append(sh)
                if w.blast_r > 0.0:
                    ballistics.blast(sh.impact, w.blast_r, w, [player], rng, now)
        emit_cb(self.x, self.y, w.sound_reach_m * self.cpm, f"{self.id}/fire")
        return shots

    # -- top level ------------------------------------------------

    def update(self, dt, now, m, cost, pen, blocks_sight, blocks_bullets,
               sounds, player, rng, emit_cb):
        """Advance one guard. Returns the list of ballistics.Shot it fired."""
        self.fire_cd = max(0.0, self.fire_cd - dt)
        if self.reload_t > 0.0:
            self.reload_t -= dt
            if self.reload_t <= 0.0:
                self.mag = self.weapon.mag
        elif self.mode != Mode.COMBAT and self.mag < self.weapon.mag:
            self.reload_t = self.weapon.reload_s      # top up between fights

        band = self.sees(blocks_sight, player)
        heard = self._hear(sounds, now)

        # taking fire is its own stimulus: a guard shot from outside its
        # vision cone should whirl toward the hit, not amble off searching
        shot_recently = (now - self.last_hit_t < SHOT_MEMORY
                         and self.hit_from is not None)
        if shot_recently:
            self.alert = 1.0
            self.investigate = self.hit_from
            self.search_until = now + SEARCH_TIME

        if band == 1:
            self.alert = min(1.0, self.alert + SIGHT_GAIN * dt)
            self.investigate = (player.x, player.y)
            self.search_until = now + SEARCH_TIME
        elif band == 3:
            self.alert = min(1.0, self.alert + PERIPH_GAIN * dt)
            self.investigate = (player.x, player.y)
            self.search_until = now + SEARCH_TIME
        elif heard is not None:
            self.alert = min(1.0, self.alert + SOUND_GAIN * heard[0] * dt)
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
        if band in (1, 3) and player.alive and self.alert >= A_COMBAT:
            self.mode = Mode.COMBAT
        elif shot_recently and player.alive and pdist <= BLIND_ENGAGE_M:
            self.mode = Mode.COMBAT                 # point-blank attacker
        elif self.mode == Mode.COMBAT and band in (1, 3) and player.alive:
            pass                                   # hold the fight
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
        if prev == Mode.COMBAT and self.mode != Mode.COMBAT:
            self.alert = max(self.alert, LOSE_SIGHT_ALERT)
            self.search_until = now + SEARCH_TIME
            if self.investigate is None:
                self.investigate = (player.x, player.y)
        if self.mode == Mode.SEARCH and prev != Mode.SEARCH:
            self.search_until = max(self.search_until, now + SEARCH_TIME)
            self._poke_next = now + 2.0

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

        self._turn(dt)
        return shots
