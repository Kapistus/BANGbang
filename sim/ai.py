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

    def _step_toward(self, m, tx, ty, dt, emit_cb, gait_energy) -> float:
        """Move toward (tx, ty), sliding on walls. Returns distance still to go."""
        dx, dy = tx - self.x, ty - self.y
        d = math.hypot(dx, dy)
        if d < 1e-4:
            return 0.0
        self._face_target = math.atan2(dy, dx)
        step = min(d, self._speed() * dt)
        ux, uy = dx / d * step, dy / d * step
        if m.can_stand(self.x + ux, self.y, self.body_r):
            self.x += ux
        if m.can_stand(self.x, self.y + uy, self.body_r):
            self.y += uy
        self.since_step += step
        if self.since_step >= STRIDE:
            self.since_step = 0.0
            emit_cb(self.x, self.y,
                    gait_energy * (0.7 + 0.6 * self.alert), f"{self.id}/step")
        return d - step

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
        if rem < 0.10:
            self.i = (self.i + 1) % len(self.pts)
            self.wait = rng.uniform(0.6, 1.8)

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
        if rem < 0.5 or now >= self._poke_next:
            self._poke_next = now + rng.uniform(1.5, 3.0)
            for _ in range(6):
                nx = self.x + rng.uniform(-3.5, 3.5)
                ny = self.y + rng.uniform(-3.5, 3.5)
                if m.can_stand(nx, ny, self.body_r):
                    self.investigate = (nx, ny)
                    break

    def _do_combat(self, dt, now, m, pen, blocks_sight, blocks_bullets,
                   player, rng, emit_cb):
        dx, dy = player.x - self.x, player.y - self.y
        dist = math.hypot(dx, dy)
        self._face_target = math.atan2(dy, dx)
        moving = dist > KEEP_MAX_M or dist < KEEP_MIN_M
        if dist > KEEP_MAX_M:
            self._step_toward(m, player.x, player.y, dt, emit_cb,
                              GUARD_FOOTSTEP_M * self.cpm)
        elif dist < KEEP_MIN_M and dist > 1e-3:
            self._step_toward(m, self.x - dx, self.y - dy, dt, emit_cb,
                              GUARD_FOOTSTEP_M * self.cpm)

        w = self.weapon
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
        if band == 1 and player.alive and self.alert >= A_COMBAT:
            self.mode = Mode.COMBAT
        elif self.mode == Mode.COMBAT and band == 1 and player.alive:
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
