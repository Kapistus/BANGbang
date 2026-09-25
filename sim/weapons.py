"""Weapon roster and the accuracy model, ported from mechanics/Items/Weapons.txt.

Two things live here:

* ``Weapon`` - a flat stat block plus the derived firing parameters the
  simulation needs (refire delay from rate of fire, wall-penetration budget,
  shield/health damage multipliers, blast radius, gunshot loudness).

* the accuracy model from mechanics/Mechanics/Weapon mechanics.txt. A weapon's
  ``accuracy`` in [0, 0.999] maps to a spread factor ``r = 0.5 - 0.389*a``;
  the per-shot angular jitter is ``atan2(r, range)``. Past the optimal range
  the factor blows up ("significantly increased spread"); moving multiplies it
  by 1.10 (the 10% accuracy penalty). There is deliberately no damage falloff
  with distance - the bible models range purely through spread.

Ranges in the bible are "units"; maps are authored at one metre per unit, so
a unit is a metre here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DMG_SCALE = 4.0         # global multiplier on all weapon damage. The bible's
                        # base figures (pistol 3-4) were written for a
                        # different pacing; this brings time-to-kill into
                        # single-digit seconds against bible-scale health.
SPREAD_SCALE = 1.0       # global multiplier on all spread; tune for feel
SPREAD_WIDEN = 1.33      # every gun but the rails is this much less precise
                         # than its rating alone would say: a wider cone is
                         # what gives the person being shot at time to move.
                         # Rail weapons (steady=True) are exempt - the slow,
                         # deliberate shot is the whole point of them.
SOUND_SCALE = 0.8        # global multiplier on how far every shot is audible
HARD_RANGE_M = 50.0      # flat max distance any shot registers a hit, metres
MIN_DEV_M = 0.06         # every gun (bar the flamethrower) misses the exact
                         # cursor point by at least this much, even point-blank
MUZZLE_M = 1.10          # shot origin fallback: grip (body centre) -> muzzle, forward
                        # (per-gun barrel points live in sprites.MUZZLE_PX)
WALL_DMG_RETAIN = 0.6    # damage kept per piece of cover a bullet pierces
BODY_DMG_RETAIN = 0.8    # damage kept per body a piercing beam passes through


@dataclass(frozen=True)
class Weapon:
    name: str
    category: str          # ballistic | heavy | energy | laser | plasma
    dmg_lo: float
    dmg_hi: float
    range_m: float
    rof: float             # trigger pulls per second
    accuracy: float        # 0 .. 0.999
    mag: int
    reload_s: float
    reserve: int = -1          # rounds available beyond the starting mag; -1 = unlimited.
                               # Every gun is capped: a firefight you cannot
                               # walk away from is one you have to finish, and
                               # an ammo pack is worth crossing a room for.
                               # Quoted in whole magazines in the roster below.
    backblast: bool = False    # vents exhaust behind the shooter on firing
    pen: float = 0.0            # wall-penetration budget (spends tile pen_cost)
    steady: bool = False        # exempt from SPREAD_WIDEN (the rail weapons)
    melee_range: float = 0.0    # >0 = a blade, not a gun: no projectile, no
                                # spread, just a reach and an arc
    melee_arc_deg: float = 0.0  # half-angle of the swing
    backstab: float = 1.0       # damage multiplier from behind the target
    fuse_s: float = 0.0         # >0 = thrown on a fuse: this many seconds from
                                # pulling the pin to going off, whether it is
                                # still in your hand, in the air or on the
                                # floor. Holding fire cooks it.
    pen_charged: float = 0.0    # >0: the budget of a fully charged shot instead
    pellets: int = 1           # >1 for shotguns
    spread_deg: float = 0.0    # half-angle of the pellet cone (0 = all on the aim line)
    shell_reload_s: float = 0.0  # >0: load ONE round per this many seconds, interruptible
    burst: int = 1             # rounds per trigger pull in primary fire
    burst_cycle: float = 0.0   # >0: seconds between bursts (overrides the default)
    auto_rof: float = 0.0      # >0: weapon has a full-auto alternate mode, this rof
    auto_accuracy: float = 0.0  # accuracy used in full-auto (lower than primary)
    pierce_bodies: bool = False  # beam continues through unshielded targets
    shield_mult: float = 1.0     # damage multiplier while hitting shields
    health_mult: float = 1.0     # damage multiplier while hitting health
    blast_r: float = 0.0         # >0 = explosive, area damage at impact
    projectile_speed: float = 0.0  # >0 = travels at this m/s instead of hitscan
    burst_pellets: int = 0       # >0: the round comes apart where it lands,
                                 # throwing this many pellets outward through
                                 # the full circle. Cover stops them like any
                                 # other shot; nothing is penetrated.
    burst_range_m: float = 0.0   # how far those pellets carry
    recharge_s: float = 0.0      # >0: the magazine puts a round back every
                                 # this many seconds, on its own, with no
                                 # reload and no reserve to draw on
    sound_reach_m: float = 90.0  # how far the shot is audible, in metres
    min_dev_m: float = MIN_DEV_M  # spread floor at any range (0 = perfectly on aim)
    weight: float = 1.0
    str_req: float = 0.0
    module_slots: int = 0

    @property
    def single_load(self) -> bool:
        """Everything it will ever fire is already in it: no reserve to draw
        on and nothing that puts rounds back, so there is no reload - a belt
        of three grenades, not one grenade and two spares."""
        return self.reserve == 0 and self.recharge_s <= 0.0

    @property
    def refire(self) -> float:
        return 1.0 / self.rof if self.rof > 0 else 0.2

    @property
    def burst_time(self) -> float:
        """Seconds between primary-fire trigger pulls."""
        if self.burst_cycle > 0.0:
            return self.burst_cycle
        return self.refire * self.burst + (0.15 if self.burst > 1 else 0.0)

    @property
    def auto_refire(self) -> float:
        return 1.0 / self.auto_rof if self.auto_rof > 0 else self.refire

    @property
    def has_auto(self) -> bool:
        return self.auto_rof > 0.0

    @property
    def hard_range_m(self) -> float:
        return HARD_RANGE_M          # flat 50 m hit-registration cap for every gun

    @property
    def shell_reload(self) -> bool:
        return self.shell_reload_s > 0.0

    @property
    def reload_step(self) -> float:
        """Time to the next reload increment: one shell, or the whole mag."""
        return self.shell_reload_s if self.shell_reload_s > 0.0 else self.reload_s

    @property
    def is_melee(self) -> bool:
        return self.melee_range > 0.0

    @property
    def is_cooked(self) -> bool:
        """Held to cook, thrown on release, and it goes off when the fuse
        does - in your hand if you leave it that long."""
        return self.fuse_s > 0.0

    def fire_modes(self) -> list[str]:
        primary = "burst" if self.burst > 1 else "semi"
        return [primary, "auto"] if self.has_auto else [primary]


def spread_factor(w: Weapon, dist_m: float, moving: bool,
                  accuracy: float | None = None) -> float:
    a = min(max(w.accuracy if accuracy is None else accuracy, 0.0), 0.999)
    r = (0.5 - 0.389 * a) * SPREAD_SCALE * (1.0 if w.steady else SPREAD_WIDEN)
    if dist_m > w.range_m:
        r *= 1.0 + 2.0 * (dist_m / w.range_m - 1.0)
    if moving:
        r *= 1.10
    return r


def jitter(w: Weapon, dist_m: float, moving: bool, rng,
           accuracy: float | None = None) -> float:
    """A single shot's angular error, radians. Triangular, so it clusters.

    Pass ``accuracy`` to override the weapon's rating - full-auto uses the
    lower ``w.auto_accuracy``.

    The metre-miss the model implies at ``dist_m`` is floored at
    ``w.min_dev_m`` so no gun (bar the flamethrower, min_dev_m 0) ever lands
    exactly on the cursor, even at point-blank range.
    """
    sf = spread_factor(w, dist_m, moving, accuracy)      # miss in m at optimal range
    floor = w.min_dev_m * (1.0 if w.steady else SPREAD_WIDEN)
    lin = max(sf * dist_m / max(w.range_m, 1e-3), floor)
    half = math.atan2(lin, max(dist_m, 0.3))
    return (rng.random() + rng.random() - 1.0) * half


def pellet_offset(w: Weapon, rng) -> float:
    """Angular offset (radians) of one pellet within the weapon's cone.
    Triangular over [-spread_deg, +spread_deg] so the pattern is denser in
    the middle. Zero for anything that is not a spread weapon."""
    if w.spread_deg <= 0.0:
        return 0.0
    half = math.radians(w.spread_deg) * (1.0 if w.steady else SPREAD_WIDEN)
    return (rng.random() + rng.random() - 1.0) * half


def roll_damage(w: Weapon, rng) -> float:
    return rng.uniform(w.dmg_lo, w.dmg_hi) * DMG_SCALE


# --- the roster --------------------------------------------------------
# pen budgets: wall/pillar cost 3.5, door 1.8, low cover 1.2 (see tiles.toml).

ROSTER: dict[str, Weapon] = {
    # sidearm identity: the quietest firearm, precise, one carefully-placed
    # shot at a time - poor sustained fire (small mag, slow), but quiet enough
    # to drop an isolated guard without bringing the room
    "pistol": Weapon(
        # the trigger is as fast as the hand: a sidearm you can empty in three
        # seconds, still one carefully-placed shot at a time
        "pistol", "ballistic", 7, 10, 16, 2.6, 0.97, 8, 1.0,
        reserve=48, pen=2.0, sound_reach_m=50.0, weight=1.0, str_req=8, module_slots=1),
    # the generalist baseline: no weakness, no specialty, most customisable
    # (3 module slots); burst is the reliable mid-range answer, auto for panic
    "combat_rifle": Weapon(
        "combat rifle", "ballistic", 4, 6, 20, 3.0, 0.96, 25, 1.75,
        reserve=125, pen=2.2, burst=3, burst_cycle=0.50, auto_rof=6.0, auto_accuracy=0.83,
        sound_reach_m=105.0, weight=3.8, str_req=12.5, module_slots=3),
    "smg": Weapon(
        "sub-machinegun", "ballistic", 2, 4.5, 16, 8.0, 0.87, 25, 2.0,
        reserve=150, pen=1.8, auto_rof=13.0, auto_accuracy=0.70,
        sound_reach_m=100.0, weight=2.7, str_req=12.5, module_slots=2),
    # the close-quarters answer: ten pellets, and what decides a fight is how
    # many of them are still on the body. All ten kills anything but a Heavy
    # outright; at 5 m - across a room, down a hallway - half of them land and
    # it takes two shells; past 12 m the pattern is wider than a person and it
    # is the wrong gun.
    "combat_shotgun": Weapon(
        "combat shotgun", "ballistic", 5, 6.25, 20, 2.0, 0.92, 8, 3.0,
        reserve=40, pen=1.6, pellets=10, spread_deg=9.0, shell_reload_s=1.0,
        sound_reach_m=115.0, weight=3.5, str_req=12.5, module_slots=1),
    # rail: anti-armour, shreds cover, feeble against shields. pistol = the
    # quick peek-and-wallbang sidearm (punches one wall), rifle = the heavy
    # anti-materiel piece (the wall is irrelevant; slow, deliberate, deletes)
    "rail_pistol": Weapon(
        "rail pistol", "ballistic", 11, 16, 36, 1.6, 0.95, 8, 2.5,
        reserve=32, pen=2.5, pierce_bodies=True, shield_mult=0.4, health_mult=1.0,
        steady=True, pen_charged=3.3,    # full charge only: through a blast door (3.2), not a wall (3.5)
        sound_reach_m=95.0, weight=1.4, str_req=8, module_slots=1),
    "rail_rifle": Weapon(
        "rail rifle", "ballistic", 20, 28, 40, 0.8, 0.97, 5, 3.2,
        reserve=20, pen=6.0, pierce_bodies=True, shield_mult=0.4, health_mult=1.0,
        steady=True,
        sound_reach_m=110.0, weight=4.5, str_req=10, module_slots=2),
    # the reliable hard hitter: one big ballistic round, the longest dependable
    # ballistic range, best per-shot against a shielded target (normal mults)
    "heavy_rifle": Weapon(
        "heavy rifle", "heavy", 15, 25, 38, 0.9, 0.98, 8, 2.5,
        reserve=32, pen=2.5, sound_reach_m=120.0, weight=10.2, str_req=16, module_slots=2),
    # breacher: a 3-round burst of frag rounds, each a small blast in a tight
    # cone - clears a doorway, murders anything close, hazardous to fire point
    # blank (the shooter is in its own blast)
    # a shell rather than a burst: it flies, and where it stops - a wall, a
    # body, or the end of its run - it comes apart into a full circle of
    # pellets. Point blank it is the shell that kills; across a doorway it is
    # the ring, and the ring does not go through cover. range_m here is how
    # far the shell carries before it comes apart on its own: short enough
    # that it is a room weapon, long enough that it crosses the room rather
    # than going off in front of your face.
    "flak_cannon": Weapon(
        "flak cannon", "heavy", 5.5, 8, 10, 1.0, 0.86, 5, 2.4,
        reserve=15, pen=0.0, blast_r=0.7, projectile_speed=20.0,
        burst_pellets=36, burst_range_m=2.5,
        sound_reach_m=120.0, weight=9.7, str_req=17, module_slots=1),
    # bible lists 2 shots before reload; overridden per request - reload every shot.
    # bible blast radius is "2-5 units"; 4 m radius = 8 m kill diameter.
    "rocket_launcher": Weapon(
        "rocket launcher", "heavy", 35, 120, 35, 0.3, 0.91, 1, 6.0,
        reserve=2, backblast=True,          # 1 loaded + 2 spare = 3 rockets, ever
        pen=0.0, blast_r=4.0, projectile_speed=28.0, sound_reach_m=140.0,
        weight=15.2, str_req=15, module_slots=1),
    # the Saboteur's blade: silent, and murderous from behind. Nothing about
    # it works at a distance, which is the whole trade
    "combat_knife": Weapon(
        # two swings from the front, one from behind: a blade you have to walk
        # into someone to use should settle it when you get there
        "combat knife", "melee", 26, 34, 1.6, 1.6, 1.0, 1, 0.0,
        reserve=-1, melee_range=1.6, melee_arc_deg=50.0, backstab=4.5,
        sound_reach_m=8.0, min_dev_m=0.0, weight=0.6, str_req=6,
        module_slots=0),
    # thrown on a three-second fuse that starts when you pull the pin, not
    # when it lands. Hold to cook it: a cooked grenade gives the room no time,
    # and one held too long goes off in your hand
    # three of them, and no reload at all: what you are carrying is what you
    # have, and one that goes off at your feet kills anything but a braced
    # Heavy outright
    "frag_grenade": Weapon(
        "frag grenade", "heavy", 55, 72, 14, 1.0, 0.93, 3, 0.0,
        reserve=0, pen=0.0, blast_r=3.5, projectile_speed=16.0, fuse_s=3.0,
        sound_reach_m=130.0, weight=0.5, str_req=8, module_slots=0),
    # corridor denial: a short wide cone of fire, high flesh multiplier - no
    # one walks down this hallway and nothing stays in that doorway
    "flamethrower": Weapon(
        "flamethrower", "heavy", 3, 6, 8, 10.0, 0.92, 200, 10.0,
        reserve=300, pen=0.0, pellets=4, spread_deg=14.0, health_mult=1.5,
        sound_reach_m=60.0, min_dev_m=0.0, weight=20.0, str_req=15,
        module_slots=1),
    # shield sniper: hitscan, the longest reach and tightest accuracy of any
    # gun, low mag - strip shielded targets from across the map
    "pulse_carbine": Weapon(
        "pulse carbine", "energy", 5, 12, 46, 2.0, 0.99, 8, 2.5,
        reserve=32, pen=1.5, shield_mult=1.9, health_mult=1.0, sound_reach_m=70.0,
        weight=4.5, str_req=13, module_slots=3),
    # near-silent anti-shield sidearm that passes through an unshielded body -
    # line two guards up and drop both without a sound
    "laser_pistol": Weapon(
        "laser pistol", "laser", 3, 7, 30, 2.5, 0.985, 20, 3.0,
        reserve=100, pen=1.0, pierce_bodies=True, shield_mult=1.9, health_mult=1.0,
        sound_reach_m=45.0, weight=1.0, str_req=8, module_slots=1),
    # a slow-ish travelling plasma bolt (like the rocket, small splash) that
    # hits hard, especially against shields
    # five bolts and no spare cells: the weapon makes its own, one every five
    # seconds, so it is never empty for long and never fires for long either
    "laser_rifle": Weapon(
        "plasma rifle", "plasma", 12, 18, 40, 1.6, 0.99, 5, 3.5,
        reserve=0, recharge_s=5.0,
        pen=0.0, blast_r=1.2, projectile_speed=46.0, shield_mult=1.9,
        health_mult=1.0, sound_reach_m=75.0, weight=4.7, str_req=11,
        module_slots=2),
    # plasma family = travelling bolts w/ splash, balanced mults. pistol: a
    # fast light bolt with a small splash for clumped guards. cannon: a slow
    # heavy lob with a big splash - the rocket's repeatable understudy.
    "plasma_pistol": Weapon(
        "plasma pistol", "plasma", 4, 8, 25, 2.0, 0.96, 12, 2.5,
        reserve=60, pen=0.0, blast_r=1.3, projectile_speed=55.0,
        shield_mult=1.2, health_mult=1.2,
        sound_reach_m=80.0, weight=1.1, str_req=10, module_slots=1),
    "plasma_rifle": Weapon(
        "plasma cannon", "plasma", 10, 12, 36, 1.0, 0.97, 20, 3.0,
        reserve=60, pen=0.0, blast_r=2.6, projectile_speed=34.0,
        shield_mult=1.2, health_mult=1.2,
        sound_reach_m=95.0, weight=6.2, str_req=13, module_slots=2),
}

# Apply the global audible-range trim to every weapon in one place, so the
# per-weapon figures above stay readable as relative loudness.
for _w in ROSTER.values():
    object.__setattr__(_w, "sound_reach_m", round(_w.sound_reach_m * SOUND_SCALE, 1))
del _w


# What the player carries in the prototype - one of each fire type so every
# code path (single, burst, pellets, piercing beam, explosive) is reachable.
DEFAULT_LOADOUT = [
    "pistol", "combat_rifle", "smg", "combat_shotgun",
    "rail_rifle", "laser_rifle", "rocket_launcher",
]


def recharge_step(loadout, mags, charges, dt) -> bool:
    """Weapons that make their own ammunition, one round at a time.

    A plasma rifle carries no spare cells: the magazine puts a round back every
    `recharge_s` seconds whether it is in your hands or slung, which is why it
    has five of them rather than twenty-four. `charges` is one timer per slot,
    the same length as `loadout`; both modes step it with this, so the bar on
    the HUD is the same clock the shot comes out of. Returns True if a round
    was added.
    """
    added = False
    for i, key in enumerate(loadout):
        w = ROSTER[key]
        if w.recharge_s <= 0.0 or mags[i] >= w.mag:
            charges[i] = 0.0
            continue
        charges[i] += dt
        while charges[i] >= w.recharge_s and mags[i] < w.mag:
            charges[i] -= w.recharge_s
            mags[i] += 1
            added = True
        if mags[i] >= w.mag:
            charges[i] = 0.0
    return added


def recharge_frac(key: str, mag_now: int, charge: float) -> float:
    """How far along the next self-made round is, 0..1 (0 = not recharging)."""
    w = ROSTER[key]
    if w.recharge_s <= 0.0 or mag_now >= w.mag:
        return 0.0
    return max(0.0, min(1.0, charge / w.recharge_s))
