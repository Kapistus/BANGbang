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
HARD_RANGE_MULT = 2.5    # a shot is discarded past optimal_range * this
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
    pen: float = 0.0            # wall-penetration budget (spends tile pen_cost)
    pellets: int = 1           # >1 for shotguns
    burst: int = 1             # rounds per trigger pull in primary fire
    burst_cycle: float = 0.0   # >0: seconds between bursts (overrides the default)
    auto_rof: float = 0.0      # >0: weapon has a full-auto alternate mode, this rof
    auto_accuracy: float = 0.0  # accuracy used in full-auto (lower than primary)
    pierce_bodies: bool = False  # beam continues through unshielded targets
    shield_mult: float = 1.0     # damage multiplier while hitting shields
    health_mult: float = 1.0     # damage multiplier while hitting health
    blast_r: float = 0.0         # >0 = explosive, area damage at impact
    sound_reach_m: float = 90.0  # how far the shot is audible, in metres
    weight: float = 1.0
    str_req: float = 0.0
    module_slots: int = 0

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
        return self.range_m * HARD_RANGE_MULT

    def fire_modes(self) -> list[str]:
        primary = "burst" if self.burst > 1 else "semi"
        return [primary, "auto"] if self.has_auto else [primary]


def spread_factor(w: Weapon, dist_m: float, moving: bool,
                  accuracy: float | None = None) -> float:
    a = min(max(w.accuracy if accuracy is None else accuracy, 0.0), 0.999)
    r = (0.5 - 0.389 * a) * SPREAD_SCALE
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
    """
    half = math.atan2(spread_factor(w, dist_m, moving, accuracy),
                      max(w.range_m, 1e-3))
    return (rng.random() + rng.random() - 1.0) * half


def roll_damage(w: Weapon, rng) -> float:
    return rng.uniform(w.dmg_lo, w.dmg_hi) * DMG_SCALE


# --- the roster --------------------------------------------------------
# pen budgets: wall/pillar cost 3.5, door 1.8, low cover 1.2 (see tiles.toml).

ROSTER: dict[str, Weapon] = {
    "pistol": Weapon(
        "pistol", "ballistic", 3, 4, 16, 2.0, 0.93, 10, 1.2,
        pen=2.0, sound_reach_m=85.0, weight=1.0, str_req=8, module_slots=1),
    "combat_rifle": Weapon(
        "combat rifle", "ballistic", 3, 5, 20, 3.0, 0.96, 25, 1.75,
        pen=2.2, burst=3, burst_cycle=0.55, auto_rof=6.0, auto_accuracy=0.83,
        sound_reach_m=105.0, weight=3.8, str_req=12.5, module_slots=3),
    "smg": Weapon(
        "sub-machinegun", "ballistic", 2, 5, 16, 8.0, 0.87, 25, 2.0,
        pen=1.8, auto_rof=13.0, auto_accuracy=0.70,
        sound_reach_m=100.0, weight=2.7, str_req=12.5, module_slots=2),
    "combat_shotgun": Weapon(
        "combat shotgun", "ballistic", 2, 4, 26, 1.0, 0.92, 8, 3.0,
        pen=1.6, pellets=5, sound_reach_m=115.0, weight=3.5, str_req=12.5,
        module_slots=1),
    "rail_pistol": Weapon(
        "rail pistol", "ballistic", 15, 20, 36, 2.0, 0.95, 10, 2.5,
        pen=4.0, pierce_bodies=True, shield_mult=0.4, health_mult=1.0,
        sound_reach_m=95.0, weight=1.4, str_req=8, module_slots=1),
    "rail_rifle": Weapon(
        "rail rifle", "ballistic", 20, 28, 40, 1.0, 0.97, 15, 2.5,
        pen=5.0, pierce_bodies=True, shield_mult=0.4, health_mult=1.0,
        sound_reach_m=110.0, weight=4.5, str_req=10, module_slots=2),
    "heavy_rifle": Weapon(
        "heavy rifle", "heavy", 15, 25, 30, 0.7, 0.97, 8, 2.5,
        pen=3.0, sound_reach_m=120.0, weight=10.2, str_req=16, module_slots=2),
    "flak_cannon": Weapon(
        "flak cannon", "heavy", 10, 15, 12, 1.0, 0.80, 5, 2.0,
        pen=2.0, burst=3, sound_reach_m=120.0, weight=9.7, str_req=17,
        module_slots=1),
    # bible lists 2 shots before reload; overridden per request - reload every shot.
    "rocket_launcher": Weapon(
        "rocket launcher", "heavy", 35, 120, 35, 0.3, 0.91, 1, 6.0,
        pen=0.0, blast_r=3.5, sound_reach_m=140.0, weight=15.2, str_req=15,
        module_slots=1),
    "flamethrower": Weapon(
        "flamethrower", "heavy", 1, 3, 7, 10.0, 0.92, 200, 10.0,
        pen=0.0, pellets=3, sound_reach_m=60.0, weight=20.0, str_req=15,
        module_slots=1),
    "pulse_carbine": Weapon(
        "pulse carbine", "energy", 5, 12, 42, 2.0, 0.98, 10, 2.5,
        pen=1.5, shield_mult=1.9, health_mult=1.0, sound_reach_m=70.0,
        weight=4.5, str_req=13, module_slots=3),
    "laser_pistol": Weapon(
        "laser pistol", "laser", 3, 7, 30, 2.0, 0.985, 20, 3.0,
        pen=1.0, pierce_bodies=True, shield_mult=1.9, health_mult=1.0,
        sound_reach_m=45.0, weight=1.0, str_req=8, module_slots=1),
    "laser_rifle": Weapon(
        "laser rifle", "laser", 8, 12, 40, 1.0, 0.99, 30, 3.5,
        pen=1.2, pierce_bodies=True, shield_mult=1.9, health_mult=1.0,
        sound_reach_m=50.0, weight=4.7, str_req=11, module_slots=2),
    "plasma_pistol": Weapon(
        "plasma pistol", "plasma", 4, 8, 25, 2.0, 0.96, 12, 2.5,
        pen=0.0, blast_r=1.6, shield_mult=1.2, health_mult=1.2,
        sound_reach_m=80.0, weight=1.1, str_req=10, module_slots=1),
    "plasma_rifle": Weapon(
        "plasma rifle", "plasma", 10, 12, 36, 1.0, 0.97, 20, 3.0,
        pen=0.0, blast_r=2.2, shield_mult=1.2, health_mult=1.2,
        sound_reach_m=95.0, weight=6.2, str_req=13, module_slots=2),
}

# What the player carries in the prototype - one of each fire type so every
# code path (single, burst, pellets, piercing beam, explosive) is reachable.
DEFAULT_LOADOUT = [
    "pistol", "combat_rifle", "smg", "combat_shotgun",
    "rail_rifle", "laser_rifle", "rocket_launcher",
]
