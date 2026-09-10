"""Damage model from the mechanics bible: shields, armor, health.

Resolution order for a hit:

  1. Shields absorb first. Energy weapons hit them harder and rail weapons
     barely dent them (``shield_mult``). When shields break, the leftover
     crosses to health at the raw rate.
  2. Armor is a percentage damage reduction on the health portion only
     (1 armor point = 1%). Enemy armor is a per-hit roll inside a band.
  3. Health is what remains. Zero health kills.

Shields regenerate after ``shield_delay`` seconds without being hit. Call
``tick`` every frame for that.

Profiles at the bottom build a Combatant from the bible's class and enemy
stat blocks (level 1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Combatant:
    x: float
    y: float
    max_health: float = 100.0
    max_shields: float = 0.0
    armor_lo: float = 0.0            # percent damage reduction, low end
    armor_hi: float = 0.0           # percent damage reduction, high end
    shield_regen: float = 0.0       # shields per second once regen resumes
    shield_delay: float = 4.0       # seconds after a hit before regen resumes
    move_penalty: float = 0.0       # fraction, from body armor
    incoming_mult: float = 1.0      # all damage taken is scaled by this
    faction: str = "guard"
    body_r: float = 0.28
    speed: float = 1.0              # base m/s (used by the AI, not the player)
    health: Optional[float] = None
    shields: Optional[float] = None
    alive: bool = True
    concealed: bool = False            # hidden (e.g. still inside a bush) - AI sight skips them
    last_hit_t: float = -1e9
    hit_from: Optional[tuple] = None   # world (x, y) the last damage came from
    _regen_at: float = field(default=-1e9, repr=False)

    def __post_init__(self):
        if self.health is None:
            self.health = self.max_health
        if self.shields is None:
            self.shields = self.max_shields

    # back-compat aliases for code that still says hp / max_hp / hurt
    @property
    def hp(self) -> float:
        return self.health

    @property
    def max_hp(self) -> float:
        return self.max_health

    def had_shields(self) -> bool:
        return self.shields > 0.0

    def take(self, dmg: float, now: float,
             shield_mult: float = 1.0, health_mult: float = 1.0,
             src: Optional[tuple] = None) -> None:
        if not self.alive or dmg <= 0.0:
            return
        self.last_hit_t = now
        if src is not None:
            self.hit_from = src
        self._regen_at = now + self.shield_delay

        remaining = dmg * self.incoming_mult
        if self.shields > 0.0:
            applied = remaining * shield_mult
            if applied < self.shields:
                self.shields -= applied
                return
            over = applied - self.shields
            self.shields = 0.0
            remaining = over / max(shield_mult, 1e-6)

        arm = random.uniform(self.armor_lo, self.armor_hi) / 100.0
        arm = min(max(arm, 0.0), 0.80)
        self.health -= remaining * health_mult * (1.0 - arm)
        if self.health <= 0.0:
            self.health = 0.0
            self.alive = False

    # kept so older call sites keep working
    def hurt(self, dmg: float, now: float = 0.0) -> None:
        self.take(dmg, now)

    def tick(self, dt: float, now: float) -> None:
        if (self.alive and now >= self._regen_at
                and self.shields < self.max_shields):
            self.shields = min(self.max_shields,
                               self.shields + self.shield_regen * dt)

    def heal_full(self) -> None:
        self.health = self.max_health
        self.shields = self.max_shields
        self.alive = True


# --- profiles --------------------------------------------------------

def player_commando(x: float, y: float, level: int = 1) -> Combatant:
    """Commando class, level 1.

    Health is a small core: 70 + level*1.25*Conditioning(22.5) -> ~98 at L1.
    Survival is meant to come from armor and shields, not the health bar.

    DEBUG STARTER KIT (not the bare-Commando design default): because the
    prototype has no loot yet, the player wears ~20% armor and a modest
    shield rig. Shields are a thin one-fight buffer that crawls back only
    after a long lull; armor carries the sustained mitigation. A naked
    Commando (Leather 3% / ~17 shields) folds in about a second of fire.
    """
    cond, logic = 22.5, 15.0
    return Combatant(
        x=x, y=y,
        max_health=round(50 + level * 1.25 * cond),
        max_shields=round(level * 1.15 * logic) + 40,
        armor_lo=20.0, armor_hi=20.0,
        shield_regen=3.5, shield_delay=6.0,
        move_penalty=0.05, faction="player", speed=4.2,
    )


# base_health, base_shields, armor band, base speed, weapon, tier.
# Human factions run thin: a soldier is a couple of rifle bursts, a
# scavenger is a burst. Consortium keeps its big shield pools - that
# archetype is meant to force energy weapons.
ENEMY_PROFILES: dict[str, dict] = {
    "imperium_soldier":     dict(hp=70,  sh=14,  a=(18, 35), spd=0.85,
                                 weapon="combat_rifle", tier="tough"),
    "imperium_squad_leader": dict(hp=95, sh=14,  a=(18, 35), spd=0.85,
                                  weapon="combat_rifle", tier="tough"),
    "imperium_spec_ops":    dict(hp=100, sh=350, a=(8, 8),   spd=6.0,
                                 weapon="smg", tier="elite"),
    "pirate":               dict(hp=75,  sh=40,  a=(11, 23), spd=0.72,
                                 weapon="smg", tier="normal"),
    "pirate_leader":        dict(hp=100, sh=55,  a=(15, 30), spd=0.69,
                                 weapon="combat_rifle", tier="tough"),
    "scavenger":            dict(hp=35,  sh=0,   a=(0, 11),  spd=0.99,
                                 weapon="pistol", tier="normal"),
    "consortium_drone":     dict(hp=50,  sh=200, a=(20, 20), spd=0.85,
                                 weapon="pulse_carbine", tier="tough"),
    "consortium_controller": dict(hp=80, sh=360, a=(20, 20), spd=0.85,
                                  weapon="pulse_carbine", tier="elite"),
}


def enemy(name: str, x: float, y: float, difficulty: float = 1.0) -> Combatant:
    p = ENEMY_PROFILES[name]
    return Combatant(
        x=x, y=y,
        max_health=p["hp"] * difficulty,
        max_shields=p["sh"],
        armor_lo=p["a"][0], armor_hi=p["a"][1],
        shield_regen=2.0, shield_delay=7.0,
        faction=name, speed=p["spd"],
    )
