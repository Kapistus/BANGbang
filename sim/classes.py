"""Multiplayer player classes: what each one is, and the body it spawns in.

From the class sheets in mechanics/Classes, brought to the scale the
prototype Commando already plays at (78 health, 57 shields, 20% armour,
4.2 m/s):

  health   each class's design health at level 1, scaled by the ratio the
           prototype Commando runs at (78 of a design 178), then by
           TOUGHNESS. The design formula taken literally leaves Tech on about
           12 health.
  shields  level 1 Logic shields plus the same 40-point starter rig the
           prototype Commando wears, then by TOUGHNESS
  armour   the starting armour: leather 20%, Heavy support's metal 30%
  speed    the design speed formula. Tech's comes out at 2.1 m/s - half the
           Commando - so it is raised to 3.4.

The fields above are the design numbers. What the game and the class card use
are the rounded ones: health and shields to the nearest whole point, speed to
the nearest half a metre per second, since a card reading 4.2 m/s against
4.1 m/s says nothing anyone can feel. Read them off `hp`, `sh` and `spd`.

Movement in multiplayer runs on stance speeds (walk, run, crawl) rather than
one number, so a class moves at `speed / 4.2` of every stance speed: the
Commando is the baseline, a Heavy walks and runs about a quarter slower.

Each class starts with a pistol and two guns. Only stats and weapons differ;
class skills (a Medic healing, say) are not built.
"""
from __future__ import annotations

from dataclasses import dataclass

from sim import combat

BASE_SPEED = 4.0        # the Commando's rounded speed; stance speeds are
                        # tuned for it, so the Commando runs at 1.0
TOUGHNESS = 1.25        # health and shields, over what the class sheets give.
                        # Part of giving people time to react: with the
                        # widened spread and the trimmed SMG it puts most
                        # firefights between one and two seconds rather than
                        # under one. Single-player's Commando is not scaled.


ABILITY_COOLDOWN_S = 30.0    # every class, from the moment the ability ENDS


@dataclass(frozen=True)
class Ability:
    """What a class's one special does. The server runs all of it; the client
    predicts the Commando's speed and draws the Tech's ping."""
    key: str                # "brace" | "heal" | "blitz" | "ping"
    name: str
    duration: float         # seconds it lasts, or channels for
    hint: str               # one line, for the HUD and the class card


ABILITIES = {
    "heal": Ability("heal", "Field dressing", 3.0,
                    "channeled heal; moving, firing or a hit stops it"),
    "blitz": Ability("blitz", "Blitz", 6.0,
                     "double speed; an empty gun reloads itself"),
    "ping": Ability("ping", "Echo ping", 3.0,
                    "walls and people through walls, 3 s"),
    "brace": Ability("brace", "Brace", 6.0,
                     "half damage taken while still"),
    "vanish": Ability("vanish", "Vanish", 4.0,
                      "unseen past 3 m, silent, twice as fast; "
                      "firing ends it"),
}

# tuning, one place
HEAL_FRACTION = 0.5        # of max health, paid out over the channel
BLITZ_SPEED = 2.0          # multiplier on top of the class's own speed
VANISH_SPEED = 2.0         # the Saboteur crosses ground while unseen
PING_RADIUS_M = 12.0
BRACE_MITIGATION = 0.5     # damage taken while braced and still
VANISH_SEEN_M = 3.0        # close enough and a vanished player is drawn anyway


@dataclass(frozen=True)
class PlayerClass:
    key: str
    name: str
    health: float
    shields: float
    armour: float           # percent damage reduction on health
    speed: float            # m/s, against the Commando's 4.2
    loadout: tuple          # weapons.ROSTER keys, slot order
    ability: str            # key into ABILITIES
    blurb: str
    art: str = ""           # sprite stem for this class ("" = the base soldier)

    @property
    def spd(self) -> float:
        """Speed as the game uses it: the design number to the nearest half."""
        return round(self.speed * 2.0) / 2.0

    @property
    def speed_mult(self) -> float:
        return self.spd / BASE_SPEED

    @property
    def hp(self) -> float:
        return float(round(self.health * TOUGHNESS))

    @property
    def sh(self) -> float:
        return float(round(self.shields * TOUGHNESS))


CLASSES: dict[str, PlayerClass] = {c.key: c for c in (
    PlayerClass("commando", "Commando", 78, 57, 20, 4.2,
                ("pistol", "combat_rifle", "smg"), "blitz",
                "all-rounder: range and rooms"),
    PlayerClass("heavy_support", "Heavy support", 103, 52, 30, 2.9,
                ("pistol", "flak_cannon", "rocket_launcher"), "brace",
                "toughest, slowest; flak and rockets",
                art="heavy_support"),
    PlayerClass("medic", "Medic", 75, 60, 20, 4.1,
                ("pistol", "smg", "combat_shotgun"), "heal",
                "close quarters; the only healer", art="medic"),
    PlayerClass("tech", "Tech", 49, 78, 20, 3.4,
                ("pistol", "rail_rifle", "laser_rifle"), "ping",
                "thin, big shields, rail and plasma",
                art="tech"),
    # the art file is still assets/characters/unnamed_idle.png; rename the
    # file and this one field together when it gets its own name
    PlayerClass("saboteur", "Saboteur", 70, 45, 10, 5.0,
                ("combat_knife", "frag_grenade"), "vanish",
                "fastest and thinnest; no gun at all", art="unnamed"),
)}
ORDER = list(CLASSES)
DEFAULT = "commando"


def get(key: str | None) -> PlayerClass:
    """The class for a key; anything unknown is the Commando."""
    return CLASSES.get(key or DEFAULT, CLASSES[DEFAULT])


def cycle(key: str | None, step: int) -> str:
    """The class `step` places along from `key`, wrapping."""
    i = ORDER.index(get(key).key)
    return ORDER[(i + step) % len(ORDER)]


def make_body(key: str | None, x: float, y: float) -> "combat.Combatant":
    """A fresh body for this class. Shield regeneration is the prototype's
    for everyone: 3.5/s, after 6 s without taking damage."""
    c = get(key)
    return combat.Combatant(
        x=x, y=y,
        max_health=float(c.hp), max_shields=float(c.sh),
        armor_lo=float(c.armour), armor_hi=float(c.armour),
        shield_regen=3.5, shield_delay=6.0,
        move_penalty=0.05, faction="player", speed=c.spd,
    )


def ability_of(key: str | None) -> Ability:
    return ABILITIES[get(key).ability]


# abilities that move you faster while they run, and by how much. One table,
# because the server decides it and the client predicts it: if these two ever
# disagree the player rubber-bands for the whole six seconds.
SPEED_ABILITIES = {"blitz": BLITZ_SPEED, "vanish": VANISH_SPEED}


def ability_speed(key: str | None, active: bool) -> float:
    """The speed multiplier this class's ability is worth right now."""
    if not active:
        return 1.0
    return SPEED_ABILITIES.get(ability_of(key).key, 1.0)


def stat_line(key: str | None) -> str:
    c = get(key)
    return (f"{c.hp:.0f} health · {c.sh:.0f} shields · "
            f"{c.armour:.0f}% armour · {c.spd:.1f} m/s")


def weapon_line(key: str | None) -> str:
    from sim import weapons
    return ", ".join(weapons.ROSTER[k].name for k in get(key).loadout)
