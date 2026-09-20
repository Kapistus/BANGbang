"""
sim/pickups.py — health packs and ammo packs: what they do, and when they come back.

Placed in the editor like any other interactable, so where the ammo is on a map
is a design decision rather than an accident. Taken by walking over one, which
is the only version that works the same in both modes: an interact key would
mean a round trip in multiplayer, and a powerup you have to ask for twice is
not a powerup.

Two rules make them worth crossing a room for:

  a pack you cannot use does not vanish — full health walks over a health pack
  and leaves it for somebody who needs it

  ammo is a top-up, not a resupply — 30% of each carried weapon's cap, so it
  softens a dry spell without ending the pressure the caps create

The state machine is here and pygame-free because single-player, the server and
every client have to agree on it. In multiplayer the SERVER decides who got
there first; clients only draw what they are told.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

HEALTH = "health"
AMMO = "ammo"
KINDS = (HEALTH, AMMO)

TAKE_RADIUS_M = 0.55      # body radius is 0.28: you take it by standing on it
HEALTH_AMOUNT = 35.0      # health restored, capped at the body's maximum
AMMO_FRACTION = 0.30      # of each carried weapon's reserve cap, rounded up
RESPAWN_S = {HEALTH: 30.0, AMMO: 20.0}
ARRIVE_EPS = 1e-6


@dataclass
class Pickup:
    """One pack on the map, and whether it is there right now."""
    id: str
    kind: str
    x: float
    y: float
    live: bool = True
    left: float = 0.0          # seconds until it comes back

    @property
    def pos(self) -> tuple:
        return (self.x, self.y)

    @property
    def respawn_s(self) -> float:
        return RESPAWN_S.get(self.kind, 20.0)


# ---- what a pack actually does ---------------------------------------
#
# Both return None when the pack would do nothing, which is what stops a
# player at full health or full ammo from wasting one.

def health_gain(body, amount: float = HEALTH_AMOUNT) -> "float | None":
    """The health a Combatant would have after a health pack, or None if it is
    already at maximum."""
    if body is None or body.health >= body.max_health:
        return None
    return min(body.max_health, body.health + amount)


def ammo_gain(loadout, reserves, roster,
              fraction: float = AMMO_FRACTION) -> "list | None":
    """Reserves after an ammo pack: every carried weapon gains `fraction` of
    its own cap, rounded up, clamped to that cap. None if no weapon in the
    loadout can take any.

    Per weapon, not per gun-you-happen-to-hold: switching weapons to farm a
    pack would be a chore, and a chore is not a decision. A weapon with an
    unlimited reserve gains nothing — there is nothing to fill."""
    out = list(reserves)
    changed = False
    for i, key in enumerate(loadout):
        if i >= len(out):
            break
        w = roster.get(key)
        if w is None or w.reserve < 0:
            continue                      # unlimited: nothing to top up
        if out[i] >= w.reserve:
            continue                      # already full
        out[i] = min(w.reserve, out[i] + max(1, math.ceil(w.reserve * fraction)))
        changed = True
    return out if changed else None


class PickupSet:
    """Every pack on a map, and what each one is doing."""

    def __init__(self, m=None):
        self.packs: dict[str, Pickup] = {}
        if m is not None:
            self.build(m)

    def build(self, m) -> None:
        """Read the map's interactables. A pickup is just an interactable whose
        kind is one of ours, so the editor needed no new concept to place one."""
        self.packs.clear()
        for e in getattr(m, "interactables", ()) or ():
            if e.kind in KINDS:
                self.packs[e.id] = Pickup(id=e.id, kind=e.kind,
                                          x=float(e.pos[0]), y=float(e.pos[1]))

    def __len__(self) -> int:
        return len(self.packs)

    def __iter__(self):
        return iter(self.packs.values())

    def get(self, pid: str) -> "Pickup | None":
        return self.packs.get(pid)

    def live(self):
        return (p for p in self.packs.values() if p.live)

    # ---- taking one --------------------------------------------------

    def at(self, x: float, y: float,
           radius: float = TAKE_RADIUS_M) -> "Pickup | None":
        """The nearest live pack a body at (x, y) is standing on, or None."""
        best, bd = None, radius * radius
        for p in self.packs.values():
            if not p.live:
                continue
            d = (p.x - x) ** 2 + (p.y - y) ** 2
            if d <= bd:
                best, bd = p, d
        return best

    def take(self, pid: str) -> bool:
        """Mark a pack taken and start its clock. False if it was not there —
        which is what two players reaching it on the same tick looks like."""
        p = self.packs.get(pid)
        if p is None or not p.live:
            return False
        p.live = False
        p.left = p.respawn_s
        return True

    def step(self, dt: float) -> list:
        """Advance every pack that is away. Returns the ids that came back."""
        back = []
        for pid, p in self.packs.items():
            if p.live or p.left <= 0.0:
                continue
            p.left -= dt
            if p.left <= ARRIVE_EPS:
                p.left = 0.0
                p.live = True
                back.append(pid)
        return back

    # ---- over the wire -----------------------------------------------

    def wire(self) -> list:
        """The packs that are NOT as the map file has them, as [id, live, left]
        — what a latecomer needs so a pack that is away stays away for them
        too, and comes back at the same moment."""
        return [[p.id, bool(p.live), round(p.left, 2)]
                for p in self.packs.values() if not p.live]

    def apply_wire(self, rows) -> None:
        for row in rows or ():
            p = self.packs.get(str(row[0]))
            if p is None:
                continue
            p.live = bool(row[1])
            p.left = float(row[2]) if len(row) > 2 else 0.0
