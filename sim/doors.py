"""
sim/doors.py — sliding doors, and how long they take.

Two kinds, both split down the middle and retracting into the jambs:

    door         powered, open before you have finished the thought
    blast door   thick, slow, and five seconds of standing there while it moves

Neither is passable, transparent or quiet until the panels are fully home. A
door in motion is a wall, and that is the point of the slow one: the five
seconds are a cost, and you pay them out in the open where anyone can see you
paying. A door already moving ignores the key — once a blast door starts, it
finishes.

The state machine is here and pygame-free because single-player, the server and
every client have to agree on it to the tick. Drawing is main.draw_door.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_SLIDE = 0.35      # a door with no authored time
ARRIVE_EPS = 1e-6


def axis_from_solid(is_solid, r: int, c: int) -> str:
    """Which way the panels retract, given a way to ask whether a neighbouring
    cell is solid: "h" into the jambs left and right, "v" into the jambs above
    and below.

    Read off the wall the door sits in, because that is where the panels go: a
    door in a horizontal wall run has solid tiles either side of it, and those
    are the only places a panel can disappear into. A doorway in open ground
    has no answer, so it takes "h".

    Split out from `door_axis` so the map editor can run the same rule over its
    own grid and show the mapper which way a door will open."""
    horiz = bool(is_solid(r, c - 1)) + bool(is_solid(r, c + 1))
    vert = bool(is_solid(r - 1, c)) + bool(is_solid(r + 1, c))
    return "v" if vert > horiz else "h"


def door_axis(m, r: int, c: int) -> str:
    """`axis_from_solid` over a loaded TileMap."""
    rows, cols = m.chars.shape

    def solid(rr: int, cc: int) -> bool:
        if not (0 <= rr < rows and 0 <= cc < cols):
            return True                    # off-map reads as wall: doors on an
        t = m.tiles.get(m.chars[rr, cc])   # edge still have something to slide into
        return bool(t is not None and t.blocks_move and not t.door)

    return axis_from_solid(solid, r, c)


@dataclass
class Door:
    """One doorway: where it is, which way it opens, and what it is doing."""
    r: int
    c: int
    axis: str = "h"
    dur: float = DEFAULT_SLIDE       # seconds the panels take end to end
    is_open: bool = False            # is the DOORWAY open: panels fully home
                                     # and still. Never true during a travel
    target: bool = False             # what they are travelling toward
    left: float = 0.0                # seconds of travel remaining

    @property
    def moving(self) -> bool:
        return self.left > 0.0

    @property
    def frac(self) -> float:
        """0 shut, 1 fully open — what gets drawn.

        This is the only place the in-between exists. Sight, movement, bullets
        and sound all read `is_open`, which is False for every instant of the
        travel, in both directions: a door part-way open is not a gap you can
        squeeze through, and a door part-way shut is not one you can dive
        back out of."""
        if not self.moving:
            return 1.0 if self.is_open else 0.0
        done = 1.0 - self.left / max(self.dur, ARRIVE_EPS)
        return done if self.target else 1.0 - done

    @property
    def heavy(self) -> bool:
        """A door slow enough that waiting for it is a decision."""
        return self.dur >= 1.0


class DoorSet:
    """Every door on a map, and what each one is doing.

    Quacks like the `{(r, c): is_open}` dict this used to be — `get`, `items`,
    iteration and `in` all still work, so roofing and anything else that only
    wants the open/shut answer needs no changes."""

    def __init__(self, m=None):
        self.doors: dict[tuple[int, int], Door] = {}
        if m is not None:
            self.build(m)

    def build(self, m) -> None:
        rows, cols = m.chars.shape
        for r in range(rows):
            for c in range(cols):
                t = m.tiles.get(m.chars[r, c])
                if t is not None and t.door:
                    self.doors[(r, c)] = Door(
                        r=r, c=c, axis=door_axis(m, r, c),
                        dur=max(0.0, float(getattr(t, "door_time",
                                                   DEFAULT_SLIDE))))

    # ---- dict face ---------------------------------------------------

    def __contains__(self, key) -> bool:
        return key in self.doors

    def __iter__(self):
        return iter(self.doors)

    def __len__(self) -> int:
        return len(self.doors)

    def __getitem__(self, key) -> bool:
        return self.doors[key].is_open

    def get(self, key, default=False) -> bool:
        d = self.doors.get(key)
        return default if d is None else d.is_open

    def items(self):
        return ((k, d.is_open) for k, d in self.doors.items())

    def values(self):
        return (d.is_open for d in self.doors.values())

    def door(self, key) -> "Door | None":
        return self.doors.get(key)

    # ---- the state machine -------------------------------------------

    def begin(self, key, target: bool):
        """Start a door moving.

        Returns `(door, changed)`: the Door if it set off and None if there was
        nothing to do (already there, already moving, no such door), and
        whether the DOORWAY changed this instant. Opening changes nothing yet —
        the wall stays a wall until the panels arrive. Closing changes it
        straight away, because the gap starts shrinking the moment the panels
        move. Callers that keep map geometry in step apply `door.is_open` when
        `changed`, and again for every arrival out of `step`.

        A door in motion is deliberately not interruptible. Cancelling a blast
        door halfway would hand back the five seconds that are the whole reason
        to place one."""
        d = self.doors.get(key)
        if d is None or d.moving or d.is_open == target:
            return None, False
        if d.dur <= 0.0:
            d.is_open = d.target = target
            d.left = 0.0
            return d, True
        d.target = target
        d.left = d.dur
        if not target:
            d.is_open = False        # sealing: the doorway shuts on departure
            return d, True
        return d, False

    def resume(self, key, target: bool, left: float):
        """Put a door into a move that is already under way — a latecomer
        catching up on a blast door somebody started four seconds ago. Same
        `(door, changed)` contract as `begin`."""
        d = self.doors.get(key)
        if d is None:
            return None, False
        was = d.is_open
        if left <= 0.0:
            d.is_open = d.target = target
            d.left = 0.0
            return d, d.is_open != was
        d.target = target
        # the sender is authoritative about how long this one takes: if their
        # tile table says shorter than ours, believe them rather than holding
        # the door shut after everyone else has walked through
        d.dur = max(d.dur, left)
        d.left = left
        if not target:
            d.is_open = False
        return d, d.is_open != was

    def step(self, dt: float) -> list:
        """Advance every door in motion.

        Returns `[(key, is_open, changed)]` for the ones that ARRIVED this
        step. `changed` says whether the DOORWAY changed at this end of the
        travel, and it is asymmetric on purpose: an opening door changes it
        here, a closing door changed it when it set off. Either way the
        arrival is worth knowing about — it is when the panels land, and when
        they can be heard landing."""
        done = []
        for key, d in self.doors.items():
            if not d.moving:
                continue
            d.left -= dt
            if d.left <= ARRIVE_EPS:
                d.left = 0.0
                was = d.is_open
                d.is_open = d.target
                done.append((key, d.is_open, d.is_open != was))
        return done

    def moving_keys(self) -> list:
        return [k for k, d in self.doors.items() if d.moving]

    # ---- over the wire -----------------------------------------------

    def wire(self) -> list:
        """Everything about the doors that has drifted from the map file, as
        [r, c, open, left] rows: what a latecomer needs to arrive with the same
        walls as everyone else, including the one that is still travelling."""
        return [[d.r, d.c, bool(d.target if d.moving else d.is_open),
                 round(d.left, 3)]
                for d in self.doors.values()
                if d.is_open or d.moving]

    def apply_wire(self, rows, on_change=None) -> None:
        """Take that list back apart. `on_change(key, is_open)` is called for
        each door that is open right now, so the caller can update the map."""
        for row in rows or ():
            r, c = int(row[0]), int(row[1])
            is_open = bool(row[2])
            left = float(row[3]) if len(row) > 3 else 0.0
            d, _changed = self.resume((r, c), is_open, left)
            if d is not None and d.is_open and on_change is not None:
                on_change((r, c), True)
