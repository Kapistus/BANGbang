"""
sim/perception.py — what a listener actually perceives of a sound.

This is the misdirection model, and it is the reason the game works: a sound is
perceived as arriving from the route it travelled, not from where it was made.
Round a corner, a shot comes at you from the corner. The bearing is the negative
gradient of arrival time across the propagation field, so the geometry does the
lying for you.

Split out of main()'s loop so single-player and multiplayer share one
definition. Given a solved field and a listener's cell it answers three
questions in order: has it got here yet, was there anything left when it did,
and which way did it seem to come from.

No pygame, no rendering — safe to import anywhere.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass

# Below this much energy on arrival there is no usable direction left: the
# listener knows something happened but not where, and the cue opens out into a
# full circle instead of an arc.
CUE_MIN_ENERGY = 0.10

# Perceived loudness per unit of remaining energy. A sound that arrives with
# 77% of its energy intact is already at full volume.
GAIN_PER_ENERGY = 1.3


# --------------------------------------------------------------- loudness
# Sound energies are the METRES a sound carries in open air; callers convert to
# cell units with the map's cells_per_metre. They live here, beside the model
# that decides what a listener makes of them, so the server emitting a footstep
# and the client hearing it are working from one set of numbers.

# Metres travelled per footstep: the cadence of both the propagation emission
# and the audible step rhythm. Walking patters, running lands heavier and
# further apart, crawling is slow and makes no audible step at all.
STRIDE = {"crawl": 1.50, "walk": 0.70, "run": 1.10}

FOOTSTEP_REACH_M = {"crawl": 2.0, "walk": 5.5, "run": 11.0}
KNOCK_REACH_M = 22.0
DRYFIRE_REACH_M = 3.5
MAGDROP_REACH_M = 7.5
GLASS_BREAK_REACH_M = 15.0


class Arrival(enum.Enum):
    PENDING = "pending"        # still travelling — ask again next frame
    INAUDIBLE = "inaudible"    # it will never be heard from here
    HEARD = "heard"


@dataclass
class Heard:
    """One sound, as perceived at one listening position."""
    gain: float                # 0..1, for the mixer
    pan: float                 # -1 left .. +1 right
    angle: float               # radians: which way it SEEMS to come from
    remaining: float           # energy left on arrival, 0..1
    directional: bool          # False = loud enough to notice, too faint to place

    @property
    def cue_half_deg(self) -> float:
        """Half-angle of the on-screen arc. A strong arrival gives a tight
        bearing, a faint one a vague smear, and no direction at all gives the
        full circle."""
        if not self.directional:
            return 180.0
        return 14.0 + (75.0 - 14.0) * (1.0 - min(1.0, self.remaining / 0.55))


def perceive(field, cell: tuple[int, int], energy: float,
             elapsed_cells: float,
             min_dir_energy: float = CUE_MIN_ENERGY
             ) -> tuple[Arrival, Heard | None]:
    """What a listener at `cell` makes of a sound, `elapsed_cells` of travel
    after it was made.

    PENDING means the wavefront has not reached them yet — keep asking until the
    sound expires. INAUDIBLE means it reached them with nothing left, or cannot
    reach them at all. Sound is not instant here: a distant shot is heard after
    it is fired, and through the walls it actually went around.
    """
    cx, cy = cell
    travel = field.arrival_time(cx, cy)
    if not math.isfinite(travel):
        # never reaches this cell — but the field may still be solving, so this
        # stays PENDING and the caller drops it when the sound expires
        return Arrival.PENDING, None
    if field.arrival(cx, cy) > energy:
        return Arrival.INAUDIBLE, None
    if elapsed_cells < travel:
        return Arrival.PENDING, None

    remaining = field.remaining(cx, cy)
    b = field.bearing(cx, cy)
    return Arrival.HEARD, Heard(
        gain=min(1.0, remaining * GAIN_PER_ENERGY),
        pan=float(b[0]) if b else 0.0,
        angle=math.atan2(b[1], b[0]) if b else 0.0,
        remaining=remaining,
        directional=b is not None and remaining >= min_dir_energy,
    )
