"""Where things are on tests/range.map, the one map the tests share.

The range is laid out so each test has its own corner of it:

    top strip, rows 1-5   the door bay: a powered door at (3, 6) in a
                          vertical wall - the first door on the map - and a
                          window at (3, 20). Open floor either side for the
                          firing lines the network tests look for
    row 6                 a powered door at (6, 3) in a horizontal wall
    hall, rows 7-13       open floor, two of the four spawns
    row 14                the blast door at (14, 19) in a horizontal wall
    cover lanes, 17-23    one lane per cover type, the cover in column 8:
                          window (17), door (19), blast door (21), wall (23)
    lower right           a health pack and an ammo pack

Positions are world metres (x, y); doors and tiles are (row, col). A test that
depends on the layout calls check() first, so an edit to the map fails with
what moved rather than with a confusing result somewhere downstream.

    py main.py tests/range.map          play it
    py editor.py tests/range.map        edit it
"""
MAPS_DIR = "tests"
MAP_ID = "range"
PATH = "tests/range.map"

FIRST_DOOR = (3, 6)          # powered, vertical wall; stand east of it at:
FIRST_DOOR_STAND = (7.7, 3.5)
SP_DOOR = (6, 3)             # powered, horizontal wall
SP_DOOR_STAND = (3.5, 5.4)   # just above it, within reach
WINDOW = (3, 20)

BLAST_DOOR = (14, 19)        # horizontal wall: the slit opens down x = 19.5
BLAST_SHOOTER = (19.5, 13.3)  # above it, within reach of it
BLAST_TARGET = (19.5, 17.5)   # below it, straight down the slit

COVER_COL = 8                # cover lanes: row of each, keyed by what's in it
LANES = {"window": 17, "door": 19, "blast door": 21, "wall": 23}

HEALTH = ("med1", (30.5, 19.5))
AMMO = ("ammo1", (34.5, 19.5))
AWAY = (25.5, 22.5)          # somewhere open, clear of every test above

# two ends of the hall, 35 m apart: further than a shot's sound can cross
# before a full-auto burst would have filled the active-sound list
FAR_PAIR = ((2.5, 10.5), (37.5, 10.5))

SPAWNS = [(10.5, 10.5), (29.5, 10.5), (33.5, 3.5), (27.5, 21.5)]


def check(m):
    """Fail loudly if the range no longer has what the tests rely on."""
    from sim import doors as D
    ds = D.DoorSet(m)
    want = {FIRST_DOOR: (False, "v"), SP_DOOR: (False, "h"),
            BLAST_DOOR: (True, "h"),
            (LANES["door"], COVER_COL): (False, "v"),
            (LANES["blast door"], COVER_COL): (True, "v")}
    for key, (heavy, axis) in want.items():
        d = ds.door(key)
        assert d is not None, f"range.map: no door at {key} any more"
        assert d.heavy == heavy and d.axis == axis, \
            f"range.map: the door at {key} changed (heavy={d.heavy}, axis={d.axis})"
    assert next(iter(ds.doors)) == FIRST_DOOR, \
        "range.map: a door now comes before (3, 6) - tests use the first door"

    def tile(rc):
        return m.tiles[m.chars[rc]]
    assert tile(WINDOW).glass, f"range.map: no window at {WINDOW}"
    lane_kind = {"window": lambda t: t.glass,
                 "door": lambda t: t.door and t.door_time < 1.0,
                 "blast door": lambda t: t.door and t.door_time >= 1.0,
                 "wall": lambda t: t.blocks_bullets and not t.door and not t.glass}
    for kind, r in LANES.items():
        assert lane_kind[kind](tile((r, COVER_COL))), \
            f"range.map: lane {r} no longer has a {kind} in column {COVER_COL}"
        for c in list(range(1, COVER_COL)) + list(range(COVER_COL + 1, 17)):
            assert not tile((r, c)).blocks_bullets, \
                f"range.map: lane {r} is blocked at column {c}"
    spots = [*FAR_PAIR, FIRST_DOOR_STAND, SP_DOOR_STAND, BLAST_SHOOTER, BLAST_TARGET,
             HEALTH[1], AMMO[1], AWAY] + SPAWNS
    for x, y in spots:
        assert m.can_stand(x, y, 0.28), f"range.map: ({x}, {y}) is not open floor"
    ids = {p.id: (p.kind, tuple(p.pos)) for p in m.interactables
           if getattr(p, "kind", "") in ("health", "ammo")}
    assert ids == {HEALTH[0]: ("health", HEALTH[1]), AMMO[0]: ("ammo", AMMO[1])}, \
        f"range.map: the packs changed: {ids}"
