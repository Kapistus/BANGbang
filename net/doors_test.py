"""Headless test of sliding doors. No pygame, no rendering.

Doors used to be a boolean: press f, the tile flips, done. They are now two
panels that take time to travel, and the whole design rests on one rule —
a door in motion is still a wall. A blast door's five seconds only cost
something if you cannot walk through, see through or shoot through it while
they run.

What this checks:

  * which way the panels retract, read off the wall the door sits in
  * a fast door and a blast door get the travel times their tiles declare
  * five seconds means five seconds, and 4.99 is not five
  * a door already moving ignores the key — no cancelling a blast door
  * a door mid-travel survives the trip to a latecomer, with its remaining
    time intact
  * over a real connection: the travel time reaches clients, the server's
    geometry does not change until the panels arrive, and a client's own
    copy of the door lands at the same moment
  * both map formats: the char grid (maps/tiles.toml) and the editor's .map
    (assets/tiles/tileset.toml) have to agree about what a door is, or a door
    placed in the editor comes out as a different door in the game

Run from the project root:   python -m net.doors_test
"""
import tempfile
import time
from pathlib import Path

from sim import doors as D
from sim.tilemap import load_map

# '+' is the powered door, 'B' the blast door. The '+' sits in a vertical wall
# run so its panels go up and down; the 'B' sits in a horizontal one so its
# panels go left and right. Anything else would not be testing the rule.
GRID = """\
#########
#...#...#
#...+...#
#...#...#
#########
#.......#
###B#####
#.......#
#########
"""
FAST = (2, 4)
BLAST = (6, 3)


def _map():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "test.grid").write_text(GRID)
    (tmp / "test.toml").write_text('name = "door test"\ngrid = "test.grid"\n')
    return load_map(tmp / "test.toml", Path("maps/tiles.toml"))


def state_machine_test():
    m = _map()
    ds = D.DoorSet(m)
    assert len(ds) == 2, f"expected two doors, found {len(ds)}"

    fast, blast = ds.door(FAST), ds.door(BLAST)
    print(f"  powered door at {FAST}: panels travel "
          f"{'up/down' if fast.axis == 'v' else 'left/right'}, "
          f"{fast.dur:.2f}s")
    print(f"  blast door at {BLAST}: panels travel "
          f"{'up/down' if blast.axis == 'v' else 'left/right'}, "
          f"{blast.dur:.2f}s")
    assert fast.axis == "v", "a door in a vertical wall run slides the wrong way"
    assert blast.axis == "h", "a door in a horizontal wall run slides the wrong way"
    assert abs(fast.dur - 0.35) < 1e-6, f"powered door takes {fast.dur}s"
    assert abs(blast.dur - 5.0) < 1e-6, f"blast door takes {blast.dur}s"
    assert blast.heavy and not fast.heavy

    # --- five seconds is five seconds
    assert ds.begin(BLAST, True)[0] is not None, "the blast door would not start"
    assert blast.moving and not blast.is_open
    assert not ds.step(4.99), "a blast door opened in under five seconds"
    assert not blast.is_open, "the doorway opened before the panels arrived"
    assert blast.frac > 0.99, "the panels are not nearly home at 4.99s"
    done = ds.step(0.02)
    print(f"  blast door: shut through 4.99s, open at 5.01s (arrivals {done})")
    assert done == [(BLAST, True, True)], f"the door did not arrive: {done}"
    assert blast.is_open and not blast.moving

    # --- the drawn opening tracks the travel, and nothing else does
    _d, _chg = ds.begin(BLAST, False)
    assert _chg, "sealing did not shut the doorway on departure"
    ds.step(2.5)
    print(f"  half-sealed: drawn {blast.frac:.2f} open, "
          f"but is_open={blast.is_open}")
    assert 0.4 < blast.frac < 0.6, "the drawn panels are not where they should be"
    assert not blast.is_open, "a door is passable while it is closing"

    # --- not interruptible
    assert ds.begin(BLAST, True)[0] is None, \
        "a moving blast door took a new order — the five seconds are free"
    ds.step(3.0)
    assert not blast.is_open and not blast.moving, "it did not finish sealing"

    # --- a door already moving survives the trip to a latecomer
    ds.begin(BLAST, True)
    ds.step(1.5)
    wire = ds.wire()
    print(f"  mid-travel on the wire: {wire}")
    late = D.DoorSet(m)
    late.apply_wire(wire)
    lb = late.door(BLAST)
    assert lb.moving and lb.target is True
    assert abs(lb.left - blast.left) < 1e-3, \
        f"the latecomer's door has {lb.left:.2f}s left, not {blast.left:.2f}s"
    late.step(lb.left + 0.01)
    assert lb.is_open, "the latecomer's door never arrived"

    # a door that finished before they joined arrives already open
    applied = []
    late2 = D.DoorSet(m)
    late2.apply_wire([[BLAST[0], BLAST[1], True, 0.0]],
                     on_change=lambda k, o: applied.append((k, o)))
    assert late2.door(BLAST).is_open and applied == [(BLAST, True)]

    # --- a doorway with nothing either side still picks an axis
    assert D.door_axis(m, 5, 4) in ("h", "v")
    print("\nSTATE MACHINE CHECKS PASSED")


def editor_format_test():
    """A door placed in the editor has to be the same door the game runs.

    There are two tile tables — the char grid reads `maps/tiles.toml`, the
    editor's `.map` format reads `assets/tiles/tileset.toml` — and a door's
    travel time has to survive both. It was possible to add a tile to one and
    have the editor keep offering the old single door, which is exactly the
    sort of drift `sim/doors.py` exists to stop."""
    import json
    from sim import mapfile
    from sim.tileset import load_tileset

    ts = load_tileset()
    doors = {tid: td for tid, td in ts.tiles.items() if td.door}
    print(f"  editor palette offers {len(doors)}: "
          + ", ".join(t.name for t in doors.values()))
    assert set(doors) == {"door", "blast_door"}, \
        f"the editor's door tiles are {sorted(doors)}"
    assert doors["door"].door_time == 0.35
    assert doors["blast_door"].door_time == 5.0
    # the id has to stay `door`: five maps already have that tile placed
    assert "door" in ts, "renaming the door id would orphan every placed door"

    # --- place one of each in a .map and check what comes out the far end
    rows, cols = 7, 7
    floor = [["blank"] * cols for _ in range(rows)]
    obj = [[None] * cols for _ in range(rows)]
    for c in range(cols):
        obj[0][c] = obj[rows - 1][c] = "metal_00_00"
    for r in range(rows):
        obj[r][0] = obj[r][cols - 1] = "metal_00_00"
    # a powered door in a horizontal wall run: panels go left and right
    obj[2][1] = obj[2][3] = "metal_00_00"
    obj[2][2] = "door"
    # a blast door in a vertical one: panels go up and down
    obj[3][4] = obj[5][4] = "metal_00_00"
    obj[4][4] = "blast_door"
    doc = {"format": "bangbang-map/1", "name": "editor doors",
           "size": [cols, rows], "cell_m": 1.0, "subdiv": 8,
           "floor": floor, "object": obj, "player_spawn": [1.5, 1.5],
           "guards": [], "idle_spots": [], "lights": []}
    tmp = Path(tempfile.mkdtemp()) / "editor_doors.map"
    tmp.write_text(json.dumps(doc))
    m = mapfile.load_map(tmp)
    ds = D.DoorSet(m)
    print(f"  a .map with one of each -> {len(ds)} doors, "
          + ", ".join(f"{d.dur:g}s {d.axis}" for d in ds.doors.values()))
    assert len(ds) == 2, f"placed two doors, loaded {len(ds)}"
    times = sorted(round(d.dur, 2) for d in ds.doors.values())
    assert times == [0.35, 5.0], f"travel times came out as {times}"
    quick, slow = ds.door((2, 2)), ds.door((4, 4))
    assert quick is not None and not quick.heavy, "the powered door got heavy"
    assert quick.axis == "h", "the powered door reads the wrong wall run"
    assert slow is not None and slow.heavy, "the blast door lost its five seconds"
    assert slow.axis == "v", "the blast door reads the wrong wall run"
    ds.begin((4, 4), True)
    assert not ds.step(4.9), "an editor-placed blast door opened early"
    assert ds.step(0.2), "an editor-placed blast door never opened"
    print("\nEDITOR FORMAT CHECKS PASSED")


def wire_test():
    """The same door, over a real connection: does the client end up with the
    server's door, at the server's moment?"""
    from net import GameServer, GameClient
    from net import maps as netmaps
    from net.protocol import BTN_INTERACT, ServerState
    from net.combat_test import clear_pair

    port = 47994
    m, _ = netmaps.load_with_spawns("compound")
    stand, away = clear_pair(m, 6.0)
    srv = GameServer([stand, away], port=port, map_id="compound",
                     duration_s=300)
    srv.start()
    a = GameClient("127.0.0.1", port)
    b = GameClient("127.0.0.1", port)
    try:
        assert a.connect("Opener", (220, 40, 40)), a.reject_reason
        assert b.connect("Watcher", (40, 40, 220)), b.reject_reason
        a.set_ready(True)
        b.set_ready(True)
        t0 = time.monotonic()
        while a.world.state != ServerState.MATCH and time.monotonic() - t0 < 3:
            time.sleep(0.02)
        assert a.world.state == ServerState.MATCH, "no match"

        key = min(srv.doors, key=lambda k: (k[0] - stand[1]) ** 2
                  + (k[1] - stand[0]) ** 2)
        door = srv.doors.door(key)
        sub = srv.map.subdiv
        fine = (key[0] * sub, key[1] * sub)
        srv.players[b.world.my_id].x = away[0]
        srv.players[b.world.my_id].y = away[1]
        sp = srv.players[a.world.my_id]
        sp.x, sp.y = key[1] + 0.5 + 1.2, key[0] + 0.5
        time.sleep(0.2)
        a.drain_events()

        a.send_input(0.0, 0.0, 0.0, BTN_INTERACT)
        t0 = time.monotonic()
        while not door.moving and time.monotonic() - t0 < 1.0:
            time.sleep(0.005)
        a.send_input(0.0, 0.0, 0.0, 0)
        assert door.moving, "the door never started"
        assert srv.map.blocks_move[fine], \
            "the server opened the doorway the instant the key went down"

        # the client builds its own copy from the message and runs the same
        # clock, which is what keeps prediction honest across the travel
        ev = None
        while ev is None and time.monotonic() - t0 < 2.0:
            for e in a.drain_events():
                if e["t"] == "door":
                    ev = e
            time.sleep(0.01)
        assert ev is not None, "no door event reached the client"
        print(f"  the client was told: open={ev['open']}, dur={ev['dur']}s")
        assert abs(ev["dur"] - door.dur) < 1e-3

        mirror = D.DoorSet(srv.map)
        mirror.resume(key, ev["open"], ev["dur"])
        while door.moving and time.monotonic() - t0 < 3.0:
            time.sleep(0.01)
        assert not srv.map.blocks_move[fine], "the doorway never opened"
        mirror.step(ev["dur"] + 0.01)
        assert mirror.door(key).is_open, \
            "a client running the same clock did not land on the same state"
        print("  server and a client clock landed on the same state")
        print("\nWIRE CHECKS PASSED")
    finally:
        a.disconnect()
        b.disconnect()
        srv.stop()


if __name__ == "__main__":
    state_machine_test()
    print()
    editor_format_test()
    print()
    wire_test()
