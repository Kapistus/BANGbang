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

Run from the project root:   python -m tests.doors_test
"""
import math
import tempfile
import time
from pathlib import Path

from sim import doors as D
from tests import range_spots as R

# The range's first powered door sits in a vertical wall run, so its panels go
# up and down; its blast door sits in a horizontal one, so its panels go left
# and right. Anything else would not be testing the rule.
FAST = R.FIRST_DOOR
BLAST = R.BLAST_DOOR


def _map():
    from net import maps as netmaps
    m, _ = netmaps.load_with_spawns(R.MAP_ID, R.MAPS_DIR)
    R.check(m)
    return m


def state_machine_test():
    m = _map()
    ds = D.DoorSet(m)
    assert ds.door(FAST) and ds.door(BLAST), "the range lost a door"

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
    assert D.door_axis(m, 10, 20) in ("h", "v")
    print("\nSTATE MACHINE CHECKS PASSED")


def sight_test():
    """While the panels move you can see through the gap between them, both
    ways; on a blast door you can shoot through it too; nobody walks through
    until it is fully open. The blast door is committed; the quick door turns
    round."""
    from sim.tilemap import set_door, set_door_gap
    from sim.vision import line_of_sight

    m = _map()
    ds = D.DoorSet(m)
    sub = m.subdiv
    blast = ds.door(BLAST)                    # horizontal wall: panels go l/r
    r, c = BLAST
    y0, x0 = r * sub, c * sub
    mid_x = x0 + sub // 2                      # the fine column the slit opens on
    above = (y0 - 2, mid_x)                    # just north of the door
    below = (y0 + sub + 1, mid_x)              # just south of it

    def apply():
        for key, gap in ds.gap_changes(sub):
            dd = ds.door(key)
            set_door_gap(m, key[0], key[1], gap, dd.axis,
                         bullets=dd.shoot_through)

    def sees():
        return line_of_sight(m.blocks_sight, above[1], above[0],
                             below[1], below[0])

    assert not sees(), "a shut blast door lets sight through"
    ds.begin(BLAST, True)
    gaps = []
    for _ in range(49):                        # 4.9 s of the 5 s travel
        for key, is_open, changed in ds.step(0.1):
            if changed:
                set_door(m, None, key[0], key[1], is_open)
        apply()
        gaps.append(blast._gap)
    print(f"  blast door slit, fine cells wide, each 0.5 s: {gaps[4::5]}")
    assert gaps[0] == 0, "the slit opened before the panels had moved"
    assert gaps == sorted(gaps), "the slit narrowed while the door was opening"
    assert max(gaps) > 0, "the slit never opened"
    assert sees(), "sight does not pass the gap of a nearly open blast door"
    assert m.blocks_move[y0 + sub // 2, mid_x], \
        "a blast door still opening can already be walked through"
    assert not m.blocks_bullets[y0 + sub // 2, mid_x], \
        "a blast door's gap stops bullets"
    assert m.blocks_bullets[y0 + sub // 2, x0], \
        "bullets pass the panels, not just the gap"
    assert ds.begin(BLAST, False)[0] is None, \
        "a blast door took a new order mid-travel"
    print("  4.9 s in: you can see and shoot through the gap; you still cannot "
          "walk through or turn it round")

    # --- closing: the gap narrows, stays see-through and shootable, and
    # only seals when the panels meet
    for key, is_open, changed in ds.step(0.2):
        if changed:
            set_door(m, None, key[0], key[1], is_open)
    apply()
    assert blast.is_open and sees()
    _d, changed = ds.begin(BLAST, False)
    if changed:
        set_door(m, None, BLAST[0], BLAST[1], False)
    apply()
    closing = []
    for _ in range(52):
        ds.step(0.1)
        apply()
        closing.append((blast._gap, sees(), bool(m.blocks_move[y0 + sub // 2, mid_x])))
    widths = [g for g, _s, _m in closing]
    print(f"  closing slit, each 0.5 s: {widths[4::5]}")
    assert widths == sorted(widths, reverse=True), "the slit widened while closing"
    assert closing[0][1], "a door that has only just started closing is sealed"
    assert all(mv for _g, _s, mv in closing), "a closing door could be walked through"
    assert not blast.moving and widths[-1] == 0 and not sees(), \
        "the door finished closing but its last sliver stayed open"
    print("  see-through while it closes, sealed once the panels meet, and "
          "never walkable")

    # --- a quick door's gap is for looking, not shooting
    fast = ds.door(FAST)
    fr, fc = FAST
    ds.begin(FAST, True)
    ds.step(0.30)
    apply()
    f_mid = (fr * sub + sub // 2, fc * sub + sub // 2)
    assert not m.blocks_sight[f_mid], "a quick door's gap hides nothing"
    assert m.blocks_bullets[f_mid], "bullets pass a quick door mid-travel"
    # ...and back to shut through the real API: let it land open, close it,
    # let that land, so the reversal check below starts from rest
    for key, is_open, changed in ds.step(0.10):
        if changed:
            set_door(m, None, key[0], key[1], is_open)
    apply()
    assert fast.is_open and not fast.moving
    _d, changed = ds.begin(FAST, False)
    if changed:
        set_door(m, None, fr, fc, False)
    ds.step(fast.dur + 0.01)
    apply()
    assert not fast.is_open and not fast.moving and fast._gap == 0
    print("  a quick door mid-travel: see-through, not shoot-through")

    # --- the quick door turns round, and the panels do not jump
    ds.begin(FAST, True)
    ds.step(0.20)
    frac_before = fast.frac
    d, _changed = ds.begin(FAST, False)
    assert d is not None, "a quick door would not turn round"
    assert abs(fast.frac - frac_before) < 1e-6, \
        f"reversing jumped the panels from {frac_before:.2f} to {fast.frac:.2f}"
    assert not fast.target and abs(fast.left - 0.20) < 1e-6, \
        f"it should take the 0.20 s it had covered to get back, not {fast.left:.2f}"
    assert not ds.step(0.19), "it arrived shut too early"
    assert ds.step(0.02) == [(FAST, False, False)], "it never got back"
    print(f"  quick door turned round at {frac_before:.0%} open, back shut "
          f"after the 0.20 s it had covered")
    print("\nSIGHT CHECKS PASSED")


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
    from tests.combat_test import clear_pair

    port = 47994
    m, _ = netmaps.load_with_spawns(R.MAP_ID, R.MAPS_DIR)
    stand, away = clear_pair(m, 6.0)
    srv = GameServer([stand, away], port=port, map_id=R.MAP_ID,
                     maps_dir=R.MAPS_DIR, duration_s=300)
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

        # --- turned round mid-travel, the client is told what is LEFT
        time.sleep(0.1)
        a.drain_events()
        a.send_input(0.0, 0.0, 0.0, 0)
        time.sleep(0.05)
        a.send_input(0.0, 0.0, 0.0, BTN_INTERACT)       # start it closing
        t0 = time.monotonic()
        while not door.moving and time.monotonic() - t0 < 1.0:
            time.sleep(0.005)
        a.send_input(0.0, 0.0, 0.0, 0)
        time.sleep(0.10)                                 # partway through
        a.send_input(0.0, 0.0, 0.0, BTN_INTERACT)       # ...and change our mind
        t0 = time.monotonic()
        while door.target is False and time.monotonic() - t0 < 1.0:
            time.sleep(0.005)
        a.send_input(0.0, 0.0, 0.0, 0)
        events = []
        t0 = time.monotonic()
        while len(events) < 2 and time.monotonic() - t0 < 2.0:
            events += [e for e in a.drain_events() if e["t"] == "door"]
            time.sleep(0.01)
        assert len(events) >= 2, f"expected a close and a reversal, got {events}"
        back = events[1]
        print(f"  closed then turned round: told open={back['open']}, "
              f"{back['dur']}s left of {door.dur}s")
        assert back["open"] is True
        assert 0.0 < back["dur"] < door.dur, \
            "a reversed door was announced as a full-length trip"
        print("\nWIRE CHECKS PASSED")
    finally:
        a.disconnect()
        b.disconnect()
        srv.stop()


def blast_fire_test():
    """On the server, where it counts: a shot through the gap of a moving blast
    door lands, the same shot at the shut door does not, and nobody walks
    through either way.

    Uses the range's blast door at (14, 19), which sits in a horizontal
    wall, so its slit is a vertical line down x = 19.5. The pistol
    is deliberate: its penetration is below the blast door's, so the control
    shots genuinely stop at a shut door. A rail rifle punches through one
    regardless and would prove nothing."""
    from net import GameServer, GameClient
    from net.protocol import BTN_FIRE, BTN_INTERACT, ServerState

    port = 47992
    shooter_at, target_at = R.BLAST_SHOOTER, R.BLAST_TARGET
    srv = GameServer([shooter_at, target_at], port=port, map_id=R.MAP_ID,
                     maps_dir=R.MAPS_DIR, duration_s=300)
    srv.start()
    a = GameClient("127.0.0.1", port)
    b = GameClient("127.0.0.1", port)
    try:
        assert a.connect("Shooter", (220, 40, 40)), a.reject_reason
        assert b.connect("Target", (40, 40, 220)), b.reject_reason
        a.set_ready(True)
        b.set_ready(True)
        t0 = time.monotonic()
        while a.world.state != ServerState.MATCH and time.monotonic() - t0 < 4:
            time.sleep(0.05)
        assert a.world.state == ServerState.MATCH, "no match"
        sp, tp = srv.players[a.world.my_id], srv.players[b.world.my_id]
        door = srv.doors.door(R.BLAST_DOOR)
        assert door is not None and door.heavy, "the blast door is not where expected"
        down = math.pi / 2

        def place():
            sp.x, sp.y = shooter_at
            tp.x, tp.y = target_at

        def total():
            return tp.body.health + tp.body.shields

        def volley(seconds):
            """Pull the pistol's trigger once per pull cycle, straight at the
            target, for `seconds`. Returns the damage the target took."""
            before = total()
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                place()
                a.send_input(0.0, 0.0, down, BTN_FIRE, wep=0, aim_dist=4.2)
                time.sleep(0.05)
                place()
                a.send_input(0.0, 0.0, down, 0, wep=0, aim_dist=4.2)
                time.sleep(0.6)
            return before - total()

        place()
        time.sleep(0.3)
        shut = volley(2.0)
        print(f"  pistol at the shut blast door for 2 s: {shut:.0f} damage")
        assert shut == 0, "bullets went through a shut blast door"

        tp.body.health, tp.body.shields = tp.body.max_health, tp.body.max_shields
        a.send_input(0.0, 0.0, down, BTN_INTERACT)
        t0 = time.monotonic()
        while not door.moving and time.monotonic() - t0 < 1.0:
            time.sleep(0.005)
        a.send_input(0.0, 0.0, down, 0)
        assert door.moving and door.target, "the blast door did not start opening"
        time.sleep(1.5)                               # let the slit open up
        through = volley(2.5)                         # ...and fire while it moves
        still_moving = door.moving
        br, bc = R.BLAST_DOOR
        mid = (br * srv.map.subdiv + srv.map.subdiv // 2,
               bc * srv.map.subdiv + srv.map.subdiv // 2)
        walkable = not srv.map.blocks_move[mid]
        print(f"  pistol through the opening gap for 2.5 s: {through:.0f} "
              f"damage; door still moving: {still_moving}; walkable: {walkable}")
        assert still_moving, "the door finished before the volley did"
        assert through > 0, "no shot got through the gap of a moving blast door"
        assert not walkable, "a moving blast door can be walked through"
        print("\nBLAST FIRE CHECKS PASSED")
    finally:
        a.disconnect()
        b.disconnect()
        srv.stop()


if __name__ == "__main__":
    state_machine_test()
    print()
    editor_format_test()
    print()
    sight_test()
    print()
    wire_test()
    print()
    blast_fire_test()
