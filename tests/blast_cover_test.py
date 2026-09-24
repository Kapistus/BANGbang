"""Headless test of what explosions and a charged rail shot get through.

  * a blast is stopped by cover the way a bullet is, against BLAST_PEN:
    glass lets it through; doors, blast doors and walls do not
  * a wall-stopped explosive detonates on the near face, so the side it hit
    still takes the blast and the far side does not
  * the rail pistol punches a shut blast door at full charge, not below it,
    and never a wall - in the sim and on a live server

All of it on the cover lanes of tests/range.map, bar two tiles the editor has
no way to place - the thin wall and low cover of the char-grid maps - which
get a small grid of their own.

Run from the project root:   python -m tests.blast_cover_test
"""
import dataclasses
import math
import random
import tempfile
import time
from pathlib import Path

from net import maps as netmaps
from sim import ballistics, combat, weapons
from sim.tilemap import load_map
from tests import range_spots as R

# where each lane's action happens, relative to its cover (column 8, x 8..9)
NEAR_X, BLAST_X, FAR_X = 6.5, 7.2, 9.5
MUZZLE_X, TARGET_X = 2.5, 10.5
PASSES = {"window", "thin wall"}

# the char-grid-only cover: thin wall (') and low cover (n)
GRID = """\
#########
#...#...#
#...X...#
#...#...#
#########
"""


def _range():
    m, _ = netmaps.load_with_spawns(R.MAP_ID, R.MAPS_DIR)
    R.check(m)
    return m


def _grid(ch):
    tmp = Path(tempfile.mkdtemp())
    (tmp / "t.grid").write_text(GRID.replace("X", ch))
    (tmp / "t.toml").write_text('name = "blast"\ngrid = "t.grid"\n')
    return load_map(tmp / "t.toml", Path("maps/tiles.toml"))


def _lanes():
    """(name, map, y, cover's left edge x) for every kind of cover."""
    m = _range()
    out = [(k, m, r + 0.5, float(R.COVER_COL)) for k, r in R.LANES.items()]
    for ch, name in (("'", "thin wall"), ("n", "low cover")):
        out.append((name, _grid(ch), 2.5, 4.0))
    return out


def _body(x, y):
    b = combat.player_commando(x, y)
    b.x, b.y = x, y
    b.health = b.max_health = 10000.0
    b.shields = 0.0
    return b


def cover_test():
    w = weapons.ROSTER["rocket_launcher"]
    for name, m, y, cx in _lanes():
        dx = cx - R.COVER_COL
        near, far = _body(NEAR_X + dx, y), _body(FAR_X + dx, y)
        hits = ballistics.blast((BLAST_X + dx, y), w.blast_r, w, [near, far],
                                random.Random(1), 0.0, m=m)
        got = {id(h.target): h.damage for h in hits}
        print(f"  {name:10s}: near {got.get(id(near), 0):5.1f}  "
              f"far {got.get(id(far), 0):5.1f}")
        assert id(near) in got, f"the blast missed the body on its own side ({name})"
        if name in PASSES:
            assert id(far) in got, f"a {name} stopped the blast"
        else:
            assert id(far) not in got, f"the blast went through a {name}"
    print("\nBLAST COVER CHECKS PASSED")


def near_face_test():
    """A rocket into a wall goes off on the shooter's side of it."""
    w = weapons.ROSTER["rocket_launcher"]
    m = _range()
    face = float(R.COVER_COL)
    for kind in ("wall", "blast door"):
        y = R.LANES[kind] + 0.5
        sh = ballistics.fire_shot(m, m.blocks_bullets, m.pen_cost, m.glass,
                                  (MUZZLE_X, y), 0.0, w, [], random.Random(1),
                                  0.0, apply_damage=False)
        assert sh.impact[0] >= face and sh.blast_at[0] < face, \
            (sh.impact, sh.blast_at)
        near, far = _body(face - 0.5, y), _body(face + 1.6, y)
        hits = {id(h.target) for h in ballistics.blast(
            sh.blast_at, w.blast_r, w, [near, far], random.Random(1), 0.0, m=m)}
        print(f"  rocket into the {kind}: impact x={sh.impact[0]:.3f}, "
              f"detonates at x={sh.blast_at[0]:.3f}; near hit "
              f"{id(near) in hits}, far hit {id(far) in hits}")
        assert id(near) in hits and id(far) not in hits
    print("\nNEAR FACE CHECKS PASSED")


def rail_sim_test():
    w = weapons.ROSTER["rail_pistol"]
    full = dataclasses.replace(w, pen=w.pen_charged)
    m = _range()
    for kind, want_plain, want_full in (("blast door", False, True),
                                        ("wall", False, False),
                                        ("door", True, True)):
        y = R.LANES[kind] + 0.5
        res = []
        for wp in (w, full):
            tgt = _body(TARGET_X, y)
            sh = ballistics.fire_shot(m, m.blocks_bullets, m.pen_cost, m.glass,
                                      (MUZZLE_X, y), 0.0, wp, [tgt],
                                      random.Random(1), 0.0)
            res.append(bool(sh.hits))
        print(f"  rail pistol at the {kind}: uncharged through {res[0]}, "
              f"full charge through {res[1]}")
        assert res == [want_plain, want_full], kind
    print("\nRAIL SIM CHECKS PASSED")


def rail_server_test():
    """Hold the trigger on a live server: half a charge stops on the range's
    blast door, a full one goes through."""
    from net import GameServer, GameClient
    from net.protocol import BTN_FIRE, ServerState

    port = 47994
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
        assert door is not None and door.heavy and not door.is_open
        sp.loadout[0] = "rail_pistol"
        sp.mags[0] = weapons.ROSTER["rail_pistol"].mag
        down = math.pi / 2

        def shot(hold):
            before = tp.body.health + tp.body.shields
            t_end = time.monotonic() + hold
            while time.monotonic() < t_end:
                sp.x, sp.y = shooter_at
                tp.x, tp.y = target_at
                a.send_input(0.0, 0.0, down, BTN_FIRE, wep=0, aim_dist=4.2)
                time.sleep(0.05)
            for _ in range(4):
                sp.x, sp.y = shooter_at
                tp.x, tp.y = target_at
                a.send_input(0.0, 0.0, down, 0, wep=0, aim_dist=4.2)
                time.sleep(0.1)
            return before - (tp.body.health + tp.body.shields)

        half = shot(1.5)
        time.sleep(1.0)
        full = shot(3.6)
        print(f"  server: half charge through the blast door {half:.0f} damage, "
              f"full charge {full:.0f}")
        assert half == 0, "a half-charged rail pistol went through a blast door"
        assert full > 0, "a fully charged rail pistol did not go through"
    finally:
        a.disconnect()
        b.disconnect()
        srv.stop()
    print("\nRAIL SERVER CHECKS PASSED")


if __name__ == "__main__":
    cover_test()
    near_face_test()
    rail_sim_test()
    rail_server_test()
    print("\nALL BLAST COVER TESTS PASSED")
