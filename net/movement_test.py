"""Headless test of authoritative movement. No pygame, no rendering.

Boots a server on a real map, connects two clients, drives one of them with
real input, and checks the things that would silently rot:

  * a player actually moves, at roughly the speed their stance allows
  * run is faster than walk, crawl is slower
  * a diagonal is not 1.41x faster than a cardinal (input is clamped)
  * a player cannot walk through a wall
  * every position the server reports is somewhere a body can stand

Run from the project root:   python -m net.movement_test
"""
import math
import time

from sim import movement
from net import GameServer, GameClient
from net import maps as netmaps
from net.protocol import BTN_CRAWL, BTN_RUN, ServerState

PORT = 47998
MAP = "arena"
DRIVE_S = 1.0                  # how long each movement sample runs
INPUT_HZ = 60                  # client send rate


def drive(cli, mx, my, buttons=0, seconds=DRIVE_S):
    """Hold an input for `seconds`, then report (displacement, elapsed)."""
    me = cli.world.players[cli.world.my_id]
    # let any in-flight snapshot land, then take the starting point
    time.sleep(0.15)
    x0, y0 = me.x, me.y
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        cli.send_input(mx, my, 0.0, buttons)
        time.sleep(1.0 / INPUT_HZ)
    cli.send_input(0.0, 0.0, 0.0, 0)
    time.sleep(0.15)            # let the last snapshot arrive
    elapsed = time.monotonic() - t0
    return math.hypot(me.x - x0, me.y - y0), elapsed, (x0, y0), (me.x, me.y)


def near(got, want, tol, label):
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'FAIL'} {label}: {got:.2f} (want {want:.2f} +-{tol:.2f})")
    assert ok, label


def main():
    m, spawns = netmaps.load_with_spawns(MAP)
    print(f"map {m.name} {m.width_m:.0f}x{m.height_m:.0f}m, {len(spawns)} spawns")

    # Put the driven player somewhere with room to move, and the second player
    # out of the way. Pinning spawns keeps the test independent of how
    # derivation happens to order them.
    # room to walk both ways, and far enough from a wall that a second of
    # running does not end against one
    open_spot = next((x, y) for x, y in spawns
                     if all(m.can_stand(x + d, y, 0.28)
                            for d in (-7, -4, -2, 2, 4, 7)))
    srv = GameServer([open_spot, spawns[-1]], port=PORT, duration_s=120,
                     map_id=MAP)
    srv.start()

    a = GameClient("127.0.0.1", PORT)
    b = GameClient("127.0.0.1", PORT)
    assert a.connect("Driver", (220, 40, 40)), a.reject_reason
    assert b.connect("Idle", (40, 40, 220)), b.reject_reason
    a.set_ready(True)
    b.set_ready(True)
    time.sleep(0.4)
    assert a.world.state == ServerState.MATCH, a.world.state
    # Spawns are handed out at random, so the driven player could just as
    # easily have got the other one — which has no clearance guarantee, and a
    # test that measured walking speed into a wall would fail for the wrong
    # reason. Put the driver on the open spot deliberately.
    srv.players[a.world.my_id].x, srv.players[a.world.my_id].y = open_spot
    srv.players[b.world.my_id].x = open_spot[0]
    srv.players[b.world.my_id].y = open_spot[1] + 6.0
    time.sleep(0.3)
    print(f"match started, driver placed at {open_spot}")

    try:
        # --- walk east
        d, el, p0, p1 = drive(a, 1.0, 0.0)
        print(f"walk  {p0} -> {p1}")
        near(d / el, movement.SPEED_WALK, movement.SPEED_WALK * 0.25, "walk speed m/s")

        # --- run west (back across the same ground)
        dr, elr, p0, p1 = drive(a, -1.0, 0.0, BTN_RUN)
        print(f"run   {p0} -> {p1}")
        assert dr / elr > d / el * 1.5, "run should be clearly faster than walk"
        print(f"  ok  run {dr/elr:.2f} m/s > walk {d/el:.2f} m/s")

        # --- crawl
        dc, elc, _, _ = drive(a, 1.0, 0.0, BTN_CRAWL)
        assert dc / elc < d / el, "crawl should be slower than walk"
        print(f"  ok  crawl {dc/elc:.2f} m/s < walk {d/el:.2f} m/s")

        # --- both stance bits held: crawl must win, not run
        db, elb, _, _ = drive(a, -1.0, 0.0, BTN_RUN | BTN_CRAWL)
        assert db / elb < d / el, "crawl must beat run when both are held"
        print(f"  ok  run+crawl resolves to crawl ({db/elb:.2f} m/s)")

        # --- diagonal is clamped to the unit disc
        dd, eld, _, _ = drive(a, 1.0, 1.0)
        near(dd / eld, movement.SPEED_WALK, movement.SPEED_WALK * 0.3,
             "diagonal speed m/s (clamped)")

        # --- an over-range input vector buys no extra speed
        dh, elh, _, _ = drive(a, 5.0, 0.0)
        near(dh / elh, movement.SPEED_WALK, movement.SPEED_WALK * 0.3,
             "speed on a 5.0 input vector")

        # --- walls: run at the map edge for a good while, end up standable
        for mx, my in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            drive(a, mx, my, BTN_RUN, seconds=2.0)
            me = a.world.players[a.world.my_id]
            assert m.can_stand(me.x, me.y, 0.28), \
                f"ended inside geometry at ({me.x:.2f}, {me.y:.2f})"
            assert 0 <= me.x <= m.width_m and 0 <= me.y <= m.height_m, \
                f"left the map at ({me.x:.2f}, {me.y:.2f})"
        print("  ok  never ended inside geometry or off the map")

        # --- and specifically: cannot cross a known wall
        blocked = None
        for x, y in [(x, y) for x in [i * 0.5 for i in range(2, int(m.width_m * 2))]
                     for y in [j * 0.5 for j in range(2, int(m.height_m * 2))]]:
            if m.can_stand(x, y, 0.28) and not m.can_stand(x + 0.8, y, 0.28) \
                    and not m.can_stand(x + 1.6, y, 0.28):
                blocked = (x, y)
                break
        if blocked is None:
            print("  -- no wall found on this map to push against, skipped")
        else:
            srv.players[a.world.my_id].x = blocked[0]
            srv.players[a.world.my_id].y = blocked[1]
            time.sleep(0.2)
            drive(a, 1.0, 0.0, BTN_RUN, seconds=1.5)
            me = a.world.players[a.world.my_id]
            assert me.x < blocked[0] + 0.8, \
                f"walked through a wall: {blocked} -> ({me.x:.2f}, {me.y:.2f})"
            print(f"  ok  stopped at a wall: {blocked[0]:.2f} -> {me.x:.2f} "
                  f"(wall at {blocked[0] + 0.8:.2f})")

        print("\nALL CHECKS PASSED")
    finally:
        a.disconnect()
        b.disconnect()
        srv.stop()


if __name__ == "__main__":
    main()
