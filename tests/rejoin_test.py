"""Joining, leaving and rejoining a match in progress.

A connection used to be the same thing as playing: latecomers were turned away
with "match in progress", and leaving a match meant leaving the server. Now a
player can arrive late, step out, and come back, and the match carries on
without them in the meantime.

What this checks:

  * a latecomer is accepted, and waits in the lobby rather than being dropped
    into a firefight unasked
  * they spawn at the point furthest from anyone still fighting
  * they are told about doors and windows that changed before they arrived —
    otherwise they are playing on a different map from everyone else
  * leaving keeps the connection and the scoreboard entry, and the match
    carries on
  * rejoining puts them back in, healed, at a fresh spawn
  * a match everybody has walked out of ends by itself

Run from the project root:   python -m tests.rejoin_test
"""
import math
import time

from net import GameServer, GameClient
from net import maps as netmaps
from tests.combat_test import clear_pair
from net.protocol import BTN_INTERACT, ServerState
from sim.tilemap import find_doors
from tests import range_spots as R

PORT = 47993


def wait_for(predicate, seconds=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def main():
    m, spawns = netmaps.load_with_spawns(R.MAP_ID, R.MAPS_DIR)
    a_pos, b_pos = clear_pair(m, 4.0)
    # the map's own spawn points, so "furthest from the fighting" has a real
    # choice to make — pinning two would put every option under someone's feet
    srv = GameServer(spawns, port=PORT, map_id=R.MAP_ID, maps_dir=R.MAPS_DIR,
                     duration_s=600)
    srv.start()

    a = GameClient("127.0.0.1", PORT)
    b = GameClient("127.0.0.1", PORT)
    c = None
    try:
        assert a.connect("Alice", (220, 40, 40)), a.reject_reason
        assert b.connect("Bob", (40, 40, 220)), b.reject_reason
        a.set_ready(True)
        b.set_ready(True)
        assert wait_for(lambda: a.world.state == ServerState.MATCH), "no match"
        aid, bid = a.world.my_id, b.world.my_id
        srv.players[aid].x, srv.players[aid].y = a_pos
        srv.players[bid].x, srv.players[bid].y = b_pos
        assert a.world.playing and b.world.playing, "starters are not playing"
        print(f"match running on {R.MAP_ID} with 2 players")

        # --- change the map before the latecomer arrives: open a door
        door = next(iter(find_doors(m)))
        _d = srv.doors.door(door)
        _d.is_open = _d.target = False
        _d.left = 0.0
        sp = srv.players[aid]
        old_pos = (sp.x, sp.y)
        sp.x, sp.y = door[1] + 0.5 + 1.2, door[0] + 0.5
        time.sleep(0.2)
        a.send_input(0.0, 0.0, 0.0, BTN_INTERACT)
        time.sleep(0.15)
        a.send_input(0.0, 0.0, 0.0, 0)
        assert wait_for(lambda: srv.doors.get(door) is True), \
            "could not open a door to test the catch-up with"
        sp.x, sp.y = old_pos
        print(f"  a door at {door} is open before the latecomer connects")

        # --- a third player connects mid-match
        c = GameClient("127.0.0.1", PORT)
        joined = c.connect("Carol", (40, 200, 60))
        assert joined, f"a latecomer was turned away: {c.reject_reason}"
        assert wait_for(lambda: c.world.state == ServerState.MATCH), \
            "the latecomer was never told a match is running"
        assert not c.world.playing, \
            "the latecomer was dropped into the match without asking"
        print(f"  connected mid-match, sitting out: playing={c.world.playing}")

        # they were handed the map as it stands
        assert wait_for(lambda: c.world.map_doors or c.world.map_glass), \
            "no map state was sent to the latecomer"
        assert any((row[0], row[1]) == door and row[2]
                   for row in c.world.map_doors), \
            f"the open door was not in the catch-up: {c.world.map_doors}"
        print(f"  told about {len(c.world.map_doors)} open door(s), "
              f"{len(c.world.map_glass)} broken pane(s)")

        # --- joining puts them in, away from the fighting
        c.join_match()
        assert wait_for(lambda: c.world.playing), "join_match did nothing"
        cid = c.world.my_id
        spawn = c.world.my_spawn
        fighters = [srv.players[aid], srv.players[bid]]
        got = min(math.hypot(spawn[0] - q.x, spawn[1] - q.y) for q in fighters)
        best = max(min(math.hypot(sx - q.x, sy - q.y) for q in fighters)
                   for sx, sy in srv.spawn_points)
        print(f"  joined at {spawn}: {got:.1f} m from the nearest fighter "
              f"(best of {len(srv.spawn_points)} spawns: {best:.1f} m)")
        assert abs(got - best) < 0.01, "dropped in nearer the fighting than needed"
        assert got > 8.0, \
            f"spawned {got:.1f} m from a firefight — too close to be fair"
        assert srv.players[cid].body is not None, "joined without a body"
        assert srv.players[cid].alive, "joined dead"

        # the others can see them
        assert wait_for(lambda: a.world.players.get(cid) is not None
                        and a.world.players[cid].playing), \
            "the other players never saw the newcomer join"
        print("  the other players see them in the match")

        # --- leaving: connection and score survive, the match does not stop
        srv.players[bid].kills = 3
        srv._broadcast_score()
        time.sleep(0.2)
        b.leave_match()
        assert wait_for(lambda: not b.world.playing), "leave_match did nothing"
        assert bid in srv.players, "leaving the match dropped the connection"
        assert srv.state == ServerState.MATCH, "the match stopped when one left"
        assert b.world.players[bid].kills == 3, "their score was thrown away"
        snap_before = a.world.players[aid].snap_t
        assert wait_for(lambda: a.world.players[aid].snap_t > snap_before), \
            "the remaining players stopped getting snapshots"
        print(f"  Bob left: still connected, still on the board with "
              f"{b.world.players[bid].kills} kills, match still running")

        # a player who is out is not simulated
        assert not srv.players[bid].alive and not srv.players[bid].playing
        assert srv.players[bid].body is None, "a player who left still has a body"

        # --- rejoining
        b.join_match()
        assert wait_for(lambda: b.world.playing), "could not rejoin"
        body = srv.players[bid].body
        assert body is not None and body.health == body.max_health, \
            "rejoined wounded"
        assert srv.players[bid].kills == 3, "rejoining reset their score"
        print(f"  Bob rejoined at {b.world.my_spawn}, healed, score kept")

        # --- everybody walks out: the match should not run on empty
        a.leave_match()
        b.leave_match()
        c.leave_match()
        assert wait_for(lambda: srv.state != ServerState.MATCH, 4.0), \
            "a match with nobody in it kept running"
        print(f"  everyone left -> server state {srv.state.name}")
        assert wait_for(lambda: srv.state == ServerState.LOBBY, 12.0), \
            "never came back to the lobby"
        assert all(pid in srv.players for pid in (aid, bid, cid)), \
            "walking out of a match dropped connections"
        print("  back in the lobby with all three still connected")

        print("\nALL CHECKS PASSED")
    finally:
        for cl in (a, b, c):
            if cl is not None:
                cl.disconnect()
        srv.stop()


if __name__ == "__main__":
    main()
