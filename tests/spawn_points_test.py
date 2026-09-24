"""Multiplayer spawn points as map objects.

Spawns used to be derived from geometry alone: farthest-point sampling knows
the ground is standable and nothing about sightlines, so it will happily drop
you in the open at the end of a long clean lane. A map can now declare where
matches start, with a facing and an optional team tag, and derivation is the
fallback for maps that declare nothing.

Checks the whole path: the format carries them, the editor writes them, the
loader parses them, the server prefers them over derived ones, puts each side on
its own spawns in team modes, and faces players the way the map says.

Run from the project root:   python -m tests.spawn_points_test
"""
import math
import os
import shutil
import tempfile
import time

from net import GameServer, GameClient
from net import maps as netmaps
from net.protocol import GameMode, ServerState, Team
from sim import mapfile
from sim.tilemap import SpawnPoint, parse_spawn_points, validate_spawns
from tests import range_spots as R

PORT = 47992
# The test range. Its own spawns are replaced by the ones each check writes in,
# and the positions are found on the map rather than assumed.
SOURCE_MAP = R.PATH


def wait_for(predicate, seconds=4.0):
    """Poll rather than sleep a guessed interval: a fixed wait makes a test
    that fails on a slow machine and proves nothing on a fast one."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def authored_map(tmp, spawns, name="spawntest"):
    """A copy of the range with spawn points written into it."""
    path = os.path.join(tmp, f"{name}.map")
    shutil.copy(SOURCE_MAP, path)
    doc = mapfile.load_doc(path)
    doc["spawn_points"] = spawns
    mapfile.save(doc, path)
    return path


def standable_spots(m, n, clearance=0.45):
    out = []
    step = 1.0
    y = step
    while y < m.height_m and len(out) < n:
        x = step
        while x < m.width_m and len(out) < n:
            if m.can_stand(x, y, clearance):
                if all(math.hypot(x - a, y - b) > 3.0 for a, b in out):
                    out.append((x, y))
            x += step
        y += step
    assert len(out) == n, f"only found {len(out)} spots on this map"
    return out


def main():
    tmp = tempfile.mkdtemp()
    try:
        base = mapfile.load_map(SOURCE_MAP)
        spots = standable_spots(base, 4)

        # --- the parser takes every shape a map might use
        mixed = parse_spawn_points([
            [1, 2],
            {"pos": [3, 4], "facing_deg": 90, "team": "B"},
            {"x": 5, "y": 6},
            "nonsense",
            {"pos": [7]},
            {"pos": [8, 9], "team": "purple"},
        ])
        print(f"parser: {len(mixed)} of 6 entries survived, "
              f"teams {[sp.team for sp in mixed]}")
        assert len(mixed) == 4, "the parser dropped or invented entries"
        assert mixed[1].team == "b" and mixed[1].facing_deg == 90
        assert mixed[3].team == "", "an unknown team tag was kept"

        # --- authored spawns beat derived ones
        authored = [
            {"pos": list(spots[0]), "facing_deg": 0, "team": "a"},
            {"pos": list(spots[1]), "facing_deg": 180, "team": "a"},
            {"pos": list(spots[2]), "facing_deg": 90, "team": "b"},
            {"pos": list(spots[3]), "facing_deg": 270, "team": "b"},
        ]
        path = authored_map(tmp, authored)
        m = mapfile.load_map(path)
        assert len(m.spawn_points) == 4, "the map did not carry them"
        got = netmaps.spawn_points("spawntest", m, maps_dir=tmp)
        print(f"map declares {len(m.spawn_points)} spawns; server will use "
              f"{len(got)} (derivation would give {netmaps.WANT_SPAWNS})")
        assert len(got) == 4, "derived spawns were used despite authored ones"
        assert {sp.team for sp in got} == {"a", "b"}
        assert not validate_spawns(m), f"authored spawns fail validation: " \
                                       f"{validate_spawns(m)}"

        # --- a spawn inside a wall is dropped, and the map says so
        bad = authored + [{"pos": [0.1, 0.1], "team": "a"}]
        bad_path = authored_map(tmp, bad, name="spawnbad")
        bm = mapfile.load_map(bad_path)
        problems = validate_spawns(bm)
        kept = netmaps.spawn_points("spawnbad", bm, maps_dir=tmp)
        print(f"a spawn inside geometry: {len(problems)} validation "
              f"warning(s), {len(kept)} of {len(bm.spawn_points)} kept")
        assert problems, "a spawn inside a wall raised no warning"
        assert len(kept) == 4, "the unusable spawn was handed to the server"

        # --- one-sided tagging is called out
        lopsided = [dict(a, team="a") for a in authored]
        lop_path = authored_map(tmp, lopsided, name="spawnlop")
        lm = mapfile.load_map(lop_path)
        assert any("not" in p and "both" in p for p in validate_spawns(lm)), \
            "a map tagging only one team raised no warning"
        print("  a map that tags only one side is flagged")

        # --- a team match starts each side on its own spawns, facing as told
        srv = GameServer(port=PORT, map_id="spawntest", maps_dir=tmp,
                         mode=GameMode.TEAM, duration_s=300)
        srv.start()
        a = GameClient("127.0.0.1", PORT)
        b = GameClient("127.0.0.1", PORT)
        try:
            assert a.connect("Ay", (220, 40, 40)), a.reject_reason
            assert b.connect("Bee", (40, 40, 220)), b.reject_reason
            a.set_team(Team.A)
            b.set_team(Team.B)
            assert wait_for(lambda: srv.players[a.world.my_id].team == Team.A
                            and srv.players[b.world.my_id].team == Team.B), \
                "teams were never applied"
            a.set_ready(True)
            b.set_ready(True)
            assert wait_for(lambda: a.world.state == ServerState.MATCH), \
                f"the match never started (state {a.world.state})"
            by_team = {"a": [sp for sp in srv.spawn_points if sp.team == "a"],
                       "b": [sp for sp in srv.spawn_points if sp.team == "b"]}
            for cli, tag in ((a, "a"), (b, "b")):
                p = srv.players[cli.world.my_id]
                here = (round(p.x, 2), round(p.y, 2))
                mine = [(round(sp.x, 2), round(sp.y, 2)) for sp in by_team[tag]]
                theirs = [(round(sp.x, 2), round(sp.y, 2))
                          for sp in by_team["b" if tag == "a" else "a"]]
                print(f"  team {tag.upper()} spawned at {here}, "
                      f"aim {math.degrees(p.aim):.0f} deg")
                assert here in mine, f"team {tag} spawned outside its own pool"
                assert here not in theirs, f"team {tag} spawned on the other side"
                sp = next(s for s in by_team[tag]
                          if (round(s.x, 2), round(s.y, 2)) == here)
                assert abs(p.aim - math.radians(sp.facing_deg)) < 1e-3, \
                    "the player is not facing the way the spawn does"
                assert cli.world.my_spawn_aim is not None, \
                    "the client was not told which way to face"
            print("  both sides on their own spawns, facing as authored")

            # A team match is also the only thing that puts a team_scores dict
            # on the wire. Keyed by team NUMBER it was unpackable by msgpack,
            # which killed each client's receive thread on the first score
            # broadcast — the client went silent and nothing said why. Check
            # the clients are still hearing the server after one.
            srv.players[a.world.my_id].kills = 2
            srv._broadcast_score()
            assert wait_for(lambda: a.world.team_scores), \
                "team scores never arrived: the wire format is broken again"
            before = a.world.players[a.world.my_id].snap_t
            assert wait_for(lambda: a.world.players[a.world.my_id].snap_t > before), \
                "snapshots stopped after a score broadcast"
            assert a.connected and b.connected, "a client dropped in team mode"
            print(f"  team scores reached the clients: {a.world.team_scores}, "
                  f"snapshots still flowing")
        finally:
            a.disconnect()
            b.disconnect()
            srv.stop()

        # --- a map that declares none still plays, on derived spawns
        plain_path = authored_map(tmp, [], name="plain")
        plain = netmaps.spawn_points("plain", mapfile.load_map(plain_path),
                                     maps_dir=tmp)
        print(f"a map with no authored spawns still yields {len(plain)} derived")
        assert len(plain) >= 2 and all(isinstance(sp, SpawnPoint) for sp in plain)

        print("\nALL CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
