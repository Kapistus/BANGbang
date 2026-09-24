"""Headless test that nothing unplayable reaches a match.

The game closed the moment a host pressed start: the lobby hosts `arena` by
default, arena.grid had stopped loading, and the server's fallback started the
match anyway "without collision". Every client then tried to load the map to
draw it and crashed out. These checks pin the three places that now refuse:

  * the lobby's map list leaves out anything net.maps.validate rejects
  * a server asked to host an unplayable map switches to a playable one
  * a host picking an unplayable map mid-lobby stays on the map it has

Run from the project root:   python -m tests.map_validity_test
"""
import json
import shutil
import tempfile
import time
from pathlib import Path

from net import GameServer, GameClient
from net import maps as netmaps
from net.mapcatalog import load_catalog, rejected
from net.protocol import ServerState
from tests import range_spots as R


def _dir():
    """A maps folder with one good map and two broken ones."""
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(R.PATH, tmp / "good.map")
    doc = json.load(open(R.PATH))
    doc["object"][3][3] = "no_such_wall"
    json.dump(doc, open(tmp / "badtile.map", "w"))
    (tmp / "broken.grid").write_text("####\n#.x#\n####\n")
    (tmp / "broken.toml").write_text('grid = "broken.grid"\nname = "broken"\n')
    shutil.copy("maps/tiles.toml", tmp / "tiles.toml")
    return tmp


def tests_folder_check():
    """The maps in tests/ are for the tests and for trying things by hand.
    None of them may reach the lobby - not listed, and not hostable by a
    crafted id either."""
    real = Path(R.MAPS_DIR)
    test_ids = sorted(q.stem for q in real.glob("*.map"))
    assert R.MAP_ID in test_ids, "tests/range.map is missing"
    listed = {e.map_id for e in load_catalog("maps")}
    print(f"  tests/ holds {test_ids}; the lobby lists {sorted(listed)}")
    assert not (listed & set(test_ids)), "a test map is in the lobby"
    for crafted in [f"../tests/{t}" for t in test_ids] + \
            [f"..\\tests\\{t}" for t in test_ids] + \
            ["tests/range.map", "/etc/passwd"]:
        assert netmaps.validate(crafted, "maps"), \
            f"{crafted!r} would be hostable"
    print("  every crafted path into tests/ is refused")
    # ...but they are sound maps in their own right, bar the one that is
    # there to be unplayable
    for t in test_ids:
        ok = not netmaps.validate(t, real)
        assert ok == (t != "not_playable"), f"tests/{t} playable={ok}"
    print("  the range is playable on its own; not_playable is not")


def main():
    tests_folder_check()
    d = _dir()
    try:
        listed = [e.map_id for e in load_catalog(d)]
        print(f"  folder holds good, badtile, broken -> lobby lists {listed}")
        assert listed == ["good"], f"the list offered {listed}"
        r = rejected()
        assert set(r) == {"badtile", "broken"}, f"rejected {sorted(r)}"
        for mid, why in r.items():
            print(f"    left out {mid}: {why[0][:80]}")
        assert netmaps.first_playable(d) == "good"
        assert netmaps.first_playable(d, prefer="broken") == "good"

        # --- asked to host something unplayable, the server picks one it can
        srv = GameServer(port=47962, map_id="broken", maps_dir=str(d),
                         duration_s=60)
        srv.start()
        a = GameClient("127.0.0.1", 47962)
        b = GameClient("127.0.0.1", 47962)
        try:
            assert a.connect("Host", (220, 40, 40)), a.reject_reason
            assert b.connect("B", (40, 40, 220)), b.reject_reason
            time.sleep(0.4)
            print(f"  hosting 'broken' -> the server is on {srv.map_id!r}")
            assert srv.map is not None and srv.map_id == "good"
            assert a.world.map_id == "good", "the lobby advertises a broken map"

            # --- and a host who picks a broken map stays where they are
            a.set_config(map_id="badtile")
            time.sleep(0.4)
            print(f"  host picks 'badtile' -> still on {srv.map_id!r}")
            assert srv.map_id == "good", "the server switched to a broken map"

            a.set_ready(True)
            b.set_ready(True)
            t0 = time.monotonic()
            while a.world.state != ServerState.MATCH and time.monotonic() - t0 < 4:
                time.sleep(0.05)
            assert a.world.state == ServerState.MATCH, "the match never started"
            netmaps.load(a.world.map_id, d)       # what every client does next
            print(f"  started on {a.world.map_id!r}, and it loads for a client")
        finally:
            a.disconnect()
            b.disconnect()
            srv.stop()

        # --- a folder with nothing playable: refuse, don't start broken
        only_bad = Path(tempfile.mkdtemp())
        for f in ("broken.grid", "broken.toml", "tiles.toml"):
            shutil.copy(d / f, only_bad / f)
        srv = GameServer(port=47963, map_id="broken", maps_dir=str(only_bad),
                         duration_s=60)
        srv.start()
        a = GameClient("127.0.0.1", 47963)
        b = GameClient("127.0.0.1", 47963)
        try:
            a.connect("Host", (220, 40, 40))
            b.connect("B", (40, 40, 220))
            a.set_ready(True)
            b.set_ready(True)
            time.sleep(1.0)
            print(f"  nothing playable at all -> state {a.world.state.name}")
            assert a.world.state == ServerState.LOBBY, \
                "a match started with no playable map"
        finally:
            a.disconnect()
            b.disconnect()
            srv.stop()
            shutil.rmtree(only_bad, ignore_errors=True)
        print("\nMAP VALIDITY CHECKS PASSED")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
