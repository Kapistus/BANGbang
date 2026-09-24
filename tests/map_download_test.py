"""Headless test that a player gets the host's map when they do not have it.

A client used to load the host's map from its own maps/ folder, so a map that
only the host had could not be played at all - and a different map under the
same name was worse, because it loaded and drew the wrong walls. Now the lobby
carries a fingerprint of the host's files, and a client whose copy is missing
or different downloads the host's before it can play.

What this checks:

  * a client without the map downloads it, and readying up waits for that
  * a client with a different map under the same name gets the host's, and
    its own file is left alone
  * a copy downloaded once is used again next time, not fetched again
  * the host changing map in the lobby sends the new one
  * joining a match in progress waits for the download
  * the server only ever sends the map the host selected, and the client
    only ever writes plain file names into its own download folder

Run from the project root:   python -m tests.map_download_test
"""
import json
import shutil
import tempfile
import time
from pathlib import Path

from net import GameServer, GameClient
from net import maps as netmaps
from net import protocol as P
from net.protocol import ServerState
from tests import range_spots as R

PORT = 47971


def wait_for(pred, seconds=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _host_dir():
    """The host's maps: alpha and beta, two versions of the range."""
    d = Path(tempfile.mkdtemp(prefix="host_maps_"))
    shutil.copy(R.PATH, d / "alpha.map")
    doc = json.load(open(R.PATH))
    doc["object"][10][20] = "metal_00_00"         # one extra wall in the hall
    json.dump(doc, open(d / "beta.map", "w"))
    return d


def _client(own_maps, downloads, name):
    c = GameClient("127.0.0.1", PORT)
    c.maps_dir, c.download_dir = Path(own_maps), Path(downloads)
    assert c.connect(name, (200, 60, 60)), c.reject_reason
    return c


def refusal_test():
    """What a client will not write, whatever a host sends it."""
    files = netmaps.map_files("range", "tests")
    sha = netmaps.files_sha(files)
    tmp = Path(tempfile.mkdtemp())
    try:
        bad = [
            ("../escape", sha, files, "a map id with a path in it"),
            ("range", sha, {"../range.map": files["range.map"]}, "a file name with a path"),
            ("range", "0" * 64, files, "files that do not match the fingerprint"),
            ("range", sha, {"range.exe": files["range.map"]}, "an unexpected file type"),
            ("other", netmaps.files_sha({"range.map": files["range.map"]}),
             {"range.map": files["range.map"]}, "a file for a different map"),
        ]
        for map_id, s, fs, what in bad:
            try:
                netmaps.save_download(map_id, s, fs, tmp)
            except ValueError as e:
                print(f"  refused {what}: {e}")
            else:
                raise AssertionError(f"accepted {what}")
        assert not any(tmp.rglob("*.map")), "a refused download left a file behind"
        folder = netmaps.save_download("range", sha, files, tmp)
        assert netmaps.map_sha("range", folder) == sha
        print(f"  the real thing is written to {folder.name}/ and loads")

        # a char-grid map is three files: sidecar, grid, and the tile table
        # that says what each character is
        src = tmp / "src"
        src.mkdir()
        (src / "cq.grid").write_text("\n".join(
            ["#" * 16] + ["#" + "." * 14 + "#"] * 6 + ["#" * 16]) + "\n")
        (src / "cq.toml").write_text('name = "cq"\ngrid = "cq.grid"\n')
        shutil.copy("maps/tiles.toml", src / "tiles.toml")
        gfiles = netmaps.map_files("cq", src)
        assert sorted(gfiles) == ["cq.grid", "cq.toml", "tiles.toml"], sorted(gfiles)
        gfolder = netmaps.save_download("cq", netmaps.files_sha(gfiles),
                                        gfiles, tmp / "dl")
        assert netmaps.map_sha("cq", gfolder) == netmaps.files_sha(gfiles)
        print(f"  a char-grid map travels as {sorted(gfiles)} and loads")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nREFUSAL CHECKS PASSED")


def lobby_test():
    host = _host_dir()
    mine = Path(tempfile.mkdtemp(prefix="my_maps_"))       # has no alpha
    theirs = Path(tempfile.mkdtemp(prefix="their_maps_"))  # a different alpha
    doc = json.load(open(host / "alpha.map"))
    doc["object"][9][9] = "metal_00_00"
    json.dump(doc, open(theirs / "alpha.map", "w"))
    their_own = (theirs / "alpha.map").read_bytes()
    dl_a = Path(tempfile.mkdtemp(prefix="dl_a_"))
    dl_b = Path(tempfile.mkdtemp(prefix="dl_b_"))

    srv = GameServer(port=PORT, map_id="alpha", maps_dir=str(host),
                     duration_s=120)
    srv.start()
    a = b = None
    try:
        sent = []
        real_send = srv._send_map
        srv._send_map = lambda p, mid: (sent.append(mid), real_send(p, mid))[1]

        # --- nobody but the host has alpha: it comes down the wire
        a = _client(mine, dl_a, "Ann")
        assert wait_for(lambda: a.world.map_status == "ready"), \
            f"never got the map: {a.world.map_status!r}"
        print(f"  Ann had no alpha: downloaded to "
              f"{Path(a.world.map_dir).relative_to(dl_a.parent)}")
        assert Path(a.world.map_dir) == dl_a / "alpha"
        assert netmaps.map_sha("alpha", a.world.map_dir) == srv.map_sha

        # --- a different alpha is not the host's alpha
        b = _client(theirs, dl_b, "Bob")
        assert wait_for(lambda: b.world.map_status == "ready")
        print(f"  Bob had his own alpha: got the host's instead "
              f"({Path(b.world.map_dir).name}/), his file untouched")
        assert Path(b.world.map_dir) == dl_b / "alpha"
        assert (theirs / "alpha.map").read_bytes() == their_own, \
            "the download overwrote Bob's own map"
        assert sent.count("alpha") == 2

        # --- readying waits for the map: both have it now, so this starts
        a.set_ready(True)
        b.set_ready(True)
        assert wait_for(lambda: a.world.state == ServerState.MATCH), "no match"
        m = netmaps.load(a.world.map_id, a.world.map_dir)
        assert m.width_m == srv.map.width_m
        print("  both ready with the host's map: the match started")

        # --- back in the lobby, the host switches to beta
        srv._begin_end_countdown()
        assert wait_for(lambda: a.world.state == ServerState.LOBBY, 15.0)
        a.set_config(map_id="beta")
        assert wait_for(lambda: a.world.map_id == "beta"
                        and a.world.map_status == "ready"), a.world.map_status
        assert Path(a.world.map_dir) == dl_a / "beta"
        print("  host picked beta: it followed")
        # and back to alpha: the copy from before is still good
        before = len(sent)
        a.set_config(map_id="alpha")
        assert wait_for(lambda: a.world.map_id == "alpha"
                        and a.world.map_status == "ready")
        time.sleep(0.2)
        assert len(sent) == before, \
            f"alpha was downloaded again: {sent[before:]}"
        print("  back to alpha: used the copy from before, nothing fetched")

        # --- the server sends the selected map and nothing else
        with srv._lock:
            p = srv.players[a.world.my_id]
            got = []
            real_p_send = p.send
            p.send = lambda obj: (got.append(obj), True)[1]
            srv._send_map(p, "beta")
            srv._send_map(p, "../alpha")
            p.send = real_p_send
        assert all("error" in g and "files" not in g for g in got), got
        print("  asked for anything but the selected map: refused")
    finally:
        for c in (a, b):
            if c:
                c.disconnect()
        srv.stop()
        for d in (host, mine, theirs, dl_a, dl_b):
            shutil.rmtree(d, ignore_errors=True)
    print("\nLOBBY CHECKS PASSED")


def gate_test():
    """Readying up starts nothing while somebody is still downloading, and a
    latecomer's join waits for the map."""
    host = _host_dir()
    empty = Path(tempfile.mkdtemp(prefix="empty_"))
    dls = [Path(tempfile.mkdtemp(prefix=f"dl{i}_")) for i in range(3)]
    srv = GameServer(port=PORT + 1, map_id="alpha", maps_dir=str(host),
                     duration_s=120)
    srv.start()
    held = []
    real_send = srv._send_map
    srv._send_map = lambda p, mid: held.append((p, mid))   # sit on requests
    clients = []
    try:
        for i, name in enumerate(("Ann", "Bob")):
            c = GameClient("127.0.0.1", PORT + 1)
            c.maps_dir, c.download_dir = empty, dls[i]
            assert c.connect(name, (60, 60, 200)), c.reject_reason
            clients.append(c)
        a, b = clients
        assert wait_for(lambda: len(held) == 2)
        a.set_ready(True)
        b.set_ready(True)
        time.sleep(0.5)
        assert srv.state == ServerState.LOBBY, \
            "a match started while both players were still downloading the map"
        roster = a.world.players
        assert not any(p.has_map for p in roster.values()), \
            "the lobby shows them as having the map"
        print("  both ready, neither has the map yet: still in the lobby")
        with srv._lock:
            for p, mid in held:
                real_send(p, mid)
        held.clear()
        assert wait_for(lambda: srv.state == ServerState.MATCH), \
            "the downloads finished and the match still did not start"
        print("  downloads land: the match starts on its own")

        # --- a latecomer: join is asked for at once, sent when the map lands
        c = GameClient("127.0.0.1", PORT + 1)
        c.maps_dir, c.download_dir = empty, dls[2]
        assert c.connect("Cat", (60, 200, 60)), c.reject_reason
        clients.append(c)
        assert wait_for(lambda: held), "the latecomer never asked for the map"
        c.join_match()
        time.sleep(0.4)
        assert not srv.players[c.world.my_id].playing, \
            "joined a match without the map to draw it"
        with srv._lock:
            for p, mid in held:
                real_send(p, mid)
        assert wait_for(lambda: c.world.playing), \
            "the join never went through once the map arrived"
        print("  a latecomer's join waited for the download, then went in")
    finally:
        for c in clients:
            c.disconnect()
        srv.stop()
        for d in [host, empty] + dls:
            shutil.rmtree(d, ignore_errors=True)
    print("\nGATE CHECKS PASSED")


if __name__ == "__main__":
    refusal_test()
    print()
    lobby_test()
    print()
    gate_test()
    print("\nALL MAP DOWNLOAD CHECKS PASSED")
