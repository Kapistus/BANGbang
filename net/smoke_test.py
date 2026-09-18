"""Headless smoke test of the net stack. No pygame, no sim.
Boots a server, connects two clients, drives lobby -> match -> kill ->
respawn -> host end -> back to lobby, printing what each side sees."""
import time
from net import GameServer, GameClient
from net.protocol import GameMode, ServerState, Team

PORT = 47999


def dump(tag, cli):
    w = cli.world
    print(f"  [{tag}] state={w.state.name} time_left={w.time_left} "
          f"players={[(p.name, p.kills, p.alive, p.respawn_in) for p in w.players.values()]}")


def main():
    spawns = [(5, 3), (51, 31), (10, 10), (40, 5)]
    srv = GameServer(spawns, port=PORT, mode=GameMode.FFA, duration_s=120)
    srv.start()
    print(f"server up on {srv.lan_ip()}:{PORT}")

    a = GameClient("127.0.0.1", PORT)
    b = GameClient("127.0.0.1", PORT)
    assert a.connect("Kapistus", (220, 40, 40)), a.reject_reason
    assert b.connect("Bob", (40, 40, 220)), b.reject_reason
    time.sleep(0.2)
    print(f"a.is_host={a.world.is_host} b.is_host={b.world.is_host}")
    assert a.world.is_host and not b.world.is_host

    # both ready -> autostart
    a.set_ready(True)
    b.set_ready(True)
    time.sleep(0.3)
    for c, n in ((a, "A"), (b, "B")):
        evs = [e["t"] for e in c.drain_events()]
        print(f"  {n} events: {evs}")
    assert a.world.state == ServerState.MATCH, a.world.state
    print("MATCH started; spawns:",
          a.world.my_spawn, b.world.my_spawn)
    assert a.world.my_spawn != b.world.my_spawn, "spawns must be unique"

    # simulate a kill: reach into server and call kill() directly
    with srv._lock:
        pa = srv.players[a.world.my_id]
        pb = srv.players[b.world.my_id]
        srv.kill(pa, pb)
    time.sleep(0.2)
    a.drain_events(); b.drain_events()
    dump("A after kill", a)
    assert a.scoreboard()[0] == ("Kapistus", 1), a.scoreboard()
    # Bob should be dead with respawn countdown
    bob = a.world.players[b.world.my_id]
    assert not bob.alive and bob.respawn_in > 0, (bob.alive, bob.respawn_in)
    print(f"  Bob respawn_in={bob.respawn_in}")

    # kill feed present
    print(f"  killfeed: {[(k.killer, k.verb, k.victim) for k in a.world.killfeed]}")
    assert a.world.killfeed

    # shorten respawn to verify respawn actually fires
    with srv._lock:
        srv.players[b.world.my_id].respawn_at = time.monotonic() + 0.3
    time.sleep(0.7)
    a.drain_events()
    bob = a.world.players[b.world.my_id]
    print(f"  Bob after respawn: alive={bob.alive} respawn_in={bob.respawn_in}")
    assert bob.alive, "Bob should have respawned"

    # host ends match
    a.end_match()
    time.sleep(0.2)
    seen = []
    ev = []
    for e in a.drain_events():
        ev.append(e["t"])
        if e["t"] == "end_count":
            seen.append(e["n"])
    print(f"  A events after end_match: {ev}")
    assert "match_end" in ev

    # let end countdown run to 0
    print("  end countdown:", *seen, end=" ")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 7:
        for e in a.drain_events():
            if e["t"] == "end_count":
                seen.append(e["n"])
                print(e["n"], end=" ", flush=True)
        if a.world.state == ServerState.LOBBY and 0 in seen:
            break
        time.sleep(0.05)
    print()
    assert seen == [5, 4, 3, 2, 1, 0], seen
    assert a.world.state == ServerState.LOBBY
    print("back in LOBBY, scores reset:",
          [(p.name, p.kills) for p in a.world.players.values()])
    assert all(p.kills == 0 for p in a.world.players.values())

    # test host migration: disconnect host, Bob should be promoted
    a.disconnect()
    time.sleep(1.6)          # exceed the 1s lobby reaper cadence
    print(f"after host left: b.is_host={b.world.is_host} host_id={b.world.host_id}")
    assert b.world.is_host, "Bob should be promoted to host"

    b.disconnect()
    srv.stop()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
