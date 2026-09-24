"""One slow client must not stop the world.

Broadcasts used to go out on the tick thread with a blocking sendall per
player, so a single connection that stopped draining — wifi hiccup, a laptop
going to sleep, a client paused in a debugger — froze the simulation for
everybody. Sending now happens on a per-connection thread behind an outbox.

This boots a match, adds a "deadbeat" client that completes the handshake and
then never reads another byte, floods the server with gunfire, and checks that:

  * the server keeps ticking at its normal rate
  * healthy clients keep receiving snapshots
  * the deadbeat's backlog is shed rather than grown: at most one snapshot is
    ever queued for it, because an old one is worthless
  * it is eventually cut off instead of buffering forever

Run from the project root:   python -m tests.backpressure_test
"""
import math
import socket
import time

from net import GameServer, GameClient
from net import maps as netmaps
from tests.combat_test import clear_pair
from net import protocol as P
from net.protocol import BTN_FIRE, ServerState
from net.server import SEND_QUEUE_HARD, SEND_QUEUE_SOFT, TICK_HZ
from tests import range_spots as R

PORT = 47995
STALL_S = 6.0
TINY_BUF = 2048          # squeeze the socket so backpressure arrives in seconds
                         # rather than after megabytes of kernel buffering


def deadbeat(port: int) -> socket.socket:
    """A client that joins and then never reads. It still SENDS, so the server
    has no reason to think it is gone — it simply stops consuming."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, TINY_BUF)
    sock.connect(("127.0.0.1", port))
    P.send_msg(sock, {"t": P.C_JOIN, "name": "Deadbeat",
                      "colour": [90, 90, 90], "version": P.PROTOCOL_VERSION})
    return sock


def main():
    m, _ = netmaps.load_with_spawns(R.MAP_ID, R.MAPS_DIR)
    a_pos, b_pos = clear_pair(m, 6.0)
    srv = GameServer([a_pos, b_pos, a_pos], port=PORT, map_id=R.MAP_ID,
                     maps_dir=R.MAPS_DIR, duration_s=300)

    ticks = {"n": 0}
    real_tick = GameServer._tick_match

    def counting(self, now):
        ticks["n"] += 1
        return real_tick(self, now)

    GameServer._tick_match = counting
    srv.start()

    a = GameClient("127.0.0.1", PORT)
    b = GameClient("127.0.0.1", PORT)
    dead = None
    try:
        assert a.connect("Healthy", (220, 40, 40)), a.reject_reason
        assert b.connect("Shooter", (40, 40, 220)), b.reject_reason
        dead = deadbeat(PORT)
        time.sleep(0.4)
        dead_id = next((p.id for p in srv.players.values()
                        if p.name == "Deadbeat"), None)
        assert dead_id is not None, "the deadbeat never joined"
        # squeeze the server side too, so its kernel buffer fills in seconds
        srv.players[dead_id].sock.setsockopt(socket.SOL_SOCKET,
                                             socket.SO_SNDBUF, TINY_BUF)
        print(f"{len(srv.players)} players, one of them not reading "
              f"(soft cap {SEND_QUEUE_SOFT}, hard cap {SEND_QUEUE_HARD})")

        a.force_start()
        time.sleep(0.5)
        assert a.world.state == ServerState.MATCH, a.world.state
        aid, bid = a.world.my_id, b.world.my_id
        srv.players[aid].x, srv.players[aid].y = a_pos
        srv.players[bid].x, srv.players[bid].y = b_pos

        heading = math.atan2(a_pos[1] - b_pos[1], a_pos[0] - b_pos[0])
        ticks["n"] = 0
        snaps_before = a.world.players[aid].snap_t
        t0 = time.monotonic()
        peak_queue = 0
        worst_snaps_queued = 0
        while time.monotonic() - t0 < STALL_S:
            # both live players hold the trigger: plenty of shot and sound
            # messages for the stalled connection to choke on
            b.send_input(0.0, 0.0, heading, BTN_FIRE, wep=2, mode=1,
                         aim_dist=6.0)
            a.send_input(0.0, 0.0, -heading, BTN_FIRE, wep=2, mode=1,
                         aim_dist=6.0)
            p = srv.players.get(dead_id)
            if p is not None:
                peak_queue = max(peak_queue, len(p.out))
                worst_snaps_queued = max(
                    worst_snaps_queued,
                    sum(1 for msg in p.out if msg.get("t") == P.S_SNAPSHOT))
            a.drain_events()
            b.drain_events()
            time.sleep(1 / 60)
        elapsed = time.monotonic() - t0

        rate = ticks["n"] / elapsed
        still_here = dead_id in srv.players
        p = srv.players.get(dead_id)
        shed = p.dropped_msgs if p else "(dropped)"
        snap_age = time.monotonic() - a.world.players[aid].snap_t

        print(f"  server ticked {rate:.1f}/s during the stall "
              f"(nominal {TICK_HZ})")
        print(f"  healthy client's last snapshot: {snap_age * 1000:.0f} ms old")
        print(f"  deadbeat queue peaked at {peak_queue}, shed {shed} message(s)")
        print(f"  snapshots ever queued for it at once: {worst_snaps_queued}")
        print(f"  still connected: {still_here}")

        assert rate > TICK_HZ * 0.8, \
            f"the server slowed to {rate:.1f} ticks/s with one client stalled"
        assert snap_age < 1.0, \
            f"the healthy client's snapshots stopped ({snap_age:.1f}s old)"
        assert a.world.players[aid].snap_t > snaps_before, \
            "the healthy client received nothing at all"
        assert worst_snaps_queued <= 1, \
            f"{worst_snaps_queued} stale snapshots queued: coalescing is broken"
        assert peak_queue <= SEND_QUEUE_HARD, \
            f"the outbox grew to {peak_queue}, past the hard cap"
        print("\nALL CHECKS PASSED")
    finally:
        GameServer._tick_match = real_tick
        if dead is not None:
            dead.close()
        a.disconnect()
        b.disconnect()
        srv.stop()


if __name__ == "__main__":
    main()
