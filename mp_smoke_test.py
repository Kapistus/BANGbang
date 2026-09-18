"""Headless check that the networked client actually runs.

Boots a server, connects two clients in this process, force-starts a match, and
drives one of them through MatchView for a couple of hundred frames with
scripted input — the render path, prediction, reconciliation and fog all
execute, against a dummy video driver so no window opens.

It will not tell you the game looks right. It will tell you the client still
starts, still draws, and still agrees with the server about where you are,
which is what breaks silently when the sim or the protocol moves underneath it.

    python mp_smoke_test.py                  # on the arena map
    python mp_smoke_test.py vessel_interior  # or any other map id
"""
import math
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from net import GameServer, GameClient
from net import maps as netmaps
from net.protocol import BTN_FIRE, BTN_RUN, ServerState

PORT = 47997
RUN_S = 6.0                  # how long to drive, in real seconds
MAP = sys.argv[1] if len(sys.argv) > 1 else "arena"


class ScriptedInput:
    """Stands in for keyboard and mouse: walk out, run back, then shoot.

    Phases are keyed to REAL elapsed time, not frame count. The server
    integrates in real time, so a test that drove simulated frames faster than
    the clock would predict further than the server ever moved and then watch
    reconciliation drag it back — measuring the harness, not the game."""

    def __init__(self, view, t0):
        self.view = view
        self.t0 = t0

    def __call__(self):
        t = time.monotonic() - self.t0
        if t < RUN_S * 0.35:
            return 1.0, 0.0, 0
        if t < RUN_S * 0.7:
            return -1.0, 0.0, BTN_RUN
        return 0.0, 0.0, BTN_FIRE


def main():
    import mp_client
    from sim import audio

    # Watch every clip that gets played. Your own footsteps used to go out at
    # 0.9 gain — six times what single-player uses — which is what "sprinting is
    # too loud" turned out to mean.
    played = []
    real_play = audio.play
    audio.play = lambda name, gain=1.0, pan=0.0: (played.append((name, gain)),
                                                  real_play(name, gain, pan))[1]

    # Put the driver somewhere with room to move either way, so the test
    # measures movement rather than how quickly it meets a wall.
    m, spawns = netmaps.load_with_spawns(MAP)
    open_spot = next(((x, y) for x, y in spawns
                      if m.can_stand(x + 4, y, 0.28)
                      and m.can_stand(x - 4, y, 0.28)), spawns[0])
    srv = GameServer([open_spot, spawns[-1]], port=PORT, map_id=MAP,
                     duration_s=120)
    srv.start()
    a = GameClient("127.0.0.1", PORT)
    b = GameClient("127.0.0.1", PORT)
    assert a.connect("Driver", (220, 40, 40)), a.reject_reason
    assert b.connect("Other", (40, 40, 220)), b.reject_reason
    a.set_ready(True)
    b.set_ready(True)
    time.sleep(0.5)
    assert a.world.state == ServerState.MATCH, f"no match: {a.world.state}"
    print(f"match started on {a.world.map_id}, spawn {a.world.my_spawn}")

    # Put the second player within earshot of the first. Spawns are spread
    # across the map on purpose, and on a big one like vessel_interior they can
    # be further apart than a pistol carries — so the test would be asserting
    # that sound does not travel 60 m, which is correct and useless.
    ax, ay = a.world.my_spawn
    near = next(((ax + dx, ay + dy) for dx, dy in
                 ((5, 0), (-5, 0), (0, 5), (0, -5), (3.5, 3.5), (-3.5, -3.5))
                 if m.can_stand(ax + dx, ay + dy, 0.3)), (ax, ay))
    srv.players[b.world.my_id].x, srv.players[b.world.my_id].y = near
    time.sleep(0.25)
    print(f"other player placed at ({near[0]:.1f}, {near[1]:.1f}), "
          f"{math.hypot(near[0] - ax, near[1] - ay):.1f} m away")

    pygame.init()
    view = mp_client.MatchView(a, "maps")
    print(f"MatchView up: {view.m.name} "
          f"world {view.world_w}x{view.world_h}px view {view.view_w}x{view.view_h}")

    t0 = time.monotonic()
    view.read_input = ScriptedInput(view, t0)
    start = (view.px, view.py)
    worst_drift = 0.0
    fired_shots = 0      # peak live tracers: effects fade, so watch the peak
    heard_any = 0        # peak propagation fields in flight
    cues_seen = 0        # arrival cues raised: a sound that actually REACHED us
    frames = 0
    last = time.monotonic()
    while time.monotonic() - t0 < RUN_S:
        now = time.monotonic()
        dt = min(now - last, 0.1)
        last = now
        why = view.frame(dt)
        assert why is None, f"frame {frames} bailed out: {why}"
        fired_shots = max(fired_shots, len(view.tracers))
        heard_any = max(heard_any, len(view.sounds))
        cues_seen = max(cues_seen, len(view.cues))
        # only judge drift once the first snapshots have been exchanged
        if now - t0 > 0.5:
            worst_drift = max(worst_drift, view.corrected_m)
        frames += 1
        # the other player shoots back, so this client has someone else's
        # gunfire to solve a propagation field for and hear
        if frames % 20 == 0:
            them = b.world.players[b.world.my_id]
            mine = b.world.players.get(a.world.my_id)
            heading = (math.atan2(mine.y - them.y, mine.x - them.x)
                       if mine else 0.0)
            b.send_input(0.0, 0.0, heading, BTN_FIRE, wep=0, aim_dist=4.0)
        elif frames % 20 == 10:
            b.send_input(0.0, 0.0, 0.0, 0, wep=0, aim_dist=4.0)
        time.sleep(0.002)
    elapsed = time.monotonic() - t0
    fps = frames / elapsed

    moved = math.hypot(view.px - start[0], view.py - start[1])
    print(f"weapon: {view.loadout[view.wep]}, "
          f"sprites {'on' if view.bank.ok else 'off (circle fallback)'}, "
          f"audio {'on' if view.audio_on else 'off (no device)'}")
    me = a.world.players[a.world.my_id]
    gap = math.hypot(me.x - view.px, me.y - view.py)
    seen_b = b.world.players.get(a.world.my_id)

    print(f"drove {frames} frames in {elapsed:.1f}s at {fps:.0f} fps "
          f"(render + fog + sound + net)")
    print(f"fired: peak {fired_shots} tracer(s) on screen at once")
    print(f"heard: peak {heard_any} propagation field(s) in flight, "
          f"peak {cues_seen} arrival cue(s) — a cue means a sound solved its "
          f"way to this player and was played")
    print(f"moved {moved:.2f} m from {start} to ({view.px:.2f}, {view.py:.2f})")
    # `gap` is not error: it is how far the client's prediction legitimately
    # leads the newest snapshot, which at a run is several centimetres per
    # unacked command. `worst_drift` is the real measure — how far off the
    # prediction turned out to be once the server acked it.
    print(f"worst correction after ack {worst_drift * 100:.2f} cm; "
          f"prediction leads last snapshot by {gap * 100:.1f} cm")

    # walked out for half the run, ran back for the other half: the net
    # displacement is small, so measure the path actually travelled instead
    assert view.travelled > 3.0, \
        f"scripted input produced almost no movement ({view.travelled:.2f} m)"
    print(f"travelled {view.travelled:.2f} m along the path")
    assert gap < 1.0, f"client is {gap:.2f} m ahead of the server — too far"
    assert worst_drift < 0.25, \
        f"prediction was wrong by {worst_drift * 100:.0f} cm after being acked"
    assert seen_b is not None and seen_b.x != 0.0, \
        "the other client never saw the driver move"
    assert view.m.can_stand(view.px, view.py, 0.28), "ended inside geometry"
    steps = [g for name, g in played if name == "footstep"]
    if steps:
        print(f"footstep clips played: {len(steps)}, loudest {max(steps):.2f} "
              f"(own steps are {mp_client.OWN_STEP_GAIN['walk']} walking, "
              f"{mp_client.OWN_STEP_GAIN['run']} running)")
        assert max(steps) <= 0.45, \
            f"a footstep played at {max(steps):.2f} — far too loud"
    assert cues_seen <= mp_client.CUE_MAX, \
        (f"{cues_seen} cues on screen at once: arcs are stacking into a ring "
         f"again (cap {mp_client.CUE_MAX})")
    assert fired_shots, "the client never saw its own shots"
    assert heard_any, "the other player's gunfire never reached this client"
    assert cues_seen, \
        "gunfire was emitted but never arrived: the perception path is dead"
    print("\nALL CHECKS PASSED")

    a.disconnect()
    b.disconnect()
    srv.stop()
    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
