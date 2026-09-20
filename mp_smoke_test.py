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
from sim import sprites

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
    flashed = 0          # peak muzzle flashes riding a body
    lights = 0           # peak muzzle-flash LIGHTS being cast
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
        flashed = max(flashed, len(view.shots))
        lights = max(lights, len(view.mlights))
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
    print(f"muzzle: peak {flashed} flash(es) drawn at a gun, "
          f"{lights} of them lighting the room")
    assert flashed > 0, "nobody's gun produced a muzzle flash"
    assert lights > 0, "a muzzle flash cast no light"
    if view.bank.ok:
        # the flash is two sprites pinned to the barrel, not a disc drawn at a
        # point in the world — if the art for this weapon has no registered
        # muzzle the flash silently does not happen
        art = sprites.weapon_art(view.loadout[view.wep])
        assert art in sprites.MUZZLE_PX, \
            f"{art} has no registered muzzle: the flash would not draw"
        assert view.bank.flash(view.surf, art, "big", 100, 100, 0.0), \
            "the muzzle-flash sprite would not draw"
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


def sound_reuse_test():
    """Solving a propagation field is the expensive thing this client does.

    A weapon on full auto makes ten noises a second from nearly one place, and
    solving each one separately queued 23 fields, pinned the active-sound list
    at its cap (so shots went unheard) and spiked frames to 54 ms. Each source
    now keeps a reusable field per kind of sound. These checks are about the
    things reuse could quietly break: that a source which MOVES still gets a
    fresh field, that a footstep never inherits a gunshot's loudness, and that
    changing the walls throws every cached field away.
    """
    import math
    import mp_client
    from net.combat_test import clear_pair
    from net.protocol import BTN_FIRE, BTN_INTERACT
    from sim import sound as sound_mod
    import main as renderer

    m, _ = netmaps.load_with_spawns("compound")
    a_pos, b_pos = clear_pair(m, 6.0)
    srv = GameServer([a_pos, b_pos], port=PORT + 3, map_id="compound",
                     duration_s=300)
    srv.start()
    a = GameClient("127.0.0.1", PORT + 3)
    b = GameClient("127.0.0.1", PORT + 3)
    assert a.connect("Listener", (220, 40, 40)), a.reject_reason
    assert b.connect("Noisy", (40, 40, 220)), b.reject_reason
    a.set_ready(True)
    b.set_ready(True)
    time.sleep(0.5)
    aid, bid = a.world.my_id, b.world.my_id
    srv.players[aid].x, srv.players[aid].y = a_pos
    srv.players[bid].x, srv.players[bid].y = b_pos
    time.sleep(0.2)

    pygame.init()
    view = mp_client.MatchView(a, "maps")
    view.px, view.py = a_pos
    view.rx, view.ry = a_pos
    view.read_input = lambda: (0.0, 0.0, 0)

    solves = {"n": 0}
    real_begin = sound_mod.begin

    def counting(cost, origin, energy, *args, **kw):
        solves["n"] += 1
        return real_begin(cost, origin, energy, *args, **kw)

    sound_mod.begin = counting
    renderer.sound.begin = counting

    heading = math.atan2(a_pos[1] - b_pos[1], a_pos[0] - b_pos[0])

    def blast_away(seconds, mx=0.0, my=0.0):
        t0 = time.monotonic()
        shots = 0
        while time.monotonic() - t0 < seconds:
            view.frame(1 / 60)
            b.send_input(mx, my, heading, BTN_FIRE, wep=2, mode=1, aim_dist=6.0)
            shots += 1
            time.sleep(0.004)
        return shots

    try:
        # --- a stationary shooter on full auto: one field, reused
        blast_away(4.0)
        stationary = solves["n"]
        fired = len([1 for s in view.sounds if s.label.endswith("/fire")])
        print(f"  4 s of full auto from one spot -> {stationary} field solve(s), "
              f"{len(view.sounds)} sounds tracked")
        assert stationary <= 2, \
            f"{stationary} solves for a shooter who never moved"
        assert view.sounds, "no sounds survived at all"
        assert len(view.sounds) < mp_client.MAX_ACTIVE_SOUNDS, \
            "the active-sound list is pinned at its cap again: sounds are " \
            "being dropped before they can be heard"

        # --- now make them run while firing: reuse must not follow them
        before = solves["n"]
        blast_away(3.0, mx=0.0, my=1.0)
        moved = solves["n"] - before
        print(f"  3 s of the same, while running -> {moved} new field solve(s)")
        assert moved >= 2, \
            "a shooter crossing the room kept reusing one field: sounds would " \
            "come from where they used to be"

        # --- a footstep must not ride a gunshot's field and play loud
        b.send_input(0.0, 0.0, heading, 0, wep=2, mode=1)
        for _ in range(30):
            view.frame(1 / 60)
            time.sleep(0.004)
        kinds = {k for (_id, k) in view.caches}
        print(f"  cached field kinds for this source: {sorted(kinds)}")
        assert "fire" in kinds and "step" in kinds, \
            f"footsteps and gunfire share a cached field: {sorted(kinds)}"

        # --- changing the walls throws the cache away, but not before the
        # walls actually change: a door with its panels still travelling is
        # still a wall, and every field solved against it is still correct
        assert view.caches, "no cached fields to invalidate"
        dkey = next(iter(view.doors))
        dd = view.doors.door(dkey)
        view._on_door({"r": dkey[0], "c": dkey[1], "open": True,
                       "dur": dd.dur, "id": bid, "blocked": False},
                      time.monotonic())
        assert dd.moving, "the door did not start travelling"
        assert view.caches, \
            "the cache was thrown away before the door had finished moving"
        t0 = time.monotonic()
        while dd.moving and time.monotonic() - t0 < 3.0:
            view.frame(1 / 60)
            time.sleep(0.004)
        print(f"  cache survived {dd.dur:.2f}s of travel, then the door "
              f"landed: {len(view.caches)} cached field(s), "
              f"{len(view.sounds)} sounds, {len(view.jobs)} jobs")
        assert not view.caches and not view.sounds and not view.jobs, \
            "fields solved against the old walls survived a door opening"

        print("\nSOUND REUSE CHECKS PASSED")
    finally:
        sound_mod.begin = real_begin
        renderer.sound.begin = real_begin
        a.disconnect()
        b.disconnect()
        srv.stop()


def late_join_test():
    """The path a real latecomer walks: connect during a match, land in it,
    step out with esc, and go back in from the lobby screen.

    net/rejoin_test.py covers the server side of this. What is checked here is
    the client half — that the lobby auto-joins a fresh connection, does NOT
    bounce someone who deliberately stepped out straight back in, and that the
    match render picks up a map that changed before it existed."""
    import pygame
    import lobby as L
    import mp_client
    from net.combat_test import clear_pair
    from net.protocol import BTN_INTERACT, ServerState
    from sim.tilemap import find_doors

    m, spawns = netmaps.load_with_spawns("compound")
    a_pos, _ = clear_pair(m, 4.0)
    srv = GameServer(spawns, port=PORT + 4, map_id="compound", duration_s=600)
    srv.start()
    a = GameClient("127.0.0.1", PORT + 4)
    b = GameClient("127.0.0.1", PORT + 4)
    late = None
    pygame.init()
    pygame.display.set_mode((L.W, L.H))
    fonts = L._make_fonts()
    surf = pygame.Surface((L.W, L.H))
    try:
        assert a.connect("Alice", (220, 40, 40)), a.reject_reason
        assert b.connect("Bob", (40, 40, 220)), b.reject_reason
        a.set_ready(True)
        b.set_ready(True)
        time.sleep(0.6)
        assert a.world.state == ServerState.MATCH
        aid = a.world.my_id

        # open a door before the latecomer exists
        door = next(iter(find_doors(m)))
        sp = srv.players[aid]
        sp.x, sp.y = door[1] + 0.5 + 1.2, door[0] + 0.5
        time.sleep(0.2)
        a.send_input(0.0, 0.0, 0.0, BTN_INTERACT)
        time.sleep(0.2)
        a.send_input(0.0, 0.0, 0.0, 0)
        time.sleep(0.3)
        assert srv.doors.get(door) is True, "could not open a door"

        # --- connect mid-match: the lobby should put us straight in
        late = GameClient("127.0.0.1", PORT + 4)
        assert late.connect("Carol", (40, 200, 60)), late.reject_reason
        lob = L.Lobby(late, fonts, maps_dir="maps", auto_join=True)
        t0 = time.monotonic()
        while lob.result is None and time.monotonic() - t0 < 4.0:
            lob.pump_events()
            lob.draw(surf)
            time.sleep(1 / 60)
        print(f"  fresh connection during a match -> lobby result "
              f"{lob.result.value if lob.result else None}")
        assert lob.result == L.LobbyResult.START, "a latecomer was left waiting"
        assert late.world.playing

        # --- the match render must inherit the door that is already open
        view = mp_client.MatchView(late, "maps")
        sub_n = view.m.subdiv
        assert view.doors.get(door) is True, \
            "the latecomer's client still thinks the door is shut"
        assert not view.m.blocks_sight[door[0] * sub_n, door[1] * sub_n], \
            "the open door still blocks sight for the latecomer"
        print("  its client inherited the open door, sight included")

        # --- esc: step out, and DON'T get dragged back in
        late.leave_match()
        t0 = time.monotonic()
        while late.world.playing and time.monotonic() - t0 < 3.0:
            time.sleep(0.05)
        assert not late.world.playing, "leaving did nothing"
        lob2 = L.Lobby(late, fonts, maps_dir="maps", auto_join=False)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.5:
            lob2.pump_events()
            lob2.draw(surf)
            time.sleep(1 / 60)
        print(f"  after stepping out, lobby result stays {lob2.result} "
              f"with {len(lob2._buttons)} buttons on screen")
        assert lob2.result is None, \
            "the lobby threw them straight back into the match they just left"

        # --- and the way back in
        lob2._join_match()
        t0 = time.monotonic()
        while lob2.result is None and time.monotonic() - t0 < 4.0:
            lob2.pump_events()
            lob2.draw(surf)
            time.sleep(1 / 60)
        print(f"  pressing JOIN MATCH -> {lob2.result.value if lob2.result else None}")
        assert lob2.result == L.LobbyResult.START, "could not rejoin"
        assert late.world.playing
        print("\nLATE JOIN CHECKS PASSED")
    finally:
        for cl in (a, b, late):
            if cl is not None:
                cl.disconnect()
        srv.stop()


def colour_test():
    """Players wear their colour as ART, not as a filter over one sprite.

    The filter was built first and measured well on paper — a multiply plus an
    additive wash put the closest pair of swatches 36.5 apart on the real khaki
    art. It still looked like what it was, a coloured pane held over a
    photograph, so the scheme is now one painted sprite per colour. What is
    worth guarding is the wiring: every swatch has a name of its own, the name
    picks the file, and a missing file falls back instead of crashing.
    """
    import mp_client
    from lobby import SWATCHES
    from sim import sprites

    pygame.init()
    pygame.display.set_mode((320, 320))

    names = [sprites.colour_name(c) for c in SWATCHES]
    print(f"{len(SWATCHES)} lobby colours -> {', '.join(names)}")
    assert len(set(names)) == len(SWATCHES), \
        f"two lobby colours share one sprite set: {names}"
    for name, rgb in sprites.PALETTE:
        assert sprites.colour_name(rgb) == name, \
            f"{rgb} is the {name} swatch but resolves to {sprites.colour_name(rgb)}"
    # a colour nobody authored still has to land somewhere sensible
    assert sprites.colour_name((210, 30, 30)) == "red"

    bank = sprites.SpriteBank()
    bank.load()
    if not bank.ok:
        print("no sprite bank available, art check skipped")
        return
    bank.set_scale(48 / sprites.ART_PPM)

    have = bank.colours()
    print(f"  coloured soldier art present: {have or 'none yet'}")
    missing = [n for n, _ in sprites.PALETTE if n not in have]
    for name in missing:
        assert bank.body_art("soldier_ready", name) == "soldier_ready", \
            f"{name} has no art but body_art did not fall back"
    for name in have:
        assert bank.body_art("soldier_ready", name) == f"soldier_ready_{name}"

    # prove the lookup actually reaches the file: stand in a fake red set and
    # check the body that comes out is a different body
    if "red" in missing:
        red = bank.raw["soldier_ready"].copy()
        red.fill((150, 0, 0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        bank.raw["soldier_ready_red"] = red
        bank._cache.clear()
        assert bank.body_art("soldier_ready", "red") == "soldier_ready_red"

        def mean(art):
            surf = pygame.Surface((200, 200))
            surf.fill((0, 0, 0))
            bank.blit(surf, art, 100, 100, 0.0)
            a = pygame.surfarray.array3d(surf).astype(float)
            return a[a.sum(axis=2) > 24].mean(axis=0)

        base, painted = mean("soldier_ready"), mean("soldier_ready_red")
        print(f"  base {base.round(0)} vs a red set {painted.round(0)}")
        assert painted[0] > base[0] + 40, "the coloured set is not being used"
        del bank.raw["soldier_ready_red"]
        bank._cache.clear()

    # light, not colour, decides how bright a body is drawn
    dark = mp_client.MatchView._light_tint(0.0)
    lit = mp_client.MatchView._light_tint(1.0)
    half = mp_client.MatchView._light_tint(0.5)
    print(f"  body tint: unlit {dark[0]}, half-lit {half[0]}, lit {lit[0]}")
    assert dark[0] < 10, "an unlit body is not a silhouette"
    assert lit[0] > 245, "a fully lit body is not drawn at full brightness"
    assert dark[0] < half[0] < lit[0], "the light curve is not monotonic"
    print("\nCOLOUR CHECKS PASSED")


def lighting_test():
    """A dark room is cover, and a light is a decision with a cost.

    Multiplayer used to draw every player at full brightness wherever they
    stood, which quietly deleted the stealth half of the game: there was no
    point in a flashlight because there was never any dark. These checks are on
    the consequences — that darkness actually hides, that a beam actually
    reveals, that somebody else's beam reaches you, and that a muzzle flash
    lights a room without committing it to memory.
    """
    import numpy as np
    import mp_client
    from net.combat_test import clear_pair
    from net.protocol import BTN_FIRE, BTN_LIGHT

    m, _ = netmaps.load_with_spawns("compound")
    a_pos, b_pos = clear_pair(m, 6.0)
    srv = GameServer([a_pos, b_pos], port=PORT + 5, map_id="compound",
                     duration_s=300)
    srv.start()
    a = GameClient("127.0.0.1", PORT + 5)
    b = GameClient("127.0.0.1", PORT + 5)
    try:
        assert a.connect("Looker", (220, 40, 40)), a.reject_reason
        assert b.connect("Lurker", (40, 40, 220)), b.reject_reason
        a.set_ready(True)
        b.set_ready(True)
        time.sleep(0.5)
        aid, bid = a.world.my_id, b.world.my_id
        srv.players[aid].x, srv.players[aid].y = a_pos
        srv.players[bid].x, srv.players[bid].y = b_pos
        time.sleep(0.2)

        pygame.init()
        view = mp_client.MatchView(a, "maps")
        view.px, view.py = view.rx, view.ry = a_pos
        view.facing = math.atan2(b_pos[1] - a_pos[1], b_pos[0] - a_pos[0])
        view.read_input = lambda: (0.0, 0.0, 0)
        # every lamp off: whatever this corner of compound happens to be lit
        # like, the test is about the dark
        view.m.lightmap = np.full(view.m.blocks_sight.shape, 0.05, np.float32)

        def torch(cli, pid, want):
            """Hold the light key until the server agrees. The toggle is on the
            key going down, so a release has to go out first."""
            cli.send_input(0.0, 0.0, view.facing, 0, aim_dist=6.0)
            time.sleep(0.06)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 2.0:
                p = a.world.players.get(pid)
                if p is not None and p.flashlight == want:
                    return True
                cli.send_input(0.0, 0.0, view.facing, BTN_LIGHT, aim_dist=6.0)
                time.sleep(0.02)
            return False

        def look():
            vf, raw = view._visible_mask()
            gated, add = view._light_pass(vf, raw, time.monotonic())
            return vf, raw, gated, add

        vf, raw, dark, add = look()
        print(f"  line of sight reaches {int((raw > 0.03).sum())} cells; "
              f"unlit, {int((dark > 0.03).sum())} of them register")
        assert raw.max() > 0.5, "the vision cone itself is empty"
        assert dark.max() == 0.0, "an unlit room is still being seen"
        assert add is None, "something is brightening a room with no light in it"
        assert not view.known.any(), "darkness wrote itself into fog memory"

        # --- your own torch
        assert torch(a, aid, True), "the server never lit the lamp"
        vf, raw, lit, add = look()
        print(f"  torch on -> {int((lit > 0.03).sum())} cells register, "
              f"world brighten peaks at {add.max():.0f}")
        assert lit.max() > 0.5, "the flashlight reveals nothing"
        assert (lit > 0.03).sum() < (raw > 0.03).sum(), \
            "the beam lights everything in sight — it is not a beam"
        assert add is not None and add.max() > 1.0

        # a beam is directional: look the other way, lose what you were seeing
        seen_ahead = int((lit > 0.03).sum())
        view.facing += math.pi
        _vf, _raw, behind, _add = look()
        view.facing -= math.pi
        print(f"  turning away: {seen_ahead} cells -> "
              f"{int((behind > 0.03).sum())}")

        # --- somebody else's torch, cast from where THEY are
        assert torch(b, bid, True), "the other lamp never lit"
        look()
        assert view.beams is not None, \
            "another player's flashlight casts no light on this client"
        print(f"  the other player's beam covers "
              f"{int((view.beams > 0.03).sum())} cells of what you can see")

        # --- a muzzle flash: light now, nothing remembered. Both lamps off
        # first, or the beams are what grows fog memory and the check proves
        # nothing.
        assert torch(a, aid, False) and torch(b, bid, False), \
            "a lamp would not go out"
        look()
        view.mlights.clear()
        known_before = int(view.known.sum())
        t0 = time.monotonic()
        while not view.mlights and time.monotonic() - t0 < 2.0:
            b.send_input(0.0, 0.0, view.facing + math.pi, BTN_FIRE,
                         wep=2, mode=1, aim_dist=6.0)
            for e in a.drain_events():
                if e["t"] == "shot":
                    view._on_shot(e, time.monotonic())
            time.sleep(0.01)
        assert view.mlights, "nobody's gun made any light"
        look()
        assert view.mflash is not None, "a muzzle flash is not lighting anything"
        lit_by_flash = int((view.mflash > 0.004).sum())
        print(f"  a shot lights {lit_by_flash} cells for "
              f"{view.mlights[0]['life'] * 1000:.0f} ms")
        assert lit_by_flash > 0
        assert int(view.known.sum()) == known_before, \
            "a muzzle flash wrote a room into fog memory"

        # a body under that flash is drawn brighter than one in the dark
        fcx, fcy = view.m.cell_of(view.mlights[0]["x"], view.mlights[0]["y"])
        near = view._illum_at(vf, fcx, fcy)
        print(f"  body light at the muzzle {near:.2f} vs ambient 0.05")
        assert near > 0.05, "the flash does not light the shooter"
        print("\nLIGHTING CHECKS PASSED")
    finally:
        a.disconnect()
        b.disconnect()
        srv.stop()


if __name__ == "__main__":
    rc = main()
    print()
    colour_test()
    print()
    sound_reuse_test()
    print()
    late_join_test()
    print()
    lighting_test()
    sys.exit(rc)
