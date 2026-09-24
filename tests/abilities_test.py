"""Headless test of the class abilities.

One special per class, space to use it, thirty seconds between uses measured
from the moment it ENDS:

  Medic   channels a heal while standing still. It pays out as it goes, and
          moving, firing or being hit stops it where it is.
  Commando sprints at double speed, and an empty gun reloads itself out of
          the ammo actually carried.
  Tech    pings: the client opens the fog around it for three seconds and
          keeps those cells in fog memory afterwards.
  Heavy   braces: half damage taken, but only while standing still.

What this checks:

  * each class gets its own ability, and the cooldown holds
  * the heal pays out over the channel and stops when interrupted, keeping
    what it already gave
  * blitz doubles the speed the server moves you at, refills an empty gun
    from the reserve, and stops when the reserve is empty
  * brace halves damage while still, and only while still
  * a ping opens the veil around it and leaves those cells remembered
  * knocking works over the network (it used to be single-player only)

Run from the project root:   python -m tests.abilities_test
"""
import math
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from net import GameServer, GameClient
from net.protocol import BTN_ABILITY, BTN_FIRE, BTN_KNOCK, ServerState
from sim import classes, weapons
from tests import range_spots as R

PORT = 47978
OPEN_SPOT = (10.5, 10.5)
AWAY = (29.5, 10.5)


def wait_for(pred, seconds=4.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if pred():
            return True
        time.sleep(0.02)
    return False


class Match:
    """Two players in a match on the range, with whatever classes you name."""

    def __init__(self, cls_a="medic", cls_b="commando", port=PORT):
        self.srv = GameServer([OPEN_SPOT, AWAY], port=port, map_id=R.MAP_ID,
                              maps_dir=R.MAPS_DIR, duration_s=600)
        self.srv.start()
        self.a = GameClient("127.0.0.1", port)
        self.b = GameClient("127.0.0.1", port)
        assert self.a.connect("A", (220, 40, 40)), self.a.reject_reason
        assert self.b.connect("B", (40, 40, 220)), self.b.reject_reason
        self.a.set_class(cls_a)
        self.b.set_class(cls_b)
        time.sleep(0.2)
        self.a.set_ready(True)
        self.b.set_ready(True)
        assert wait_for(lambda: self.a.world.state == ServerState.MATCH), "no match"
        self.pa = self.srv.players[self.a.world.my_id]
        self.pb = self.srv.players[self.b.world.my_id]

    def close(self):
        self.a.disconnect()
        self.b.disconnect()
        self.srv.stop()

    def hold(self, cli, seconds, mx=0.0, my=0.0, buttons=0, wep=0):
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            cli.send_input(mx, my, 0.0, buttons, wep=wep, aim_dist=5.0)
            time.sleep(1 / 30)

    def use(self, cli):
        """Press space, release it."""
        self.hold(cli, 0.12, buttons=BTN_ABILITY)
        self.hold(cli, 0.12)


def table_test():
    got = {k: classes.ability_of(k).key for k in classes.ORDER}
    print(f"  {got}")
    assert got == {"commando": "blitz", "heavy_support": "brace",
                   "medic": "heal", "tech": "ping", "saboteur": "vanish"}
    assert classes.get("heavy_support").loadout == (
        "pistol", "flak_cannon", "rocket_launcher")
    assert classes.get("heavy_support").spd < 3.2, \
        "the Heavy was supposed to slow down"
    print(f"  the Heavy: {classes.stat_line('heavy_support')} | "
          f"{classes.weapon_line('heavy_support')}")
    print("\nTABLE CHECKS PASSED")


def heal_test():
    m = Match("medic", "commando")
    try:
        p = m.pa
        full = p.body.max_health
        p.body.health = full * 0.2
        m.use(m.a)
        assert wait_for(lambda: p.ability_until > 0.0), "the heal never started"
        m.hold(m.a, 1.0)
        part = p.body.health
        print(f"  a second into the channel: {full * 0.2:.0f} -> {part:.0f} "
              f"of {full:.0f}")
        assert part > full * 0.25, "the heal pays out nothing as it goes"
        # --- moving stops it, and what it already gave stays given
        m.hold(m.a, 0.3, mx=1.0)
        assert wait_for(lambda: p.ability_until == 0.0, 1.0), \
            "walking did not interrupt the channel"
        stopped = p.body.health
        m.hold(m.a, 0.5)
        assert abs(p.body.health - stopped) < 1e-6, "it kept healing after it stopped"
        assert stopped >= part, "the interrupted heal took back what it gave"
        print(f"  walked away at {stopped:.0f}: no more, and none taken back")

        # --- and the cooldown holds, from the moment it stopped
        m.use(m.a)
        time.sleep(0.2)
        assert p.ability_until == 0.0, "used it again inside the cooldown"
        print(f"  {p.ability_ready_at - time.monotonic():.0f}s still to wait")

        # --- being shot also stops it
        p.ability_ready_at = 0.0
        p.body.health = full * 0.2
        m.use(m.a)
        assert wait_for(lambda: p.ability_until > 0.0)
        m.hold(m.a, 0.3)
        p.body.take(10.0, time.monotonic())
        m.hold(m.a, 0.3)
        assert p.ability_until == 0.0, "a heal survived being shot"
        print("  shot mid-channel: the heal stops")
    finally:
        m.close()
    print("\nHEAL CHECKS PASSED")


def blitz_test():
    m = Match("commando", "medic", port=PORT + 1)
    try:
        p = m.pa

        def run_east(seconds):
            p.x, p.y = OPEN_SPOT
            start = p.x
            m.hold(m.a, seconds, mx=1.0)
            return p.x - start
        plain = run_east(1.0)
        m.use(m.a)
        assert wait_for(lambda: p.ability_until > 0.0), "blitz never started"
        fast = run_east(1.0)
        print(f"  a second's walk: {plain:.2f} m normally, {fast:.2f} m "
              f"blitzed ({fast / plain:.2f}x)")
        assert 1.8 < fast / plain < 2.2, "blitz is not double speed"

        # --- an empty gun refills itself, out of what is carried
        p.x, p.y = OPEN_SPOT
        w = weapons.ROSTER[p.loadout[0]]
        p.mags[0] = 0
        p.reserves[0] = w.mag + 2
        m.hold(m.a, 0.4, buttons=BTN_FIRE)
        print(f"  empty magazine during blitz: {p.mags[0]} rounds back, "
              f"{p.reserves[0]} left in reserve")
        assert p.mags[0] > 0, "an empty gun did not reload itself"
        assert p.reserves[0] < w.mag + 2, "the rounds came from nowhere"

        # --- with nothing in reserve it stays empty
        p.mags[0] = 0
        p.reserves[0] = 0
        m.hold(m.a, 0.4, buttons=BTN_FIRE)
        assert p.mags[0] == 0, "blitz invented ammo"
        print("  with an empty reserve it stays empty")

        # --- and it ends by itself
        assert wait_for(lambda: p.ability_until == 0.0,
                        classes.ABILITIES["blitz"].duration + 2.0), \
            "blitz never ended"
        p.x, p.y = OPEN_SPOT
        after = run_east(0.6)
        assert after / 0.6 < plain / 1.0 * 1.2, "still fast after it ended"
        print("  ended on its own; back to normal speed")
    finally:
        m.close()
    print("\nBLITZ CHECKS PASSED")


def brace_test():
    m = Match("heavy_support", "commando", port=PORT + 2)
    try:
        p = m.pa
        p.body.shields = 0.0

        def shoot(dmg=20.0):
            before = p.body.health
            p.body.take(dmg, time.monotonic())
            return before - p.body.health
        plain = shoot()
        m.use(m.a)
        assert wait_for(lambda: p.ability_until > 0.0), "brace never started"
        m.hold(m.a, 0.2)
        braced = shoot()
        print(f"  20 damage: {plain:.1f} standing, {braced:.1f} braced")
        assert braced < plain * 0.75, "brace did not soften the hit"

        # --- moving gives it up
        m.hold(m.a, 0.3, mx=1.0)
        moved_hit = shoot()
        print(f"  the same hit while walking: {moved_hit:.1f}")
        assert moved_hit > braced * 1.4, "brace still protected a moving player"

        # --- standing still again re-applies it, until the six seconds are up
        m.hold(m.a, 0.3)
        assert shoot() < plain * 0.75, "standing still again did not brace"
        assert wait_for(lambda: p.ability_until == 0.0,
                        classes.ABILITIES["brace"].duration + 2.0)
        assert p.body.incoming_mult == 1.0, "brace outlived its duration"
        print("  it ends with the timer, whatever you are doing")
    finally:
        m.close()
    print("\nBRACE CHECKS PASSED")


def ping_test():
    import mp_client
    import pygame
    m = Match("tech", "commando", port=PORT + 3)
    try:
        pygame.init()
        view = mp_client.MatchView(m.a, R.MAPS_DIR)
        view.read_input = lambda: (0.0, 0.0, 0)
        m.pa.x, m.pa.y = OPEN_SPOT
        view.px, view.py = view.rx, view.ry = OPEN_SPOT
        for _ in range(8):
            view.frame(1 / 60)
        # somewhere in range but behind us, so line of sight is not what shows it
        cx, cy = view.m.cell_of(OPEN_SPOT[0] - 8.0, OPEN_SPOT[1])
        view.known[cy, cx] = False
        assert view.ping is None
        m.use(m.a)
        assert wait_for(lambda: m.pa.ability_until > 0.0), "the ping never fired"
        for _ in range(10):
            view.frame(1 / 60)
            time.sleep(0.01)
        assert view.ping is not None, "the client never drew the ping"
        assert view.known[cy, cx], \
            "the ping did not show a cell 8 m away with no line of sight"
        assert view._ping_visible(OPEN_SPOT[0] - 8.0, OPEN_SPOT[1])
        assert not view._ping_visible(OPEN_SPOT[0] - 20.0, OPEN_SPOT[1]), \
            "the ping reaches further than it should"
        print(f"  ping up: cells {classes.PING_RADIUS_M:.0f} m away are shown "
              f"and remembered")
        t0 = time.monotonic()
        while view.ping is not None and time.monotonic() - t0 < 6.0:
            view.frame(1 / 60)
            time.sleep(0.01)
        assert view.ping is None, "the ping never faded"
        assert view.known[cy, cx], "what the ping showed was forgotten again"
        print(f"  faded after {time.monotonic() - t0:.1f}s, and what it showed "
              f"stays in fog memory")
    finally:
        m.close()
    print("\nPING CHECKS PASSED")


def knock_test():
    m = Match("commando", "medic", port=PORT + 4)
    try:
        m.b.drain_events()
        m.hold(m.a, 0.15, buttons=BTN_KNOCK)
        m.hold(m.a, 0.15)
        heard = [e for e in m.b.drain_events()
                 if e["t"] == "sound" and e.get("clip") == "knock"]
        print(f"  knocking over the network: the other client got "
              f"{len(heard)} knock event(s)")
        assert heard, "nobody heard the knock"
    finally:
        m.close()
    print("\nKNOCK CHECKS PASSED")


if __name__ == "__main__":
    table_test()
    print()
    heal_test()
    print()
    blitz_test()
    print()
    brace_test()
    print()
    ping_test()
    print()
    knock_test()
    print("\nALL ABILITY CHECKS PASSED")
