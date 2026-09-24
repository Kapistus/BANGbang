"""Headless test of the heavy and energy weapons, the ones that do not simply
put a line between a muzzle and a hole.

Three of them go through the travelling-round path, and that path is where the
bugs live: a rocket that goes off at the muzzle, a plasma bolt that never
arrives, a flak shell that behaves like a burst of frag rounds. What this
checks:

  * the rocket flies. It takes the time its speed says it should to cross a
    room, it goes off where it arrives, and the man who fired it is not the
    one it kills
  * the plasma rifle carries five bolts and no spare cells, makes itself a new
    one every five seconds whether it is held or slung, and tells the client
    how far along that round is so the bar can be drawn
  * the flak shell bursts where it stops - at what it hit, or at the end of
    its run - into a ring of pellets that catches somebody standing off to one
    side, and that a wall stops dead
  * the pistol's trigger is as fast as the hand

Run from the project root:   python -m tests.weapons_test
"""
import math
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from net import GameServer, GameClient
from net.protocol import BTN_FIRE, ServerState
from sim import classes, weapons
from tests import range_spots as R

PORT = 47985
SPOT = (10.5, 10.5)
AWAY = (29.5, 10.5)

HEAVY = classes.get("heavy_support").loadout
TECH = classes.get("tech").loadout
FLAK = HEAVY.index("flak_cannon")
ROCKET = HEAVY.index("rocket_launcher")
PLASMA = TECH.index("laser_rifle")


def wait_for(pred, seconds=4.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if pred():
            return True
        time.sleep(0.02)
    return False


class Match:
    def __init__(self, port=PORT, cls_a="heavy_support", cls_b="commando"):
        self.srv = GameServer([SPOT, AWAY], port=port, map_id=R.MAP_ID,
                              maps_dir=R.MAPS_DIR, duration_s=600)
        self.srv.start()
        self.a = GameClient("127.0.0.1", port)
        self.b = GameClient("127.0.0.1", port)
        assert self.a.connect("Gun", (220, 40, 40)), self.a.reject_reason
        assert self.b.connect("Mark", (40, 40, 220)), self.b.reject_reason
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

    def hold(self, cli, seconds, mx=0.0, my=0.0, aim=0.0, buttons=0, wep=0):
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            cli.send_input(mx, my, aim, buttons, wep=wep, aim_dist=12.0)
            time.sleep(1 / 30)

    def arm(self, wep):
        """Get `wep` into the shooter's hands and ready to fire. A weapon
        still coming up does not fire, which reads as a weapon that is broken.
        """
        self.hold(self.a, 0.8, wep=wep)
        t0 = time.monotonic()
        while ((self.pa.fire_cd > 0.0 or self.pa.swap_t > 0.0)
               and time.monotonic() - t0 < 3.0):
            self.hold(self.a, 0.05, wep=wep)
        assert self.pa.wi == wep, f"the weapon never came up: {self.pa.wi}"

    def shoot(self, wep, aim=0.0, hold=0.2):
        self.hold(self.a, hold, aim=aim, buttons=BTN_FIRE, wep=wep)
        self.hold(self.a, 0.05, aim=aim, wep=wep)


def rocket_test():
    m = Match()
    try:
        p, q = m.pa, m.pb
        w = weapons.ROSTER["rocket_launcher"]
        p.x, p.y = 3.5, 10.5
        q.x, q.y = 23.5, 10.5           # twenty metres down the hall
        q.body.shields = 0.0
        before_q, before_p = q.body.health, p.body.health
        m.arm(ROCKET)
        t0 = time.monotonic()
        m.shoot(ROCKET)
        assert m.srv._projectiles, "the rocket never left the tube"

        # it is in the air, and it is not going off next to the man holding it
        m.hold(m.a, 0.25, wep=ROCKET)
        assert p.body.health == before_p, "it went off in the shooter's face"

        assert wait_for(lambda: q.body.health < before_q, 4.0), "it never arrived"
        flight = time.monotonic() - t0
        want = 20.0 / w.projectile_speed
        print(f"  twenty metres in {flight:.2f}s (the tube says {want:.2f}s), "
              f"{before_q - q.body.health:.0f} damage")
        assert want * 0.5 < flight < want + 0.6, "that is not its flight time"
        assert p.body.health == before_p, "the shooter took his own rocket"
        print("\nROCKET CHECKS PASSED")
    finally:
        m.close()


def plasma_test():
    m = Match(port=PORT + 1, cls_a="tech")
    try:
        p, q = m.pa, m.pb
        w = weapons.ROSTER["laser_rifle"]
        assert w.mag == 5 and w.reserve == 0 and w.recharge_s > 0.0, \
            "five bolts, no spare cells, and it makes its own"
        p.x, p.y = 3.5, 10.5
        q.x, q.y = 23.5, 10.5
        q.body.shields = 0.0
        before = q.body.health
        m.arm(PLASMA)
        assert p.mags[PLASMA] == 5

        # empty it, and check the bolts are still arriving twenty metres out
        for _ in range(5):
            m.shoot(PLASMA)
            t0 = time.monotonic()
            while p.fire_cd > 0.0 and time.monotonic() - t0 < 2.0:
                m.hold(m.a, 0.05, wep=PLASMA)
        assert wait_for(lambda: q.body.health < before, 3.0), \
            "not one bolt crossed the hall"
        print(f"  five bolts at twenty metres: {before - q.body.health:.0f} damage")
        assert wait_for(lambda: p.mags[PLASMA] == 0, 3.0), \
            f"it still has {p.mags[PLASMA]} rounds"

        # empty, no reserve, and nothing a reload can do about it
        assert p.reserves[PLASMA] == 0
        part = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < w.recharge_s * 1.4:
            m.hold(m.a, 0.2, wep=PLASMA)
            seen = m.b.world.players[m.a.world.my_id].recharge
            if 0.15 < seen < 0.95:
                part = seen
            if p.mags[PLASMA] > 0:
                break
        assert p.mags[PLASMA] == 1, f"no round came back: {p.mags[PLASMA]}"
        assert part is not None, "the client was never told it was charging"
        print(f"  the bar was seen part-way at {part:.2f}")

        # and the rate: the round after that is a clean five seconds
        t1 = time.monotonic()
        while p.mags[PLASMA] < 2 and time.monotonic() - t1 < w.recharge_s * 2:
            m.hold(m.a, 0.2, wep=PLASMA)
        gap = time.monotonic() - t1
        print(f"  the next round took {gap:.1f}s against {w.recharge_s:.0f}s")
        assert p.mags[PLASMA] == 2 and abs(gap - w.recharge_s) < 1.0, \
            "that is not one round every five seconds"
        print("\nPLASMA CHECKS PASSED")
    finally:
        m.close()


def flak_test():
    m = Match(port=PORT + 2)
    try:
        p, q = m.pa, m.pb
        w = weapons.ROSTER["flak_cannon"]
        assert w.burst_pellets > 0 and w.projectile_speed > 0.0, \
            "the shell should fly and come apart, not fire three frag rounds"

        # --- the ring: it bursts at the end of its run, and catches somebody
        # standing off the line it flew down
        p.x, p.y = 3.5, 10.5
        q.x, q.y = 3.5 + w.range_m, 12.4     # two metres off the shell's path
        q.body.shields = 0.0
        before = q.body.health
        m.arm(FLAK)
        m.shoot(FLAK)
        assert wait_for(lambda: q.body.health < before, 3.0), \
            "the burst missed a man standing two metres off it"
        print(f"  off the line by two metres: {before - q.body.health:.0f} damage")

        # --- and the shooter is not in his own ring at twelve metres
        assert p.body.health == p.body.max_health, "he caught his own burst"

        # --- cover: a wall between you and the burst is the whole answer
        p.x, p.y = 2.5, 23.5                 # the wall lane
        q.x, q.y = 10.5, 23.5                # behind the wall in column 8
        q.body.health = q.body.max_health
        q.body.shields = 0.0
        safe = q.body.health
        m.arm(FLAK)
        m.shoot(FLAK)
        time.sleep(1.5)
        m.hold(m.a, 0.3, wep=FLAK)
        assert q.body.health == safe, \
            f"pellets went through a wall: {safe - q.body.health:.0f} damage"
        print("  behind a wall: nothing")
        print("\nFLAK CHECKS PASSED")
    finally:
        m.close()


def pistol_test():
    m = Match(port=PORT + 3, cls_a="commando")
    try:
        p = m.pa
        w = weapons.ROSTER["pistol"]
        assert w.rof >= 2.5, "the sidearm should not be the slow part"
        p.x, p.y = 3.5, 10.5
        m.arm(0)
        fired = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            was = p.mags[0]
            m.hold(m.a, 1 / 30, buttons=BTN_FIRE, wep=0)
            m.hold(m.a, 1 / 30, wep=0)
            fired += max(0, was - p.mags[0])
        rate = fired / (time.monotonic() - t0)
        print(f"  {fired} shots in two seconds: {rate:.1f}/s against {w.rof:.1f}")
        assert rate > w.rof * 0.7, "the trigger is slower than the weapon says"
        print("\nPISTOL CHECKS PASSED")
    finally:
        m.close()


if __name__ == "__main__":
    rocket_test()
    plasma_test()
    flak_test()
    pistol_test()
    print("\nALL WEAPON TESTS PASSED")
