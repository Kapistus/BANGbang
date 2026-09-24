"""Headless test of sprint fuel in multiplayer, and of what a thrown round
looks like on the wire.

Sprinting is not free. The same model runs in both modes (sim/movement.py), but
only the server's copy decides anything: a client that kept sprinting on an
empty tank would simply be corrected. This checks the server's copy, and that
the number reaches the other machine to be drawn.

What this checks:

  * running burns fuel, and empty locks the sprint out
  * a locked player moves at walking speed even while holding shift
  * standing still refills faster than walking does, and the lock lifts at
    STAMINA_UNLOCK rather than at the first drop of fuel
  * the snapshot carries it, so the bar under the health bar is the server's
    figure and not a guess
  * respawning hands back a full tank
  * a thrown grenade broadcasts no tracer segments and a fuse, and goes off
    exactly once - the throw draws the round, a later message blows it up

Run from the project root:   python -m tests.stamina_test
"""
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from net import GameServer, GameClient
from net.protocol import BTN_FIRE, BTN_RUN, ServerState
from sim import classes, movement
from tests import range_spots as R

PORT = 47983
SPOT = (10.5, 10.5)
AWAY = (29.5, 10.5)
# the grenade's slot, read off the class rather than written down: the
# Saboteur carries no gun, so it is not where it used to be
NADE = classes.get("saboteur").loadout.index("frag_grenade")


def wait_for(pred, seconds=4.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if pred():
            return True
        time.sleep(0.02)
    return False


class Match:
    def __init__(self, port=PORT, cls_a="commando", cls_b="commando"):
        self.srv = GameServer([SPOT, AWAY], port=port, map_id=R.MAP_ID,
                              maps_dir=R.MAPS_DIR, duration_s=600)
        self.srv.start()
        self.a = GameClient("127.0.0.1", port)
        self.b = GameClient("127.0.0.1", port)
        assert self.a.connect("Runner", (220, 40, 40)), self.a.reject_reason
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
            cli.send_input(mx, my, aim, buttons, wep=wep, aim_dist=3.0)
            time.sleep(1 / 30)

    def ready(self, wep=0):
        t0 = time.monotonic()
        while ((self.pa.fire_cd > 0.0 or self.pa.swap_t > 0.0)
               and time.monotonic() - t0 < 3.0):
            self.hold(self.a, 0.05, wep=wep)


def model_test():
    """The shared model on its own, before any networking is involved."""
    s, locked = 1.0, False
    for _ in range(int(6.5 / 0.05)):
        s, locked = movement.step_stamina(s, locked, "run", True, 0.05)
    assert s == 0.0 and locked, f"six seconds of running should empty it: {s}"
    assert movement.allowed_stance("run", s, locked) == "walk"

    # half a second of standing still is fuel, but not permission
    for _ in range(10):
        s, locked = movement.step_stamina(s, locked, "walk", False, 0.05)
    assert s > 0.0 and locked, "the lock lifts at a quarter tank, not at a drop"
    assert movement.allowed_stance("run", s, locked) == "walk"

    while locked:
        s, locked = movement.step_stamina(s, locked, "walk", False, 0.05)
    assert s >= movement.STAMINA_UNLOCK
    assert movement.allowed_stance("run", s, locked) == "run"

    # standing still beats walking it off
    idle, _ = movement.step_stamina(0.5, False, "walk", False, 1.0)
    walk, _ = movement.step_stamina(0.5, False, "walk", True, 1.0)
    assert idle > walk, "standing still should refill faster than walking"
    print("  empty in 6.5s, locked until a quarter tank, idle refills fastest")
    print("\nMODEL CHECKS PASSED")


def drain_test():
    m = Match()
    try:
        p = m.pa
        full = p.stamina
        assert full == 1.0, f"a match should start with a full tank: {full}"

        # sprint into the open: two seconds of it should cost real fuel
        m.hold(m.a, 2.0, mx=1.0, buttons=BTN_RUN)
        after = p.stamina
        spent = full - after
        print(f"  two seconds of sprinting spent {spent:.2f} of the tank")
        assert 0.2 < spent < 0.45, f"drain rate looks wrong: {spent:.2f}"
        assert not p.sprint_locked

        # the client is told, rather than left to work it out
        seen = m.b.world.players[m.a.world.my_id].stamina
        assert abs(seen - after) < 0.12, \
            f"the other client sees {seen:.2f}, the server has {after:.2f}"

        # run it dry. Once it bottoms out the sprint locks, the stance drops
        # to a walk and the tank starts refilling, so the lock - not a reading
        # of exactly zero - is what says it ran out.
        t0 = time.monotonic()
        while not p.sprint_locked and time.monotonic() - t0 < 12.0:
            m.hold(m.a, 0.1, mx=1.0, my=0.3, buttons=BTN_RUN)
        assert p.sprint_locked, "it should bottom out and lock"
        assert p.stamina < 0.1, f"locked with {p.stamina:.2f} still in it"
        print(f"  emptied after {time.monotonic() - t0:.1f}s more of running")

        # shift held, tank empty: this is a walk, whatever the keyboard says.
        # Back the way it came, so a wall is not what is holding it up.
        x0, y0 = p.x, p.y
        m.hold(m.a, 0.6, mx=-1.0, buttons=BTN_RUN)
        speed = ((p.x - x0) ** 2 + (p.y - y0) ** 2) ** 0.5 / 0.6
        print(f"  empty and holding shift moves at {speed:.1f} m/s")
        assert speed > 0.5, "it did not move at all: this measures nothing"
        assert speed < movement.SPEED_WALK * 1.35, "still sprinting on empty"

        # stand still and it comes back, and the lock lifts part-way up
        m.hold(m.a, 1.8)
        assert p.stamina > 0.0
        assert not p.sprint_locked, "the lock should lift by a quarter tank"
        print(f"  1.8s of standing still refilled it to {p.stamina:.2f}")
        print("\nDRAIN CHECKS PASSED")
    finally:
        m.close()


def respawn_test():
    m = Match(port=PORT + 1)
    try:
        p = m.pa
        t0 = time.monotonic()
        while p.stamina > 0.3 and time.monotonic() - t0 < 12.0:
            m.hold(m.a, 0.2, mx=1.0, my=0.2, buttons=BTN_RUN)
        assert p.stamina <= 0.3, "could not run it down"
        m.srv.kill(m.pb, p)
        assert wait_for(lambda: not p.alive, 2.0), "did not die"
        p.respawn_at = time.monotonic() + 0.3      # no need to sit out the wait
        assert wait_for(lambda: p.alive, 8.0), "never came back"
        time.sleep(0.2)
        assert p.stamina > 0.9 and not p.sprint_locked, \
            f"a fresh body should have a fresh tank: {p.stamina:.2f}"
        print(f"  came back with {p.stamina:.2f} in the tank")
        print("\nRESPAWN CHECKS PASSED")
    finally:
        m.close()


def thrown_round_test():
    """What a grenade looks like to the other machine.

    A bullet is a line drawn between where it left and where it hit. A thrown
    round is not: the round itself is the thing you see, so sending segments
    for one draws a tracer out of an arm. And it must go off once - the throw
    message carries the fuse so the round can be drawn lying there, and the
    server sends the blast when the fuse ends.
    """
    m = Match(port=PORT + 2, cls_a="saboteur")
    try:
        m.b.drain_events()
        m.hold(m.a, 0.8, wep=NADE)        # let the grenade actually come out
        m.ready(wep=NADE)
        m.hold(m.a, 0.2, aim=0.0, buttons=BTN_FIRE, wep=NADE)
        m.hold(m.a, 0.25, aim=0.0, wep=NADE)

        throws, blasts = [], []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6.0:
            for e in m.b.drain_events():
                if e["t"] != "shot" or e.get("wep") != "frag_grenade":
                    continue
                if e.get("travel", 0.0) > 0.0:
                    throws.append(e)
                elif e.get("blast", 0.0) > 0.0:
                    blasts.append(e)
            m.hold(m.a, 0.1, wep=NADE)

        assert len(throws) == 1, f"{len(throws)} throws for one grenade"
        t = throws[0]
        assert t["segs"] == [], "a thrown round should draw no tracer"
        assert t.get("fuse", 0.0) > 0.0, \
            "the throw must carry its fuse, or the round is drawn as a rocket"
        assert t["blast"] == 0.0, \
            "the throw itself must not blow up: the fuse has not run yet"
        assert len(blasts) == 1, f"{len(blasts)} explosions for one grenade"
        print(f"  one throw (fuse {t['fuse']:.1f}s, no segments), one blast")
        print("\nTHROWN ROUND CHECKS PASSED")
    finally:
        m.close()


if __name__ == "__main__":
    model_test()
    drain_test()
    respawn_test()
    thrown_round_test()
    print("\nALL STAMINA TESTS PASSED")
