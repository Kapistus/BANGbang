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
from sim import ballistics, classes, weapons
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
        q.x, q.y = 3.5 + w.range_m, 12.0     # a metre and a half off the path
        q.body.shields = 0.0
        before = q.body.health
        m.arm(FLAK)
        m.shoot(FLAK)
        assert wait_for(lambda: q.body.health < before, 3.0), \
            "the burst missed a man standing next to it"
        print(f"  off the line by a metre and a half: "
              f"{before - q.body.health:.0f} damage")

        # --- and the shooter is not in his own ring
        assert p.body.health == p.body.max_health, "he caught his own burst"

        # --- it carries to what it hits, rather than coming apart in the air
        # short of it: the wall lane has a wall in column 8
        p.x, p.y = 2.5, 23.5
        m.pb.x, m.pb.y = 30.5, 3.5           # out of it entirely
        m.b.drain_events()
        m.arm(FLAK)
        m.shoot(FLAK)
        time.sleep(1.0)
        m.hold(m.a, 0.3, wep=FLAK)
        bursts = [e for e in m.b.drain_events()
                  if e["t"] == "shot" and e.get("wep") == "flak_cannon"
                  and e.get("travel", 0.0) == 0.0]
        assert bursts, "the shell never went off"
        bx = bursts[-1]["impact"][0]
        print(f"  fired at a wall 5.5 m away: burst at x={bx:.1f} "
              f"(the wall is at 8.0, its range would carry it to "
              f"{p.x + 1.1 + w.range_m:.1f})")
        assert 7.0 < bx < 8.2, \
            "it came apart in mid-air instead of on the wall it was fired at"

        # --- cover: the same distance, once in the open and once with a wall
        # in between, so the second half is not passing for want of range
        p.x, p.y = 5.5, 23.5                 # the wall lane, wall in column 8
        q.x, q.y = 9.5, 23.5                 # just behind it
        gap = q.x - (p.x + 2.5)              # how far the burst is from him
        q.body.health = q.body.max_health
        q.body.shields = 0.0
        safe = q.body.health
        m.arm(FLAK)
        m.shoot(FLAK)
        time.sleep(1.2)
        m.hold(m.a, 0.3, wep=FLAK)
        assert q.body.health == safe, \
            f"pellets went through a wall: {safe - q.body.health:.0f} damage"

        # the control: nothing between them, same spacing, down the open hall
        p.x, p.y = 5.5, 10.5
        q.x, q.y = 9.5, 10.5
        q.body.health = q.body.max_health
        q.body.shields = 0.0
        open_before = q.body.health
        m.arm(FLAK)
        m.shoot(FLAK)
        assert wait_for(lambda: q.body.health < open_before, 3.0), \
            "the control shot did nothing either: the wall proved nothing"
        print(f"  {gap:.1f} m past the burst: "
              f"{open_before - q.body.health:.0f} damage in the open, "
              f"none through the wall")
        print("  behind a wall: nothing")
        print("\nFLAK CHECKS PASSED")
    finally:
        m.close()


def no_tracer_test():
    """Nothing that travels draws a line when it is fired.

    A bullet's visual is the tracer between the muzzle and the hole. A rocket,
    a shell or a grenade is the round itself, and a line drawn to where it is
    going arrives before the round does - it gives the shot away and reads as
    a streak out of the barrel. The server decides this, so it is checked on
    the wire rather than in the drawing code.
    """
    m = Match(port=PORT + 4)
    try:
        p = m.pa
        p.x, p.y = 3.5, 10.5
        m.pb.x, m.pb.y = 33.5, 10.5
        for slot, key in ((ROCKET, "rocket_launcher"), (FLAK, "flak_cannon"),
                          (0, "pistol")):
            w = weapons.ROSTER[key]
            travels = w.blast_r > 0.0 and w.projectile_speed > 0.0
            m.b.drain_events()
            m.arm(slot)
            m.shoot(slot)
            time.sleep(1.5)
            m.hold(m.a, 0.3, wep=slot)
            shots = [e for e in m.b.drain_events()
                     if e["t"] == "shot" and e.get("wep") == key]
            assert shots, f"{key}: nothing was broadcast at all"
            fired = [e for e in shots if e.get("travel", 0.0) > 0.0] \
                if travels else shots
            assert fired, f"{key}: no firing message"
            segs = len(fired[0].get("segs", []))
            print(f"  {w.name:16} fired with {segs} tracer segment(s)")
            if travels:
                assert segs == 0, \
                    f"{w.name} still draws a line to where it is going"
            else:
                assert segs > 0, "a bullet with no tracer: the test is wrong"
            # the burst of a flak shell is its own message, and what it draws
            # is stubs at the burst rather than the pellets' full reach
            for e in shots:
                if not travels or e.get("travel", 0.0) > 0.0 or not e.get("segs"):
                    continue
                longest = max(math.hypot(g[2] - g[0], g[3] - g[1])
                              for g in e["segs"])
                print(f"  {w.name:16} burst: {len(e['segs'])} stubs, "
                      f"longest {longest:.2f} m")
                assert longest <= ballistics.BURST_DRAW_M + 0.01, \
                    "the burst is drawn as full-length pellet tracers"
        print("\nTRACER CHECKS PASSED")
    finally:
        m.close()


def one_blast_test():
    """One round, one explosion.

    The server decides when a round that travels goes off and says so. The
    client used to work the arrival out for itself as well and draw its own
    fireball, so a rocket or a flak shell went off twice - once on the
    client's arithmetic, once when the message landed. Two blasts that close
    together overlap, so the test watches how many are on screen at once.
    """
    import pygame
    import mp_client

    m = Match(port=PORT + 5, cls_a="commando", cls_b="heavy_support")
    try:
        watcher, shooter = m.pa, m.pb
        # the watcher stands off the firing line: in front of a flak shell he
        # is what it bursts on, and a rocket at two metres kills him, and a
        # dead man's client is not what this is measuring
        watcher.x, watcher.y = 6.5, 13.5
        shooter.x, shooter.y = 6.5, 10.5
        pygame.init()
        view = mp_client.MatchView(m.a, R.MAPS_DIR, window=(640, 480))
        view.read_input = lambda: (0.0, 0.0, 0)
        view.audio_on = False
        for slot, key in ((ROCKET, "rocket_launcher"), (FLAK, "flak_cannon")):
            # the weapon has to be in hand AND out of its last recovery
            # before the trigger means anything - a rocket launcher takes
            # three seconds between shots
            t0 = time.monotonic()
            while time.monotonic() - t0 < 8.0:
                m.b.send_input(0.0, 0.0, 0.0, 0, wep=slot, aim_dist=12.0)
                view.frame(1 / 60)
                time.sleep(1 / 60)
                if (shooter.wi == slot and shooter.swap_t <= 0.0
                        and shooter.fire_cd <= 0.0
                        and time.monotonic() - t0 > 0.5):
                    break
            assert shooter.wi == slot and shooter.fire_cd <= 0.0, \
                f"{key} never came up ready"
            most = 0
            t0 = time.monotonic()
            while time.monotonic() - t0 < 2.5:
                # held for a moment, not for a single command: one frame of
                # trigger can be the one the tick does not get to
                btn = BTN_FIRE if time.monotonic() - t0 < 0.2 else 0
                m.b.send_input(0.0, 0.0, 0.0, btn, wep=slot, aim_dist=12.0)
                view.frame(1 / 60)
                most = max(most, len(view.blasts))
                time.sleep(1 / 60)
            print(f"  {weapons.ROSTER[key].name:16} {most} explosion(s) "
                  f"on screen at once")
            assert most == 1, \
                f"{key} drew {most} explosions for one round"
    finally:
        m.close()
    print("\nONE BLAST CHECKS PASSED")


def sound_test():
    """What each of these is heard as, which is not always a gunshot.

    The sound a weapon makes is decided twice over: the server says what
    happened and how far it carries, and the client turns that into a clip. A
    grenade used to do both wrongly - it announced itself as a gunshot with
    the blast's own reach, so throwing one sounded like the explosion it had
    not had yet.
    """
    from sim import audio
    import mp_client

    assert audio.fire_clip(weapons.ROSTER["combat_knife"]) == "whoosh"
    assert audio.fire_clip(weapons.ROSTER["frag_grenade"]) == "throw"
    assert audio.fire_clip(weapons.ROSTER["flak_cannon"]) != "boom", \
        "a cannon firing a shell is not an explosion"
    assert mp_client.CLIP_BY_KIND["melee"] == "whoosh"
    assert mp_client.CLIP_BY_KIND["blast"] == "boom"

    m = Match(port=PORT + 6, cls_a="saboteur", cls_b="heavy_support")
    try:
        p = m.pa
        nade = classes.get("saboteur").loadout.index("frag_grenade")
        p.x, p.y = 10.5, 10.5
        m.pb.x, m.pb.y = 25.5, 10.5

        # --- the knife: a swing is a swing, not a shot
        m.b.drain_events()
        m.arm(0)
        m.shoot(0, hold=0.2)
        time.sleep(0.4)
        m.hold(m.a, 0.2)
        clips = [e.get("clip") for e in m.b.drain_events() if e["t"] == "sound"
                 and e.get("id") == m.a.world.my_id]
        print(f"  a knife swing is heard as: {sorted(set(clips))}")
        assert "melee" in clips, "the swing made no sound at all"
        assert "fire" not in clips, "a blade should not be heard as a gunshot"

        # --- the grenade: quiet going out, loud going off
        m.b.drain_events()
        m.arm(nade)
        # thrown the other way: a grenade that lands on the other player kills
        # him, and a corpse cannot fire the rocket this test needs next
        m.shoot(nade, aim=math.pi, hold=0.2)
        throw = blast = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0:
            for e in m.b.drain_events():
                if e["t"] != "sound" or e.get("id") != m.a.world.my_id:
                    continue
                if e.get("clip") == "throw":
                    throw = e
                elif e.get("clip") == "blast":
                    blast = e
                elif e.get("clip") == "fire":
                    raise AssertionError("a thrown grenade fired a gunshot")
            m.hold(m.a, 0.1, wep=nade)
        assert throw is not None, "the throw made no sound"
        assert blast is not None, "the grenade going off made no sound"
        print(f"  a grenade: throw carries {throw['energy']:.0f}, "
              f"the blast {blast['energy']:.0f}")
        assert throw["energy"] < blast["energy"] * 0.4, \
            "the throw is nearly as loud as the explosion"

        # --- and a rocket: a launch, then a blast
        m.a.drain_events()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6.0:
            m.b.send_input(0.0, 0.0, 0.0, 0, wep=ROCKET, aim_dist=12.0)
            time.sleep(1 / 30)
            if (m.pb.wi == ROCKET and m.pb.swap_t <= 0.0
                    and m.pb.fire_cd <= 0.0 and time.monotonic() - t0 > 0.7):
                break
        assert m.pb.wi == ROCKET, "the launcher never came up"
        loaded = m.pb.mags[ROCKET]
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0:
            btn = BTN_FIRE if time.monotonic() - t0 < 0.3 else 0
            m.b.send_input(0.0, 0.0, 0.0, btn, wep=ROCKET, aim_dist=12.0)
            time.sleep(1 / 30)
        assert m.pb.mags[ROCKET] < loaded, "the launcher never fired"
        heard = [e.get("clip") for e in m.a.drain_events()
                 if e["t"] == "sound" and e.get("id") == m.b.world.my_id]
        print(f"  a rocket is heard as: {sorted(set(heard))}")
        assert "fire" in heard and "blast" in heard, \
            f"a launch and a detonation, not {sorted(set(heard))}"
        print("\nSOUND CHECKS PASSED")
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
    no_tracer_test()
    one_blast_test()
    sound_test()
    pistol_test()
    print("\nALL WEAPON TESTS PASSED")
