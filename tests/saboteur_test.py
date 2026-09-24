"""Headless test of the Saboteur: the knife, the grenade and Vanish.

The class trades armour and health for speed, a blade that kills from behind,
and three grenades. None of the three existed before it: a melee swing has no
projectile, a grenade goes off on a fuse rather than on contact, and Vanish
turns off the two things that actually find you in this game — being seen and
being heard.

What this checks:

  * the knife: reach, arc, three times damage from behind, and that it does
    not swing through a wall or spend ammo
  * the grenade: the fuse runs from pulling the pin, so cooking it shortens
    the wait, letting go too late puts the blast in the air between you and
    them, and holding it all the way goes off in your hand — and cover still
    stops the blast
  * Vanish: no footstep sounds while it runs, the snapshot says so, and
    firing or swinging ends it early
  * the class art: each class asks for its own sprite, and falls back to its
    own idle rather than to the khaki soldier

Run from the project root:   python -m tests.saboteur_test
"""
import math
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from net import GameServer, GameClient
from net.protocol import BTN_ABILITY, BTN_FIRE, ServerState
from sim import classes, weapons
from tests import range_spots as R

PORT = 47982
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
    def __init__(self, port=PORT, cls_a="saboteur", cls_b="commando"):
        self.srv = GameServer([SPOT, AWAY], port=port, map_id=R.MAP_ID,
                              maps_dir=R.MAPS_DIR, duration_s=600)
        self.srv.start()
        self.a = GameClient("127.0.0.1", port)
        self.b = GameClient("127.0.0.1", port)
        assert self.a.connect("Sab", (220, 40, 40)), self.a.reject_reason
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
        """Wait out the last throw's recovery, holding the same weapon the
        next one will use: holding fire while the weapon is blocked - by the
        recovery or by a weapon swap the wep number started - cooks nothing,
        which reads as a much shorter fuse."""
        t0 = time.monotonic()
        while ((self.pa.fire_cd > 0.0 or self.pa.swap_t > 0.0)
               and time.monotonic() - t0 < 3.0):
            self.hold(self.a, 0.05, wep=wep)

    def swing(self, aim=0.0, wep=0):
        """One swing, then wait out the blade's own recovery - swinging again
        inside it does nothing at all, which would read as a miss."""
        self.hold(self.a, 0.15, aim=aim, buttons=BTN_FIRE, wep=wep)
        self.hold(self.a, 0.15, aim=aim, wep=wep)
        t0 = time.monotonic()
        while self.pa.fire_cd > 0.0 and time.monotonic() - t0 < 2.0:
            self.hold(self.a, 0.05, aim=aim, wep=wep)


def kit_test():
    c = classes.get("saboteur")
    print(f"  {c.name}: {classes.stat_line('saboteur')} | "
          f"{classes.weapon_line('saboteur')} | "
          f"{classes.ability_of('saboteur').name}")
    assert c.loadout == ("combat_knife", "frag_grenade"), \
        "no gun: the blade and the grenades are the whole kit"
    assert c.spd > classes.get("commando").spd, "it should be the fastest"
    assert c.armour < 20, "it should be the thinnest"
    knife = weapons.ROSTER["combat_knife"]
    nade = weapons.ROSTER["frag_grenade"]
    assert knife.is_melee and knife.backstab > 1.0
    assert nade.fuse_s > 0.0 and nade.blast_r > 0.0
    assert nade.mag + nade.reserve == 3, "three grenades, ever"
    print("\nKIT CHECKS PASSED")


def knife_test():
    m = Match()
    try:
        p, q = m.pa, m.pb

        def one_swing(target_aim, aim=0.0):
            """A swing at a target with a deep health pool, so what comes back
            is the damage rolled rather than whatever was left of them."""
            q.body.max_health = 5000.0
            q.body.health = 5000.0
            q.body.shields = 0.0
            q.aim = target_aim
            before = q.body.health
            m.swing(aim=aim)
            return before - q.body.health

        p.x, p.y = SPOT
        q.x, q.y = SPOT[0] + 1.0, SPOT[1]
        front = one_swing(math.pi)             # looking back at the knife
        back = one_swing(0.0)                  # looking away
        print(f"  at 1 m: {front:.0f} damage face to face, {back:.0f} from "
              f"behind ({back / max(front, 1):.1f}x)")
        assert front > 0.0, "the swing missed a body right in front of it"
        assert back > front * 2.0, "a backstab is not worth more than a poke"

        # --- and what that means against a real body: one from behind is a
        # kill, one from the front is not
        for target_aim, want_dead in ((math.pi, False), (0.0, True)):
            q.body.max_health = classes.get("commando").hp
            q.body.heal_full()
            q.aim = target_aim
            m.swing(aim=0.0)
            where = "behind" if want_dead else "in front"
            print(f"  one swing from {where}: alive={q.body.alive}")
            assert q.body.alive != want_dead, \
                f"a single swing from {where} should{'' if want_dead else ' not'} kill"
        q.body.heal_full()

        # --- out of reach, and behind you
        q.body.heal_full()
        q.x, q.y = SPOT[0] + 3.0, SPOT[1]
        before = q.body.health
        m.swing(aim=0.0)
        assert q.body.health == before, "the knife reached 3 m"
        q.x, q.y = SPOT[0] - 1.0, SPOT[1]
        m.swing(aim=0.0)
        assert q.body.health == before, "the knife hit somebody behind the swing"
        print("  no reach at 3 m, and none behind you")

        # --- it never runs out
        assert p.mags[0] == weapons.ROSTER["combat_knife"].mag, \
            "swinging spent ammo"
    finally:
        m.close()
    print("\nKNIFE CHECKS PASSED")


def grenade_test():
    m = Match(port=PORT + 1)
    try:
        p, q = m.pa, m.pb
        nade = weapons.ROSTER["frag_grenade"]
        p.wi = NADE
        p.x, p.y = SPOT
        q.x, q.y = SPOT[0] + 4.0, SPOT[1]
        q.body.shields = 0.0
        before = q.body.health
        t0 = time.monotonic()
        m.hold(m.a, 0.2, aim=0.0, buttons=BTN_FIRE, wep=NADE)
        m.hold(m.a, 0.2, aim=0.0, wep=NADE)
        assert m.srv._projectiles or q.body.health < before, "nothing was thrown"
        hurt_at = None
        while time.monotonic() - t0 < 6.0:
            if q.body.health < before:
                hurt_at = time.monotonic() - t0
                break
            time.sleep(0.02)
        assert hurt_at is not None, "the grenade never went off"
        print(f"  a quick throw 4 m: went off {hurt_at:.2f}s later for "
              f"{before - q.body.health:.0f} damage (fuse {nade.fuse_s}s)")
        assert hurt_at > nade.fuse_s * 0.5, "it went off far too early"
        assert p.mags[NADE] == 0 and p.reserves[NADE] == 2, "the belt did not count"

        # --- cooked: the same throw, held for half the fuse, goes off in
        # about half the time
        m.ready(2)
        p.mags[NADE], p.reserves[NADE] = 1, 1
        q.body.heal_full()
        q.body.shields = 0.0
        before = q.body.health
        cook = nade.fuse_s * 0.55
        t0 = time.monotonic()
        m.hold(m.a, cook, aim=0.0, buttons=BTN_FIRE, wep=NADE)
        m.hold(m.a, 0.1, aim=0.0, wep=NADE)
        cooked_at = None
        while time.monotonic() - t0 < 6.0:
            if q.body.health < before:
                cooked_at = time.monotonic() - t0
                break
            time.sleep(0.02)
        assert cooked_at is not None, "the cooked grenade never went off"
        # the fuse is absolute: what cooking buys is the time AFTER it leaves
        # your hand, which is what the person being thrown at gets
        print(f"  cooked {cook:.1f}s first: {cooked_at:.2f}s after the pin, "
              f"but only {cooked_at - cook:.2f}s after the throw "
              f"(uncooked: {hurt_at - 0.2:.2f}s)")
        assert cooked_at - cook < (hurt_at - 0.2) * 0.6, \
            "cooking did not shorten the wait after the throw"

        # --- held all the way: it goes off in your hand
        m.ready(2)
        p.mags[NADE], p.reserves[NADE] = 1, 0
        p.body.heal_full()
        p.x, p.y = SPOT
        q.x, q.y = AWAY
        mine = p.body.health + p.body.shields
        m.hold(m.a, nade.fuse_s + 0.4, aim=0.0, buttons=BTN_FIRE, wep=NADE)
        m.hold(m.a, 0.2, aim=0.0, wep=NADE)
        hurt = mine - (p.body.health + p.body.shields)
        print(f"  held the whole {nade.fuse_s}s: {hurt:.0f} damage to the "
              f"thrower, {p.mags[NADE]} left in hand")
        assert hurt > 0.0, "holding a grenade to the end cost nothing"
        assert p.mags[NADE] == 0, "the grenade that went off is still on the belt"
        assert not m.srv._projectiles, "it was thrown as well as going off"

        # --- let go too late and it goes off between you and them
        m.ready(2)
        p.mags[NADE], p.reserves[NADE] = 1, 0
        p.body.heal_full()
        q.body.heal_full()
        q.body.shields = 0.0
        p.x, p.y = SPOT
        q.x, q.y = SPOT[0] + 12.0, SPOT[1]     # further than it can fly in time
        theirs = q.body.health
        m.b.drain_events()
        m.hold(m.a, nade.fuse_s - 0.25, aim=0.0, buttons=BTN_FIRE, wep=NADE)
        m.hold(m.a, 1.0, aim=0.0, wep=NADE)
        blasts = [e for e in m.b.drain_events()
                  if e["t"] == "shot" and e.get("blast", 0.0) > 0.0]
        assert blasts, "nothing went off at all"
        bx, by = blasts[-1]["impact"]
        went = math.hypot(bx - SPOT[0], by - SPOT[1])
        print(f"  let go with 0.25s left, target 12 m away: it went off "
              f"{went:.1f} m out, and they took "
              f"{theirs - q.body.health:.0f} damage")
        assert 0.5 < went < 11.0, \
            f"a grenade with a quarter second left detonated {went:.1f} m out"
        assert q.body.health == theirs, "it still reached a target 12 m away"

        # --- cover stops it: the blast is traced like a bullet
        m.ready(2)
        p.mags[NADE], p.reserves[NADE] = 1, 0
        q.body.heal_full()
        q.body.shields = 0.0
        p.x, p.y = R.BLAST_SHOOTER          # the blast door between them
        q.x, q.y = R.BLAST_TARGET
        before = q.body.health
        m.hold(m.a, 0.2, aim=math.pi / 2, buttons=BTN_FIRE, wep=NADE)
        m.hold(m.a, 3.2, aim=math.pi / 2, wep=NADE)
        print(f"  through a shut blast door: {before - q.body.health:.0f} damage")
        assert q.body.health == before, "a grenade went through a blast door"
    finally:
        m.close()
    print("\nGRENADE CHECKS PASSED")


def vanish_test():
    m = Match(port=PORT + 2)
    try:
        p = m.pa
        m.b.drain_events()
        m.hold(m.a, 0.8, mx=1.0)
        steps = [e for e in m.b.drain_events()
                 if e["t"] == "sound" and e.get("clip") == "footstep"]
        assert steps, "the harness cannot hear footsteps at all"
        print(f"  walking normally: {len(steps)} footstep(s) heard")

        # how far a walk carries normally, to measure the vanished one against
        p.x, p.y = SPOT
        x0 = p.x
        m.hold(m.a, 0.8, mx=1.0)
        plain = p.x - x0

        m.hold(m.a, 0.12, buttons=BTN_ABILITY)
        assert wait_for(lambda: p.ability_until > 0.0), "Vanish never started"
        m.b.drain_events()
        x0 = p.x
        m.hold(m.a, 0.8, mx=1.0)
        fast = p.x - x0
        print(f"  the same walk: {plain:.2f} m normally, {fast:.2f} m vanished")
        assert fast > plain * 1.7, \
            "Vanish is meant to carry you across ground, at double speed"
        quiet = [e for e in m.b.drain_events()
                 if e["t"] == "sound" and e.get("clip") == "footstep"]
        print(f"  vanished: {len(quiet)} footstep(s) heard")
        assert not quiet, "a vanished Saboteur was still audible"
        assert wait_for(lambda: m.b.world.players[m.a.world.my_id].vanished), \
            "the other client was never told they are unseen"

        # --- swinging gives you away
        m.swing(aim=0.0)
        assert p.ability_until == 0.0, "swinging did not end Vanish"
        assert not m.srv._vanished(p)
        print("  a swing ends it")
    finally:
        m.close()
    print("\nVANISH CHECKS PASSED")


def art_test():
    """Each class asks for its own sprite set, and a class with only an idle
    pose keeps its own armour rather than falling back to the soldier."""
    import pygame
    from sim import sprites
    pygame.init()
    pygame.display.set_mode((16, 16))
    bank = sprites.SpriteBank()
    bank.load()
    assert bank.ok, "no sprites loaded"
    for key in classes.ORDER:
        art = classes.get(key).art
        ready = bank.body_art("soldier_ready", None, art)
        idle = bank.body_art("soldier_idle", None, art)
        print(f"  {classes.get(key).name:14} ready -> {ready:22} idle -> {idle}")
        if not art:
            assert ready == "soldier_ready" and idle == "soldier_idle"
            continue
        assert idle.startswith(art), f"{key} does not use its own idle art"
        assert ready.startswith(art), \
            f"{key} falls back to the soldier for its ready pose"
    print("\nART CHECKS PASSED")


if __name__ == "__main__":
    kit_test()
    print()
    knife_test()
    print()
    grenade_test()
    print()
    vanish_test()
    print()
    art_test()
    print("\nALL SABOTEUR CHECKS PASSED")
