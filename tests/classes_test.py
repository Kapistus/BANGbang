"""Headless test of multiplayer classes.

A player picks a class - from the prompt when they first connect, the class
row in the lobby, or the esc menu mid-match - and it takes effect the next
time they spawn. The class sets their health, shields, armour, speed and the
three weapons they carry.

What this checks:

  * the four classes: Commando is exactly the old prototype body, every kit
    is a pistol and two guns that exist
  * a class picked in the lobby is what the match starts with: body, loadout
    and speed, on the server and in what clients are told
  * a class picked mid-match does NOT change anything until a respawn, and
    then changes everything
  * the esc menu: opens, the wheel steps through classes, leave still leaves,
    and while it is open the player neither moves nor fires
  * the client's prediction runs at its class's speed, so a slow class does
    not rubber-band

Run from the project root:   python -m tests.classes_test
"""
import math
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from net import GameServer, GameClient
from net.protocol import BTN_FIRE, ServerState
from sim import classes, combat, weapons
from tests import range_spots as R

PORT = 47975


def wait_for(pred, seconds=4.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if pred():
            return True
        time.sleep(0.02)
    return False


def table_test():
    old = combat.player_commando(0, 0)
    new = classes.make_body("commando", 0, 0)
    # multiplayer bodies are the prototype Commando with the rebalance
    # toughness on top; everything else about it is untouched
    for attr in ("armor_lo", "armor_hi", "shield_regen", "shield_delay"):
        assert getattr(old, attr) == getattr(new, attr), \
            f"the Commando changed: {attr} {getattr(old, attr)} -> {getattr(new, attr)}"
    for attr in ("max_health", "max_shields"):
        want = round(getattr(old, attr) * classes.TOUGHNESS)
        assert getattr(new, attr) == want, \
            f"{attr}: {getattr(new, attr)} is not the prototype's {want}"
    # every class carries the rounded numbers, so the card and the body agree
    for k in classes.ORDER:
        c = classes.get(k)
        assert c.spd == round(c.speed * 2.0) / 2.0 and c.spd % 0.5 == 0.0, \
            f"{k}: speed {c.spd} is not on a half"
        assert c.hp == round(c.hp) and c.sh == round(c.sh), \
            f"{k}: {c.hp} health, {c.sh} shields are not whole"
        assert classes.make_body(k, 0, 0).speed == c.spd
    assert classes.get("commando").spd == classes.BASE_SPEED, \
        "the Commando is the baseline: it must come out at 1.0"
    print(f"  the Commando is the prototype body x{classes.TOUGHNESS} "
          f"health and shields: {new.max_health:.0f} + {new.max_shields:.0f}")
    assert classes.ORDER == ["commando", "heavy_support", "medic", "tech",
                             "saboteur"]
    for k in classes.ORDER:
        c = classes.get(k)
        # a sidearm and two guns each - except the Saboteur, who carries a
        # blade and grenades and no gun at all
        assert len(c.loadout) == (2 if c.key == "saboteur" else 3), c.loadout
        assert c.loadout[0] in ("pistol", "combat_knife"), c.loadout
        assert all(w in weapons.ROSTER for w in c.loadout), c.loadout
        print(f"  {c.name:14} {classes.stat_line(k)}  |  {classes.weapon_line(k)}")
    assert classes.get("nonsense").key == "commando"
    assert classes.cycle(classes.ORDER[-1], 1) == classes.ORDER[0]
    assert classes.cycle(classes.ORDER[0], -1) == classes.ORDER[-1]
    print("\nTABLE CHECKS PASSED")


def server_test():
    srv = GameServer([R.SPAWNS[0], R.SPAWNS[1]], port=PORT, map_id=R.MAP_ID,
                     maps_dir=R.MAPS_DIR, duration_s=300)
    srv.start()
    a, b = GameClient("127.0.0.1", PORT), GameClient("127.0.0.1", PORT)
    try:
        assert a.connect("Heavy", (200, 60, 60)), a.reject_reason
        assert b.connect("Comm", (60, 60, 200)), b.reject_reason
        a.set_class("heavy_support")
        assert wait_for(lambda: b.world.players.get(a.world.my_id) is not None
                        and b.world.players[a.world.my_id].cls == "heavy_support"), \
            "the other player never saw the pick in the lobby"
        a.set_ready(True)
        b.set_ready(True)
        assert wait_for(lambda: a.world.state == ServerState.MATCH), "no match"
        pa, pb = srv.players[a.world.my_id], srv.players[b.world.my_id]
        heavy = classes.get("heavy_support")
        assert pa.cls == "heavy_support" and pb.cls == "commando"
        assert pa.body.max_health == heavy.hp
        assert pa.body.max_shields == heavy.sh
        assert pa.body.armor_lo == heavy.armour
        assert pa.loadout == list(heavy.loadout)
        assert wait_for(lambda: a.world.players[a.world.my_id].cls_now
                        == "heavy_support"), "snapshots do not say the class"
        print(f"  lobby pick -> match: Heavy spawns with "
              f"{pa.body.max_health:.0f} health, {pa.loadout}")

        # --- speed: both run east for a second from the same row
        def run_east(cli, p, start):
            p.x, p.y = start
            t_end = time.monotonic() + 1.0
            while time.monotonic() < t_end:
                cli.send_input(1.0, 0.0, 0.0, 0)
                time.sleep(1 / 30)
            cli.send_input(0.0, 0.0, 0.0, 0)
            time.sleep(0.15)
            return p.x - start[0]
        da = run_east(a, pa, (3.5, 8.5))
        db = run_east(b, pb, (3.5, 11.5))
        ratio = da / db
        print(f"  a second's walk: Heavy {da:.2f} m, Commando {db:.2f} m, "
              f"ratio {ratio:.2f} (class says {heavy.speed_mult:.2f})")
        assert abs(ratio - heavy.speed_mult) < 0.08, "class speed not applied"

        # --- mid-match change: nothing until the respawn
        b.set_class("tech")
        assert wait_for(lambda: pb.cls_next == "tech")
        time.sleep(0.3)
        assert pb.cls == "commando" and pb.loadout == list(
            classes.get("commando").loadout), "the class changed before a respawn"
        assert pb.body.max_health == classes.get("commando").hp
        print("  picked Tech mid-match: still a Commando until respawn")
        srv.kill(pa, pb)
        with srv._lock:
            pb.respawn_at = time.monotonic() + 0.2
        assert wait_for(lambda: pb.alive and pb.cls == "tech", 5.0), \
            "respawned without the new class"
        tech = classes.get("tech")
        assert pb.body.max_health == tech.hp and pb.loadout == list(tech.loadout)
        assert wait_for(lambda: b.world.players[b.world.my_id].cls_now == "tech")
        print(f"  respawned as Tech: {pb.body.max_health:.0f} health, {pb.loadout}")

        # --- a bad class name changes nothing
        a.set_class("wizard")
        time.sleep(0.2)
        assert pa.cls_next == "heavy_support"
    finally:
        a.disconnect()
        b.disconnect()
        srv.stop()
    print("\nSERVER CHECKS PASSED")


def menu_test():
    import mp_client
    srv = GameServer([R.SPAWNS[0], R.SPAWNS[1]], port=PORT + 1,
                     map_id=R.MAP_ID, maps_dir=R.MAPS_DIR, duration_s=300)
    srv.start()
    a, b = GameClient("127.0.0.1", PORT + 1), GameClient("127.0.0.1", PORT + 1)
    try:
        assert a.connect("Menu", (200, 60, 60)), a.reject_reason
        assert b.connect("Other", (60, 60, 200)), b.reject_reason
        a.set_class("heavy_support")
        time.sleep(0.2)
        a.set_ready(True)
        b.set_ready(True)
        assert wait_for(lambda: a.world.state == ServerState.MATCH)
        pygame.init()
        view = mp_client.MatchView(a, R.MAPS_DIR)
        assert wait_for(lambda: (view._sync_class(), view.cls_now)[1]
                        == "heavy_support")
        assert view.loadout == list(classes.get("heavy_support").loadout)
        p = srv.players[a.world.my_id]

        # prediction at the class's speed: walk, and see how far the
        # server had to correct us
        held = {"k": (1.0, 0.0, 0)}
        view.read_input = lambda: held["k"]
        worst = 0.0
        t_end = time.monotonic() + 1.5
        while time.monotonic() < t_end:
            view.frame(1 / 60)
            worst = max(worst, view.corrected_m)
            time.sleep(1 / 90)
        print(f"  walking as Heavy: worst correction {worst * 100:.1f} cm")
        # at the Commando's speed this comes out around 3-4 cm; right, ~0.1
        assert worst < 0.02, "the client predicts at the wrong speed"

        # --- esc opens the menu; walking and firing stop
        def key(k):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0,
                                                 unicode=""))
        key(pygame.K_ESCAPE)
        view.frame(1 / 60)
        assert view.menu_open, "esc did not open the menu"
        held["k"] = (1.0, 0.0, BTN_FIRE)
        # let the commands sent before the menu opened finish arriving, or
        # the sample below races the last of them
        for _ in range(10):
            view.frame(1 / 60)
            time.sleep(1 / 60)
        mag0 = p.mags[p.wi]
        x0 = p.x
        for _ in range(40):
            view.frame(1 / 60)
            time.sleep(1 / 90)
        assert abs(p.x - x0) < 0.05 and p.mags[p.wi] == mag0, \
            "moved or fired with the menu open"
        print("  menu open: standing still, not firing")

        # --- the wheel steps through classes; it waits for a respawn
        pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))
        view.frame(1 / 60)
        assert wait_for(lambda: p.cls_next == "medic"), p.cls_next
        assert p.cls == "heavy_support"
        pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))
        view.frame(1 / 60)
        assert wait_for(lambda: p.cls_next == "tech"), p.cls_next
        print(f"  wheel: heavy -> medic -> tech picked, still playing {p.cls}")

        # --- esc again closes; the menu's leave still leaves
        key(pygame.K_ESCAPE)
        view.frame(1 / 60)
        assert not view.menu_open
        key(pygame.K_ESCAPE)
        view.frame(1 / 60)
        key(pygame.K_DOWN)
        view.frame(1 / 60)
        key(pygame.K_DOWN)
        view.frame(1 / 60)
        key(pygame.K_RETURN)
        assert view.frame(1 / 60) == "leave", "leave match did not leave"
        print("  esc closes it; the menu's Leave match returns 'leave'")
    finally:
        a.disconnect()
        b.disconnect()
        srv.stop()
    print("\nMENU CHECKS PASSED")


if __name__ == "__main__":
    table_test()
    print()
    server_test()
    print()
    menu_test()
    print("\nALL CLASS CHECKS PASSED")
