"""Headless test of the damage balance: how long it takes to kill somebody.

The point of the numbers in sim/weapons.py and sim/classes.py is that a
firefight lasts long enough to react to. Before the rebalance a shotgun killed
a Tech in ONE trigger pull with no warning at all, and an SMG emptied a
Commando in 0.77s - under a second, which is about three human reaction times.

This pins what came out of that, at point blank with every shot hitting, which
is the fastest any of it can happen:

  * nothing kills a full-health class in under half a second, bar the one
    shell the shotgun is allowed
  * the shotgun: all ten pellets kill anything but a Heavy outright, seven do
    not, and the pattern still puts enough on a body across a hallway
  * every gun but the rails is SPREAD_WIDEN less precise than its rating
  * classes carry TOUGHNESS times what the class sheets give them, and
    single-player's Commando is deliberately NOT scaled

Run from the project root:   python -m tests.balance_test
"""
import random
import statistics

from sim import classes, combat, weapons

# every gun a player can carry, the ones the classes actually hand out first
GUNS = ["pistol", "combat_rifle", "smg", "combat_shotgun", "heavy_rifle",
        "rail_pistol", "rail_rifle", "pulse_carbine", "laser_pistol"]
FLOOR_S = 0.5             # nothing may kill faster than this...
SHOTGUN_EXEMPT = "combat_shotgun"   # ...except one point-blank shell


def ttk(wkey, cls, pellets=None, runs=120):
    """Median seconds to kill, point blank, every shot on the body."""
    w = weapons.ROSTER[wkey]
    rng = random.Random(4)
    times = []
    for _ in range(runs):
        body = classes.make_body(cls, 0.0, 0.0)
        t, fired = 0.0, 0
        auto = w.auto_rof > 0.0
        rate = w.auto_rof if auto else w.rof
        per_pull = 1 if auto else w.burst
        npel = w.pellets if pellets is None else pellets
        while body.alive and t < 30.0:
            for _r in range(per_pull):
                for _p in range(npel):
                    body.take(weapons.roll_damage(w, rng), t,
                              w.shield_mult, w.health_mult)
                fired += 1
                if fired % w.mag == 0:
                    t += w.reload_s
                if not body.alive:
                    break
            if not body.alive:
                break
            t += 1.0 / rate
        times.append(t)
    return statistics.median(times)


def ttk_test():
    print(f"  {'weapon':16}" + "".join(f"{classes.get(c).name:>14}"
                                       for c in classes.ORDER))
    for g in GUNS:
        row = {c: ttk(g, c) for c in classes.ORDER}
        print(f"  {weapons.ROSTER[g].name:16}"
              + "".join(f"{row[c]:>13.2f}s" for c in classes.ORDER))
        if g == SHOTGUN_EXEMPT:
            continue
        worst = min(row.values())
        assert worst >= FLOOR_S - 1e-9, \
            f"{g} kills a {min(row, key=row.get)} in {worst:.2f}s"
    print("\nTIME-TO-KILL CHECKS PASSED")


def shotgun_test():
    """One shell with the whole pattern on the body is a kill. What makes it a
    close-range gun is how few of the ten pellets are still on a person a few
    metres out - so this checks the curve, not just the muzzle."""
    w = weapons.ROSTER[SHOTGUN_EXEMPT]
    assert w.pellets == 10, f"the shotgun fires {w.pellets} pellets"
    for cls in classes.ORDER:
        alls = ttk(SHOTGUN_EXEMPT, cls, pellets=10)
        most = ttk(SHOTGUN_EXEMPT, cls, pellets=7)
        few = ttk(SHOTGUN_EXEMPT, cls, pellets=3)
        print(f"  {classes.get(cls).name:14} ten pellets {alls:.2f}s, "
              f"seven {most:.2f}s, three {few:.2f}s")
        if cls == "heavy_support":
            assert alls > 0.0, "a Heavy died to one shell"
        else:
            assert alls == 0.0, f"a point-blank shell did not kill a {cls}"
            # the two thin classes die to a pattern that is mostly on
            # them; the rest need the whole thing
            assert most > 0.0 or cls in ("tech", "saboteur"), \
                f"seven pellets killed a {cls} - the margin is gone"
        assert few >= 1.0, f"three pellets still kill a {cls} too fast"
    print("\nSHOTGUN CHECKS PASSED")


def pattern_test():
    """The pattern, in pellets on a body, at the ranges that matter."""
    import math
    w = weapons.ROSTER[SHOTGUN_EXEMPT]
    rng = random.Random(11)
    body_r = 0.28
    got = {}
    for d in (1.0, 3.0, 5.0, 8.0, 12.0):
        hits = []
        for _ in range(600):
            k = 0
            for _p in range(w.pellets):
                off = (weapons.pellet_offset(w, rng)
                       + weapons.jitter(w, d, False, rng))
                if abs(math.tan(off)) * d <= body_r:
                    k += 1
            hits.append(k)
        got[d] = statistics.mean(hits)
        print(f"  {d:>4.0f} m: {got[d]:.1f} of {w.pellets} pellets on the body")
    assert got[1.0] > 9.0, "the muzzle pattern has holes in it"
    assert 5.5 < got[3.0] < 8.5, "3 m should cost you about a third of it"
    assert 4.0 < got[5.0] < 6.0, \
        "5 m is hallway range: about half the pattern should land"
    assert got[8.0] > 2.5, "it should still be worth firing down a hallway"
    assert got[12.0] < 2.6, "it is still too good at 12 m"
    print("\nPATTERN CHECKS PASSED")


def spread_test():
    """Every gun but the rails shoots wider than its accuracy rating says."""
    def dev(w, widen):
        was = weapons.SPREAD_WIDEN
        weapons.SPREAD_WIDEN = widen
        try:
            rng = random.Random(3)
            # at its own effective range, where the cone rather than the
            # point-blank floor decides the miss
            return statistics.median(
                abs(weapons.jitter(w, w.range_m, False, rng))
                for _ in range(2000))
        finally:
            weapons.SPREAD_WIDEN = was
    for key, w in weapons.ROSTER.items():
        if w.min_dev_m <= 0.0:
            continue                      # the flamethrower has no cone to widen
        ratio = dev(w, weapons.SPREAD_WIDEN) / dev(w, 1.0)
        want = 1.0 if w.steady else weapons.SPREAD_WIDEN
        assert abs(ratio - want) < 0.05, \
            f"{w.name}: spread is {ratio:.2f}x its rating, wanted {want:.2f}x"
    steady = sorted(k for k, w in weapons.ROSTER.items() if w.steady)
    print(f"  every gun is {weapons.SPREAD_WIDEN}x wider except {steady}")
    shot = weapons.ROSTER["combat_shotgun"]
    rng = random.Random(1)
    widest = max(abs(weapons.pellet_offset(shot, rng)) for _ in range(4000))
    import math
    print(f"  the shotgun cone reaches {math.degrees(widest):.1f} deg of "
          f"{shot.spread_deg * weapons.SPREAD_WIDEN:.1f}")
    assert math.degrees(widest) > shot.spread_deg * 1.05, \
        "the pellet cone did not widen with everything else"
    print("\nSPREAD CHECKS PASSED")


def toughness_test():
    sp = combat.player_commando(0.0, 0.0)
    mp = classes.make_body("commando", 0.0, 0.0)
    print(f"  single-player Commando {sp.max_health:.0f} + {sp.max_shields:.0f}, "
          f"multiplayer {mp.max_health:.0f} + {mp.max_shields:.0f} "
          f"(x{classes.TOUGHNESS})")
    assert mp.max_health == round(sp.max_health * classes.TOUGHNESS)
    assert mp.max_shields == round(sp.max_shields * classes.TOUGHNESS)
    assert sp.max_health == 78 and sp.max_shields == 57, \
        "single-player's Commando was supposed to stay where it was"
    for c in classes.ORDER:
        cc = classes.get(c)
        assert cc.hp == round(cc.health * classes.TOUGHNESS)
        assert cc.sh == round(cc.shields * classes.TOUGHNESS)
    print("\nTOUGHNESS CHECKS PASSED")


if __name__ == "__main__":
    ttk_test()
    print()
    shotgun_test()
    print()
    pattern_test()
    print()
    spread_test()
    print()
    toughness_test()
    print("\nALL BALANCE CHECKS PASSED")
