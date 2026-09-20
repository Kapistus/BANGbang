"""Headless test of health and ammo packs, and the caps that make them matter.

Ammo used to be infinite, so a firefight only ever ended when somebody died or
walked away. Every gun is capped now, and a pack tops each carried weapon up by
30% of its own cap. That turns "where is the ammo on this map" into a question
with an answer, which is the whole point.

What this checks:

  * the caps are real, and in a sane band of whole magazines
  * a pack gives 30% of each carried weapon's cap, rounded up, never over it
  * a pack that would do nothing is NOT consumed — full health walks over a
    health pack and leaves it standing
  * the respawn clock, and that a pack away from its spot stays away
  * over a real connection: the server decides who got there first, both
    clients are told, and a latecomer inherits a pack that is still away

Run from the project root:   python -m net.pickups_test
"""
import math
import shutil
import tempfile
import time
from pathlib import Path

from sim import pickups as pk
from sim import weapons
from sim.pickups import PickupSet
from sim.tilemap import load_map

# a small arena with a health pack and an ammo pack out in the open
GRID = """\
##########
#........#
#........#
#........#
#........#
##########
"""
SIDECAR = '''grid = "pk.grid"
name = "pickup test"
subdiv = 4
player_spawn = [1.5, 1.5]

[[interactables]]
id = "med1"
kind = "health"
pos = [4.5, 2.5]
name = "health"

[[interactables]]
id = "ammo1"
kind = "ammo"
pos = [7.5, 2.5]
name = "ammo"
'''


def _mapdir():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "pk.grid").write_text(GRID)
    (tmp / "pk.toml").write_text(SIDECAR)
    shutil.copy("maps/tiles.toml", tmp / "tiles.toml")
    return tmp


def caps_test():
    unlimited = [k for k, w in weapons.ROSTER.items() if w.reserve < 0]
    print(f"  {len(weapons.ROSTER)} weapons, {len(unlimited)} with unlimited "
          f"reserve: {unlimited or 'none'}")
    assert not unlimited, f"these guns still have infinite ammo: {unlimited}"
    for k, w in weapons.ROSTER.items():
        mags = w.reserve / w.mag
        assert 1.0 <= mags <= 8.0, \
            f"{k} carries {mags:.1f} spare magazines, which is not a cap"
    worst = min(weapons.ROSTER.items(), key=lambda kv: kv[1].reserve / kv[1].mag)
    best = max(weapons.ROSTER.items(), key=lambda kv: kv[1].reserve / kv[1].mag)
    print(f"  tightest {worst[0]} ({worst[1].reserve / worst[1].mag:.1f} mags), "
          f"loosest {best[0]} ({best[1].reserve / best[1].mag:.1f} mags)")
    print("\nCAP CHECKS PASSED")


def effects_test():
    loadout = list(weapons.DEFAULT_LOADOUT)
    caps = [weapons.ROSTER[k].reserve for k in loadout]

    # --- an empty loadout gets 30% of each cap, rounded up
    empty = [0] * len(loadout)
    got = pk.ammo_gain(loadout, empty, weapons.ROSTER)
    want = [max(1, math.ceil(c * 0.30)) for c in caps]
    print(f"  dry -> {got}")
    print(f"  (30% of {caps})")
    assert got == want, f"expected {want}"
    assert all(g <= c for g, c in zip(got, caps)), "a pack went over a cap"

    # --- it never overfills
    nearly = [c - 1 for c in caps]
    got = pk.ammo_gain(loadout, nearly, weapons.ROSTER)
    assert got == caps, f"topping up a nearly-full loadout gave {got}"

    # --- and a full loadout does not consume one
    assert pk.ammo_gain(loadout, list(caps), weapons.ROSTER) is None, \
        "a full player would have eaten an ammo pack"

    # --- a weapon with unlimited reserve has nothing to fill
    fake = dict(weapons.ROSTER)
    assert pk.ammo_gain(["nonexistent"], [0], fake) is None

    # --- health
    class Body:
        health, max_health = 40.0, 100.0
    b = Body()
    assert pk.health_gain(b) == 75.0, "health pack is not worth 35"
    b.health = 80.0
    assert pk.health_gain(b) == 100.0, "a health pack overhealed"
    b.health = 100.0
    assert pk.health_gain(b) is None, "a full player would have eaten a medkit"
    print("  health: 40 -> 75, 80 -> 100 (capped), 100 -> left alone")
    print("\nEFFECT CHECKS PASSED")


def set_test():
    m = load_map(_mapdir() / "pk.toml")
    ps = PickupSet(m)
    print(f"  map declares {len(ps)} packs: "
          + ", ".join(f"{p.id}={p.kind}" for p in ps))
    assert len(ps) == 2

    # --- you take one by standing on it, not near it
    assert ps.at(4.5, 2.5) is not None, "standing on the pack missed it"
    assert ps.at(4.5, 2.5).id == "med1"
    assert ps.at(5.4, 2.5) is None, "a pack was taken from most of a metre away"

    # --- taking is once
    assert ps.take("med1") is True
    assert ps.take("med1") is False, "the same pack was taken twice"
    assert ps.at(4.5, 2.5) is None, "a taken pack is still on the floor"

    # --- the clock
    med = ps.get("med1")
    assert not ps.step(pk.RESPAWN_S[pk.HEALTH] - 0.1), "it came back early"
    assert not med.live
    assert ps.step(0.2) == ["med1"], "it never came back"
    assert med.live and ps.at(4.5, 2.5) is not None

    # --- a pack mid-respawn survives the trip to a latecomer
    ps.take("ammo1")
    ps.step(5.0)
    wire = ps.wire()
    print(f"  five seconds into a respawn, on the wire: {wire}")
    late = PickupSet(m)
    late.apply_wire(wire)
    lp = late.get("ammo1")
    assert not lp.live and abs(lp.left - ps.get("ammo1").left) < 1e-6
    late.step(lp.left + 0.01)
    assert lp.live, "the latecomer's pack never came back"
    print("\nSET CHECKS PASSED")


def wire_test():
    from net import GameServer, GameClient
    from net.protocol import ServerState

    port = 47993
    mdir = _mapdir()
    srv = GameServer([(1.5, 1.5), (8.5, 4.5)], port=port, map_id="pk",
                     maps_dir=str(mdir), duration_s=300)
    srv.start()
    a = GameClient("127.0.0.1", port)
    b = GameClient("127.0.0.1", port)
    try:
        assert a.connect("Taker", (220, 40, 40)), a.reject_reason
        assert b.connect("Watcher", (40, 40, 220)), b.reject_reason
        a.set_ready(True)
        b.set_ready(True)
        t0 = time.monotonic()
        while a.world.state != ServerState.MATCH and time.monotonic() - t0 < 3:
            time.sleep(0.02)
        assert a.world.state == ServerState.MATCH, "no match"
        assert len(srv.packs) == 2, f"the server loaded {len(srv.packs)} packs"

        aid = a.world.my_id
        pa = srv.players[aid]
        srv.players[b.world.my_id].x = 8.5
        srv.players[b.world.my_id].y = 4.5
        a.drain_events()
        b.drain_events()

        # --- hurt them, empty them, then stand them on the health pack
        pa.body.health = 30.0
        pa.reserves = [0] * len(pa.reserves)
        before_res = list(pa.reserves)
        pa.x, pa.y = 4.5, 2.5
        t0 = time.monotonic()
        while pa.body.health <= 30.0 and time.monotonic() - t0 < 2.0:
            time.sleep(0.02)
        print(f"  stood on the health pack: {30.0} -> {pa.body.health}")
        assert pa.body.health == 65.0, "the server did not hand over the pack"
        assert not srv.packs.get("med1").live, "the pack is still on the floor"

        mine = [e for e in a.drain_events() if e["t"] == "pickup"]
        theirs = [e for e in b.drain_events() if e["t"] == "pickup"]
        assert mine and theirs, "the clients were not told"
        assert mine[0]["pid"] == "med1" and mine[0]["by"] == aid
        assert mine[0]["live"] is False and mine[0]["dur"] > 0
        print(f"  both clients told: by={mine[0]['by']}, "
              f"back in {mine[0]['dur']}s")

        # --- standing on it again while it is away changes nothing
        health_now = pa.body.health
        time.sleep(0.3)
        assert pa.body.health == health_now, "an absent pack healed somebody"

        # --- the ammo pack tops every carried weapon up
        pa.x, pa.y = 7.5, 2.5
        t0 = time.monotonic()
        while sum(pa.reserves) == 0 and time.monotonic() - t0 < 2.0:
            time.sleep(0.02)
        caps = [weapons.ROSTER[k].reserve for k in pa.loadout]
        want = [max(1, math.ceil(c * 0.30)) for c in caps]
        print(f"  ammo pack: {before_res} -> {pa.reserves}")
        assert pa.reserves == want, f"expected {want}"

        # --- and a latecomer is told the packs are away
        c = GameClient("127.0.0.1", port)
        assert c.connect("Late", (40, 200, 60)), c.reject_reason
        t0 = time.monotonic()
        while not c.world.map_pickups and time.monotonic() - t0 < 2.0:
            time.sleep(0.02)
        print(f"  a latecomer was handed: {c.world.map_pickups}")
        assert len(c.world.map_pickups) == 2, "the latecomer sees full packs"
        assert all(not live for _pid, live, _left in c.world.map_pickups)
        c.disconnect()
        print("\nWIRE CHECKS PASSED")
    finally:
        a.disconnect()
        b.disconnect()
        srv.stop()


if __name__ == "__main__":
    caps_test()
    print()
    effects_test()
    print()
    set_test()
    print()
    wire_test()
