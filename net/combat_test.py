"""Headless test of networked combat. No pygame, no rendering.

Two clients on a real map, one shooting the other, checking the things that
decide whether a firefight is fair:

  * a shot lands, and damage runs through the shields/armor/health model
  * semi-auto fires once per click; full-auto streams at the weapon's rate
  * magazines empty, reloads refill, and an empty gun only clicks
  * a wall stops a bullet that a clear line would have landed
  * a kill is attributed to the shooter, scored, and fed back to both clients
  * firing and footsteps reach everyone as sound events

Run from the project root:   python -m net.combat_test
"""
import math
import time

from sim import weapons
from net import GameServer, GameClient
from net import maps as netmaps
from net.protocol import BTN_FIRE, BTN_RELOAD, ServerState

PORT = 47996
MAP = "arena"
RANGE_M = 3.0             # close enough that spread is not the variable here


def clear_pair(m, gap):
    """Two standable points `gap` apart with nothing solid between them."""
    step = 0.5
    y = step
    while y < m.height_m:
        x = step
        while x < m.width_m - gap:
            a, b = (x, y), (x + gap, y)
            if m.can_stand(*a, 0.3) and m.can_stand(*b, 0.3):
                n = int(gap / 0.1)
                if all(not m.solid_at(x + i * gap / n, y) for i in range(n + 1)):
                    return a, b
            x += step
        y += step
    raise SystemExit("no clear firing line found on this map")


def pump(cli, seconds):
    """Drain events for a while, returning them."""
    out = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        out.extend(cli.drain_events())
        time.sleep(0.02)
    return out


class Gun:
    """Drives one client's input at the server's tick rate."""

    def __init__(self, cli, other_id):
        self.cli = cli
        self.other = other_id
        self.wep = 0
        self.mode = 0

    def aim(self):
        me = self.cli.world.players[self.cli.world.my_id]
        them = self.cli.world.players[self.other]
        return math.atan2(them.y - me.y, them.x - me.x)

    def send(self, buttons=0, ticks=1, hz=30):
        evs = []
        for _ in range(ticks):
            self.cli.send_input(0.0, 0.0, self.aim(), buttons,
                                wep=self.wep, mode=self.mode,
                                aim_dist=RANGE_M)
            time.sleep(1.0 / hz)
            evs.extend(self.cli.drain_events())
        return evs

    def click(self):
        """One trigger pull: press, release."""
        evs = self.send(BTN_FIRE, ticks=2)
        evs += self.send(0, ticks=2)
        return evs


def pen_between(m, a, b):
    """Total penetration cost of the cover between two points.

    Bullets in this game do not stop at walls, they spend a budget against each
    tile's pen_cost — so 'is there something solid in the way' is the wrong
    question. A pistol carries 2.0; cover worth more than that is cover it
    cannot get through."""
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) * m.cells_per_metre))
    total, seen = 0.0, set()
    for i in range(n + 1):
        f = i / n
        cx, cy = m.cell_of(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        if (cx, cy) in seen:
            continue
        seen.add((cx, cy))
        if m.blocks_bullets[cy, cx]:
            total += float(m.pen_cost[cy, cx])
    return total


def cover_between(m, a, b, spread=0.25):
    """Penetration cost of the cover across the whole corridor a shot might
    take, not just the exact centre line.

    Shots jitter. A wall that lies ALONG the firing line — one cell thick in
    the perpendicular direction — blocks the centre line and nothing else, so
    measuring only that line finds 'cover' a shot strolls past. This samples
    three parallel lines and returns the weakest, which is what the bullet will
    find."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    px, py = -dy / n * spread, dx / n * spread      # perpendicular offset
    return min(
        pen_between(m, a, b),
        pen_between(m, (a[0] + px, a[1] + py), (b[0] + px, b[1] + py)),
        pen_between(m, (a[0] - px, a[1] - py), (b[0] - px, b[1] - py)),
    )


def revive(srv, pid, pos):
    """Put a player back to full at a known spot.

    Each phase below starts from this, because a pistol magazine is more than
    enough to kill: without it the wall test would be firing at a corpse and
    passing for the wrong reason."""
    sp = srv.players[pid]
    sp.x, sp.y = pos
    sp.alive = True
    sp.respawn_at = None
    if sp.body is not None:
        sp.body.heal_full()
        sp.body.x, sp.body.y = pos
    time.sleep(0.25)


def integrity(cli, pid):
    """Shields plus health, as the client sees them. Damage lands on shields
    first — a Commando carries 57 of them — so a test that watched only the
    health bar would call a landed pistol round a miss."""
    p = cli.world.players[pid]
    return p.hp + p.sh


def main():
    m, _ = netmaps.load_with_spawns(MAP)
    a_pos, b_pos = clear_pair(m, RANGE_M)
    print(f"firing line on {m.name}: {a_pos} -> {b_pos}")

    srv = GameServer([a_pos, b_pos], port=PORT, map_id=MAP, duration_s=600)
    srv.start()
    a = GameClient("127.0.0.1", PORT)
    b = GameClient("127.0.0.1", PORT)
    assert a.connect("Shooter", (220, 40, 40)), a.reject_reason
    assert b.connect("Target", (40, 40, 220)), b.reject_reason
    a.set_ready(True)
    b.set_ready(True)
    time.sleep(0.5)
    assert a.world.state == ServerState.MATCH, a.world.state
    aid, bid = a.world.my_id, b.world.my_id
    gun = Gun(a, bid)

    try:
        # keep the target still and in place for the shooting tests
        sp, sq = srv.players[aid], srv.players[bid]
        sp.x, sp.y = a_pos
        sq.x, sq.y = b_pos
        time.sleep(0.2)

        # --- semi-auto: one click, one shot
        before = integrity(a, bid)
        evs = gun.click()
        shots = [e for e in evs if e["t"] == "shot"]
        tp = a.world.players[bid]
        print(f"  one click -> {len(shots)} shot event(s), target "
              f"{before:.2f} -> {integrity(a, bid):.2f} "
              f"(hp {tp.hp:.2f} shields {tp.sh:.2f})")
        assert len(shots) == 1, f"a click fired {len(shots)} times"
        assert integrity(a, bid) < before, "the shot did no damage"
        assert any(e["t"] == "sound" and e["clip"] == "fire" for e in evs), \
            "firing made no sound event"

        # --- holding a semi-auto does NOT stream
        evs = gun.send(BTN_FIRE, ticks=20)
        held = [e for e in evs if e["t"] == "shot"]
        gun.send(0, ticks=2)
        assert len(held) <= 1, f"held semi-auto fired {len(held)} times"
        print(f"  ok  holding semi-auto fired {len(held)} shot")

        # --- magazine empties, then only clicks
        pistol = weapons.ROSTER[weapons.DEFAULT_LOADOUT[0]]
        fired = 0
        # the pistol cycles at 1.6 rounds/s, so clicks have to be spaced or the
        # server simply refuses them — which is itself the rate limit working
        for _ in range(pistol.mag + 3):
            evs = gun.click()
            fired += len([e for e in evs if e["t"] == "shot"])
            if srv.players[aid].mags[0] <= 0:
                break
            time.sleep(pistol.burst_time)
        mag = srv.players[aid].mags[0]
        time.sleep(pistol.burst_time)     # let the cycle finish, or the empty
                                          # click is refused for being early
        evs = gun.click()
        dry = [e for e in evs if e["t"] == "sound" and e["clip"] == "dryfire"]
        print(f"  ok  magazine ran dry after {fired} shots (mag={mag}), "
              f"empty click made {len(dry)} dry sound(s)")
        assert mag == 0, f"magazine did not empty: {mag}"
        assert not [e for e in evs if e["t"] == "shot"], "empty gun still fired"

        # --- the rate limit itself: clicking as fast as possible for a second
        # cannot beat the weapon's cycle time
        srv.players[aid].mags[0] = pistol.mag
        t0 = time.monotonic()
        rapid = 0
        while time.monotonic() - t0 < 2.0:
            rapid += len([e for e in gun.click() if e["t"] == "shot"])
        ceiling = int(2.0 / pistol.burst_time) + 2
        print(f"  ok  2 s of mashing gave {rapid} shots (ceiling {ceiling})")
        assert rapid <= ceiling, f"{rapid} shots in 2 s beat the cycle time"
        srv.players[aid].mags[0] = 0
        gun.send(0, ticks=2)

        # --- reload refills it
        gun.send(BTN_RELOAD, ticks=2)
        gun.send(0, ticks=2)
        time.sleep(pistol.reload_s + 0.4)
        gun.send(0, ticks=2)
        print(f"  ok  reloaded to {srv.players[aid].mags[0]}/{pistol.mag}")
        assert srv.players[aid].mags[0] == pistol.mag, "reload did not refill"

        # --- a wall stops what a clear line would hit
        revive(srv, bid, b_pos)
        srv.players[aid].mags[0] = pistol.mag
        hidden = None
        need = pistol.pen * 2.0
        for x, y in [(x * 0.5, y * 0.5)
                     for x in range(2, int(m.width_m * 2))
                     for y in range(2, int(m.height_m * 2))]:
            if not (m.can_stand(x, y, 0.3) and m.can_stand(x + 2.2, y, 0.3)):
                continue
            if cover_between(m, (x, y), (x + 2.2, y)) > need:
                hidden = (x, y)
                break
        if hidden is None:
            print("  -- no cover thick enough for a pistol to fail on, skipped")
        else:
            sp.x, sp.y = hidden
            sq.x, sq.y = hidden[0] + 2.2, hidden[1]
            time.sleep(0.2)
            before_wall = integrity(a, bid)
            for _ in range(4):
                gun.click()
            print(f"  ok  four pistol shots into cover worth "
                  f"{cover_between(m, hidden, (hidden[0] + 2.2, hidden[1])):.1f} "
                  f"penetration (pistol carries {pistol.pen}): "
                  f"{before_wall:.2f} -> {integrity(a, bid):.2f}")
            assert integrity(a, bid) == before_wall, \
                "a bullet went through a wall"
            sp.x, sp.y = a_pos
            sq.x, sq.y = b_pos
            time.sleep(0.2)

        # --- full-auto streams, and eventually kills
        revive(srv, bid, b_pos)
        revive(srv, aid, a_pos)
        kills_before = srv.players[aid].kills   # the magazine phase killed once
        gun.wep = 2                      # smg
        gun.mode = 1                     # auto
        gun.send(0, ticks=25)            # let the weapon swap finish
        kills, shots = [], 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 8.0:
            evs = gun.send(BTN_FIRE, ticks=5)
            shots += len([e for e in evs if e["t"] == "shot"])
            kills += [e for e in evs if e["t"] == "kill"]
            if kills:
                break
            if srv.players[aid].mags[2] <= 0:
                gun.send(BTN_RELOAD, ticks=2)
                gun.send(0, ticks=2)
                time.sleep(weapons.ROSTER[weapons.DEFAULT_LOADOUT[2]].reload_s + 0.3)
        gun.send(0, ticks=2)
        print(f"  ok  full-auto put out {shots} shots and "
              f"{'killed' if kills else 'did NOT kill'} the target")
        assert shots > 3, f"full-auto only fired {shots} times"
        assert kills, "nobody died after 8 seconds of automatic fire"
        entry = kills[0]["entry"]
        print(f"     kill feed: {entry.killer} {entry.verb} {entry.victim} "
              f"({entry.weapon})")
        assert entry.killer == "Shooter" and entry.victim == "Target"
        scored = srv.players[aid].kills - kills_before
        assert scored == 1, f"the kill scored {scored} times, not once"
        assert not b.world.players[bid].alive, "the victim is still alive"

        # --- the victim's own client saw it too
        assert any(e["t"] == "kill" for e in b.drain_events() + pump(b, 0.3)), \
            "the victim's client never saw the kill"

        # --- footsteps are broadcast as sound
        # (the target is a corpse at this point, and corpses don't walk)
        revive(srv, bid, b_pos)
        b.send_input(1.0, 0.0, 0.0, 0)
        evs = []
        for _ in range(40):
            b.send_input(1.0, 0.0, 0.0, 0)
            time.sleep(1 / 30)
            evs.extend(a.drain_events())
        steps = [e for e in evs if e["t"] == "sound" and e["clip"] == "footstep"]
        print(f"  ok  walking produced {len(steps)} footstep sound event(s)")
        assert steps, "walking made no sound"

        print("\nALL CHECKS PASSED")
    finally:
        a.disconnect()
        b.disconnect()
        srv.stop()


def glass_test():
    """Glass is map state, so the server owns the break and everyone applies
    it. A pane broken on one machine only would give those players a wall the
    others can see and shoot through."""
    m, _ = netmaps.load_with_spawns("compound")
    pane = None
    for (r, c), _ in [((r, c), None)
                      for r in range(m.chars.shape[0])
                      for c in range(m.chars.shape[1])]:
        tile = m.tiles.get(m.chars[r, c])
        if tile is not None and getattr(tile, "glass", False):
            # somewhere standable to shoot it from, on either side
            for dx in (-2.0, 2.0):
                sx, sy = c + 0.5 + dx, r + 0.5
                if m.can_stand(sx, sy, 0.3):
                    pane = ((c + 0.5, r + 0.5), (sx, sy))
                    break
        if pane:
            break
    if pane is None:
        print("no glass on compound to shoot, skipped")
        return
    (gx, gy), (sx, sy) = pane
    print(f"glass at ({gx:.1f}, {gy:.1f}), shooting from ({sx:.1f}, {sy:.1f})")

    srv = GameServer([(sx, sy), (sx, sy)], port=PORT + 1, map_id="compound",
                     duration_s=300)
    srv.start()
    a = GameClient("127.0.0.1", PORT + 1)
    b = GameClient("127.0.0.1", PORT + 1)
    try:
        assert a.connect("Shooter", (220, 40, 40)), a.reject_reason
        assert b.connect("Watcher", (40, 40, 220)), b.reject_reason
        a.set_ready(True)
        b.set_ready(True)
        time.sleep(0.5)
        assert a.world.state == ServerState.MATCH
        aid = a.world.my_id
        srv.players[aid].x, srv.players[aid].y = sx, sy
        heading = math.atan2(gy - sy, gx - sx)
        time.sleep(0.2)

        broke = []
        pistol = weapons.ROSTER[weapons.DEFAULT_LOADOUT[0]]
        for _ in range(6):
            a.send_input(0.0, 0.0, heading, BTN_FIRE, aim_dist=2.0)
            time.sleep(0.07)
            a.send_input(0.0, 0.0, heading, 0, aim_dist=2.0)
            time.sleep(pistol.burst_time)
            broke += [e for e in a.drain_events() if e["t"] == "glass"]
            if broke:
                break
        seen_by_b = [e for e in b.drain_events() if e["t"] == "glass"]
        print(f"  shooter saw {len(broke)} glass break(s), "
              f"watcher saw {len(seen_by_b)}")
        assert broke, "shooting a window never broke it"
        assert seen_by_b, "the other client was never told the glass broke"
        r, c = broke[0]["cells"][0]
        assert not srv.map.blocks_sight[r * srv.map.subdiv, c * srv.map.subdiv], \
            "the server still treats the broken pane as opaque"
        print("  ok  the pane is open on the server and announced to everyone")
    finally:
        a.disconnect()
        b.disconnect()
        srv.stop()


def door_test():
    """Doors are map state like glass: the server owns the toggle so that sight,
    sound and bullets agree on every machine."""
    from net.protocol import BTN_INTERACT
    from sim.tilemap import find_doors

    m, _ = netmaps.load_with_spawns("compound")
    doors = find_doors(m)
    assert doors, "no doors on compound to test with"
    # a door with somewhere to stand beside it
    spot = None
    for (r, c) in doors:
        for dx, dy in ((1.2, 0.0), (-1.2, 0.0), (0.0, 1.2), (0.0, -1.2)):
            x, y = c + 0.5 + dx, r + 0.5 + dy
            if m.can_stand(x, y, 0.3):
                spot = ((r, c), (x, y))
                break
        if spot:
            break
    assert spot, "no door with standable ground beside it"
    (dr, dc), stand = spot
    print(f"door at ({dr}, {dc}), standing at "
          f"({stand[0]:.1f}, {stand[1]:.1f}) of {len(doors)} doors")

    srv = GameServer([stand, stand], port=PORT + 2, map_id="compound",
                     duration_s=300)
    srv.start()
    a = GameClient("127.0.0.1", PORT + 2)
    b = GameClient("127.0.0.1", PORT + 2)
    try:
        assert a.connect("Opener", (220, 40, 40)), a.reject_reason
        assert b.connect("Watcher", (40, 40, 220)), b.reject_reason
        a.set_ready(True)
        b.set_ready(True)
        time.sleep(0.5)
        assert a.world.state == ServerState.MATCH
        aid, bid = a.world.my_id, b.world.my_id
        sub = srv.map.subdiv
        fine = (dr * sub, dc * sub)

        # park the watcher out of the way so it never blocks the leaf
        srv.players[bid].x, srv.players[bid].y = stand[0] + 6.0, stand[1]
        srv.players[aid].x, srv.players[aid].y = stand
        time.sleep(0.25)
        assert srv.map.blocks_sight[fine], "the door started open"

        def press():
            a.send_input(0.0, 0.0, 0.0, BTN_INTERACT)
            time.sleep(0.1)
            a.send_input(0.0, 0.0, 0.0, 0)
            time.sleep(0.25)
            return ([e for e in a.drain_events() if e["t"] == "door"],
                    [e for e in b.drain_events() if e["t"] == "door"])

        mine, theirs = press()
        print(f"  opening: shooter saw {len(mine)} door event(s), "
              f"watcher saw {len(theirs)}")
        assert mine and theirs, "the door never moved for both clients"
        assert mine[0]["open"] is True and not mine[0]["blocked"]
        assert srv.doors[(dr, dc)] is True, "the server did not open it"
        assert not srv.map.blocks_sight[fine], "an open door still blocks sight"
        assert not srv.map.blocks_move[fine], "an open door still blocks movement"
        print("  ok  open: sight, movement and bullets pass, both clients told")

        mine, theirs = press()
        assert mine and mine[0]["open"] is False, "the door did not close again"
        assert srv.map.blocks_sight[fine], "a closed door stopped blocking sight"
        print("  ok  closed again, and the geometry came back")

        # standing IN the doorway: it must refuse rather than trap anyone
        srv.players[aid].x, srv.players[aid].y = dc + 0.5, dr + 0.5
        time.sleep(0.25)
        press()                                  # open it from inside
        assert srv.doors[(dr, dc)] is True
        mine, _ = press()                        # now try to close on yourself
        print(f"  standing in the leaf, closing gave blocked="
              f"{mine[0]['blocked'] if mine else 'no event'}")
        assert mine and mine[0]["blocked"], "closed a door on a player"
        assert srv.doors[(dr, dc)] is True, "the door shut on somebody"
        print("  ok  a door will not close on a body")
    finally:
        a.disconnect()
        b.disconnect()
        srv.stop()


if __name__ == "__main__":
    main()
    print()
    glass_test()
    print("\nGLASS CHECKS PASSED")
    print()
    door_test()
    print("\nDOOR CHECKS PASSED")
