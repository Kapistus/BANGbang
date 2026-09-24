"""Headless check that a downed single-player character is a body on the floor.

It used to be that going down stopped your gun and nothing else: you could still
walk, turn, open doors, knock, reload and pick up a health pack from the floor.
The multiplayer server always refused a dead player's input, so this is about
main.py's own loop.

The loop reads the keyboard, mouse and event queue straight from pygame, so this
replaces those three: W, D, shift and the left button held, the mouse sweeping,
and every in-world key pressed. It runs the script twice — once on a LIVE player
first, to prove the harness can see walking, knocking, doors and pickups at all
(a freeze test that could not see a live player act would pass on anything),
then on a downed one, where all of it has to stop.

Uses copies of tests/range.map, with the spawn moved next to a door and on top
of a health pack.

    python -m tests.sp_downed_test
"""
import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from tests import range_spots as R

FRAMES = 70
SX, SY = R.SP_DOOR_STAND


def _map(src, name, spawn):
    tmp = tempfile.mkdtemp()
    doc = json.load(open(src))
    doc["player_spawn"] = list(spawn)
    path = os.path.join(tmp, name)
    json.dump(doc, open(path, "w"))
    return tmp, path


def run(map_path, downed, respawn_s=600.0):
    """Play FRAMES frames of main.py with every input held. Returns what the
    player managed to do."""
    import importlib
    import main
    importlib.reload(main)          # fresh module state per run
    from sim import audio, combat

    main.RESPAWN_DELAY = respawn_s  # 600 = stay down for the whole test
    got = {"body": None, "clips": [], "fires": 0, "sounds": []}

    real_commando = combat.player_commando

    def commando(x, y):
        b = real_commando(x, y)
        b.health = 0.0 if downed else b.max_health * 0.5
        got["body"] = b
        return b
    main.combat.player_commando = commando

    real_emit = main.emit

    def emit(m, cost, x, y, energy, label, *a, **kw):
        got["sounds"].append(label)
        return real_emit(m, cost, x, y, energy, label, *a, **kw)
    main.emit = emit
    audio.play = lambda name, *a, **kw: got["clips"].append(name)

    def fire(*a, **kw):
        got["fires"] += 1
    audio.play_fire = fire

    frame = {"n": 0}
    WALK = range(30, 56)            # act first, while still beside the door,
    held = {pygame.K_w, pygame.K_d, pygame.K_LSHIFT}   # then walk

    class Keys:
        def __getitem__(self, k):
            return frame["n"] in WALK and k in held
    pygame.key.get_pressed = lambda: Keys()
    pygame.mouse.get_pressed = lambda *a, **kw: (frame["n"] in WALK, False, False)

    def mouse_pos():
        n = frame["n"]
        return (300 + (n * 37) % 600, 200 + (n * 23) % 300)
    pygame.mouse.get_pos = mouse_pos

    # frame -> what happens. The door, the knock, the shot and the reload come
    # before any walking; the keys that would get in the live player's way
    # (sling, switch, inventory) come after it
    def key(k):
        return pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="")
    script = {
        5: [key(pygame.K_f)],
        6: [key(pygame.K_e)],
        8: [pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(640, 300), button=1)],
        14: [key(pygame.K_r)],
        20: [key(pygame.K_l)],
        22: [key(pygame.K_b)],
        57: [key(pygame.K_h)],
        59: [key(pygame.K_2)],
        61: [pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1)],
        63: [key(pygame.K_i)],
    }

    def events():
        n = frame["n"]
        frame["n"] += 1
        out = list(script.get(n, []))
        if n >= FRAMES:
            out.append(pygame.event.Event(pygame.QUIT))
        return out
    pygame.event.get = events

    main.main(map_path)
    b = got["body"]
    return {"start": None, "x": b.x, "y": b.y, "health": b.health,
            "clips": got["clips"], "fires": got["fires"],
            "sounds": got["sounds"]}


def main():
    pygame.init()
    tmps = []
    try:
        # next to the powered door in the horizontal wall
        t1, doors = _map(R.PATH, "range.map", R.SP_DOOR_STAND)
        # standing on a health pack
        t2, packs = _map(R.PATH, "range.map", R.HEALTH[1])
        tmps += [t1, t2]

        for downed in (False, True):
            who = "DOWNED" if downed else "live"
            a = run(doors, downed)
            moved = abs(a["x"] - SX) + abs(a["y"] - SY)
            knocks = a["sounds"].count("knock")
            door = sum(1 for s in a["sounds"] if s.endswith("door"))
            steps = sum(1 for s in a["sounds"] if s.startswith("step/"))
            reloads = a["clips"].count("reload")
            p = run(packs, downed)
            healed = p["health"] > (0.0 if downed else 50.0)
            print(f"  {who:6} player: moved {moved:5.2f} m, {steps} footstep(s), "
                  f"{knocks} knock(s), {door} door sound(s), {a['fires']} shot(s), "
                  f"{reloads} reload(s), health pack {'taken' if healed else 'left'}")
            if not downed:
                # the harness has to be able to see all of this happen
                assert moved > 0.3, "the harness cannot make a live player walk"
                assert knocks and door, "the harness cannot knock or open a door"
                assert a["fires"] and reloads, "the harness cannot fire or reload"
                assert healed, "the harness cannot take a health pack"
            else:
                assert moved < 1e-6, f"a downed player moved {moved:.3f} m"
                assert steps == 0, "a downed player made footsteps"
                assert knocks == 0, "a downed player knocked"
                assert door == 0, "a downed player opened a door"
                assert a["fires"] == 0 and p["fires"] == 0, "a downed player fired"
                assert reloads == 0, "a downed player reloaded"
                assert not healed, "a health pack revived a downed player"
        # --- and the freeze lets go: down at the start, back up almost at
        # once, then the walking frames have to move the player again
        r = run(doors, True, respawn_s=0.05)
        moved = abs(r["x"] - SX) + abs(r["y"] - SY)
        print(f"  downed, then respawned: moved {moved:.2f} m afterwards, "
              f"health {r['health']:.0f}")
        assert r["health"] > 0.0, "the player never respawned"
        assert moved > 0.3, "the player stayed frozen after respawning"
        print("\nDOWNED CHECKS PASSED")
    finally:
        for t in tmps:
            shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
