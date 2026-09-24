"""Headless check that the map editor opens, draws and edits.

Written after shipping a spawn-placement mode whose status bar had no entry for
it: every model-level test passed, because they called apply() and save() and
never rendered a frame. The editor crashed on the first frame after pressing m.

So this renders. For every map in tests/ it opens the editor, draws a complete
frame in every mode — grid, panel and status bar — places an object, and draws
again. That is the path a person takes, and it is the one that was untested.

    python -m tests.editor_test
"""
import os
import shutil
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

import editor as ed
from sim import mapfile
from sim.tilemap import validate_spawns
from tests import range_spots as R

MODES = ["paint", "spawn", "mpspawn", "guard", "light", "entity"]


def frame(app):
    """One whole frame, the way run() draws it."""
    app.draw_grid()
    app.draw_panel()
    app.draw_status()


def main():
    pygame.init()
    pygame.display.set_mode((1200, 800))
    maps = sorted(p for p in os.listdir(R.MAPS_DIR) if p.endswith(".map"))
    assert maps, "no .map files to test with"
    print(f"{len(maps)} map(s): {', '.join(maps)}")

    for name in maps:
        app = ed.Editor(os.path.join(R.MAPS_DIR, name))
        for mode in MODES:
            app.mode = mode
            frame(app)
        # a mode nobody has written yet must not take the status bar down
        app.mode = "unheard_of_mode"
        frame(app)
        print(f"  {name:22} drew in all {len(MODES)} modes")

    # --- placing, turning, tagging and deleting a multiplayer spawn
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "editortest.map")
        shutil.copy(R.PATH, path)
        app = ed.Editor(path)
        app.doc["spawn_points"] = []     # the range's own four are not the point
        app.mode = "mpspawn"
        app.spawn_team = "a"
        app.spawn_facing = 90.0
        here = (ed.PANEL_W + 200, 300)
        app.apply(*here)
        app.apply(ed.PANEL_W + 260, 340)
        frame(app)
        assert len(app.doc["spawn_points"]) == 2, "placing did nothing"

        app.turn_mpspawn(*here, 45.0)
        app.cycle_mpspawn_team(*here, 1)
        first = app.doc["spawn_points"][0]
        print(f"  placed 2, first is now {first}")
        assert first["facing_deg"] == 135.0, "the wheel did not turn it"
        assert first["team"] == "b", "the bracket keys did not retag it"

        mapfile.save(app.doc, path)
        m = mapfile.load_map(path)
        assert len(m.spawn_points) == 2, "spawns did not survive a save"
        assert m.spawn_points[0].facing_deg == 135.0
        assert m.spawn_points[0].team == "b"
        print(f"  saved and reloaded: {m.spawn_points}")

        problems = validate_spawns(m)
        print(f"  validation says: {problems or 'nothing wrong'}")

        assert app.delete_mpspawn_near(*here), "delete found nothing"
        frame(app)
        assert len(app.doc["spawn_points"]) == 1
        print("  deleted one, drew again")

        # a brand new map starts with the key present and empty
        fresh = mapfile.new_map(20, 20, app.ts, "fresh")
        assert fresh.get("spawn_points") == [], \
            f"new maps should carry an empty spawn list, got {fresh.get('spawn_points')!r}"
        print("  a new map carries an empty spawn list")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    keymap_test()
    mouse_test()
    save_check_test()

    print("\nALL CHECKS PASSED")


def save_check_test():
    """Saving tells you whether the lobby will offer the map.

    The lobby only lists maps that pass net.maps.validate, so a map that fails
    it simply is not there to host. The editor runs the same check on save so
    you find out while the map is still open, not by looking for it in the list.
    Saving itself is never blocked. All of this happens in a temp folder."""
    import json
    import editor as ed

    print("\n-- save check")
    tmp = tempfile.mkdtemp()
    try:
        good = os.path.join(tmp, "good.map")
        shutil.copy(R.PATH, good)
        app = ed.Editor(good)
        app.do_save()
        assert os.path.exists(good), "saving a playable map did not write it"
        assert not app.save_problems, f"the range flagged: {app.save_problems}"
        frame(app)
        print(f"  a copy of the range: {app.toast!r}")

        # the same map with one tile the tileset has never heard of
        doc = json.load(open(good))
        doc["floor"][5][5] = "no_such_tile"
        bad = os.path.join(tmp, "bad.map")
        json.dump(doc, open(bad, "w"))
        app = ed.Editor(bad)
        app.dirty = True
        app.do_save()
        assert os.path.exists(bad), "a map with problems was not saved at all"
        assert app.save_problems, "an unknown tile was not caught"
        assert "NOT PLAYABLE" in app.toast
        frame(app)
        print(f"  with one unknown tile: saved, and {app.save_problems[0]!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def mouse_test():
    """Every handler the run loop dispatches to has to exist, and has to
    survive being clicked in every mode.

    This was missing: a rewrite of the key handler deleted `on_mousedown`,
    and every other check here still passed because none of them ever sent a
    mouse event. The editor crashed on the first click. So this does two
    things — reads `run()` and insists every `self.on_*` it calls is defined,
    then actually clicks, right-clicks, drags and scrolls in each mode.

    Works on a COPY of a real map in a temp folder, and never saves: the
    right button erases in paint mode, and this must not touch your maps.
    """
    import inspect
    import re
    import editor as ed

    print("\n-- mouse")
    src = inspect.getsource(ed.Editor.run)
    called = sorted(set(re.findall(r"self\.(on_\w+)\(", src)))
    missing = [m for m in called if not hasattr(ed.Editor, m)]
    print(f"  run() dispatches to: {' '.join(called)}")
    assert not missing, f"run() calls handlers that do not exist: {missing}"

    tmp = tempfile.mkdtemp()
    try:
        src_map = R.PATH
        work = os.path.join(tmp, "mouse_copy.map")
        shutil.copy(src_map, work)
        before = open(src_map, "rb").read()

        app = ed.Editor(work)
        app.ent_kind = "health"       # places without a name prompt
        w, h = app.screen.get_size()
        at = (ed.PANEL_W + (w - ed.PANEL_W) // 2, h // 2)
        panel = (ed.PANEL_W // 2, 200)

        def ev(kind, **kw):
            return pygame.event.Event(kind, **kw)

        for mode in ("paint", "entity", "light", "guard", "spawn", "mpspawn"):
            app.mode = mode
            app.ent_kind = "health"
            for button in (1, 3, 2):
                app.on_mousedown(ev(pygame.MOUSEBUTTONDOWN, pos=at, button=button))
                app.on_motion(ev(pygame.MOUSEMOTION, pos=(at[0] + 18, at[1]),
                                 rel=(18, 0), buttons=(button == 1, button == 2,
                                                       button == 3)))
                app.painting = app.erasing = app.panning = False
            app.on_wheel(ev(pygame.MOUSEWHEEL, x=0, y=1))
            app.on_wheel(ev(pygame.MOUSEWHEEL, x=0, y=-1))
            frame(app)
        # the panel takes clicks too
        app.on_mousedown(ev(pygame.MOUSEBUTTONDOWN, pos=panel, button=1))
        frame(app)
        print("  left/right/middle click, drag and scroll in all 6 modes, "
              "and a panel click: no crash")

        assert open(src_map, "rb").read() == before, \
            "the real map changed during the mouse test"
        print("  tests/range.map itself is byte-identical afterwards")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def keymap_test():
    """Every hotkey has to be one unshifted key on a Nordic layout.

    The editor used to drive team tagging, light tuning, entity kinds and
    guard weapons off [ ] and , . — all AltGr combinations on a Finnish
    keyboard. It is not enough to fix the bindings: nothing here pressed a key,
    so the binding table could rot without a single test noticing. This presses
    every one of them.
    """
    import editor as ed

    print("\n-- keymap")
    # AltGr on a Finnish layout, or otherwise awkward to reach
    BANNED = {pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET, pygame.K_BACKSLASH,
              pygame.K_LEFTPAREN, pygame.K_RIGHTPAREN, pygame.K_AT,
              pygame.K_DOLLAR, pygame.K_CARET, pygame.K_BACKQUOTE}
    ALLOWED = (set(range(pygame.K_0, pygame.K_9 + 1))
               | {getattr(pygame, f"K_{c}") for c in "qwertyuiop"}
               | {pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_KP_ENTER,
                  pygame.K_BACKSPACE, pygame.K_DELETE, pygame.K_TAB,
                  pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN,
                  pygame.K_SPACE})

    app = ed.Editor(R.PATH)
    bound = (set(ed.Editor.MODE_KEYS) | set(ed.Editor.CYCLE_A)
             | set(ed.Editor.CYCLE_B)
             | {pygame.K_w, pygame.K_i, pygame.K_o, pygame.K_p, pygame.K_TAB,
                pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9,
                pygame.K_0})
    names = sorted(pygame.key.name(k) for k in bound)
    print(f"  {len(bound)} bindings: {' '.join(names)}")
    assert not (bound & BANNED), \
        f"these need AltGr: {[pygame.key.name(k) for k in bound & BANNED]}"
    assert bound <= ALLOWED, \
        f"outside the number/top-letter rows: {[pygame.key.name(k) for k in bound - ALLOWED]}"

    # --- every mode key reaches its mode, and drawing survives it
    for key, want in ed.Editor.MODE_KEYS.items():
        assert app.on_key(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
        assert app.mode == want, \
            f"{pygame.key.name(key)} should enter {want}, got {app.mode}"
        frame(app)
    print(f"  all {len(ed.Editor.MODE_KEYS)} mode keys reach their mode and draw")

    # --- the adjust keys, in every mode, must not raise. 1/2 and 3/4 mean
    #     different things per mode, which is exactly where a rewrite breaks
    for mode in ("paint", "light", "entity", "mpspawn", "guard", "spawn"):
        app.mode = mode
        for key in list(ed.Editor.CYCLE_A) + list(ed.Editor.CYCLE_B) + \
                [pygame.K_5, pygame.K_6]:
            for mod in (0, pygame.KMOD_SHIFT):
                assert app.on_key(pygame.event.Event(
                    pygame.KEYDOWN, key=key, mod=mod))
        frame(app)
    print("  1-6 (and shifted) handled in every mode without raising")

    # --- the toggles actually toggle
    app.mode = "paint"
    for key, attr in ((pygame.K_i, "show_grid"), (pygame.K_o, "show_roof")):
        was = getattr(app, attr)
        app.on_key(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
        assert getattr(app, attr) is not was, f"{pygame.key.name(key)} did nothing"
    was = app.place_role
    app.on_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w, mod=0))
    assert app.place_role != was, "w no longer swaps floor/wall"
    print("  i, o and w still toggle what they say they do")

    # --- entity kinds cycle through all five, packs included
    app.mode = "entity"
    seen = {app.ent_kind}
    for _ in range(len(ed.ENT_KINDS) + 1):
        app.on_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2, mod=0))
        seen.add(app.ent_kind)
    assert seen == set(ed.ENT_KINDS), f"2 cycled through {sorted(seen)}"
    print(f"  2 cycles all {len(ed.ENT_KINDS)} entity kinds: {' '.join(ed.ENT_KINDS)}")

    # --- esc still quits, and clearing guards still clears
    app.mode = "paint"
    app.doc["guards"] = [{"id": "g", "patrol": [[1, 1]], "weapon": "pistol"}]
    app.on_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_7, mod=0))
    assert app.doc["guards"] == [], "7 did not clear the guards"
    assert app.on_key(pygame.event.Event(
        pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)) is False, "esc no longer quits"
    print("  7 clears guards, esc still quits")


if __name__ == "__main__":
    sys.exit(main())
