"""Headless test of the lobby's layout.

The lobby is drawn at absolute coordinates, so it is the screen that quietly
breaks when something is added to it: a class card with one more weapon in it,
a longer ability line, a window that changed size. None of that raises - it
just draws off the edge, or stops a line short, and the first anyone knows is
that the last few words of an ability are missing.

What this checks:

  * every button the lobby puts on screen is inside the window, in both game
    modes, with and without the class prompt open
  * the panels do not run off the bottom or overlap the button band
  * each class card has room for its whole description - the ability line and
    the blurb, wrapped to the card's own width
  * the two small screens fit the frame they are drawn into

Run from the project root:   python -m tests.lobby_layout_test
"""
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

import lobby
from lobby import Lobby, JoinScreen, StartScreen, W, H
from net import GameServer, GameClient
from net.protocol import GameMode
from sim import classes
from tests import range_spots as R

PORT = 47987


def wait_for(pred, seconds=4.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        if pred():
            return True
        time.sleep(0.02)
    return False


def cards_test():
    """Every class card holds its whole description."""
    pygame.init()
    pygame.display.set_mode((1, 1))
    f_sm = lobby._make_fonts()[2]
    n = len(classes.ORDER)
    gap, margin = 12, 18
    cw = min(lobby.CARD_W_MAX, (W - 2 * margin - gap * (n - 1)) // n)
    worst = None
    for i, key in enumerate(classes.ORDER):
        c = classes.get(key)
        r = pygame.Rect(margin + i * (cw + gap), lobby.CARD_TOP, cw, lobby.CARD_H)
        ab = classes.ability_of(key)
        lines = lobby.wrap(f"{ab.hint} · {c.blurb}", f_sm, r.w - 24)
        room = lobby.card_room(c, r)
        spare = room - len(lines)
        print(f"  {c.name:14} {cw} px wide: {len(lines)} lines of "
              f"{room} ({spare} spare)")
        assert spare >= 0, \
            f"{c.name}'s description is {-spare} line(s) too long for its card"
        worst = spare if worst is None else min(worst, spare)
    assert cw >= 180, f"the cards are down to {cw} px; the window shrank"
    assert worst >= 1, \
        "no card has a spare line: a font a shade wider than this one truncates"
    print("\nCARD CHECKS PASSED")


def screen_test():
    """Buttons stay inside the window, whatever the lobby is showing."""
    srv = GameServer([(10.5, 10.5), (29.5, 10.5)], port=PORT, map_id=R.MAP_ID,
                     maps_dir=R.MAPS_DIR)
    srv.start()
    cli = GameClient("127.0.0.1", PORT)
    assert cli.connect("Layout", (200, 60, 60)), cli.reject_reason
    pygame.init()
    surf = pygame.Surface((W, H))
    fonts = lobby._make_fonts()
    try:
        lb = Lobby(cli, fonts, maps_dir=R.MAPS_DIR, auto_join=False)
        assert wait_for(lambda: cli.world.my_id is not None), "never joined"
        frame = pygame.Rect(0, 0, W, H)
        for mode in (GameMode.FFA, GameMode.TEAM):
            cli.world.mode = mode
            for prompt in (False, True):
                lb.choosing_class = prompt
                lb.draw(surf)
                for b in lb._buttons:
                    assert frame.contains(b.rect), \
                        f"a button at {tuple(b.rect)} is outside the window"
                print(f"  {mode.name.lower():5} "
                      f"{'prompt' if prompt else 'lobby ':6}: "
                      f"{len(lb._buttons)} buttons, all inside")
        # the panels: nothing may reach into the button band
        assert lb.picker.rect.bottom < lobby.BTN_Y, "the map list runs into the buttons"
        assert 96 + lobby.LIST_H + 16 + 140 < H, "the class box runs off the bottom"
        assert lb.picker.rect.right < lobby.COL_X, "the map list is under the roster"

        # and the two small screens fit the panel they are drawn into
        panel = pygame.Surface((lobby.PANEL_W, lobby.PANEL_H))
        pane = pygame.Rect(0, 0, lobby.PANEL_W, lobby.PANEL_H)
        start = StartScreen(fonts)
        start.draw(panel)
        for b in (start.host_btn, start.join_btn, start.exit_btn):
            assert pane.contains(b.rect), f"start screen: {tuple(b.rect)}"
        join = JoinScreen(fonts, "", mode="join")
        join.draw(panel)
        for b in (join.connect_btn, join.exit_btn):
            assert pane.contains(b.rect), f"join screen: {tuple(b.rect)}"
        print("  start and join screens fit their panel")
        print("\nSCREEN CHECKS PASSED")
    finally:
        cli.disconnect()
        srv.stop()


if __name__ == "__main__":
    cards_test()
    screen_test()
    print("\nALL LOBBY LAYOUT TESTS PASSED")
