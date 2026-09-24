"""
lobby.py — BANGbang multiplayer lobby (pygame window).

Self-contained and runnable on its own for testing, and designed to fold into
main.py's state machine: the reusable part is the Lobby class, which renders one
frame and handles one event against a live GameClient. run_lobby() is the
standalone harness (own window + loop); in main.py you instead construct
Lobby(client) once and call lobby.handle_event(ev) / lobby.draw(surface) from
your existing loop while world.state == LOBBY, then switch to your match render
on the LobbyResult.START / when world.state becomes MATCH.

Standalone usage:
    python lobby.py                 # start screen: host or join
    python lobby.py 192.168.1.42    # straight to joining, IP pre-filled
    python lobby.py --host          # straight to hosting on this machine

Flow:
    START screen -> host a game here, or join one elsewhere
    CONNECT screen -> host: pick a port, start the server, connect to it on
                      loopback / join: enter host IP, name, colour -> Connect
    LOBBY screen -> live player list (name/colour/team/ready)
                    edit name, cycle colour, pick team (team mode),
                    toggle Ready; host sees Start + mode/duration.
    Match start / Exit end the lobby loop with a LobbyResult.

The lobby never runs the game. On START (or world.state==MATCH) it returns
control so main.py can begin the match render with the same GameClient.
"""
from __future__ import annotations

import enum
import sys

import pygame

from net import GameClient, GameServer
from net.protocol import GameMode, ServerState, Team, DEFAULT_PORT
from net.mappicker import MapPicker
from sim import classes, sprites


# --------------------------------------------------------------------- palette

BG = (18, 20, 24)
PANEL = (28, 31, 37)
PANEL_LT = (38, 42, 50)
LINE = (52, 57, 66)
TEXT = (222, 226, 232)
TEXT_DIM = (140, 146, 156)
ACCENT = (90, 170, 250)
GOOD = (80, 200, 110)
WARN = (230, 180, 60)
BAD = (220, 90, 90)
TEAM_A = (80, 140, 230)
TEAM_B = (230, 110, 80)

# The lobby palette is the sprite palette: picking a swatch here picks which
# coloured soldier art you wear in the match.
SWATCHES = list(sprites.SWATCHES)

# The lobby window. Everything here is laid out against these two numbers
# rather than written down as pixels: the screens people have are 1920x1080
# and up, and a 820x620 window put five class cards in 147 px each, which is
# where the descriptions ran out of room.
W, H = 1280, 800

# the right-hand column: the player list, and the class box under it
COL_X = W - 340
COL_W = 300
LIST_H = 420
# the bottom band: every button and note on the lobby screen hangs off this
BTN_Y = H - 150
# the class cards, on the prompt that opens over the lobby
CARD_TOP, CARD_H = 160, 380
CARD_W_MAX = 200          # they are read side by side, so they are columns
                          # rather than pages: wider than this and the eye
                          # stops taking two of them in at once

# the start and join screens are composed for this size and drawn centred in
# the window, rather than pinned to its top-left corner
PANEL_W, PANEL_H = 820, 620


CARD_LINE_H = 17          # one line of card description


def wrap(text: str, font, width: int) -> list:
    """`text` broken on word boundaries into lines that fit `width`."""
    out, line = [], ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if line and font.size(trial)[0] > width:
            out.append(line)
            line = word
        else:
            line = trial
    if line:
        out.append(line)
    return out


def card_text_top(cls, rect) -> int:
    """Where a class card's description starts: under the name, the four
    stats, the weapons it carries and the ability's name. Shared with
    tests/lobby_layout_test.py, which checks the description still fits under
    it - a card that cannot show the last line of an ability is a card that
    lies about the class."""
    return (rect.y + 12 + 36 + 4 * 21 + 8 + 20
            + 19 * len(cls.loadout) + 8 + 19 + 21)


def card_room(cls, rect) -> int:
    """How many lines of description that card has room for."""
    return max(0, (rect.bottom - 10 - card_text_top(cls, rect)) // CARD_LINE_H)


class LobbyResult(enum.Enum):
    START = "start"       # match began; hand off to game render
    EXIT = "exit"         # user quit the lobby / window closed
    DISCONNECT = "disc"   # lost the server


# --------------------------------------------------------------------- widgets

class Button:
    def __init__(self, rect, label, on_click, enabled=True, tone=ACCENT):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.enabled = enabled
        self.tone = tone
        self.hover = False

    def draw(self, surf, font):
        col = self.tone if self.enabled else LINE
        bg = PANEL_LT if (self.hover and self.enabled) else PANEL
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        pygame.draw.rect(surf, col, self.rect, width=2, border_radius=6)
        txt = font.render(self.label, True, TEXT if self.enabled else TEXT_DIM)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def handle(self, ev):
        if ev.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(ev.pos)
        elif (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
              and self.enabled and self.rect.collidepoint(ev.pos)):
            self.on_click()
            return True
        return False


class TextField:
    def __init__(self, rect, text="", placeholder="", max_len=20,
                 on_change=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.placeholder = placeholder
        self.max_len = max_len
        self.on_change = on_change
        self.active = False

    def draw(self, surf, font):
        pygame.draw.rect(surf, (14, 15, 18), self.rect, border_radius=5)
        pygame.draw.rect(surf, ACCENT if self.active else LINE,
                         self.rect, width=2, border_radius=5)
        show = self.text if self.text else self.placeholder
        col = TEXT if self.text else TEXT_DIM
        txt = font.render(show, True, col)
        surf.blit(txt, (self.rect.x + 10,
                        self.rect.y + (self.rect.h - txt.get_height()) // 2))
        if self.active and (pygame.time.get_ticks() // 500) % 2 == 0:
            cx = self.rect.x + 10 + font.size(self.text)[0] + 1
            pygame.draw.line(surf, TEXT, (cx, self.rect.y + 6),
                             (cx, self.rect.bottom - 6), 2)

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.active = self.rect.collidepoint(ev.pos)
        elif ev.type == pygame.KEYDOWN and self.active:
            if ev.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_TAB):
                self.active = False
            elif ev.unicode and ev.unicode.isprintable():
                if len(self.text) < self.max_len:
                    self.text += ev.unicode
            else:
                return
            if self.on_change:
                self.on_change(self.text)


# --------------------------------------------------------------------- lobby

class Lobby:
    """Reusable lobby screen. Construct once with a connected GameClient, then
    call handle_event() and draw() each frame. Poll result each frame: if it is
    not None, the lobby is done (START/EXIT/DISCONNECT)."""

    def __init__(self, client: GameClient, fonts, maps_dir="maps",
                 auto_join=False):
        """auto_join: drop straight into a match already in progress. True when
        the player has just connected — they clicked Connect, they want to play
        — and False when they have stepped out of a match on purpose and are
        sitting here deciding whether to go back in."""
        self.cli = client
        self.f_big, self.f, self.f_sm = fonts
        self.auto_join = auto_join
        self._join_sent = False
        # a fresh connection is asked to pick a class before anything else;
        # somebody back from a match already has one and can change it with
        # the class row
        self.choosing_class = auto_join
        self.result: LobbyResult | None = None
        self._colour_idx = 0
        self._buttons: list[Button] = []
        self._build_static()

        # name field seeded from our current known name
        me = self._me()
        self.name_field = TextField(
            (150, 92, 260, 40), text=(me.name if me else "Player"),
            placeholder="your name", on_change=self._on_name_change)

        # map picker occupies the left-column space below the team row
        self.picker = MapPicker(
            client, maps_dir, (self.f, self.f_sm),
            rect=(34, 240, COL_X - 58, BTN_Y - 264))

    # ---- helpers to read world ----

    def _me(self):
        w = self.cli.world
        return w.players.get(w.my_id)

    def _sorted_players(self):
        return sorted(self.cli.world.players.values(), key=lambda p: p.id)

    def _all_ready(self):
        ps = list(self.cli.world.players.values())
        return len(ps) >= 2 and all(p.ready for p in ps)

    # ---- callbacks ----

    def _on_name_change(self, text):
        self.cli.set_name(text or "Player")

    def _cycle_colour(self, step):
        self._colour_idx = (self._colour_idx + step) % len(SWATCHES)
        self.cli.set_colour(SWATCHES[self._colour_idx])

    def _pick_colour(self, idx):
        self._colour_idx = idx
        self.cli.set_colour(SWATCHES[idx])

    def _my_class(self) -> str:
        me = self._me()
        return classes.get(me.cls if me else None).key

    def pick_class(self, key: str) -> None:
        """Choose a class. From the join prompt this also lets an automatic
        join go ahead, now that there is a class to join as."""
        self.cli.set_class(key)
        self.choosing_class = False

    def _toggle_ready(self):
        me = self._me()
        self.cli.set_ready(not (me.ready if me else False))

    def _set_team(self, team):
        self.cli.set_team(team)

    def _start(self):
        # host force-start; server still requires >=2 players
        self.cli.force_start()

    def _join_match(self):
        self._join_sent = True
        self.cli.join_match()

    def _exit(self):
        self.result = LobbyResult.EXIT

    def _cycle_mode(self):
        w = self.cli.world
        new = GameMode.TEAM if w.mode == GameMode.FFA else GameMode.FFA
        self.cli.set_config(mode=new)

    def _bump_duration(self, delta):
        w = self.cli.world
        self.cli.set_config(duration_s=max(120, min(3600, w.duration_s + delta)))

    def _build_static(self):
        pass  # dynamic buttons are rebuilt per-frame in draw()

    # ---- event handling ----

    def handle_event(self, ev):
        if ev.type == pygame.QUIT:
            self.result = LobbyResult.EXIT
            return
        if not self.choosing_class:
            self.name_field.handle(ev)
            self.picker.handle_event(ev)
        for b in self._buttons:
            if b.handle(ev):
                break
        if self.choosing_class:
            return                  # only the class cards take clicks
        # colour swatch clicks
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for i, r in enumerate(self._swatch_rects()):
                if r.collidepoint(ev.pos):
                    self._pick_colour(i)

    def pump_events(self):
        """Drain GameClient events; detect match start / disconnect.

        A match being in progress is no longer the same thing as being in it:
        a player who stepped out sits here watching the roster until they
        choose to go back."""
        for e in self.cli.drain_events():
            if e["t"] == "match_start":
                self.result = LobbyResult.START
            elif e["t"] == "disconnected":
                self.result = LobbyResult.DISCONNECT
        w = self.cli.world
        if w.state == ServerState.MATCH and w.playing and self.result is None:
            self.result = LobbyResult.START
        elif (self.auto_join and not self._join_sent and not self.choosing_class
              and w.state == ServerState.MATCH and not w.playing):
            self._join_match()

    # ---- layout constants ----

    def _swatch_rects(self):
        rects = []
        x0, y0 = 100, 150
        for i in range(len(SWATCHES)):
            rects.append(pygame.Rect(x0 + i * 30, y0, 24, 24))
        return rects

    # ---- draw ----

    def draw(self, surf):
        self._buttons = []
        w = self.cli.world
        me = self._me()
        surf.fill(BG)

        title = self.f_big.render("BANGbang — Lobby", True, TEXT)
        surf.blit(title, (32, 26))
        sub = self.f_sm.render(
            f"mode: {'Team DM' if w.mode == GameMode.TEAM else 'Free-for-all'}"
            f"    match: {w.duration_s}s", True, TEXT_DIM)
        surf.blit(sub, (34, 64))

        # ---- left column: your settings ----
        lbl = self.f_sm.render("NAME", True, TEXT_DIM)
        surf.blit(lbl, (34, 100))
        self.name_field.draw(surf, self.f)

        lbl = self.f_sm.render("COLOUR", True, TEXT_DIM)
        surf.blit(lbl, (34, 132))
        for i, r in enumerate(self._swatch_rects()):
            pygame.draw.rect(surf, SWATCHES[i], r, border_radius=4)
            sel = (me and tuple(me.colour) == SWATCHES[i])
            pygame.draw.rect(surf, TEXT if sel else LINE, r,
                             width=3 if sel else 1, border_radius=4)

        # team pick (team mode only)
        if w.mode == GameMode.TEAM:
            lbl = self.f_sm.render("TEAM", True, TEXT_DIM)
            surf.blit(lbl, (34, 190))
            ta = Button((150, 186, 70, 34), "A",
                        lambda: self._set_team(Team.A),
                        tone=TEAM_A)
            tb = Button((226, 186, 70, 34), "B",
                        lambda: self._set_team(Team.B),
                        tone=TEAM_B)
            ta.draw(surf, self.f); tb.draw(surf, self.f)
            self._buttons += [ta, tb]
            if me:
                mark = self.f_sm.render(
                    f"you: {me.team.name}", True, TEXT_DIM)
                surf.blit(mark, (306, 194))

        # map picker (left column, below team row)
        self.picker.draw(surf)

        # ---- right column: player list ----
        panel = pygame.Rect(COL_X, 96, COL_W, LIST_H)
        pygame.draw.rect(surf, PANEL, panel, border_radius=8)
        pygame.draw.rect(surf, LINE, panel, width=1, border_radius=8)
        hdr = self.f_sm.render("PLAYERS", True, TEXT_DIM)
        surf.blit(hdr, (panel.x + 14, panel.y + 10))

        y = panel.y + 36
        for p in self._sorted_players():
            row = pygame.Rect(panel.x + 6, y - 2, panel.w - 12, 28)
            if p.id == w.my_id:
                pygame.draw.rect(surf, PANEL_LT, row, border_radius=4)
            sw = pygame.Rect(panel.x + 14, y + 4, 16, 16)
            pygame.draw.rect(surf, tuple(p.colour), sw, border_radius=3)
            pygame.draw.rect(surf, LINE, sw, width=1, border_radius=3)
            nm = p.name + ("  \u2605" if p.is_host else "")   # star = host
            # truncate name to fit before the tag/dot zone
            maxw = panel.w - 110
            while self.f.size(nm)[0] > maxw and len(nm) > 4:
                nm = nm[:-2]
            surf.blit(self.f.render(nm, True, TEXT), (panel.x + 38, y))
            # team tag (fixed column)
            if w.mode == GameMode.TEAM:
                tcol = TEAM_A if p.team == Team.A else TEAM_B
                tag = self.f_sm.render(p.team.name, True, tcol)
                surf.blit(tag, (panel.right - 54, y + 3))
            # ready dot (fixed far-right); a hollow one while their client
            # is still fetching the map - readying up waits for that
            dot = GOOD if p.ready else WARN
            pygame.draw.circle(surf, dot, (panel.right - 20, y + 10), 6,
                               width=0 if getattr(p, "has_map", True) else 2)
            y += 30

        cnt = self.f_sm.render(
            f"\u2605 host · {len(w.players)} connected · "
            f"{sum(p.ready for p in w.players.values())} ready",
            True, TEXT_DIM)
        surf.blit(cnt, (panel.x + 14, panel.bottom - 26))

        # ---- your class, for your next spawn ----
        box = pygame.Rect(COL_X, 96 + LIST_H + 16, COL_W, 140)
        pygame.draw.rect(surf, PANEL, box, border_radius=8)
        pygame.draw.rect(surf, LINE, box, width=1, border_radius=8)
        surf.blit(self.f_sm.render("CLASS", True, TEXT_DIM),
                  (box.x + 14, box.y + 8))
        mine = self._my_class()
        prev_b = Button((box.x + 62, box.y + 5, 28, 26), "<",
                        lambda: self.cli.set_class(classes.cycle(
                            self._my_class(), -1)))
        next_b = Button((box.right - 38, box.y + 5, 28, 26), ">",
                        lambda: self.cli.set_class(classes.cycle(
                            self._my_class(), 1)))
        prev_b.draw(surf, self.f)
        next_b.draw(surf, self.f)
        self._buttons += [prev_b, next_b]
        nm = self.f.render(classes.get(mine).name, True, TEXT)
        surf.blit(nm, nm.get_rect(center=((box.x + 90 + box.right - 38) // 2,
                                          box.y + 18)))
        c = classes.get(mine)
        short = (f"{c.hp:.0f} hp · {c.sh:.0f} sh · "
                 f"{c.armour:.0f}% armour · {c.spd:.1f} m/s")
        for j, line in enumerate((short, classes.weapon_line(mine),
                                  "space: " + classes.ability_of(mine).name)):
            t = self.f_sm.render(line, True, TEXT_DIM)
            while t.get_width() > box.w - 20 and len(line) > 8:
                line = line[:-2]
                t = self.f_sm.render(line + "…", True, TEXT_DIM)
            surf.blit(t, (box.x + 12, box.y + 38 + j * 20))
        surf.blit(self.f_sm.render("applies from your next spawn",
                                   True, TEXT_DIM), (box.x + 12, box.y + 96))

        # ---- the map: have it, fetching it, or could not get it ----
        st = w.map_status
        if st and st != "ready":
            msg = (f"downloading map '{w.map_id}' from the host..."
                   if st == "downloading" else f"map '{w.map_id}': {st}")
            col = TEXT_DIM if st == "downloading" else BAD
            surf.blit(self.f_sm.render(msg[:70], True, col), (34, BTN_Y - 24))

        # ---- bottom bar: ready / host controls / exit ----
        ready_on = bool(me and me.ready)
        mid_match = (self.cli.world.state == ServerState.MATCH
                     and not self.cli.world.playing)
        if mid_match:
            # a match is running without us: the only useful button is the one
            # that puts us in it
            join_btn = Button((34, BTN_Y, 240, 48), "JOIN MATCH IN PROGRESS",
                              self._join_match, tone=GOOD)
            join_btn.draw(surf, self.f)
            self._buttons.append(join_btn)
            surf.blit(self.f_sm.render(
                "you will spawn away from the fighting", True, TEXT_DIM),
                (34, BTN_Y + 54))
        else:
            ready_btn = Button(
                (34, BTN_Y, 170, 48),
                "READY  \u2713" if ready_on else "READY UP",
                self._toggle_ready, tone=GOOD if ready_on else ACCENT)
            ready_btn.draw(surf, self.f)
            self._buttons.append(ready_btn)

        if w.is_host:
            mode_btn = Button((220, BTN_Y, 150, 48), "Mode: "
                              + ("Team" if w.mode == GameMode.TEAM else "FFA"),
                              self._cycle_mode, tone=ACCENT)
            mode_btn.draw(surf, self.f)
            minus = Button((386, BTN_Y, 36, 48), "-",
                           lambda: self._bump_duration(-60))
            plus = Button((526, BTN_Y, 36, 48), "+",
                          lambda: self._bump_duration(60))
            minus.draw(surf, self.f); plus.draw(surf, self.f)
            dtxt = self.f_sm.render(f"{w.duration_s // 60}m",
                                    True, TEXT)
            surf.blit(dtxt, dtxt.get_rect(
                center=(484, BTN_Y + 24)))
            self._buttons += [mode_btn, minus, plus]

            can_start = len(w.players) >= 2
            start_btn = Button(
                (34, BTN_Y + 62, 300, 52),
                "START GAME" if can_start else "START (need 2+ players)",
                self._start, enabled=can_start, tone=GOOD)
            start_btn.draw(surf, self.f)
            self._buttons.append(start_btn)
            hint = self.f_sm.render(
                "auto-starts when everyone is ready", True, TEXT_DIM)
            surf.blit(hint, (34, BTN_Y + 120))
        else:
            waiting = self.f_sm.render(
                "waiting for host to start (auto when all ready)",
                True, TEXT_DIM)
            surf.blit(waiting, (34, BTN_Y + 78))

        exit_btn = Button((W - 186, BTN_Y + 62, 152, 52), "EXIT",
                          self._exit, tone=BAD)
        exit_btn.draw(surf, self.f)
        self._buttons.append(exit_btn)

        if self.choosing_class:
            self._draw_class_prompt(surf)

    def _card_rects(self):
        """One card per class, sized to fit however many there are."""
        n = len(classes.ORDER)
        gap, margin, ch = 12, 18, CARD_H
        cw = min(CARD_W_MAX, (W - 2 * margin - gap * (n - 1)) // n)
        x0 = (W - (cw * n + gap * (n - 1))) // 2
        return [pygame.Rect(x0 + i * (cw + gap), CARD_TOP, cw, ch)
                for i in range(n)]

    def _draw_class_prompt(self, surf):
        """The first thing a new arrival sees: the classes side by side, with
        what each one is and carries. Picking one closes it; the class row in
        the lobby changes it later."""
        self._buttons = []                   # nothing underneath is live
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((8, 9, 12, 225))
        surf.blit(shade, (0, 0))
        head = self.f_big.render("Choose your class", True, TEXT)
        surf.blit(head, head.get_rect(midtop=(W // 2, 70)))
        sub = self.f_sm.render("you can change it in the lobby, or with esc "
                               "during a match - it applies when you next "
                               "spawn", True, TEXT_DIM)
        surf.blit(sub, sub.get_rect(midtop=(W // 2, 114)))
        from sim import weapons
        mine = self._my_class()

        def fit(text, font, width):
            """`text`, trimmed with an ellipsis until it fits `width`."""
            if font.size(text)[0] <= width:
                return text
            while text and font.size(text + "…")[0] > width:
                text = text[:-1]
            return text + "…"

        for key, r in zip(classes.ORDER, self._card_rects()):
            c = classes.get(key)
            btn = Button(r, "", lambda k=key: self.pick_class(k),
                         tone=GOOD if key == mine else ACCENT)
            btn.draw(surf, self.f)
            self._buttons.append(btn)
            inner = r.w - 24
            y = r.y + 12
            # the name at full size if it fits, smaller if it does not
            tfont = self.f if self.f.size(c.name)[0] <= inner else self.f_sm
            t = tfont.render(fit(c.name, tfont, inner), True, TEXT)
            surf.blit(t, t.get_rect(midtop=(r.centerx, y)))
            y += 36
            for label, val in (("health", f"{c.hp:.0f}"),
                               ("shields", f"{c.sh:.0f}"),
                               ("armour", f"{c.armour:.0f}%"),
                               ("speed", f"{c.spd:.1f} m/s")):
                surf.blit(self.f_sm.render(label, True, TEXT_DIM), (r.x + 12, y))
                v = self.f_sm.render(val, True, TEXT)
                surf.blit(v, (r.right - 12 - v.get_width(), y))
                y += 21
            y += 8
            surf.blit(self.f_sm.render("weapons", True, TEXT_DIM), (r.x + 12, y))
            y += 20
            for wk in c.loadout:
                nm = fit(weapons.ROSTER[wk].name, self.f_sm, inner - 8)
                surf.blit(self.f_sm.render(nm, True, TEXT), (r.x + 18, y))
                y += 19
            y += 8
            ab = classes.ability_of(key)
            surf.blit(self.f_sm.render("space", True, TEXT_DIM), (r.x + 12, y))
            y += 19
            nm = self.f_sm.render(fit(ab.name, self.f_sm, inner), True, TEXT)
            surf.blit(nm, (r.x + 18, y))
            y += 21
            # the ability and then the blurb, wrapped to the card and stopped
            # at its bottom edge rather than spilling past it
            y = card_text_top(c, r)     # one source of truth, shared with
                                        # the test that checks it still fits
            lines = wrap(ab.hint + " · " + c.blurb, self.f_sm, inner)
            room = card_room(c, r)
            for i, text in enumerate(lines[:room]):
                if i == room - 1 and len(lines) > room:
                    text = fit(text + " " + lines[room], self.f_sm, inner)
                surf.blit(self.f_sm.render(text, True, TEXT_DIM), (r.x + 12, y))
                y += CARD_LINE_H


# --------------------------------------------------------------------- start screen

class StartScreen:
    """First screen: host a game on this machine, or join one on another."""

    def __init__(self, fonts):
        self.f_big, self.f, self.f_sm = fonts
        self.choice: str | None = None          # "host" | "join"
        self.exit = False
        self.host_btn = Button((150, 200, 260, 64), "HOST A GAME",
                               lambda: setattr(self, "choice", "host"), tone=GOOD)
        self.join_btn = Button((150, 284, 260, 64), "JOIN A GAME",
                               lambda: setattr(self, "choice", "join"))
        self.exit_btn = Button((150, 402, 130, 48), "EXIT",
                               lambda: setattr(self, "exit", True), tone=BAD)

    def handle_event(self, ev):
        if ev.type == pygame.QUIT:
            self.exit = True
            return
        self.host_btn.handle(ev)
        self.join_btn.handle(ev)
        self.exit_btn.handle(ev)

    def draw(self, surf):
        surf.fill(BG)
        surf.blit(self.f_big.render("BANGbang", True, TEXT), (32, 40))
        surf.blit(self.f_sm.render(
            "hosting runs the match on this machine - the address others need "
            "is shown on the next screen", True, TEXT_DIM), (34, 92))
        self.host_btn.draw(surf, self.f)
        self.join_btn.draw(surf, self.f)
        self.exit_btn.draw(surf, self.f)


# --------------------------------------------------------------------- connect screen

class JoinScreen:
    """Pre-connection screen.

    mode="join": enter a host IP and connect to a server elsewhere.
    mode="host": start a GameServer in this process and connect to it over
                 loopback, so the host sits in the same lobby as everyone else.

    On success .client is a connected GameClient. In host mode .server is the
    GameServer this process now owns — whoever called run_lobby must stop it.
    """

    def __init__(self, fonts, default_ip="", mode="join",
                 port=DEFAULT_PORT, map_id="arena"):
        self.f_big, self.f, self.f_sm = fonts
        self.mode = mode
        self.map_id = map_id
        self.client: GameClient | None = None
        self.server: GameServer | None = None
        self.exit = False
        self.status = ""
        self._colour_idx = 0
        self.ip_field = TextField((150, 150, 300, 42), text=default_ip,
                                  placeholder="192.168.x.x", max_len=21)
        self.port_field = TextField((150, 150, 120, 42), text=str(port),
                                    placeholder=str(DEFAULT_PORT), max_len=5)
        self.name_field = TextField((150, 214, 300, 42), text="Player",
                                    placeholder="your name")
        go_label = "START HOSTING" if mode == "host" else "CONNECT"
        self.connect_btn = Button((150, 360, 220, 50), go_label,
                                  self._go, tone=GOOD)
        self.exit_btn = Button((390, 360, 120, 50), "EXIT",
                               lambda: setattr(self, "exit", True), tone=BAD)

    def _swatch_rects(self):
        return [pygame.Rect(150 + i * 34, 288, 28, 28)
                for i in range(len(SWATCHES))]

    # ---- connect / host ----

    def _go(self):
        if self.mode == "host":
            self._do_host()
        else:
            self._do_join()

    def _name(self):
        return self.name_field.text or "Player"

    def _port(self):
        txt = self.port_field.text.strip()
        if not txt:
            return DEFAULT_PORT
        try:
            n = int(txt)
        except ValueError:
            return None
        return n if 1 <= n <= 65535 else None

    def _do_host(self):
        port = self._port()
        if port is None:
            self.status = "bad port"
            return
        self.status = "starting server ..."
        try:
            srv = GameServer(port=port, map_id=self.map_id)
            srv.start()
        except OSError as e:
            # almost always: something already listening on that port
            self.status = f"cannot listen on {port}: {e}"
            return
        cli = GameClient("127.0.0.1", port)
        if cli.connect(name=self._name(), colour=SWATCHES[self._colour_idx],
                       timeout=5.0):
            self.server = srv
            self.client = cli
        else:
            self.status = f"failed: {cli.reject_reason}"
            srv.stop()

    def _do_join(self):
        ip = self.ip_field.text.strip()
        if not ip:
            self.status = "enter a host IP"
            return
        # allow "ip:port" in the address field
        port = self._port()
        if ":" in ip:
            ip, _, ps = ip.partition(":")
            try:
                port = int(ps)
            except ValueError:
                self.status = "bad port"
                return
        if port is None:
            self.status = "bad port"
            return
        self.status = f"connecting to {ip}:{port} ..."
        cli = GameClient(ip, port)
        if cli.connect(name=self._name(), colour=SWATCHES[self._colour_idx],
                       timeout=5.0):
            self.client = cli
        else:
            self.status = f"failed: {cli.reject_reason}"

    # ---- frame ----

    def handle_event(self, ev):
        if ev.type == pygame.QUIT:
            self.exit = True
            return
        if self.mode == "host":
            self.port_field.handle(ev)
        else:
            self.ip_field.handle(ev)
        self.name_field.handle(ev)
        self.connect_btn.handle(ev)
        self.exit_btn.handle(ev)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for i, r in enumerate(self._swatch_rects()):
                if r.collidepoint(ev.pos):
                    self._colour_idx = i

    def draw(self, surf):
        surf.fill(BG)
        title = "BANGbang - Host Game" if self.mode == "host" \
            else "BANGbang - Join Game"
        surf.blit(self.f_big.render(title, True, TEXT), (32, 40))
        if self.mode == "host":
            surf.blit(self.f_sm.render("PORT", True, TEXT_DIM), (34, 158))
            self.port_field.draw(surf, self.f)
            try:
                lan = GameServer.lan_ip()
            except OSError:
                lan = "127.0.0.1"
            port_txt = self.port_field.text.strip() or str(DEFAULT_PORT)
            shown = lan if port_txt == str(DEFAULT_PORT) else f"{lan}:{port_txt}"
            surf.blit(self.f_sm.render(f"others join with:  {shown}",
                                       True, ACCENT), (290, 162))
            surf.blit(self.f_sm.render(
                "map and mode are picked in the lobby once you are in",
                True, TEXT_DIM), (34, 462))
        else:
            surf.blit(self.f_sm.render("HOST IP", True, TEXT_DIM), (34, 158))
            self.ip_field.draw(surf, self.f)
            surf.blit(self.f_sm.render("(ip, or ip:port)", True, TEXT_DIM),
                      (466, 162))
        surf.blit(self.f_sm.render("NAME", True, TEXT_DIM), (34, 222))
        self.name_field.draw(surf, self.f)
        surf.blit(self.f_sm.render("COLOUR", True, TEXT_DIM), (34, 296))
        for i, r in enumerate(self._swatch_rects()):
            pygame.draw.rect(surf, SWATCHES[i], r, border_radius=4)
            sel = i == self._colour_idx
            pygame.draw.rect(surf, TEXT if sel else LINE, r,
                             width=3 if sel else 1, border_radius=4)
        self.connect_btn.draw(surf, self.f)
        self.exit_btn.draw(surf, self.f)
        if self.status:
            col = BAD if self.status.startswith(("failed", "cannot", "bad")) \
                else TEXT_DIM
            surf.blit(self.f_sm.render(self.status, True, col), (150, 424))


# --------------------------------------------------------------------- standalone loop

def _make_fonts():
    return (pygame.font.SysFont("arial", 30, bold=True),
            pygame.font.SysFont("arial", 20),
            pygame.font.SysFont("arial", 15))


def _panel_origin():
    """Where the 820x620 start/join composition sits in the window."""
    return ((W - PANEL_W) // 2, (H - PANEL_H) // 2)


def _at_panel(ev, ox, oy):
    """The same event, in the panel's coordinates rather than the window's."""
    if not hasattr(ev, "pos"):
        return ev
    d = dict(ev.__dict__)
    d["pos"] = (ev.pos[0] - ox, ev.pos[1] - oy)
    return pygame.event.Event(ev.type, d)


def _run_panel_screen(screen, panel, clock, scr, done):
    """Drive one of the two small screens: it draws itself at its own size and
    is blitted into the middle of the window, and the mouse is moved into its
    frame so its buttons are where they look."""
    ox, oy = _panel_origin()
    while not done():
        for ev in pygame.event.get():
            scr.handle_event(_at_panel(ev, ox, oy))
        scr.draw(panel)
        screen.fill(BG)
        screen.blit(panel, (ox, oy))
        pygame.display.flip()
        clock.tick(60)


def run_lobby(default_ip="", existing_client: GameClient | None = None,
              mode: str | None = None, maps_dir="maps",
              existing_server: GameServer | None = None):
    """Open (or reuse) a window and run start -> host/join -> lobby.

    mode "host" or "join" skips the start screen. Passing default_ip implies
    joining. Returns (result, client, server): server is set only when this
    process is hosting, and the CALLER owns it — stop() it when done.

    Does not call pygame.quit(); the caller decides, because mp_client.py keeps
    the window and switches straight into the match.
    """
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("BANGbang Lobby")
    clock = pygame.time.Clock()
    fonts = _make_fonts()
    panel = pygame.Surface((PANEL_W, PANEL_H))

    cli = existing_client
    srv = existing_server

    if cli is None:
        if mode is None:
            mode = "join" if default_ip else None
        if mode is None:
            start = StartScreen(fonts)
            _run_panel_screen(screen, panel, clock, start,
                              lambda: start.choice is not None or start.exit)
            if start.exit:
                return LobbyResult.EXIT, None, None
            mode = start.choice

        join = JoinScreen(fonts, default_ip, mode=mode)
        _run_panel_screen(screen, panel, clock, join,
                          lambda: join.client is not None or join.exit)
        if join.exit:
            if join.server:
                join.server.stop()
            return LobbyResult.EXIT, None, None
        cli, srv = join.client, join.server

    # a freshly made connection joins a running match on its own; one handed
    # back to us (the player stepped out) waits to be asked
    lobby = Lobby(cli, fonts, maps_dir=maps_dir,
                  auto_join=existing_client is None)
    while lobby.result is None:
        for ev in pygame.event.get():
            lobby.handle_event(ev)
        lobby.pump_events()
        lobby.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    result = lobby.result
    # On START the client stays alive and goes to the caller, which switches to
    # the match render. On EXIT/DISCONNECT nothing is left running.
    if result != LobbyResult.START:
        cli.disconnect()
        if srv:
            srv.stop()
        return result, None, None
    return result, cli, srv


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    want_host = "--host" in args
    ip = next((a for a in args if not a.startswith("-")), "")
    res, client, server = run_lobby(default_ip=ip,
                                    mode="host" if want_host else None)
    print("lobby ended:", res)
    if res == LobbyResult.START and client:
        print("-> would now start match render with client id",
              client.world.my_id)
        client.disconnect()
        if server:
            server.stop()
    pygame.quit()
