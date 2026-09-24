"""
mp_client.py — play BANGbang over the network.

    python mp_client.py                  # start screen: host here, or join
    python mp_client.py --host           # host on this machine straight away
    python mp_client.py 192.168.1.42     # join that host
    python mp_client.py 192.168.1.42:47999

Hosting runs the server inside this process and connects your own client to it
over loopback, so the host sits in the same lobby as everyone else and the
address others need is shown on screen.

The server owns everything that decides a fight: where people are, what they
hit, what it cost them. This client predicts its own movement so the controls
feel immediate, draws everyone else from interpolated snapshots, and hides them
behind the same fog of war that hides the rest of the map.

Sound is the exception. The server says what happened and how loud; each client
solves its own propagation field and asks sim/perception.py what it can hear —
which means a shot round a corner reaches you from the corner, late and
quieter, exactly as in single-player.

Controls
    WASD / arrows   move        shift  run        ctrl  crawl
    mouse           aim         left mouse  fire (hold for full-auto or to
                                            charge a rail weapon)
    1-7 / wheel     weapon      r      reload      b   fire mode
    f               open or close a door you are standing next to
    l               flashlight — it lights the room, and it shows everyone in
                    that room exactly where you are
    tab             hold for the scoreboard
    esc             step out of the match, back to the lobby (you stay
                    connected, and can rejoin the same match)

Connecting while a match is in progress drops you straight into it, at the spawn
point furthest from anyone still fighting.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import sys
import time
from collections import deque

import numpy as np
import pygame

import main as renderer            # map art, fog, sound emission, tuning
from lobby import LobbyResult, Lobby, run_lobby, _make_fonts, W as LOBBY_W, \
    H as LOBBY_H
from net import GameClient, GameServer
from net import maps as netmaps
from net.protocol import (BTN_ABILITY, BTN_CRAWL, BTN_FIRE, BTN_INTERACT,
                          BTN_KNOCK, BTN_LIGHT,
                          BTN_RELOAD, BTN_RUN, DEFAULT_PORT, GameMode,
                          ServerState)
from net.server import TICK_DT
from sim import audio, classes, lighting, movement, perception, sprites, weapons
from sim.doors import DoorSet
from sim.pickups import PickupSet
from sim.tilemap import (break_glass_cells, compute_roof, set_door,
                         set_door_gap)
from sim.vision import ConeSpec, VisibilityCache, shadowcast

# Prediction. Input is sent as one fixed-size command per server tick, and this
# client applies each command locally the moment it sends it, so the screen
# responds immediately instead of a round trip later. Each command is kept until
# a snapshot says the server has applied it; then the server's position is taken
# as truth and the commands it had not yet seen are replayed on top.
#
# Both sides step the same function (sim/movement.py) by the same TICK_DT in the
# same order, so the replayed result matches what the server will send next —
# the correction is normally zero, not a tug backwards.
MAX_CATCHUP_TICKS = 3     # after a hitch, send at most this many commands at
                          # once; beyond that the lost time is simply lost
SNAP_M = 2.0              # a correction this large is a teleport (respawn, or
                          # we are badly out of sync): accept it, don't smooth
PENDING_MAX = 64          # commands kept for replay; ~2 s at 30 Hz
RENDER_SMOOTH_PER_S = 30.0  # the drawn position eases toward the predicted one

INTERP_DELAY = 0.07       # render remote players this far in the past, so there
                          # are always two snapshots to interpolate between

NAME_DY = -0.95           # name tag offset above a body, metres

# Players wear their lobby colour as ART: `soldier_ready_red.png` and the rest
# of the palette, chosen by sprites.body_art(). Tinting the single khaki sprite
# was tried first and read as exactly what it was — a coloured pane held over
# the art. Any pose with no coloured version falls back to the base sprite, so
# the palette can arrive one file at a time, and the ring at the feet carries
# the colour either way.
SEEN_MIN = 0.02           # cone intensity at which a remote player registers
ROCKET_R = 0.14           # drawn radius of a rocket in flight, metres

# Light. Multiplayer runs the model single-player runs, out of sim/lighting.py:
# what you can see is gated on what is lit, your flashlight and everyone else's
# are shadow-cast beams that stop at walls, and a muzzle flash is a real light
# that briefly shows the room it went off in. A dark room is cover.
FLASH_REACH_MUL = 0.9     # flashlight reach as a fraction of identify range
BODY_LIGHT_LO = 0.14      # illumination at which a body is a black silhouette
BODY_LIGHT_HI = 0.92      # ...and at which it is drawn at full brightness

# What another player's sound turns into in the mixer. The label the server
# sends ends in the kind of event, so this survives new weapons being added.
# Your own noises, at the volume single-player plays them. These are quiet on
# purpose: what matters about a footstep is that everyone ELSE can hear it, and
# a sprint that deafens the person sprinting is just noise. Crawling is silent
# to its owner, as it is in main.py.
OWN_STEP_GAIN = {"crawl": 0.0, "walk": 0.09, "run": 0.15}
OWN_GAIN = {"reload": 0.5, "dryfire": 0.5, "door": 0.6, "glass": 0.7}

# An arrival cue is one event, but a sprinting player emits five footsteps a
# second and each arc lingers for CUE_FADE. Left alone they stack into a ring
# around the listener and stop meaning anything. Arrivals from roughly the same
# direction, close together in time, refresh the arc that is already there.
CUE_MERGE_DEG = 22.0
CUE_MERGE_S = 0.9
CUE_MAX = 4
SWING_TIME = 0.18         # how long a knife arc stays on screen

# Solving a propagation field is the expensive thing this client does, and a
# weapon on full auto makes ten noises a second from what is very nearly one
# place. Each source keeps its own reusable field, per kind of sound, so a
# burst costs one solve instead of ten. Keyed per kind because a footstep
# riding a gunshot's field would inherit the gunshot's energy and play far
# louder than it should.
SOUND_CACHE_TTL = 6.0     # forget a source's field after this long unused
SOUND_CACHE_MAX = 24      # hard cap on cached fields, oldest evicted first
MAX_ACTIVE_SOUNDS = 48    # alive at once; past this, heard ones go first (main._trim_sounds).
                          # 48 keeps sustained full-auto audible to ~100 m on open ground

CLIP_BY_KIND = {
    "fire": "gunshot", "step": "footstep", "reload": "magazine",
    "dry": "dryfire", "glass": "glass", "knock": "knock", "door": "door",
    "door_heavy": "door_heavy",
    # no blade sound of its own yet: the knock is the closest thing the
    # procedural bank has to a short, dull impact
    "melee": "knock", "ability": "knock",
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Play BANGbang over the network.")
    ap.add_argument("address", nargs="?", default="",
                    help="host to join, e.g. 192.168.1.42 or 192.168.1.42:47999")
    ap.add_argument("--host", action="store_true",
                    help="host a match on this machine")
    ap.add_argument("--map", dest="map_id", default="arena",
                    help="map to host (host only; changeable in the lobby)")
    ap.add_argument("--maps-dir", default="maps")
    ap.add_argument("--window", default="",
                    help="viewport size, e.g. 900x700 — handy for running two "
                         "clients side by side on one screen")
    return ap.parse_args()


def _window_size(spec: str):
    if not spec:
        return None
    try:
        w, h = spec.lower().split("x")
        return max(480, int(w)), max(360, int(h))
    except ValueError:
        print(f"ignoring --window {spec!r}: expected WIDTHxHEIGHT")
        return None


# --------------------------------------------------------------------- match

class MatchView:
    """One networked match: input, prediction, and the render of everyone else.

    Construct when the server says the match started, then call frame() until
    it returns a reason to stop."""

    def __init__(self, cli: GameClient, maps_dir="maps", window=None):
        self.cli = cli
        w = cli.world
        # the host's exact map: ours, or the copy the client downloaded. A
        # host that force-started before the download finished is waited for
        # rather than drawn with the wrong walls.
        where = w.map_dir or cli.wait_for_map()
        if where is None:
            raise RuntimeError(f"no copy of map {w.map_id!r} "
                               f"({w.map_status or 'the host did not send it'})")
        self.m = netmaps.load(w.map_id, where)
        self.ppm = renderer.PX_PER_M
        self.cpm = self.m.cells_per_metre
        # the sound solver works on its own copy of the cost field, kept in
        # step with the map when glass breaks
        self.cost = self.m.sound_cost.astype(np.float64)

        self.world_w = round(self.m.width_m * self.ppm)
        self.world_h = round(self.m.height_m * self.ppm)
        cap = window or renderer.view_cap()
        self.view_w = min(self.world_w, cap[0])
        self.view_h = min(self.world_h, cap[1])
        self.window = pygame.display.set_mode((self.view_w, self.view_h))
        pygame.display.set_caption(f"BANGbang — {self.m.name}")
        self.surf = pygame.Surface((self.world_w, self.world_h))

        self.base = renderer.build_base_surface(self.m)
        if self.m.lightmap is not None:
            renderer.apply_lightmap(self.base, self.m.lightmap)

        self.bank = sprites.SpriteBank()
        self.bank.load()
        self.bank.set_scale(self.ppm / sprites.ART_PPM)
        self.audio_on = audio.init()

        self.vis = VisibilityCache(self.m.blocks_sight)
        self.known = np.zeros(self.m.blocks_sight.shape, dtype=bool)
        self.cone = ConeSpec(
            identify_deg=renderer.CONE_DEG[0],
            recognise_deg=renderer.CONE_DEG[1],
            peripheral_deg=renderer.CONE_DEG[2],
            identify_range=renderer.CONE_RANGE_M[0] * self.cpm,
            recognise_range=renderer.CONE_RANGE_M[1] * self.cpm,
            peripheral_range=renderer.CONE_RANGE_M[2] * self.cpm,
            near_range=renderer.CONE_RANGE_M[3] * self.cpm,
        )
        # the veil, as a multiply layer: black at alpha a is the same as
        # multiplying by (1 - a), and an opaque multiply blit is about half
        # the cost of a per-pixel-alpha one over a whole viewport
        self._ov = pygame.Surface((8, 8))
        self._ov_big = pygame.Surface((8, 8))

        self.font = pygame.font.SysFont("consolas,monospace", 14)
        self.font_mid = pygame.font.SysFont("consolas,monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("consolas,monospace", 34, bold=True)
        pygame.mouse.set_visible(False)

        sx, sy = w.my_spawn or self.m.player_spawn
        self.px, self.py = float(sx), float(sy)      # predicted position
        self.rx, self.ry = self.px, self.py          # smoothed, what is drawn
        # a spawn point can say which way you are looking when you arrive, so a
        # spawn in a corner does not start you staring at a wall
        self.facing = (w.my_spawn_aim if w.my_spawn_aim is not None
                       else -math.pi / 2)
        self.aim_dist = 10.0
        self.cam_x = self.cam_y = 0
        self.acc = 0.0                    # unspent frame time, in tick units
        self.pending: deque = deque()     # commands the server has not acked
        self.last_ack = -1
        self.show_scores = False
        self.corrected_m = 0.0
        self.travelled = 0.0

        # weapons: the client asks, the server decides. What you carry is
        # your class's kit, and the class is the one you spawned as
        _me = w.players.get(w.my_id)
        self.cls_now = classes.get(_me.cls_now if _me else None).key
        self.loadout = list(classes.get(self.cls_now).loadout)
        self.wep = 0
        self.modes = [0] * len(self.loadout)

        # sprint fuel: predicted here, corrected by every snapshot
        self.stamina = 1.0
        self.sprint_locked = False

        # the esc menu: resume, class for the next respawn, leave
        self.menu_open = False
        self.menu_sel = 0
        # a Tech's echo ping: where it went off and when it fades. The cells it
        # showed you stay in fog memory afterwards, like anything else seen
        self.ping = None

        # sound and effects
        self.sounds: list = []        # ActiveSound: fields being solved/heard
        self.jobs: list = []          # chunked solve jobs
        self.caches: dict = {}        # (source id, kind) -> reusable field
        self.cues: list = []          # arrival arcs at the listener
        self.tracers: list = []       # (segments, t0, mine)
        self.swings: list = []        # knife arcs: {id, heading, reach, hit, t0}
        self.shots: dict = {}         # player id -> {t0, art, roll, scale}
        self.mlights: list = []       # {x, y, t0, col, gain, reach, life}
        self.blasts: list = []        # {x, y, r, t0}
        self.rockets: list = []       # {x, y, ix, iy, t0, dur}
        self.rail_ch = None           # the rail spool-up loop, while charging
        self.broken: set = set()      # coarse glass cells already broken here
        self.doors = DoorSet(self.m)
        self.packs = PickupSet(self.m)
        self._apply_map_state(w)
        self.msg, self.msg_t = "", -9.0

        # lighting, rebuilt every frame by _light_pass
        self.rng = random.Random()
        self.illum = None             # illumination over the vision window
        self.mflash = None            # muzzle-flash light, mono
        self.mflash_rgb = None        # ...and its per-source colour
        self.beams = None             # other players' flashlight beams

    # ---- input -------------------------------------------------------

    def read_input(self):
        """Keys and mouse -> the same (mx, my, buttons) the server expects."""
        k = pygame.key.get_pressed()
        mx = (k[pygame.K_d] or k[pygame.K_RIGHT]) - (k[pygame.K_a] or k[pygame.K_LEFT])
        my = (k[pygame.K_s] or k[pygame.K_DOWN]) - (k[pygame.K_w] or k[pygame.K_UP])
        mx, my = movement.clamp_input(float(mx), float(my))
        buttons = 0
        if k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT]:
            buttons |= BTN_RUN
        if k[pygame.K_LCTRL] or k[pygame.K_RCTRL]:
            buttons |= BTN_CRAWL
        if k[pygame.K_r]:
            buttons |= BTN_RELOAD
        if k[pygame.K_f]:
            buttons |= BTN_INTERACT
        if k[pygame.K_e]:
            buttons |= BTN_KNOCK
        if k[pygame.K_SPACE]:
            buttons |= BTN_ABILITY
        if k[pygame.K_l]:
            # the server owns the lamp and toggles it on the rising edge, so
            # holding the key does not strobe it
            buttons |= BTN_LIGHT
        if pygame.mouse.get_pressed()[0]:
            buttons |= BTN_FIRE
        mxp, myp = pygame.mouse.get_pos()
        ax = (mxp + self.cam_x) / self.ppm
        ay = (myp + self.cam_y) / self.ppm
        dx, dy = ax - self.rx, ay - self.ry
        if self._alive() and (abs(dx) > 1e-6 or abs(dy) > 1e-6):
            # a dead player's view does not follow the mouse
            self.facing = math.atan2(dy, dx)
            self.aim_dist = max(0.5, math.hypot(dx, dy))
        return mx, my, buttons

    # ---- the esc menu ------------------------------------------------

    MENU_ITEMS = ("resume", "class", "leave")

    def _my_pick(self) -> str:
        me = self.cli.world.players.get(self.cli.world.my_id)
        return classes.get(me.cls if me else None).key

    def _cycle_class(self, step: int) -> None:
        new = classes.cycle(self._my_pick(), step)
        self.cli.set_class(new)
        when = ("now playing it" if new == self.cls_now
                else "from your next respawn")
        self._say(f"class: {classes.get(new).name}, {when}")

    @staticmethod
    def _wrap(text: str, font, width: int) -> list:
        """`text` broken into lines that fit `width`, on word boundaries."""
        if font.size(text)[0] <= width:
            return [text]
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

    def _menu_rects(self):
        """(item, rect) for each menu row, in window coordinates."""
        bw = 600
        x = self.view_w // 2 - bw // 2
        y = self.view_h // 2 - 150
        heights = {"resume": 40, "class": 130, "leave": 40}
        out = []
        yy = y + 50
        for item in self.MENU_ITEMS:
            out.append((item, pygame.Rect(x + 16, yy, bw - 32, heights[item])))
            yy += heights[item] + 10
        return out

    def _menu_activate(self, item: str) -> str | None:
        if item == "resume":
            self.menu_open = False
        elif item == "class":
            self._cycle_class(1)
        elif item == "leave":
            return "leave"
        return None

    def _menu_event(self, ev) -> str | None:
        """Everything while the menu is open. Up/down (or w/s) moves between
        rows, left/right or the mouse wheel steps through the classes, enter
        or a click picks, esc closes."""
        if ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_ESCAPE:
                self.menu_open = False
            elif k in (pygame.K_UP, pygame.K_w):
                self.menu_sel = (self.menu_sel - 1) % len(self.MENU_ITEMS)
            elif k in (pygame.K_DOWN, pygame.K_s):
                self.menu_sel = (self.menu_sel + 1) % len(self.MENU_ITEMS)
            elif k in (pygame.K_LEFT, pygame.K_a):
                self._cycle_class(-1)
            elif k in (pygame.K_RIGHT, pygame.K_d):
                self._cycle_class(1)
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                return self._menu_activate(self.MENU_ITEMS[self.menu_sel])
        elif ev.type == pygame.MOUSEWHEEL:
            self._cycle_class(-1 if ev.y > 0 else 1)
        elif ev.type == pygame.MOUSEMOTION:
            for i, (item, r) in enumerate(self._menu_rects()):
                if r.collidepoint(ev.pos):
                    self.menu_sel = i
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3):
            for item, r in self._menu_rects():
                if r.collidepoint(ev.pos):
                    if item == "class":
                        self._cycle_class(1 if ev.button == 1 else -1)
                        return None
                    return self._menu_activate(item)
        return None

    def _draw_menu(self) -> None:
        bw, bh = 600, 320
        x = self.view_w // 2 - bw // 2
        y = self.view_h // 2 - 150
        shade = pygame.Surface((self.view_w, self.view_h), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 120))
        self.window.blit(shade, (0, 0))
        panel = pygame.Surface((bw, bh), pygame.SRCALPHA)
        panel.fill((16, 18, 22, 236))
        self.window.blit(panel, (x, y))
        white, dim, gold = (222, 226, 232), (140, 146, 156), (230, 210, 140)
        self.window.blit(self.font_mid.render("MENU", True, white), (x + 16, y + 14))
        pick = self._my_pick()
        c = classes.get(pick)
        for i, (item, r) in enumerate(self._menu_rects()):
            sel = i == self.menu_sel
            pygame.draw.rect(self.window, (40, 44, 52) if sel else (26, 28, 34),
                             r, border_radius=6)
            pygame.draw.rect(self.window, gold if sel else (60, 64, 74), r,
                             width=2, border_radius=6)
            if item == "resume":
                t = self.font_mid.render("Resume", True, white)
                self.window.blit(t, t.get_rect(center=r.center))
            elif item == "leave":
                t = self.font_mid.render("Leave match", True, white)
                self.window.blit(t, t.get_rect(center=r.center))
            else:
                t = self.font_mid.render(f"<   {c.name}   >", True, white)
                self.window.blit(t, t.get_rect(midtop=(r.centerx, r.y + 8)))
                lines = [(classes.stat_line(pick), white),
                         (classes.weapon_line(pick), white),
                         (f"space: {classes.ability_of(pick).name} - "
                          f"{classes.ability_of(pick).hint}", dim)]
                if pick != self.cls_now:
                    lines.append((f"from your next respawn - now playing "
                                  f"{classes.get(self.cls_now).name}", gold))
                else:
                    lines.append(("the class you are playing", dim))
                # wrapped to the panel: an ability line is a sentence, and a
                # sentence written past the edge of the box is one you cannot
                # read the end of
                wrapped = []
                for txt, col in lines:
                    wrapped += [(ln, col)
                                for ln in self._wrap(txt, self.font, r.w - 24)]
                step = min(22, max(14, (r.h - 44) // max(1, len(wrapped))))
                for j, (txt, col) in enumerate(wrapped):
                    s_ = self.font.render(txt, True, col)
                    self.window.blit(s_, s_.get_rect(
                        midtop=(r.centerx, r.y + 34 + j * step)))
        hint = self.font.render(
            "up/down choose   left/right or wheel: class   enter select   "
            "esc close", True, dim)
        self.window.blit(hint, hint.get_rect(midtop=(x + bw // 2, y + bh - 22)))

    def _sync_class(self) -> None:
        """Pick up a class change the moment the server spawns us as it: new
        weapons, and the speed prediction steps at."""
        me = self.cli.world.players.get(self.cli.world.my_id)
        if me is None or me.cls_now == self.cls_now:
            return
        self.cls_now = classes.get(me.cls_now).key
        self.loadout = list(classes.get(self.cls_now).loadout)
        self.wep = 0
        self.modes = [0] * len(self.loadout)
        self._say(f"now playing {classes.get(self.cls_now).name}")

    def handle_event(self, ev) -> str | None:
        if ev.type == pygame.QUIT:
            return "quit"
        if self.menu_open:
            return self._menu_event(ev)
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                self.menu_open = True
                self.menu_sel = 0
                self.show_scores = False
                return None
            if ev.key == pygame.K_TAB:
                self.show_scores = True
            elif not self._alive():
                pass                  # dead: the scoreboard and esc still work
            elif pygame.K_1 <= ev.key <= pygame.K_9:
                idx = ev.key - pygame.K_1
                if idx < len(self.loadout):
                    self._select(idx)
            elif ev.key == pygame.K_b:
                w = weapons.ROSTER[self.loadout[self.wep]]
                modes = w.fire_modes()
                if len(modes) > 1:
                    self.modes[self.wep] = (self.modes[self.wep] + 1) % len(modes)
                    self._say(f"{w.name}: {modes[self.modes[self.wep]]}")
                else:
                    self._say(f"{w.name}: no alt fire")
        elif ev.type == pygame.KEYUP and ev.key == pygame.K_TAB:
            self.show_scores = False
        elif ev.type == pygame.MOUSEWHEEL and self._alive():
            self._select((self.wep - ev.y) % len(self.loadout))
        return None

    @staticmethod
    def _their_weapon(p) -> str:
        """The gun another player is holding: slot `wep` of THEIR class's kit,
        not of ours."""
        kit = classes.get(p.cls_now).loadout
        return kit[min(p.wep, len(kit) - 1)]

    def _on_ability(self, e, now):
        """Somebody used their class ability. Your own ping is the one this
        client has to draw; the rest is the server's business."""
        who = self.cli.world.players.get(e["id"])
        name = classes.ABILITIES.get(e.get("ab", ""))
        if e["id"] == self.cli.world.my_id:
            if e.get("ab") == "ping":
                self.ping = (self.rx, self.ry, now + float(e.get("dur", 3.0)))
            if name is not None:
                self._say(name.name)
        elif who is not None and name is not None and e.get("ab") == "blitz":
            pass                     # nothing to draw for somebody else's yet

    def _ability_speed(self) -> float:
        """Blitz and Vanish, predicted the way the server runs them."""
        me = self.cli.world.players.get(self.cli.world.my_id)
        if me is None:
            return 1.0
        return classes.ability_speed(me.cls_now, me.ability_t > 0.0)

    def _ping_visible(self, x: float, y: float) -> bool:
        """Is this spot inside a live echo ping?"""
        if self.ping is None:
            return False
        px, py, _t = self.ping
        return math.hypot(x - px, y - py) <= classes.PING_RADIUS_M

    def _alive(self) -> bool:
        me = self.cli.world.players.get(self.cli.world.my_id)
        return bool(me and me.alive)

    def _select(self, idx: int) -> None:
        if idx != self.wep:
            self.wep = idx
            self._say(f"switched to {weapons.ROSTER[self.loadout[idx]].name}")

    def _say(self, text: str) -> None:
        self.msg, self.msg_t = text, time.monotonic()

    # ---- prediction --------------------------------------------------

    def _stance(self, buttons):
        """The stance the SERVER will give this input. Predicting a sprint the
        server refuses - because the tank is empty - is a correction every
        frame, so the client runs the same sprint fuel the server does."""
        if buttons & BTN_CRAWL:
            return "crawl"
        want = "run" if buttons & BTN_RUN else "walk"
        return movement.allowed_stance(want, self.stamina, self.sprint_locked)

    def _apply(self, x, y, mx, my, buttons):
        """One command, stepped exactly as the server will step it."""
        moving = (mx, my) != (0.0, 0.0)
        self.stamina, self.sprint_locked = movement.step_stamina(
            self.stamina, self.sprint_locked, self._stance(buttons), moving,
            TICK_DT)
        return movement.step(
            self.m, x, y, mx, my, self._stance(buttons), TICK_DT,
            speed_mult=(classes.get(self.cls_now).speed_mult
                        * self._ability_speed()))

    def send_commands(self, mx, my, buttons, dt):
        """Emit whole input commands at the server's tick rate, predicting each
        one locally as it goes out."""
        self.acc = min(self.acc + dt, TICK_DT * MAX_CATCHUP_TICKS)
        while self.acc >= TICK_DT:
            self.acc -= TICK_DT
            x0, y0 = self.px, self.py
            self.px, self.py = self._apply(self.px, self.py, mx, my, buttons)
            self.travelled += math.hypot(self.px - x0, self.py - y0)
            seq = self.cli.send_input(mx, my, self.facing, buttons,
                                      wep=self.wep, mode=self.modes[self.wep],
                                      aim_dist=self.aim_dist)
            self.pending.append((seq, mx, my, buttons))
            while len(self.pending) > PENDING_MAX:
                self.pending.popleft()

    def reconcile(self):
        """Take the server's position as truth, then replay the commands it had
        not yet applied when it sent that snapshot."""
        me = self.cli.world.players.get(self.cli.world.my_id)
        if me is None or not me.alive:
            return
        ack = me.acked_seq
        if ack < 0 or ack == self.last_ack:
            return                       # nothing new to reconcile against
        self.last_ack = ack
        while self.pending and self.pending[0][0] <= ack:
            self.pending.popleft()
        x, y = me.x, me.y
        for _seq, mx, my, buttons in self.pending:
            x, y = self._apply(x, y, mx, my, buttons)
        err = math.hypot(x - self.px, y - self.py)
        self.corrected_m = err
        self.px, self.py = x, y
        if err > SNAP_M:
            self.rx, self.ry = x, y

    # ---- what the server tells us ------------------------------------

    def _on_shot(self, e, now):
        """Somebody fired: tracers, muzzle flash, and anything in flight."""
        mine = e["id"] == self.cli.world.my_id
        segs = [((s[0], s[1]), (s[2], s[3])) for s in e.get("segs", [])]
        if segs:
            self.tracers.append((segs, now, mine))
        wid = e.get("wep", "")
        w = weapons.ROSTER.get(wid)
        # the flash rides the shooter, not a fixed point in the world: it is
        # drawn at their gun's muzzle, after the recoil kick, exactly as
        # single-player draws yours
        self.shots[e["id"]] = {
            "t0": now, "art": sprites.weapon_art(wid) if wid else "",
            "roll": self.rng.uniform(-180, 180),
            "scale": self.rng.uniform(0.85, 1.25)}
        if w is not None:
            col, gain, reach = renderer.muzzle_light_spec(w)
            self.mlights.append({"x": e["x"], "y": e["y"], "t0": now,
                                 "col": col, "gain": gain, "reach": reach,
                                 "life": renderer.MUZZLE_LIGHT_TIME})
        ix, iy = e["impact"]
        if e.get("travel", 0.0) > 0.0:
            d = math.hypot(ix - e["x"], iy - e["y"])
            flight = max(0.05, d / e["travel"])
            fuse = float(e.get("fuse", 0.0))
            # a fused round runs the server's clock: it flies, then lies there
            # until the fuse ends. A rocket has no fuse and goes off on arrival.
            self.rockets.append({"x": e["x"], "y": e["y"], "ix": ix, "iy": iy,
                                 "t0": now, "dur": fuse if fuse > 0.0 else flight,
                                 "flight": flight, "fuse": fuse, "wep": wid})
        elif e.get("blast", 0.0) > 0.0:
            self.blasts.append({"x": ix, "y": iy, "r": e["blast"], "t0": now})
            if w is not None:
                self._blast_light(ix, iy, w, now)
        if mine:
            # your own gun is not something you work out from a sound field
            w = weapons.ROSTER.get(e.get("wep", ""))
            if w is not None and self.audio_on:
                audio.play_fire(w, gain=min(1.0, 0.8 + 0.4 * e.get("charge", 0.0)))

    def _on_sound(self, e, now):
        """Something audible happened somewhere.

        Your own noises play directly — you know you pulled the trigger. Anyone
        else's is solved as a propagation field, so what reaches you is however
        much energy survived the route, from the direction it arrived, after
        the time it took to travel. That is the whole stealth model, and it is
        the same code single-player runs."""
        if e["id"] == self.cli.world.my_id:
            if e["clip"] == "fire" or not self.audio_on:
                return                    # the gunshot is played by _on_shot
            if e["clip"] == "footstep":
                gain = OWN_STEP_GAIN.get(e.get("stance", "walk"), 0.09)
                if gain > 0.0:
                    audio.play("footstep", gain=gain)
                return
            audio.play(e["clip"], gain=OWN_GAIN.get(e["clip"], 0.6))
            return
        kind = e["label"].rsplit("/", 1)[-1]
        cache = self._cache_for(e["id"], kind, now)
        renderer.emit(self.m, self.cost, e["x"], e["y"],
                      e["energy"] * self.cpm, e["label"], self.sounds, now,
                      cache=cache, jobs=self.jobs, enemy=True,
                      max_active=MAX_ACTIVE_SOUNDS)

    def _apply_map_state(self, w) -> None:
        """Catch up on the map as it stands: doors somebody opened and windows
        somebody shot out before we arrived. Without this a latecomer sees
        doorways as shut that everyone else walked through, and hears sound
        bouncing off walls that are not there any more."""
        # `left` carries a door still travelling — a blast door somebody
        # started four seconds ago finishes on this client too, at the same
        # moment it finishes on everyone else's
        self.doors.apply_wire(
            w.map_doors,
            on_change=lambda k, o: set_door(self.m, self.cost, k[0], k[1], o))
        self.packs.apply_wire(w.map_pickups)
        if w.map_glass:
            sub_n = self.m.subdiv
            break_glass_cells(self.m,
                              [(c * sub_n, r * sub_n) for r, c in w.map_glass],
                              self.cost, self.broken)
        if w.map_doors or w.map_glass:
            compute_roof(self.m, self.doors)
            self.vis.invalidate()

    def _on_door(self, e, now):
        """A door moved (or somebody tried and could not). Applying it here
        keeps this client's sight, sound and bullet geometry in step with the
        server's, and with everyone else's."""
        if e.get("blocked"):
            if e["id"] == self.cli.world.my_id:
                self._say("stand clear to close the door")
            return
        key = (e["r"], e["c"])
        d, changed = self.doors.resume(key, e["open"], e.get("dur", 0.0))
        if d is None:
            return
        if changed or not d.moving:
            # `changed` is a door that has begun to seal: the gap is gone
            # already. Otherwise this is a door that arrived instantly.
            self._door_arrived(key, d.is_open)

    def _door_arrived(self, key, is_open) -> None:
        """The panels are home. This is the instant the wall stops being a
        wall, so every field solved against the old geometry has to go."""
        set_door(self.m, self.cost, key[0], key[1], is_open)
        compute_roof(self.m, self.doors)
        self._geometry_changed()

    def _on_pickup(self, e, now) -> None:
        """A pack changed hands, or came back. The server has already decided;
        this only matches the drawing to it, and says so if it was yours."""
        p = self.packs.get(e["pid"])
        if p is None:
            return
        p.live = bool(e["live"])
        p.left = 0.0 if p.live else float(e.get("dur", p.respawn_s))
        if not p.live and e.get("by") == self.cli.world.my_id:
            self._say("health pack" if p.kind == "health" else "ammo pack")

    def _draw_packs(self):
        """Under the fog, like everything else: a pack in a room you cannot
        see into is a pack you do not know the state of."""
        for p in self.packs:
            renderer.sprites.draw_pickup(self.surf, p.x * self.ppm,
                                         p.y * self.ppm, p.kind, self.ppm,
                                         p.live)

    def _draw_doors(self):
        """Panels parting along the wall, through the same function
        single-player draws. Under the fog, so a door you cannot see is a door
        whose state you do not know — including whether it is on its way."""
        ppm = self.ppm
        for d in self.doors.doors.values():
            rc = pygame.Rect(int(d.c * ppm), int(d.r * ppm), int(ppm), int(ppm))
            renderer.draw_door(self.surf, rc, d.frac, d.axis, d.heavy)

    def _near_door(self):
        best, bd = None, renderer.INTERACT_RANGE
        for (r, c) in self.doors:
            d = math.hypot(c + 0.5 - self.rx, r + 0.5 - self.ry)
            if d < bd:
                best, bd = (r, c), d
        return best

    def _on_glass(self, e, now):
        """The server broke a pane; break the same one here so this client's
        walls match everyone else's for sight, bullets and sound."""
        sub = self.m.subdiv
        cells = [(c * sub, r * sub) for r, c in e.get("cells", [])]
        broke = break_glass_cells(self.m, cells, self.cost, self.broken)
        if broke:
            # a shattered pane changes sight AND sound, so held fields go too
            self._geometry_changed()

    def _cache_for(self, src_id: int, kind: str, now: float) -> dict:
        """One reusable field per source per kind of sound."""
        key = (src_id, kind)
        cache = self.caches.get(key)
        if cache is None:
            if len(self.caches) >= SOUND_CACHE_MAX:
                oldest = min(self.caches, key=lambda k: self.caches[k].get("t", 0.0))
                del self.caches[oldest]
            cache = {}
            self.caches[key] = cache
        cache["t"] = now
        return cache

    def _drop_stale_caches(self, now: float) -> None:
        for key in [k for k, c in self.caches.items()
                    if now - c.get("t", now) > SOUND_CACHE_TTL]:
            del self.caches[key]

    def _geometry_changed(self) -> None:
        """A door moved or a pane broke. Every field solved against the old
        walls is now wrong, and so is every field held for reuse."""
        self.vis.invalidate()
        self.sounds.clear()
        self.jobs.clear()
        self.caches.clear()

    def _clip_for(self, label: str) -> str:
        kind = label.rsplit("/", 1)[-1]
        return CLIP_BY_KIND.get(kind, audio.enemy_clip(label))

    def _hear(self, now):
        """Step the solvers, then ask what has reached us since last frame."""
        if self.jobs:
            budget = max(200, renderer.SOLVE_BUDGET // len(self.jobs))
            self.jobs = [j for j in self.jobs if not j.step(budget)]
        pc = self.m.cell_of(self.rx, self.ry)
        for snd in self.sounds:
            if snd.cued:
                continue
            status, heard = perception.perceive(
                snd.field, pc, snd.energy, snd.elapsed_cells(now))
            if status is perception.Arrival.PENDING:
                if snd.done(now):
                    snd.cued = True
                continue
            snd.cued = True
            if heard is None:
                continue
            self._add_cue(heard, now)
            gain = heard.gain
            if snd.label.endswith("/step"):
                gain *= 0.4               # footsteps are a faint cue, not loud
            if self.audio_on:
                audio.play(self._clip_for(snd.label), gain=gain, pan=heard.pan)
        self.sounds = [s for s in self.sounds if not s.done(now)]
        self.cues = [c for c in self.cues if now - c["t"] < renderer.CUE_FADE]
        self._drop_stale_caches(now)

    def _add_cue(self, heard, now):
        """One arc per direction, refreshed — not one per footstep."""
        for c in self.cues:
            if now - c["t"] > CUE_MERGE_S:
                continue
            delta = abs((heard.angle - c["ang"] + math.pi) % (2 * math.pi)
                        - math.pi)
            if delta <= math.radians(CUE_MERGE_DEG):
                # same direction, still fresh: move it to where we are now,
                # take the louder reading, and restart its fade
                c["x"], c["y"], c["t"] = self.rx, self.ry, now
                c["ang"] = heard.angle
                c["rem"] = max(c["rem"], heard.remaining)
                c["dir"] = c["dir"] or heard.directional
                return
        self.cues.append({"x": self.rx, "y": self.ry, "t": now,
                          "ang": heard.angle, "rem": heard.remaining,
                          "dir": heard.directional})
        if len(self.cues) > CUE_MAX:
            self.cues.pop(0)

    # ---- render ------------------------------------------------------

    def _visible_mask(self):
        """Raw line of sight: the cone, before light has anything to say about
        it. _light_pass decides how much of it actually registers."""
        pcx, pcy = self.m.cell_of(self.rx, self.ry)
        vf = self.vis.get(pcx, pcy, self.cone.max_range)
        return vf, vf.cone_intensity(self.facing, self.cone)

    def _beam_reach(self) -> float:
        return max(self.cone.identify_range * FLASH_REACH_MUL, 1.0)

    def _muzzle_light(self, vf, h_, w_, now):
        """Every live muzzle flash and blast as light, each shadow-cast from
        where it went off so a wall stops it, then masked to what you can see.

        Returns (mono field for perception and the world brighten, per-source
        coloured field for the glow drawn over the fog)."""
        if not self.mlights:
            return None, None
        mf = np.zeros((h_, w_), np.float32)
        mrgb = np.zeros((h_, w_, 3), np.float32)
        for ml in self.mlights:
            b = lighting.pulse((now - ml["t0"]) / ml["life"]) * ml["gain"]
            if b <= 0.02:
                continue
            fcx, fcy = self.m.cell_of(ml["x"], ml["y"])
            rc = max(2, int(ml["reach"] * self.cpm * 1.3))
            sf = shadowcast(self.m.blocks_sight, fcx, fcy, rc)
            sl = renderer._vf_slice(sf, vf, h_, w_)
            if sl is None:
                continue
            (y0, y1, x0, x1), (sy0, sy1, sx0, sx1) = sl
            lit = lighting.point_light(sf.dist[sy0:sy1, sx0:sx1],
                                       sf.visible[sy0:sy1, sx0:sx1], rc, b)
            np.maximum(mf[y0:y1, x0:x1], lit, out=mf[y0:y1, x0:x1])
            col = np.array(ml["col"], np.float32) / 255.0
            mrgb[y0:y1, x0:x1] += lit[:, :, None] * col
        if mf.max() <= 0.0:
            return None, None
        vis = vf.visible.astype(np.float32)
        mf = renderer.box_blur(mf, renderer.MFLASH_BLUR) * vis
        mrgb *= vis[:, :, None]
        return (mf if mf.max() > 0.0 else None,
                mrgb if mrgb.max() > 0.0 else None)

    def _light_pass(self, vf, inten, now):
        """Build this frame's illumination and gate sight on it.

        Three lights: your own torch, everyone else's, and every muzzle flash
        of the last tenth of a second. All three are cast from their own
        source, so light stops at walls — and so does what the light shows you.

        Returns (cone intensity after the light gate, the additive world
        brighten to lay down before the fog, or None)."""
        h_, w_ = inten.shape
        wl = self.cli.world
        me = wl.players.get(wl.my_id)
        reach = self._beam_reach()

        fl = None
        if me is not None and me.alive and me.flashlight:
            fl = lighting.cone_beam(vf.ang, vf.dist, vf.visible, self.facing,
                                    renderer.FLASHLIGHT_HALF_DEG, reach)
            fl = renderer.box_blur(fl, renderer.FLASHLIGHT_BLUR)

        # Other people's torches, cast from where THEY stand and then masked to
        # what you can see. You get the room their beam lights and the sweeping
        # cone itself, which is the tell that someone is in there — the same
        # thing a guard's torch does to you in single-player.
        beams = None
        rc = max(2, int(reach))
        for p in wl.players.values():
            if p.id == wl.my_id or not p.alive or not p.flashlight:
                continue
            bx, by, baim = p.render_pos(now, INTERP_DELAY)
            sf = self.vis.get(*self.m.cell_of(bx, by), rc)
            sl = renderer._vf_slice(sf, vf, h_, w_)
            if sl is None:
                continue
            (y0, y1, x0, x1), (sy0, sy1, sx0, sx1) = sl
            gl = lighting.cone_beam(sf.ang[sy0:sy1, sx0:sx1],
                                    sf.dist[sy0:sy1, sx0:sx1],
                                    sf.visible[sy0:sy1, sx0:sx1],
                                    baim, renderer.FLASHLIGHT_HALF_DEG, rc)
            if beams is None:
                beams = np.zeros((h_, w_), np.float32)
            np.maximum(beams[y0:y1, x0:x1], gl, out=beams[y0:y1, x0:x1])
        if beams is not None:
            beams *= vf.visible
            beams = renderer.box_blur(beams, 1)

        self.mflash, self.mflash_rgb = self._muzzle_light(vf, h_, w_, now)
        self.beams = beams

        # A muzzle flash is deliberately NOT in `illum`: it parts the fog while
        # it burns (below, in _draw_fog) but it must not write a room into
        # memory that you only saw by its light for a twelfth of a second.
        illum = lighting.static_illumination(self.m.lightmap, vf.y0, vf.x0,
                                             h_, w_)
        if fl is not None:
            illum = np.maximum(illum, fl)
        if beams is not None:
            illum = np.maximum(illum, beams)
        self.illum = illum
        gated = inten * lighting.see_gate(illum, renderer.LIGHT_SEE_MIN,
                                          renderer.LIGHT_SEE_FULL)
        self.known[vf.y0:vf.y0 + h_, vf.x0:vf.x0 + w_] |= gated > 0.03

        add = None
        for field, gain in ((fl, renderer.FLASHLIGHT_GAIN),
                            (self.mflash, renderer.MUZZLE_LIGHT_GAIN),
                            (beams, renderer.GUARD_FLASH_GAIN)):
            if field is None:
                continue
            lit = field * gain
            add = lit if add is None else np.maximum(add, lit)
        return gated, add

    def _illum_at(self, vf, cx, cy) -> float:
        """How much light falls on one cell — a baked lamp, a beam, the pop of
        a muzzle flash. This is what a body is drawn with."""
        val = 0.0
        if self.m.lightmap is not None:
            lh, lw = self.m.lightmap.shape
            if 0 <= cy < lh and 0 <= cx < lw:
                val = float(self.m.lightmap[cy, cx])
        if self.illum is not None:
            ry, rx = cy - vf.y0, cx - vf.x0
            ih, iw = self.illum.shape
            if 0 <= ry < ih and 0 <= rx < iw:
                val = max(val, float(self.illum[ry, rx]))
                if self.mflash is not None:
                    val = max(val, float(self.mflash[ry, rx]) * 1.4)
        return val

    def _sees_cell(self, vf, inten, cx, cy) -> bool:
        h_, w_ = inten.shape
        ry, rx = cy - vf.y0, cx - vf.x0
        if not (0 <= ry < h_ and 0 <= rx < w_):
            return False
        return bool(inten[ry, rx] > SEEN_MIN)

    @staticmethod
    def _light_tint(level: float) -> tuple:
        """Illumination -> the multiply a body is drawn with: a black
        silhouette in an unlit room, the full painted sprite under a lamp.
        Same curve single-player uses on the player."""
        k = min(1.0, max(0.0, (level - BODY_LIGHT_LO) /
                         (BODY_LIGHT_HI - BODY_LIGHT_LO)))
        k = k * k * (3.0 - 2.0 * k)
        v = int(3 + 252 * k)
        return (v, v, v)

    def _draw_body(self, x, y, facing, colour, name, wep_id=None,
                   reloading=False, dead=False, light=1.0, shot=None,
                   torch=False, now=0.0, torch_light=None, cls=None):
        """A soldier in their own colour art, lit by whatever light reaches
        them, with a ring in that colour at their feet.

        The ring stays even where the coloured art exists: two players in
        neighbouring greys are still two players, and in a dark room the sprite
        is a silhouette and the ring is all there is to read."""
        ppm = self.ppm
        cx, cy = x * ppm, y * ppm
        r = max(4, int(renderer.BODY_R * ppm))
        tint = self._light_tint(light)
        # a lamp held out in front lights the front of the body, not the back:
        # the whole sprite at the room's light, then its front half again at
        # the lamp's
        front = None
        if torch and not dead:
            front = self._light_tint(max(
                light, renderer.FLASHLIGHT_SELF_SEEN if torch_light is None
                else torch_light))
        cname = sprites.colour_name(colour)
        cls_art = classes.get(cls).art if cls else ""
        pygame.draw.circle(self.surf, colour, (int(cx), int(cy)), r + 4, 2)
        drew = False
        if self.bank.ok:
            if dead:
                drew = self.bank.blit(
                    self.surf, self.bank.body_art("soldier_ded", cname, cls_art),
                    cx, cy, facing, tint=tuple(int(c * 0.55) for c in tint))
            elif reloading:
                art = self.bank.body_art("soldier_idle", cname, cls_art)
                drew = self.bank.blit(self.surf, art, cx, cy, facing, tint=tint)
                self.bank.blit(self.surf, "weapon_sling", cx, cy, facing,
                               tint=tint)
                if front and drew:
                    self.bank.blit_front(self.surf, art, cx, cy, facing,
                                         tint=front)
                    self.bank.blit_front(self.surf, "weapon_sling", cx, cy,
                                         facing, tint=front)
            else:
                art = self.bank.body_art("soldier_ready", cname, cls_art)
                drew = self.bank.blit(self.surf, art, cx, cy, facing, tint=tint)
                if front and drew:
                    self.bank.blit_front(self.surf, art, cx, cy, facing,
                                         tint=front)
                if wep_id:
                    self._draw_weapon(cx, cy, facing, wep_id, tint, shot, now,
                                      front=front)
            if drew and not dead:
                # the lamp itself is drawn full-bright when it is on, and as an
                # unlit fitting when it is off
                if torch:
                    self.bank.blit(self.surf, "light_on", cx, cy, facing)
                else:
                    self.bank.blit(self.surf, "light_off", cx, cy, facing,
                                   tint=tint)
        if not drew:
            pygame.draw.circle(self.surf, (12, 12, 14), (int(cx), int(cy)), r + 2)
            pygame.draw.circle(self.surf, colour, (int(cx), int(cy)), r)
            tip = (cx + math.cos(facing) * r * 2.1,
                   cy + math.sin(facing) * r * 2.1)
            left = (cx + math.cos(facing + 0.5) * r,
                    cy + math.sin(facing + 0.5) * r)
            right = (cx + math.cos(facing - 0.5) * r,
                     cy + math.sin(facing - 0.5) * r)
            pygame.draw.polygon(self.surf, colour, (tip, left, right))
        if name:
            tag = self.font.render(name, True, (235, 235, 235))
            self.surf.blit(tag, tag.get_rect(center=(cx, cy + NAME_DY * ppm)))

    def _draw_weapon(self, cx, cy, facing, wep_id, tint, shot, now,
                     front=None) -> None:
        """The gun, kicked back by however much recoil is left, and the muzzle
        flash pinned to its barrel.

        The gun keeps its own colours — a red rifle reads as a toy — and the
        flash is the same two-layer sprite pair single-player draws, through
        the same function."""
        art = sprites.weapon_art(wep_id)
        if not art:
            return                    # a blade or a grenade: nothing in hand yet
        kick = 0.0
        if shot:
            age = now - shot["t0"]
            kick = (max(0.0, 1.0 - age / sprites.RECOIL_TIME)
                    * sprites.RECOIL_M * self.ppm)
        gwx = cx - math.cos(facing) * kick
        gwy = cy - math.sin(facing) * kick
        self.bank.blit(self.surf, art, gwx, gwy, facing, tint=tint)
        if front is not None:
            # the lamp's half of the gun, split on the body's line, not the
            # gun's, so recoil does not move the edge
            self.bank.blit_front(self.surf, art, gwx, gwy, facing, tint=front,
                                 pivot=(cx, cy))
        if shot:
            ft = max(0.0, sprites.FLASH_TIME - (now - shot["t0"]))
            sprites.draw_muzzle(self.bank, self.surf, shot.get("art") or art,
                                gwx, gwy, facing, ft, shot["roll"],
                                shot["scale"])

    def _draw_swings(self, now, mine: bool):
        """A knife swing: a short arc where the blade went, brighter if it
        landed. There is no projectile to trace, so this is all there is."""
        w = self.cli.world
        for sw in self.swings:
            if (sw["id"] == w.my_id) != mine:
                continue
            if sw["id"] == w.my_id:
                x, y = self.rx, self.ry
            else:
                p = w.players.get(sw["id"])
                if p is None:
                    continue
                x, y, _a = p.render_pos(now, INTERP_DELAY)
            age = (now - sw["t0"]) / SWING_TIME
            fade = max(0.0, 1.0 - age)
            r = sw["reach"] * self.ppm
            col = (235, 225, 205) if sw["hit"] else (150, 156, 166)
            arc = math.radians(50.0)
            steps = 7
            pts = [(x * self.ppm, y * self.ppm)]
            for i in range(steps + 1):
                a = sw["heading"] - arc + 2 * arc * i / steps
                pts.append((x * self.ppm + math.cos(a) * r,
                            y * self.ppm + math.sin(a) * r))
            pygame.draw.lines(self.surf, tuple(int(c * fade) for c in col),
                              False, pts[1:], 2)

    def _draw_effects(self, now, mine: bool):
        """Tracers, muzzle flashes, blasts and rockets.

        Other people's are drawn UNDER the fog, so a firefight in a room you
        cannot see into stays invisible; your own go over it, because you know
        where your own gun is pointing."""
        ppm = self.ppm
        for segs, t0, is_mine in self.tracers:
            if is_mine != mine:
                continue
            age = (now - t0) / renderer.TRACER_FADE
            if age >= 1.0:
                continue
            a = int(230 * (1.0 - age))
            col = (255, 240, 190) if is_mine else (255, 214, 140)
            for p0, p1 in segs:
                pygame.draw.line(self.surf, tuple(int(c * a / 255) for c in col),
                                 (p0[0] * ppm, p0[1] * ppm),
                                 (p1[0] * ppm, p1[1] * ppm), 2)
        for b in self.blasts:
            age = (now - b["t0"]) / renderer.BLAST_FADE
            if age >= 1.0:
                continue
            rad = int(b["r"] * ppm * (0.35 + 0.65 * age))
            col = (255, int(180 - 120 * age), 60)
            pygame.draw.circle(self.surf, col,
                               (int(b["x"] * ppm), int(b["y"] * ppm)),
                               rad, max(2, int(6 * (1.0 - age))))
        for rk in self.rockets:
            age = now - rk["t0"]
            if age >= rk["dur"]:
                continue
            f = min(1.0, age / max(rk.get("flight", rk["dur"]), 1e-3))
            x = rk["x"] + (rk["ix"] - rk["x"]) * f
            y = rk["y"] + (rk["iy"] - rk["y"]) * f
            if rk.get("fuse", 0.0) > 0.0:
                # a grenade: dark, and the fuse light quickens as it burns down
                left = 1.0 - age / rk["dur"]
                blink = (now * (6.0 + 10.0 * (1.0 - left))) % 1.0 < 0.5
                pygame.draw.circle(self.surf, (70, 74, 66),
                                   (int(x * ppm), int(y * ppm)),
                                   max(2, int(ROCKET_R * ppm)))
                if blink:
                    pygame.draw.circle(self.surf, (235, 120, 60),
                                       (int(x * ppm), int(y * ppm)),
                                       max(1, int(ROCKET_R * ppm * 0.5)))
            else:
                pygame.draw.circle(self.surf, (255, 200, 120),
                                   (int(x * ppm), int(y * ppm)),
                                   max(2, int(ROCKET_R * ppm)))

    def _blast_light(self, x, y, w, now) -> None:
        """A detonation as light: brighter, wider and longer-lived than a
        muzzle flash, cast from the impact point."""
        col, gain, reach, life = renderer.blast_flash_spec(w)
        self.mlights.append({"x": x, "y": y, "t0": now, "col": col,
                             "gain": gain, "reach": reach, "life": life})

    def _detonate(self, rk, now) -> None:
        """A rocket reached its impact point. The server sends the round on its
        way and its speed; when it arrives is arithmetic, so the client does
        the fireball itself rather than waiting for a message.

        A fused round is different: the server decides when it goes off and
        says so, and drawing our own fireball as well as that one is how a
        grenade came to explode twice."""
        w = weapons.ROSTER.get(rk.get("wep", ""))
        if w is None or w.blast_r <= 0.0 or rk.get("fuse", 0.0) > 0.0:
            return
        self.blasts.append({"x": rk["ix"], "y": rk["iy"], "r": w.blast_r,
                            "t0": now})
        self._blast_light(rk["ix"], rk["iy"], w, now)

    def _rail_sound(self) -> None:
        """The rail spool-up, for as long as the trigger is held.

        Single-player starts this the moment the charge begins and cuts it on
        release; multiplayer has no local trigger state to hang it on, so it
        follows the charge the server reports - which is the same clock the
        ring at the cursor is drawn from.
        """
        if not self.audio_on:
            return
        me = self.cli.world.players.get(self.cli.world.my_id)
        w = weapons.ROSTER[self.loadout[self.wep]]
        charging = (me is not None and me.charge > 0.0 and not w.is_cooked
                    and me.alive)
        if charging and self.rail_ch is None:
            self.rail_ch = audio.play_channel("railcharge", 0.26)
        elif not charging and self.rail_ch is not None:
            self.rail_ch.stop()
            self.rail_ch = None

    def _reap_effects(self, now):
        self.tracers = [t for t in self.tracers
                        if now - t[1] < renderer.TRACER_FADE]
        self.blasts = [b for b in self.blasts
                       if now - b["t0"] < renderer.BLAST_FADE]
        self.mlights = [m for m in self.mlights
                        if now - m["t0"] < m["life"]]
        self.swings = [sw for sw in self.swings if now - sw["t0"] < SWING_TIME]
        gone = max(sprites.FLASH_TIME, sprites.RECOIL_TIME)
        self.shots = {i: sh for i, sh in self.shots.items()
                      if now - sh["t0"] < gone}
        live = []
        for rk in self.rockets:
            if now - rk["t0"] < rk["dur"]:
                live.append(rk)
            else:
                self._detonate(rk, now)
        self.rockets = live

    def _ping_into(self, ov_a, build):
        """Open the veil inside a live ping, and write those cells into fog
        memory so the room stays remembered once it fades."""
        by0, by1, bx0, bx1 = build
        px, py, _t = self.ping
        cx, cy = self.m.cell_of(px, py)
        r = classes.PING_RADIUS_M * self.cpm
        ys = np.arange(by0, by1, dtype=np.float32)[:, None] - cy
        xs = np.arange(bx0, bx1, dtype=np.float32)[None, :] - cx
        inside = (xs * xs + ys * ys) <= r * r
        ov_a[inside] = np.minimum(
            ov_a[inside],
            renderer.A_REMEMBER_F * (1.0 - renderer.CONE_REVEAL))
        self.known[by0:by1, bx0:bx1] |= inside

    def _draw_fog(self, vf, inten):
        """The veil: unknown, remembered, and what is in view right now.

        Only the fine cells under the camera (plus a blur margin) are built,
        blurred and blitted - on a big map the rest of the world costs
        nothing."""
        ppc = self.ppm / self.cpm                  # pixels per fine cell
        build, blit = renderer.fog_windows(
            self.known.shape, self.cam_x, self.cam_y, self.view_w,
            self.view_h, ppc)
        by0, by1, bx0, bx1 = build
        ov_a, _rgb = renderer.fog_veil(
            self.known, build, cone=(vf.y0, vf.x0, inten), mflash=self.mflash)
        if self.ping is not None:
            self._ping_into(ov_a, build)

        sy0, sy1, sx0, sx1 = blit
        fw, fh = sx1 - sx0, sy1 - sy0
        if fw <= 0 or fh <= 0:
            return
        if self._ov.get_size() != (fw, fh):
            self._ov = pygame.Surface((fw, fh))
        px3 = pygame.surfarray.pixels3d(self._ov)
        keep = np.transpose(np.clip(
            (1.0 - ov_a[sy0 - by0:sy1 - by0, sx0 - bx0:sx1 - bx0]) * 255.0,
            0, 255)).astype(np.uint8)
        px3[:, :, 0] = keep
        px3[:, :, 1] = keep
        px3[:, :, 2] = keep
        del px3
        big = (round(fw * ppc), round(fh * ppc))
        if self._ov_big.get_size() != big:
            self._ov_big = pygame.Surface(big)
        pygame.transform.scale(self._ov, big, self._ov_big)
        self.surf.blit(self._ov_big, (round(sx0 * ppc), round(sy0 * ppc)),
                       special_flags=pygame.BLEND_RGB_MULT)

    def _draw_cues(self, now):
        """The arc that says a sound arrived, and roughly from where. A firm
        arrival gives a tight arc; a faint one smears out; no usable direction
        at all becomes a full ring."""
        for c in self.cues:
            age = (now - c["t"]) / renderer.CUE_FADE
            alpha = (1.0 - age) ** 1.5 * 210
            if alpha < 2:
                continue
            if c["dir"]:
                half = renderer.CUE_NARROW_DEG + \
                    (renderer.CUE_WIDE_DEG - renderer.CUE_NARROW_DEG) * \
                    (1.0 - min(1.0, c["rem"] / 0.55))
            else:
                half = 180.0
            renderer.draw_cue(self.surf, c["x"] * self.ppm, c["y"] * self.ppm,
                              c["ang"], half, alpha, self.ppm)

    def _bar(self, surf, x, y, w, h, frac, col):
        pygame.draw.rect(surf, (30, 33, 38), (x, y, w, h), border_radius=3)
        if frac > 0:
            pygame.draw.rect(surf, col, (x, y, int(w * min(1.0, frac)), h),
                             border_radius=3)

    def _draw_hud(self, now):
        w = self.cli.world
        me = w.players.get(w.my_id)
        pad = 12
        t = max(0.0, w.time_left)
        clock_txt = self.font_big.render(
            f"{int(t) // 60}:{int(t) % 60:02d}", True, (235, 235, 235))
        self.window.blit(clock_txt,
                         (self.view_w // 2 - clock_txt.get_width() // 2, pad))

        rows = sorted(w.players.values(), key=lambda p: (-p.kills, p.id))
        if self.show_scores:
            bw, bh = 360, 44 + 26 * len(rows)
            panel = pygame.Surface((bw, bh), pygame.SRCALPHA)
            panel.fill((16, 18, 22, 228))
            head = "TEAM DEATHMATCH" if w.mode == GameMode.TEAM else "FREE FOR ALL"
            panel.blit(self.font_mid.render(head, True, (222, 226, 232)), (14, 10))
            for i, p in enumerate(rows):
                yy = 42 + i * 26
                pygame.draw.circle(panel, p.colour, (24, yy + 9), 7)
                tag = " (you)" if p.id == w.my_id else ""
                panel.blit(self.font.render(f"{p.name}{tag}", True,
                                            (222, 226, 232)), (40, yy))
                panel.blit(self.font.render(str(p.kills), True,
                                            (222, 226, 232)), (320, yy))
            self.window.blit(panel, (self.view_w // 2 - bw // 2, 60))
        else:
            strip = "   ".join(f"{p.name} {p.kills}" for p in rows[:6])
            self.window.blit(self.font.render(strip, True, (150, 156, 166)),
                             (pad, pad))

        # kill feed, top right
        for i, k in enumerate(w.killfeed[-5:]):
            age = now - k.at
            if age > 6.0:
                continue
            txt = f"{k.killer} {k.verb} {k.victim}"
            surf = self.font.render(txt, True, (210, 180, 150))
            surf.set_alpha(int(255 * max(0.0, 1.0 - (age - 4.0) / 2.0))
                           if age > 4.0 else 255)
            self.window.blit(surf, (self.view_w - surf.get_width() - pad,
                                    pad + 26 + i * 18))

        # bottom left: condition and gun
        wep = weapons.ROSTER[self.loadout[self.wep]]
        modes = wep.fire_modes()
        mode = modes[min(self.modes[self.wep], len(modes) - 1)]
        by = self.view_h - 88
        panel = pygame.Surface((300, 76), pygame.SRCALPHA)
        panel.fill((16, 18, 22, 210))
        self.window.blit(panel, (pad, by))
        if me:
            self._bar(self.window, pad + 10, by + 10, 150, 8, me.hp,
                      (190, 70, 70))
            self._bar(self.window, pad + 10, by + 22, 150, 6, me.sh,
                      (80, 150, 230))
            # sprint fuel: dull grey once it has bottomed out and locked
            self._bar(self.window, pad + 10, by + 31, 150, 4, self.stamina,
                      (120, 120, 130) if self.sprint_locked else (230, 205, 110))
            ammo = f"{me.mag}" if me.reload_t <= 0 else "reloading"
            if wep.is_melee:
                ammo = "blade"
            if me.charge > 0.0:
                if wep.is_cooked:
                    # what is left of the fuse, which is what you are actually
                    # deciding about while you hold it
                    ammo = f"cook {max(0.0, wep.fuse_s - me.charge):.1f}s"
                else:
                    ammo = f"charge {min(1.0, me.charge / 3.0) * 100:.0f}%"
            self.window.blit(self.font.render(
                f"{wep.name}  [{mode}]", True, (222, 226, 232)),
                (pad + 10, by + 40))
            self.window.blit(self.font_mid.render(ammo, True, (230, 210, 140)),
                             (pad + 210, by + 20))
            if wep.recharge_s > 0.0 and me.charge <= 0.0:
                # a weapon with no reserve to count: what there is to know is
                # how long until it hands you the next round
                self.window.blit(self.font.render(
                    f"/{wep.mag}", True, (120, 126, 136)),
                    (pad + 210, by + 42))
                self._bar(self.window, pad + 210, by + 58, 76, 4,
                          me.recharge, (120, 190, 235))
            # the class ability: running, cooling down, or ready
            ab = classes.ability_of(me.cls_now)
            if me.ability_t > 0.0:
                txt, col = f"{ab.name}  {me.ability_t:.1f}s", (230, 210, 140)
            elif me.ability_cd > 0.0:
                txt, col = f"{ab.name}  {me.ability_cd:.0f}s", (120, 126, 136)
            else:
                txt, col = f"space: {ab.name}", (190, 196, 206)
            self.window.blit(self.font.render(txt, True, col),
                             (pad + 10, by + 58))
        near = self._near_door()
        if near is not None:
            d = self.doors.door(near)
            name = "blast door" if d.heavy else "door"
            if d.moving and not d.reversible:
                # a blast door is committed; the countdown is what matters
                hint = f"{name} {'opening' if d.target else 'sealing'} — {d.left:.1f}s"
            elif (d.target if d.moving else d.is_open):
                hint = f"f: close {name}"
            else:
                hint = (f"f: open {name} ({d.dur:.0f}s)" if d.heavy
                        else f"f: open {name}")
            txt = self.font_mid.render(hint, True, (230, 210, 140))
            self.window.blit(txt, txt.get_rect(
                center=(self.view_w // 2, self.view_h - 120)))
        if now - self.msg_t < 2.2:
            self.window.blit(self.font.render(self.msg, True, (150, 156, 166)),
                             (pad, by - 20))
        self.window.blit(self.font.render(
            f"({self.px:5.1f}, {self.py:5.1f})  correction "
            f"{self.corrected_m * 100:4.1f} cm    tab scores   e knock   "
            f"space ability   esc menu",
            True, (90, 96, 106)), (pad, self.view_h - 20))

        mxp, myp = pygame.mouse.get_pos()
        pygame.draw.circle(self.window, (230, 230, 230), (mxp, myp), 3, 1)
        if me is not None and me.charge > 0.0:
            # the spool-up ring single-player draws, for the rail and for a
            # grenade cooking in your hand - amber, then red as the fuse runs
            full = wep.fuse_s if wep.is_cooked else 3.0
            f = min(1.0, me.charge / max(full, 1e-3))
            ring = ((235, 170, 60) if f < 0.7 else (235, 90, 60)) \
                if wep.is_cooked else (120, 190, 255)
            pygame.draw.circle(self.window, (60, 64, 74), (mxp, myp), 16, 1)
            pygame.draw.arc(self.window, ring, (mxp - 16, myp - 16, 32, 32),
                            -math.pi / 2, -math.pi / 2 + f * 2 * math.pi,
                            3 if f < 0.999 else 4)

    # ---- one frame ---------------------------------------------------

    def frame(self, dt) -> str | None:
        w = self.cli.world
        now = time.monotonic()
        for ev in pygame.event.get():
            r = self.handle_event(ev)
            if r:
                return r
        for e in self.cli.drain_events():
            t = e["t"]
            if t == "match_end":
                return "match_end"
            if t == "disconnected":
                return "disconnected"
            if t == "map_state":
                self._apply_map_state(w)
            elif t == "shot":
                self._on_shot(e, now)
            elif t == "sound":
                self._on_sound(e, now)
            elif t == "glass":
                self._on_glass(e, now)
            elif t == "door":
                self._on_door(e, now)
            elif t == "pickup":
                self._on_pickup(e, now)
            elif t == "ability":
                self._on_ability(e, now)
            elif t == "melee":
                self.swings.append({"id": e["id"], "heading": e["heading"],
                                    "reach": e.get("reach", 1.6),
                                    "hit": e.get("hit", 0), "t0": now})

        # doors travel on this client's own clock, started by the message that
        # said they had set off. Stepped BEFORE prediction, so the geometry
        # movement is predicted against is the geometry the server has.
        self.packs.step(dt)
        for _key, _open, _chg in self.doors.step(dt):
            if _chg:
                self._door_arrived(_key, _open)
        # the slit between moving panels. The client only needs it for
        # sight — the server owns bullets — but writing both keeps this map
        # identical to the server's
        _sg = self.doors.gap_changes(self.m.subdiv)
        for _key, _gap in _sg:
            _d = self.doors.door(_key)
            set_door_gap(self.m, _key[0], _key[1], _gap, _d.axis,
                         bullets=_d.shoot_through)
        if _sg:
            self.vis.invalidate()

        self._sync_class()
        mx, my, buttons = self.read_input()
        me_now = w.players.get(w.my_id)
        if me_now is not None:
            # the server's number is the truth; ease onto it rather than
            # snapping, so the bar does not jitter between snapshots
            self.stamina += (me_now.stamina - self.stamina) * min(1.0, dt * 6.0)
            if me_now.stamina <= 0.0:
                self.sprint_locked = True
            elif me_now.stamina >= movement.STAMINA_UNLOCK:
                self.sprint_locked = False
        if self.menu_open:
            # in the menu you stand still and hold fire; the match goes on
            mx = my = 0.0
            buttons = 0
        me = w.players.get(w.my_id)
        alive = bool(me and me.alive)
        if alive:
            self.send_commands(mx, my, buttons, dt)
            self.reconcile()
        elif me:
            self.px, self.py = me.x, me.y
            self.pending.clear()
            self.acc = 0.0

        f = min(1.0, RENDER_SMOOTH_PER_S * dt)
        self.rx += (self.px - self.rx) * f
        self.ry += (self.py - self.ry) * f

        if self.ping is not None and now >= self.ping[2]:
            self.ping = None          # the cells it showed stay remembered
        self._hear(now)
        self._rail_sound()
        self._reap_effects(now)

        self.cam_x = int(min(max(self.rx * self.ppm - self.view_w * 0.5, 0),
                             max(0, self.world_w - self.view_w)))
        self.cam_y = int(min(max(self.ry * self.ppm - self.view_h * 0.5, 0),
                             max(0, self.world_h - self.view_h)))

        self.surf.blit(self.base, (0, 0))
        self._draw_doors()
        self._draw_packs()
        vf, inten = self._visible_mask()
        inten, add = self._light_pass(vf, inten, now)
        h_, w_ = inten.shape
        ppc = self.ppm / self.cpm

        # everyone else, and everything they did, goes under the fog
        self._draw_effects(now, mine=False)
        self._draw_swings(now, mine=False)
        for p in w.players.values():
            if p.id == w.my_id or not p.alive:
                continue
            rx, ry, raim = p.render_pos(now, INTERP_DELAY)
            cx, cy = self.m.cell_of(rx, ry)
            if p.vanished and math.hypot(rx - self.rx, ry - self.ry) \
                    > classes.VANISH_SEEN_M:
                continue                  # vanished: nothing to see from here
            if not self._sees_cell(vf, inten, cx, cy) \
                    and not self._ping_visible(rx, ry):
                continue
            self._draw_body(rx, ry, raim, p.colour, p.name,
                            cls=p.cls_now,
                            wep_id=self._their_weapon(p),
                            reloading=p.reload_t > 0.0,
                            light=self._illum_at(vf, cx, cy),
                            shot=self.shots.get(p.id), torch=p.flashlight,
                            now=now)

        # the world brightens under whichever light is strongest here, BEFORE
        # the fog goes on — a real light falls on the floor, not on the veil
        if add is not None and add.max() > 1.0:
            renderer._additive_blit(self.surf, add, 1.0, vf, ppc, w_, h_,
                                    tint=renderer.WARM_LIGHT_TINT)

        self._draw_fog(vf, inten)

        # you, and what you did, go over it
        if alive:
            pcx, pcy = self.m.cell_of(self.rx, self.ry)
            mylight = self._illum_at(vf, pcx, pcy)
            self._draw_body(self.rx, self.ry, self.facing,
                            (me.colour if me else (220, 220, 220)), "",
                            cls=self.cls_now,
                            wep_id=self.loadout[self.wep],
                            reloading=bool(me and me.reload_t > 0.0),
                            light=mylight, shot=self.shots.get(w.my_id),
                            torch=bool(me and me.flashlight), now=now,
                            torch_light=renderer.FLASHLIGHT_SELF)
        self._draw_effects(now, mine=True)
        self._draw_swings(now, mine=True)
        if self.mflash_rgb is not None:
            # the coloured glow of each flash, already shadow-cast and masked
            # to line of sight, over the fog
            renderer._additive_blit(self.surf, self.mflash_rgb,
                                    renderer.MFLASH_POP_GAIN, vf, ppc, w_, h_)
        if self.beams is not None:
            # somebody else's beam over the fog, cool white so it reads as not
            # yours
            renderer._additive_blit(self.surf, self.beams,
                                    renderer.GUARD_FLASH_POP, vf, ppc, w_, h_,
                                    tint=renderer.GUARD_LIGHT_TINT)
        self._draw_cues(now)

        self.window.blit(self.surf, (-self.cam_x, -self.cam_y))
        if not alive and me:
            txt = self.font_big.render(
                f"respawning in {me.respawn_in:.0f}", True, (230, 180, 60))
            self.window.blit(txt, txt.get_rect(
                center=(self.view_w // 2, self.view_h // 2)))
        self._draw_hud(now)
        if self.menu_open:
            self._draw_menu()
        pygame.display.flip()
        return None


# --------------------------------------------------------------------- driver

def run_match(cli: GameClient, maps_dir="maps", window=None) -> str:
    """Play until the match ends or the player leaves. Returns why it stopped."""
    try:
        view = MatchView(cli, maps_dir, window)
    except Exception as e:
        print(f"could not start the match render: {e}")
        return "error"
    clock = pygame.time.Clock()
    while True:
        dt = clock.tick(renderer.FPS_CAP or 0) / 1000.0
        dt = min(dt, 0.1)                 # a hitch must not teleport anyone
        why = view.frame(dt)
        if why:
            pygame.mouse.set_visible(True)
            return why


def main():
    args = parse_args()
    window = _window_size(args.window)
    import net.client as _netclient
    _netclient.MAPS_DIR = Path(args.maps_dir)   # where to look for the host's map
    ip, port = args.address, DEFAULT_PORT
    if ip and ":" in ip:
        ip, _, ps = ip.partition(":")
        try:
            port = int(ps)
        except ValueError:
            print("bad port in address")
            return 2
    mode = "host" if args.host else ("join" if ip else None)

    client: GameClient | None = None
    server: GameServer | None = None
    try:
        while True:
            result, client, server = run_lobby(
                default_ip=(f"{ip}:{port}" if ip and port != DEFAULT_PORT else ip),
                existing_client=client, existing_server=server,
                mode=mode, maps_dir=args.maps_dir)
            if result != LobbyResult.START:
                print("lobby ended:", result.value)
                return 0
            why = run_match(client, args.maps_dir, window)
            print("match ended:", why)
            if why in ("quit", "disconnected", "error"):
                return 0
            pygame.display.set_mode((LOBBY_W, LOBBY_H))
            if why == "leave":
                # step out but stay connected — the lobby offers a way back in
                client.leave_match()
                mode = None
                continue
            # match_end: back to the lobby on the same connection
            deadline = time.monotonic() + 12.0
            while (client.world.state != ServerState.LOBBY
                   and time.monotonic() < deadline):
                client.drain_events()
                time.sleep(0.05)
            mode = None
    except KeyboardInterrupt:
        pass
    finally:
        if client:
            client.disconnect()
        if server:
            server.stop()
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
