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
    tab             hold for the scoreboard
    esc             quit
"""
from __future__ import annotations

import argparse
import math
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
from net.protocol import (BTN_CRAWL, BTN_FIRE, BTN_INTERACT, BTN_RELOAD,
                          BTN_RUN, DEFAULT_PORT, GameMode, ServerState)
from net.server import TICK_DT
from sim import audio, movement, perception, sprites, weapons
from sim.tilemap import break_glass_cells, compute_roof, find_doors, set_door
from sim.vision import ConeSpec, VisibilityCache

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

INTERP_DELAY = 0.10       # render remote players this far in the past, so there
                          # are always two snapshots to interpolate between

NAME_DY = -0.95           # name tag offset above a body, metres
SEEN_MIN = 0.02           # cone intensity at which a remote player registers
MUZZLE_FLASH_T = 0.07     # how long a muzzle flash is drawn, seconds
ROCKET_R = 0.14           # drawn radius of a rocket in flight, metres

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

CLIP_BY_KIND = {
    "fire": "gunshot", "step": "footstep", "reload": "magazine",
    "dry": "dryfire", "glass": "glass", "knock": "knock", "door": "door",
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
        self.m = netmaps.load(w.map_id, maps_dir)
        self.ppm = renderer.PX_PER_M
        self.cpm = self.m.cells_per_metre
        # the sound solver works on its own copy of the cost field, kept in
        # step with the map when glass breaks
        self.cost = self.m.sound_cost.astype(np.float64)

        self.world_w = round(self.m.width_m * self.ppm)
        self.world_h = round(self.m.height_m * self.ppm)
        cap = window or renderer.MAX_VIEW
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
        self._ov = pygame.Surface((8, 8), pygame.SRCALPHA)

        self.font = pygame.font.SysFont("consolas,monospace", 14)
        self.font_mid = pygame.font.SysFont("consolas,monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("consolas,monospace", 34, bold=True)
        pygame.mouse.set_visible(False)

        sx, sy = w.my_spawn or self.m.player_spawn
        self.px, self.py = float(sx), float(sy)      # predicted position
        self.rx, self.ry = self.px, self.py          # smoothed, what is drawn
        self.facing = -math.pi / 2
        self.aim_dist = 10.0
        self.cam_x = self.cam_y = 0
        self.acc = 0.0                    # unspent frame time, in tick units
        self.pending: deque = deque()     # commands the server has not acked
        self.last_ack = -1
        self.show_scores = False
        self.corrected_m = 0.0
        self.travelled = 0.0

        # weapons: the client asks, the server decides
        self.loadout = list(weapons.DEFAULT_LOADOUT)
        self.wep = 0
        self.modes = [0] * len(self.loadout)

        # sound and effects
        self.sounds: list = []        # ActiveSound: fields being solved/heard
        self.jobs: list = []          # chunked solve jobs
        self.cues: list = []          # arrival arcs at the listener
        self.tracers: list = []       # (segments, t0, mine)
        self.flashes: list = []       # {x, y, t0, mine}
        self.blasts: list = []        # {x, y, r, t0}
        self.rockets: list = []       # {x, y, ix, iy, t0, dur}
        self.broken: set = set()      # coarse glass cells already broken here
        self.doors = find_doors(self.m)
        self.msg, self.msg_t = "", -9.0

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
        if pygame.mouse.get_pressed()[0]:
            buttons |= BTN_FIRE
        mxp, myp = pygame.mouse.get_pos()
        ax = (mxp + self.cam_x) / self.ppm
        ay = (myp + self.cam_y) / self.ppm
        dx, dy = ax - self.rx, ay - self.ry
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            self.facing = math.atan2(dy, dx)
            self.aim_dist = max(0.5, math.hypot(dx, dy))
        return mx, my, buttons

    def handle_event(self, ev) -> str | None:
        if ev.type == pygame.QUIT:
            return "quit"
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                return "leave"
            if ev.key == pygame.K_TAB:
                self.show_scores = True
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
        elif ev.type == pygame.MOUSEWHEEL:
            self._select((self.wep - ev.y) % len(self.loadout))
        return None

    def _select(self, idx: int) -> None:
        if idx != self.wep:
            self.wep = idx
            self._say(f"switched to {weapons.ROSTER[self.loadout[idx]].name}")

    def _say(self, text: str) -> None:
        self.msg, self.msg_t = text, time.monotonic()

    # ---- prediction --------------------------------------------------

    @staticmethod
    def _stance(buttons):
        if buttons & BTN_CRAWL:
            return "crawl"
        return "run" if buttons & BTN_RUN else "walk"

    def _apply(self, x, y, mx, my, buttons):
        """One command, stepped exactly as the server will step it."""
        return movement.step(self.m, x, y, mx, my,
                             self._stance(buttons), TICK_DT)

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
        self.flashes.append({"x": e["x"], "y": e["y"], "t0": now, "mine": mine})
        ix, iy = e["impact"]
        if e.get("travel", 0.0) > 0.0:
            d = math.hypot(ix - e["x"], iy - e["y"])
            self.rockets.append({"x": e["x"], "y": e["y"], "ix": ix, "iy": iy,
                                 "t0": now, "dur": max(0.05, d / e["travel"])})
        elif e.get("blast", 0.0) > 0.0:
            self.blasts.append({"x": ix, "y": iy, "r": e["blast"], "t0": now})
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
        renderer.emit(self.m, self.cost, e["x"], e["y"],
                      e["energy"] * self.cpm, e["label"], self.sounds, now,
                      jobs=self.jobs, enemy=True)

    def _on_door(self, e, now):
        """A door moved (or somebody tried and could not). Applying it here
        keeps this client's sight, sound and bullet geometry in step with the
        server's, and with everyone else's."""
        if e.get("blocked"):
            if e["id"] == self.cli.world.my_id:
                self._say("stand clear to close the door")
            return
        key = (e["r"], e["c"])
        self.doors[key] = e["open"]
        set_door(self.m, self.cost, e["r"], e["c"], e["open"])
        compute_roof(self.m, self.doors)
        self.vis.invalidate()
        # fields solved against the old geometry are wrong now
        self.sounds.clear()
        self.jobs.clear()

    def _draw_doors(self):
        """Solid leaf shut, hollow green open — the same read as single-player.
        Drawn under the fog, so a door you cannot see is a door whose state you
        do not know."""
        ppm = self.ppm
        for (r, c), is_open in self.doors.items():
            rc = pygame.Rect(int(c * ppm), int(r * ppm), int(ppm), int(ppm))
            if is_open:
                pygame.draw.rect(self.surf, renderer.DOOR_FLOOR, rc)
                lw = max(3, int(ppm * 0.22))
                pygame.draw.rect(self.surf, renderer.DOOR_WOOD,
                                 (rc.x, rc.y, lw, rc.h))
                pygame.draw.rect(self.surf, renderer.DOOR_OPEN_EDGE, rc, 2)
            else:
                pygame.draw.rect(self.surf, renderer.DOOR_WOOD, rc)
                pygame.draw.rect(self.surf, renderer.DOOR_FRAME, rc, 2)
                pygame.draw.line(self.surf, renderer.DOOR_FRAME,
                                 rc.midtop, rc.midbottom, 1)

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
            self.vis.invalidate()

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
        pcx, pcy = self.m.cell_of(self.rx, self.ry)
        vf = self.vis.get(pcx, pcy, self.cone.max_range)
        inten = vf.cone_intensity(self.facing, self.cone)
        h_, w_ = inten.shape
        self.known[vf.y0:vf.y0 + h_, vf.x0:vf.x0 + w_] |= inten > 0.03
        return vf, inten

    def _sees_cell(self, vf, inten, cx, cy) -> bool:
        h_, w_ = inten.shape
        ry, rx = cy - vf.y0, cx - vf.x0
        if not (0 <= ry < h_ and 0 <= rx < w_):
            return False
        return bool(inten[ry, rx] > SEEN_MIN)

    def _draw_body(self, x, y, facing, colour, name, wep_id=None,
                   reloading=False, dead=False):
        """A soldier, with a ring in their lobby colour so you can tell who is
        who — the sprites are identical and tinting them to a player colour
        would wreck the art."""
        ppm = self.ppm
        cx, cy = x * ppm, y * ppm
        r = max(4, int(renderer.BODY_R * ppm))
        pygame.draw.circle(self.surf, colour, (int(cx), int(cy)), r + 4, 2)
        drew = False
        if self.bank.ok and not dead:
            if reloading:
                drew = self.bank.blit(self.surf, "soldier_idle", cx, cy, facing)
                self.bank.blit(self.surf, "weapon_sling", cx, cy, facing)
            else:
                drew = self.bank.blit(self.surf, "soldier_ready", cx, cy, facing)
                if wep_id:
                    self.bank.blit(self.surf, sprites.weapon_art(wep_id),
                                   cx, cy, facing)
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
        for f in self.flashes:
            if f["mine"] != mine:
                continue
            age = (now - f["t0"]) / MUZZLE_FLASH_T
            if age >= 1.0:
                continue
            rad = int((0.28 + 0.22 * age) * ppm)
            pygame.draw.circle(self.surf, (255, 236, 170),
                               (int(f["x"] * ppm), int(f["y"] * ppm)), rad)
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
            f = (now - rk["t0"]) / rk["dur"]
            if f >= 1.0:
                continue
            x = rk["x"] + (rk["ix"] - rk["x"]) * f
            y = rk["y"] + (rk["iy"] - rk["y"]) * f
            pygame.draw.circle(self.surf, (255, 200, 120),
                               (int(x * ppm), int(y * ppm)),
                               max(2, int(ROCKET_R * ppm)))

    def _reap_effects(self, now):
        self.tracers = [t for t in self.tracers
                        if now - t[1] < renderer.TRACER_FADE]
        self.flashes = [f for f in self.flashes
                        if now - f["t0"] < MUZZLE_FLASH_T]
        self.blasts = [b for b in self.blasts
                       if now - b["t0"] < renderer.BLAST_FADE]
        self.rockets = [r for r in self.rockets
                        if now - r["t0"] < r["dur"]]

    def _draw_fog(self, vf, inten):
        a_unknown = renderer.A_UNKNOWN / 255.0
        a_remember = renderer.A_REMEMBERED / 255.0
        ov_a = np.where(self.known, a_remember, a_unknown).astype(np.float32)
        h_, w_ = inten.shape
        sub = ov_a[vf.y0:vf.y0 + h_, vf.x0:vf.x0 + w_]
        lit = inten > 0.004
        sub[lit] = np.minimum(
            sub[lit], a_remember * (1.0 - renderer.CONE_REVEAL * inten[lit]))
        ov_a = renderer.box_blur(ov_a, renderer.FOG_BLUR_R)

        ppc = self.ppm / self.cpm                  # pixels per fine cell
        foh, fow = ov_a.shape
        mgn = renderer.FOG_BLUR_R + 2
        fx0 = max(0, int(self.cam_x / ppc) - mgn)
        fy0 = max(0, int(self.cam_y / ppc) - mgn)
        fx1 = min(fow, int((self.cam_x + self.view_w) / ppc) + mgn + 1)
        fy1 = min(foh, int((self.cam_y + self.view_h) / ppc) + mgn + 1)
        fw, fh = fx1 - fx0, fy1 - fy0
        if fw <= 0 or fh <= 0:
            return
        if self._ov.get_size() != (fw, fh):
            self._ov = pygame.Surface((fw, fh), pygame.SRCALPHA)
        px3 = pygame.surfarray.pixels3d(self._ov)
        pxa = pygame.surfarray.pixels_alpha(self._ov)
        px3[:, :, :] = 0
        pxa[:, :] = np.transpose(
            np.clip(ov_a[fy0:fy1, fx0:fx1] * 255.0, 0, 255)).astype(np.uint8)
        del px3, pxa
        self.surf.blit(pygame.transform.scale(
            self._ov, (round(fw * ppc), round(fh * ppc))),
            (round(fx0 * ppc), round(fy0 * ppc)))

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
        by = self.view_h - 78
        panel = pygame.Surface((300, 66), pygame.SRCALPHA)
        panel.fill((16, 18, 22, 210))
        self.window.blit(panel, (pad, by))
        if me:
            self._bar(self.window, pad + 10, by + 10, 150, 8, me.hp,
                      (190, 70, 70))
            self._bar(self.window, pad + 10, by + 22, 150, 6, me.sh,
                      (80, 150, 230))
            ammo = f"{me.mag}" if me.reload_t <= 0 else "reloading"
            if me.charge > 0.0:
                ammo = f"charge {min(1.0, me.charge / 3.0) * 100:.0f}%"
            self.window.blit(self.font.render(
                f"{wep.name}  [{mode}]", True, (222, 226, 232)),
                (pad + 10, by + 34))
            self.window.blit(self.font_mid.render(ammo, True, (230, 210, 140)),
                             (pad + 210, by + 18))
        near = self._near_door()
        if near is not None:
            hint = "f: close door" if self.doors[near] else "f: open door"
            txt = self.font_mid.render(hint, True, (230, 210, 140))
            self.window.blit(txt, txt.get_rect(
                center=(self.view_w // 2, self.view_h - 120)))
        if now - self.msg_t < 2.2:
            self.window.blit(self.font.render(self.msg, True, (150, 156, 166)),
                             (pad, by - 20))
        self.window.blit(self.font.render(
            f"({self.px:5.1f}, {self.py:5.1f})  correction "
            f"{self.corrected_m * 100:4.1f} cm    tab scores   esc quit",
            True, (90, 96, 106)), (pad, self.view_h - 20))

        mxp, myp = pygame.mouse.get_pos()
        pygame.draw.circle(self.window, (230, 230, 230), (mxp, myp), 3, 1)

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
            if t == "shot":
                self._on_shot(e, now)
            elif t == "sound":
                self._on_sound(e, now)
            elif t == "glass":
                self._on_glass(e, now)
            elif t == "door":
                self._on_door(e, now)

        mx, my, buttons = self.read_input()
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

        self._hear(now)
        self._reap_effects(now)

        self.cam_x = int(min(max(self.rx * self.ppm - self.view_w * 0.5, 0),
                             max(0, self.world_w - self.view_w)))
        self.cam_y = int(min(max(self.ry * self.ppm - self.view_h * 0.5, 0),
                             max(0, self.world_h - self.view_h)))

        self.surf.blit(self.base, (0, 0))
        self._draw_doors()
        vf, inten = self._visible_mask()

        # everyone else, and everything they did, goes under the fog
        self._draw_effects(now, mine=False)
        for p in w.players.values():
            if p.id == w.my_id or not p.alive:
                continue
            rx, ry, raim = p.render_pos(now, INTERP_DELAY)
            cx, cy = self.m.cell_of(rx, ry)
            if not self._sees_cell(vf, inten, cx, cy):
                continue
            self._draw_body(rx, ry, raim, p.colour, p.name,
                            wep_id=self.loadout[min(p.wep,
                                                    len(self.loadout) - 1)],
                            reloading=p.reload_t > 0.0)

        self._draw_fog(vf, inten)

        # you, and what you did, go over it
        if alive:
            self._draw_body(self.rx, self.ry, self.facing,
                            (me.colour if me else (220, 220, 220)), "",
                            wep_id=self.loadout[self.wep],
                            reloading=bool(me and me.reload_t > 0.0))
        self._draw_effects(now, mine=True)
        self._draw_cues(now)

        self.window.blit(self.surf, (-self.cam_x, -self.cam_y))
        if not alive and me:
            txt = self.font_big.render(
                f"respawning in {me.respawn_in:.0f}", True, (230, 180, 60))
            self.window.blit(txt, txt.get_rect(
                center=(self.view_w // 2, self.view_h // 2)))
        self._draw_hud(now)
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
            if why == "leave":
                # No mid-match spectate on the server yet, so leaving the match
                # means leaving the server.
                return 0
            # match_end: back to the lobby on the same connection
            pygame.display.set_mode((LOBBY_W, LOBBY_H))
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
