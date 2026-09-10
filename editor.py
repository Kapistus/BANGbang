"""BANGbang map editor.

    python editor.py [maps/foo.map]

Left panel: the tile palette (a scrolling list of PNG sprites, read from
assets/tiles/tileset.toml with is_see_through / is_walkable / blocks_los /
blocks_shots per entry). Pick a tile, then paint it into the grid in the
main view - it snaps to the cell. Floor tiles go on the base layer, objects
(walls, doors, windows, props) on the layer above; erasing an object shows
the floor again.

A grid cell is one metre. The player spawn and every guard route render the
actual soldier sprite plus a ring at the character's collision radius, both
at true in-game scale for the current zoom, so a room you draw here is sized
the way it will play - keep doorways and corridors at least two cells wide.

Mouse
    left           paint the selected tile (drag to paint a stroke)
    right          erase the object in that cell
    middle / space+drag   pan
    wheel          zoom (over the grid) / scroll the palette (over the panel)

Keys
    s  save        S  save as        l  load        n  new map
    g  grid on/off      p  paint mode      f  roof preview on/off
    w  toggle placing tiles as Floor / Wall (walls are solid + drawn darker)
    Tab / Shift+Tab  next / prev tileset page in the palette
    o  set player spawn (then click a cell)
    k  add guard: click waypoints, Enter to finish, Backspace undo point
    L  light mode: click to place a lamp; wheel = intensity, shift+wheel = radius
       (or r / i keys, +shift to lower); RMB/Del remove
    [ ]  cycle the guard's weapon - the route under the cursor, or the
         default for the next route (combat rifle by default)
    c  clear all guards        del  erase hovered object
    arrows  pan        esc  quit
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pygame

from sim import mapfile, sprites, weapons
from sim.tilemap import FLOOR_DARKEN, WALL_LIGHTEN, enclosed_mask
from sim.tileset import load_tileset

ROOT = Path(__file__).resolve().parent
MAPS_DIR = ROOT / "maps"
GUARD_WEAPONS = ["combat_rifle", "smg", "combat_shotgun", "pistol",
                 "rail_rifle", "laser_rifle", "rocket_launcher"]
GUARD_SKILLS = ["veteran", "seasoned", "rookie"]
PANEL_W = 240
STATUS_H = 26
TILE_ICON = 96          # palette preview size (was 24)
BG = (26, 26, 24)
PANEL_BG = (20, 20, 19)
TEXT = (222, 220, 212)
DIM = (140, 138, 132)
SEL = (239, 159, 39)
GRIDLN = (60, 60, 58)
SPAWN_C = (29, 158, 117)
GUARD_C = (216, 90, 48)
LIGHT_C = (240, 210, 120)
DEF_LIGHT_RADIUS = 5.0
DEF_LIGHT_INTENSITY = 1.0

MIN_ZOOM, MAX_ZOOM = 8, 48
DEF_ZOOM = 18             # px per cell (a cell is mapfile cell_m metres)
BODY_R_M = 0.28          # character collision radius, metres
NEW_COLS, NEW_ROWS = 80, 52   # default blank-map size, in cells


def load_sprites(ts):
    """tile id -> native Surface. Generates the placeholder PNGs on first run
    if any are missing; falls back to a flat colour square if a file is bad."""
    out, missing = {}, False
    for td in ts.tiles.values():
        p = ts.path.parent / td.png
        if not p.exists():
            missing = True
    if missing:
        try:
            from tools import gen_sprites
            gen_sprites.main(str(ts.path))
        except Exception as exc:            # pragma: no cover
            print("sprite generation failed:", exc)
    for td in ts.tiles.values():
        p = ts.path.parent / td.png
        try:
            out[td.id] = pygame.image.load(str(p)).convert_alpha()
        except Exception:
            s = pygame.Surface((ts.tile_px, ts.tile_px))
            s.fill(tuple(td.colour))
            pygame.draw.rect(s, (0, 0, 0), s.get_rect(), 1)
            out[td.id] = s
    return out


def text_prompt(screen, font, label, default=""):
    """Blocking one-line input. Returns the string, or None on escape."""
    buf = default
    clock = pygame.time.Clock()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    return buf
                if ev.key == pygame.K_ESCAPE:
                    return None
                if ev.key == pygame.K_BACKSPACE:
                    buf = buf[:-1]
                elif ev.unicode and ev.unicode.isprintable():
                    buf += ev.unicode
        w, h = screen.get_size()
        box = pygame.Surface((w, 70), pygame.SRCALPHA)
        box.fill((0, 0, 0, 210))
        screen.blit(box, (0, h // 2 - 35))
        screen.blit(font.render(label, True, SEL), (20, h // 2 - 26))
        screen.blit(font.render(buf + "_", True, TEXT), (20, h // 2 - 4))
        pygame.display.flip()
        clock.tick(60)


class Editor:
    def __init__(self, path=None):
        pygame.init()
        pygame.display.set_caption("BANGbang map editor")
        self.screen = pygame.display.set_mode((1280, 800), pygame.RESIZABLE)
        self.font = pygame.font.SysFont("consolas,monospace", 13)
        self.small = pygame.font.SysFont("consolas,monospace", 11)
        self.clock = pygame.time.Clock()
        self.ts = load_tileset()
        self.native = load_sprites(self.ts)
        self.bank = sprites.SpriteBank()
        self.bank.load()
        self.zoom = DEF_ZOOM
        self.disp = {}
        self._rescale()

        self.path = None
        self.dirty = False
        if path and Path(path).exists():
            self.doc = mapfile.load_doc(path)
            self.path = Path(path)
        else:
            self.doc = mapfile.new_map(NEW_COLS, NEW_ROWS, self.ts, "untitled")
        self.cols, self.rows = self.doc["size"]
        self._rescale()                       # pick up the doc's cell_m

        self.cam_x = PANEL_W + 20
        self.cam_y = 20
        self.sel = self.ts.default_floor
        self.place_role = "floor"           # floor | wall  (w toggles; specials ignore)
        self.groups = self._tile_groups()   # palette pages: "base" + each group
        self.tile_group = self.groups[0]
        self.mode = "paint"                 # paint | spawn | guard
        self.guard_wip = []
        self.guard_weapon = "combat_rifle"  # gun for the next route ([ ] / , . cycle)
        self.guard_skill = "veteran"        # aim tier for the next route (t cycles)
        self.light_radius = DEF_LIGHT_RADIUS      # next placed light ([ ] adjust)
        self.light_intensity = DEF_LIGHT_INTENSITY  # next placed light (, . adjust)
        self.toast = ""
        self.toast_t = 0
        self.show_grid = True
        self.show_roof = True               # f : auto-roof (enclosed area) preview
        self.panel_scroll = 0
        self.rows_hit = []                  # (rect, kind, value) filled per frame
        self.painting = self.erasing = self.panning = False
        self.pan_from = (0, 0)

    # -- geometry ----------------------------------------------------

    def _generic_tds(self):
        """Non-special TileDefs (art tiles you paint as Floor or Wall)."""
        return [td for td in self.ts.tiles.values() if not self._is_special(td)]

    def _special_tds(self):
        return [td for td in self.ts.tiles.values() if self._is_special(td)]

    def _tile_groups(self):
        """Palette pages: 'base' (ungrouped generic tiles) then each named
        group, sorted."""
        tds = self._generic_tds()
        pages = ["base"] if any(not td.group for td in tds) else []
        pages += sorted({td.group for td in tds if td.group})
        return pages or ["base"]

    def _group_tiles(self):
        """Generic TileDefs on the active palette page."""
        return [td for td in self._generic_tds()
                if (td.group or "base") == self.tile_group]

    def cycle_group(self, step):
        if len(self.groups) < 2:
            return
        i = (self.groups.index(self.tile_group) + step) % len(self.groups)
        self.tile_group = self.groups[i]
        self.panel_scroll = 0
        page = self._group_tiles()
        if page and self.sel not in {td.id for td in page}:
            self.sel, self.mode = page[0].id, "paint"
        self._flash(f"tileset: {self.tile_group}")

    def _roof_mask(self):
        """Coarse bool of cells the game will auto-roof: sealed off from the
        map border by a closed loop of enclosing object tiles (walls, window
        frames, doors). Recomputed live so painting a wall shows its roof."""
        ts = self.ts
        obj = self.doc["object"]
        enc = np.zeros((self.rows, self.cols), dtype=bool)
        for r in range(self.rows):
            row = obj[r]
            for c in range(self.cols):
                o = row[c]
                # any object cell seals a room except a bush (walkable foliage)
                if o and o in ts and not ts[o].bush:
                    enc[r, c] = True
        return enclosed_mask(enc)

    def _cell_m(self):
        return float(getattr(self, "doc", {}).get("cell_m", 1.0)) if hasattr(
            self, "doc") else 1.0

    def _rescale(self):
        z = self.zoom
        self.disp = {k: pygame.transform.smoothscale(v, (z, z))
                     for k, v in self.native.items()}
        # role-tinted copies (mirror sim/tilemap): floor darker, wall lighter
        dk = int(round(255 * FLOOR_DARKEN))
        lt = int(round(255 * WALL_LIGHTEN))
        self.disp_floor, self.disp_wall = {}, {}
        for k, s in self.disp.items():
            f = s.copy()
            f.fill((dk, dk, dk, 255), special_flags=pygame.BLEND_RGB_MULT)
            self.disp_floor[k] = f
            wv = s.copy()
            wv.fill((lt, lt, lt, 0), special_flags=pygame.BLEND_RGB_ADD)
            self.disp_wall[k] = wv
        # zoom is px per cell; a cell is cell_m metres, so px-per-metre is
        # zoom / cell_m. Characters then preview at true in-game size.
        self.bank.set_scale(z / self._cell_m() / sprites.ART_PPM)

    def cell_at(self, mx, my):
        c = int((mx - self.cam_x) / self.zoom)
        r = int((my - self.cam_y) / self.zoom)
        return c, r

    def cell_rect(self, c, r):
        return pygame.Rect(self.cam_x + c * self.zoom,
                           self.cam_y + r * self.zoom, self.zoom, self.zoom)

    def in_grid(self, c, r):
        return 0 <= c < self.cols and 0 <= r < self.rows

    # -- editing ---------------------------------------------------

    def apply(self, mx, my, erase=False):
        c, r = self.cell_at(mx, my)
        if not self.in_grid(c, r) or mx < PANEL_W:
            return
        if self.mode == "spawn":
            self.doc["player_spawn"] = [c + 0.5, r + 0.5]
            self.dirty = True
            return
        if self.mode == "guard":
            self.guard_wip.append([c + 0.5, r + 0.5])
            return
        if self.mode == "light":
            self.doc.setdefault("lights", []).append(
                {"pos": [c + 0.5, r + 0.5], "radius": self.light_radius,
                 "intensity": self.light_intensity})
            self.dirty = True
            return
        if erase:
            if self.doc["object"][r][c]:
                self.doc["object"][r][c] = ""
                self.dirty = True
            return
        td = self.ts[self.sel]
        # door / window / bush always land on the object layer; every other
        # tile follows the Floor / Wall toggle
        as_wall = self._is_special(td) or self.place_role == "wall"
        grid = self.doc["object"] if as_wall else self.doc["floor"]
        if grid[r][c] != self.sel:
            grid[r][c] = self.sel
            self.dirty = True

    @staticmethod
    def _is_special(td):
        return bool(td.door or td.glass or td.bush)

    def commit_guard(self):
        if len(self.guard_wip) >= 1:
            gid = f"g{len(self.doc['guards']) + 1}"
            self.doc["guards"].append({"id": gid, "patrol": self.guard_wip,
                                       "weapon": self.guard_weapon,
                                       "skill": self.guard_skill})
            self.dirty = True
        self.guard_wip = []

    def guard_near(self, mx, my):
        c, r = self.cell_at(mx, my)
        cx, cy = c + 0.5, r + 0.5
        for g in self.doc["guards"]:
            if any(abs(px - cx) < 0.9 and abs(py - cy) < 0.9
                   for px, py in g["patrol"]):
                return g
        return None

    def delete_guard_near(self, mx, my):
        g = self.guard_near(mx, my)
        if g is not None:
            self.doc["guards"].remove(g)
            self.dirty = True
            return True
        return False

    def light_near(self, mx, my):
        c, r = self.cell_at(mx, my)
        cx, cy = c + 0.5, r + 0.5
        best, bd = None, 1.2
        for lt in self.doc.get("lights", []):
            d = math.hypot(lt["pos"][0] - cx, lt["pos"][1] - cy)
            if d < bd:
                best, bd = lt, d
        return best

    def delete_light_near(self, mx, my):
        lt = self.light_near(mx, my)
        if lt is not None:
            self.doc["lights"].remove(lt)
            self.dirty = True
            return True
        return False

    def tune_light(self, mx, my, dr=0.0, di=0.0):
        """[ ] adjust radius, , . adjust intensity - of the light under the
        cursor, else the next-placed default."""
        lt = self.light_near(mx, my)
        if lt is not None:
            lt["radius"] = round(min(40.0, max(1.0, lt["radius"] + dr)), 1)
            lt["intensity"] = round(min(20.0, max(0.1, lt["intensity"] + di)), 2)
            self.dirty = True
            self._flash(f"light  r{lt['radius']}  i{lt['intensity']}")
        else:
            self.light_radius = round(min(40.0, max(1.0, self.light_radius + dr)), 1)
            self.light_intensity = round(
                min(20.0, max(0.1, self.light_intensity + di)), 2)
            self._flash(f"next light  r{self.light_radius}  i{self.light_intensity}")

    def _flash(self, text):
        self.toast = text
        self.toast_t = pygame.time.get_ticks()

    def cycle_guard_weapon(self, mx, my, step):
        """[ ] or , . : retag the route whose waypoint is under the cursor,
        otherwise cycle the gun the NEXT route will be created with."""
        g = self.guard_near(mx, my) if self.mode == "guard" else None
        if g is not None:
            cur = g.get("weapon", "combat_rifle")
            i = GUARD_WEAPONS.index(cur) if cur in GUARD_WEAPONS else 0
            g["weapon"] = GUARD_WEAPONS[(i + step) % len(GUARD_WEAPONS)]
            self.dirty = True
            self._flash(f"{g.get('id', 'route')}  weapon -> {g['weapon']}")
        else:
            i = GUARD_WEAPONS.index(self.guard_weapon)
            self.guard_weapon = GUARD_WEAPONS[(i + step) % len(GUARD_WEAPONS)]
            hint = "" if self.mode == "guard" else "   (k = guard mode)"
            self._flash(f"next route weapon -> {self.guard_weapon}{hint}")

    def cycle_guard_skill(self, mx, my, step):
        """t : retag the route under the cursor's aim tier, else the next one."""
        g = self.guard_near(mx, my) if self.mode == "guard" else None
        if g is not None:
            cur = g.get("skill", "veteran")
            i = GUARD_SKILLS.index(cur) if cur in GUARD_SKILLS else 0
            g["skill"] = GUARD_SKILLS[(i + step) % len(GUARD_SKILLS)]
            self.dirty = True
            self._flash(f"{g.get('id', 'route')}  skill -> {g['skill']}")
        else:
            i = GUARD_SKILLS.index(self.guard_skill)
            self.guard_skill = GUARD_SKILLS[(i + step) % len(GUARD_SKILLS)]
            hint = "" if self.mode == "guard" else "   (k = guard mode)"
            self._flash(f"next route skill -> {self.guard_skill}{hint}")

    def do_new(self):
        s = text_prompt(self.screen, self.font, "new map  cols,rows :",
                        f"{NEW_COLS},{NEW_ROWS}")
        if not s:
            return
        try:
            cols, rows = (int(x) for x in s.replace(" ", "").split(","))
        except ValueError:
            return
        cols, rows = max(4, min(cols, 400)), max(4, min(rows, 400))
        self.doc = mapfile.new_map(cols, rows, self.ts, "untitled")
        self.cols, self.rows = cols, rows
        self.path, self.dirty = None, False
        self._rescale()

    def do_save(self, as_new=False):
        if self.mode == "guard" and self.guard_wip:
            self.commit_guard()          # don't drop a route in progress
        if self.path is None or as_new:
            d = self.doc.get("name", "untitled")
            s = text_prompt(self.screen, self.font, "save as (maps/....map):",
                            f"maps/{d}.map")
            if not s:
                return
            self.path = (ROOT / s) if not Path(s).is_absolute() else Path(s)
            if self.path.suffix != ".map":
                self.path = self.path.with_suffix(".map")
        self.doc["name"] = self.path.stem
        mapfile.save(self.doc, self.path)
        self.dirty = False

    def do_load(self):
        s = text_prompt(self.screen, self.font, "load (maps/....map):", "maps/")
        if not s:
            return
        p = (ROOT / s) if not Path(s).is_absolute() else Path(s)
        try:
            self.doc = mapfile.load_doc(p)
        except Exception as exc:
            print("load failed:", exc)
            return
        self.cols, self.rows = self.doc["size"]
        self.path, self.dirty = p, False
        self._rescale()

    # -- palette panel -------------------------------------------

    def draw_panel(self):
        w, h = self.screen.get_size()
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, PANEL_W, h))
        self.rows_hit = []
        y = 8 - self.panel_scroll

        def row(label, kind, value, sprite=None, active=False):
            nonlocal y
            rh = (TILE_ICON + 8) if sprite is not None else 30
            rect = pygame.Rect(6, y, PANEL_W - 12, rh)
            if y + rh > 0 and y < h:
                if active:
                    pygame.draw.rect(self.screen, (48, 42, 20), rect)
                    pygame.draw.rect(self.screen, SEL, rect, 1)
                col = TEXT if active else DIM
                if sprite is not None:
                    self.screen.blit(
                        pygame.transform.smoothscale(sprite, (TILE_ICON, TILE_ICON)),
                        (10, y + 4))
                    self.screen.blit(self.font.render(label, True, col),
                                     (TILE_ICON + 18, y + rh // 2 - 7))
                else:
                    self.screen.blit(self.font.render(label, True, col),
                                     (12, y + 8))
            self.rows_hit.append((rect, kind, value))
            y += rh + 2

        row("[ set player spawn ]  o", "mode", "spawn", active=self.mode == "spawn")
        row("[ guard route ]  k", "mode", "guard", active=self.mode == "guard")
        row("[ light ]  L", "mode", "light", active=self.mode == "light")
        row("[ paint / erase ]  p", "mode", "erase", active=self.mode == "paint")
        y += 4

        # Floor / Wall placement toggle  (w)
        box = "[x]" if self.place_role == "wall" else "[ ]"
        row(f"{box} place as WALL   w", "role", None,
            active=self.place_role == "wall")
        y += 6

        # tileset page switcher: < name (i/N) >  (Tab / click the arrows)
        if len(self.groups) > 1:
            gr = pygame.Rect(6, y, PANEL_W - 12, 24)
            half = gr.width // 2
            if 0 <= y <= h:
                pygame.draw.rect(self.screen, (40, 40, 38), gr)
                pygame.draw.rect(self.screen, SEL, gr, 1)
                i = self.groups.index(self.tile_group) + 1
                cap = f"{self.tile_group}  {i}/{len(self.groups)}"
                ct = self.small.render(cap, True, TEXT)
                self.screen.blit(ct, (gr.centerx - ct.get_width() // 2, y + 6))
                self.screen.blit(self.font.render("<", True, SEL), (gr.x + 7, y + 4))
                self.screen.blit(self.font.render(">", True, SEL),
                                 (gr.right - 15, y + 4))
            self.rows_hit.append((pygame.Rect(gr.x, y, half, 24), "group", -1))
            self.rows_hit.append((pygame.Rect(gr.x + half, y, half, 24), "group", 1))
            y += 30

        for head, tds in (("TILES", self._group_tiles()),
                          ("SPECIAL", self._special_tds())):
            if not tds:
                continue
            if 0 <= y <= h:
                self.screen.blit(self.small.render(head, True, SEL), (10, y))
            y += 16
            for td in tds:
                row(td.name, "tile", td.id, self.native[td.id],
                    active=self.mode == "paint" and self.sel == td.id)

    def panel_click(self, mx, my):
        for rect, kind, value in self.rows_hit:
            if rect.collidepoint(mx, my):
                if kind == "tile":
                    self.sel, self.mode = value, "paint"
                elif kind == "mode":
                    self.mode = value if value != "erase" else "paint"
                elif kind == "group":
                    self.cycle_group(value)
                elif kind == "role":
                    self.place_role = ("wall" if self.place_role == "floor"
                                       else "floor")
                return

    # -- main view -----------------------------------------------

    def draw_grid(self):
        w, h = self.screen.get_size()
        view = pygame.Rect(PANEL_W, 0, w - PANEL_W, h - STATUS_H)
        self.screen.set_clip(view)
        c0 = max(0, int((PANEL_W - self.cam_x) / self.zoom))
        r0 = max(0, int((0 - self.cam_y) / self.zoom))
        c1 = min(self.cols, int((w - self.cam_x) / self.zoom) + 1)
        r1 = min(self.rows, int((h - self.cam_y) / self.zoom) + 1)
        canopy = []            # overlay tiles - drawn after the characters
        for r in range(r0, r1):
            for c in range(c0, c1):
                rect = self.cell_rect(c, r)
                fid = self.doc["floor"][r][c]
                self.screen.blit(self.disp_floor[fid if fid in self.disp_floor
                                                 else self.ts.default_floor], rect)
                obj = self.doc["object"][r][c]
                if obj and obj in self.disp:
                    otd = self.ts[obj] if obj in self.ts else None
                    if otd is not None and otd.overlay:
                        canopy.append((obj, rect))
                    elif otd is not None and self._is_special(otd):
                        self.screen.blit(self.disp[obj], rect)      # door/window
                    else:
                        self.screen.blit(self.disp_wall[obj], rect)  # wall = lighter
        if self.show_grid and self.zoom >= 10:
            for c in range(c0, c1 + 1):
                x = self.cam_x + c * self.zoom
                pygame.draw.line(self.screen, GRIDLN, (x, self.cam_y),
                                 (x, self.cam_y + self.rows * self.zoom))
            for r in range(r0, r1 + 1):
                yv = self.cam_y + r * self.zoom
                pygame.draw.line(self.screen, GRIDLN, (self.cam_x, yv),
                                 (self.cam_x + self.cols * self.zoom, yv))
        # map border
        pygame.draw.rect(self.screen, (90, 90, 86),
                         (self.cam_x, self.cam_y,
                          self.cols * self.zoom, self.rows * self.zoom), 1)
        self._draw_entities()
        for obj, rect in canopy:
            self.screen.blit(self.disp[obj], rect)
        if self.show_roof:
            # live auto-roof preview: translucent wash over every enclosed cell
            mask = self._roof_mask()
            wash = pygame.Surface((self.zoom, self.zoom), pygame.SRCALPHA)
            wash.fill((0, 0, 0, 205))
            for r in range(r0, r1):
                mr = mask[r]
                for c in range(c0, c1):
                    if mr[c]:
                        self.screen.blit(wash, self.cell_rect(c, r))
        # hover
        mx, my = pygame.mouse.get_pos()
        if mx >= PANEL_W:
            c, r = self.cell_at(mx, my)
            if self.in_grid(c, r):
                pygame.draw.rect(self.screen, SEL, self.cell_rect(c, r), 1)
        self.screen.set_clip(None)

    def _wpt(self, p):
        return (int(self.cam_x + p[0] * self.zoom),
                int(self.cam_y + p[1] * self.zoom))

    def _facing(self, a, b):
        """Screen-space heading from metre point a to metre point b."""
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def _draw_body(self, cx, cy, col):
        """The character's real collision footprint at the current zoom."""
        r = max(3, int(BODY_R_M / self._cell_m() * self.zoom))
        pygame.draw.circle(self.screen, col, (cx, cy), r, 1)

    def _draw_entities(self):
        spawn = self.doc["player_spawn"]
        sx, sy = self._wpt(spawn)
        if not self.bank.blit(self.screen, "soldier_ready", sx, sy, 0.0):
            pygame.draw.circle(self.screen, SPAWN_C, (sx, sy), 6)
        self._draw_body(sx, sy, SPAWN_C)
        pygame.draw.circle(self.screen, SPAWN_C, (sx, sy), 3)
        for g in self.doc["guards"]:
            route = g["patrol"]
            pts = [self._wpt(p) for p in route]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, GUARD_C, True, pts, 2)
            for i, p in enumerate(pts):
                pygame.draw.circle(self.screen, GUARD_C, p, 4)
            if route:
                face = self._facing(route[0], route[1]) if len(route) > 1 else 0.0
                wid = g.get("weapon", "combat_rifle")
                if self.bank.blit(self.screen, "soldier_ready", pts[0][0],
                                  pts[0][1], face):
                    self.bank.blit(self.screen, sprites.weapon_art(wid),
                                   pts[0][0], pts[0][1], face)
                self._draw_body(pts[0][0], pts[0][1], GUARD_C)
                lbl = f"{g.get('id', 'g')}  {wid}  {g.get('skill', 'veteran')}"
                self.screen.blit(self.small.render(lbl, True, TEXT),
                                 (pts[0][0] + 6, pts[0][1] - 6))
        if self.guard_wip:
            pts = [self._wpt(p) for p in self.guard_wip]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, SEL, False, pts, 2)
            for i, p in enumerate(pts):
                pygame.draw.circle(self.screen, SEL, p, 5)
                self.screen.blit(self.small.render(str(i + 1), True, (0, 0, 0)),
                                 (p[0] - 3, p[1] - 6))

        for lt in self.doc.get("lights", []):
            lx, ly = self._wpt(lt["pos"])
            rr = int(lt["radius"] / self._cell_m() * self.zoom)
            if rr > 2:
                halo = pygame.Surface((rr * 2, rr * 2), pygame.SRCALPHA)
                a = int(24 + 46 * min(1.0, lt["intensity"]))
                pygame.draw.circle(halo, (*LIGHT_C, a), (rr, rr), rr)
                self.screen.blit(halo, (lx - rr, ly - rr))
                pygame.draw.circle(self.screen, LIGHT_C, (lx, ly), rr, 1)
            pygame.draw.circle(self.screen, LIGHT_C, (lx, ly), 4)
            pygame.draw.circle(self.screen, (20, 20, 20), (lx, ly), 4, 1)

    def draw_status(self):
        w, h = self.screen.get_size()
        pygame.draw.rect(self.screen, PANEL_BG, (0, h - STATUS_H, w, STATUS_H))
        mx, my = pygame.mouse.get_pos()
        c, r = self.cell_at(mx, my)
        cell = f"{c},{r}" if self.in_grid(c, r) else "--"
        name = (self.path.name if self.path else "untitled") + ("*" if self.dirty else "")
        cur = {"paint": f"tile:{self.sel} [{self.tile_group}] as {self.place_role.upper()}",
               "spawn": "SET SPAWN (click a cell)",
               "guard": (f"GUARD  next [{self.guard_weapon} / {self.guard_skill}]  "
                         f"LMB=waypoint  Enter=save  [ ],. =weapon  t=skill  "
                         f"(hover a waypoint to retag)  RMB/Del=delete  "
                         f"wpts:{len(self.guard_wip)}"),
               "light": (f"LIGHT  next r{self.light_radius} i{self.light_intensity}  "
                         f"LMB=place  wheel=intensity  shift+wheel=radius  "
                         f"(r/i keys, +shift to lower)  RMB/Del=remove  "
                         f"({len(self.doc.get('lights', []))} placed)")
               }[self.mode]
        msg = (f"cell {cell:>7}   {cur}   |   {name}  {self.cols}x{self.rows} "
               f"z{self.zoom}   s save  l load  n new  o spawn  k guard  g grid"
               f"  f roof{'' if self.show_roof else ':off'}")
        self.screen.blit(self.font.render(msg, True, DIM), (8, h - STATUS_H + 6))
        if self.toast and pygame.time.get_ticks() - self.toast_t < 2000:
            t = self.small.render(self.toast, True, SEL)
            self.screen.blit(t, (w - t.get_width() - 10, h - STATUS_H + 7))

    # -- loop ----------------------------------------------------

    def run(self):
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(
                        (ev.w, ev.h), pygame.RESIZABLE)
                elif ev.type == pygame.KEYDOWN:
                    running = self.on_key(ev) and running
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    self.on_mousedown(ev)
                elif ev.type == pygame.MOUSEBUTTONUP:
                    if ev.button == 1:
                        self.painting = False
                    if ev.button == 3:
                        self.erasing = False
                    if ev.button == 2:
                        self.panning = False
                elif ev.type == pygame.MOUSEMOTION:
                    self.on_motion(ev)
                elif ev.type == pygame.MOUSEWHEEL:
                    self.on_wheel(ev)

            keys = pygame.key.get_pressed()
            pan = 8
            if keys[pygame.K_LEFT]:
                self.cam_x += pan
            if keys[pygame.K_RIGHT]:
                self.cam_x -= pan
            if keys[pygame.K_UP]:
                self.cam_y += pan
            if keys[pygame.K_DOWN]:
                self.cam_y -= pan

            self.screen.fill(BG)
            self.draw_grid()
            self.draw_panel()
            self.draw_status()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

    def on_key(self, ev):
        k = ev.key
        shift = ev.mod & pygame.KMOD_SHIFT
        if k == pygame.K_ESCAPE:
            if self.mode == "guard" and self.guard_wip:
                self.guard_wip = []
                return True
            return False
        if k == pygame.K_s:
            self.do_save(as_new=bool(shift))
        elif k == pygame.K_l:
            if shift:
                if self.mode == "guard":
                    self.commit_guard()
                self.mode = "light"
            else:
                self.do_load()
        elif k == pygame.K_n:
            self.do_new()
        elif k == pygame.K_g:
            self.show_grid = not self.show_grid
        elif k == pygame.K_TAB:
            self.cycle_group(-1 if shift else 1)
        elif k == pygame.K_w:
            self.place_role = "wall" if self.place_role == "floor" else "floor"
            self._flash(f"placing as {self.place_role}")
        elif k == pygame.K_f:
            self.show_roof = not self.show_roof
            self._flash(f"auto-roof preview {'on' if self.show_roof else 'off'}")
        elif k == pygame.K_p:
            if self.mode == "guard":
                self.commit_guard()          # don't lose a route in progress
            self.mode = "paint"
        elif k == pygame.K_o:
            if self.mode == "guard":
                self.commit_guard()
            self.mode = "spawn"
        elif k == pygame.K_k:
            if self.mode == "guard":
                self.commit_guard()          # k again = finish this, start next
            self.mode, self.guard_wip = "guard", []
        elif k == pygame.K_c and self.mode != "guard":
            self.doc["guards"] = []
            self.dirty = True
        elif k == pygame.K_r and self.mode == "light":
            mx, my = pygame.mouse.get_pos()
            self.tune_light(mx, my, dr=-1.0 if shift else 1.0)
        elif k == pygame.K_i and self.mode == "light":
            mx, my = pygame.mouse.get_pos()
            self.tune_light(mx, my, di=-0.25 if shift else 0.25)
        elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET,
                   pygame.K_COMMA, pygame.K_PERIOD):
            mx, my = pygame.mouse.get_pos()
            if self.mode == "light":
                self.tune_light(
                    mx, my,
                    dr={pygame.K_LEFTBRACKET: -0.5, pygame.K_RIGHTBRACKET: 0.5}
                    .get(k, 0.0),
                    di={pygame.K_COMMA: -0.1, pygame.K_PERIOD: 0.1}.get(k, 0.0))
            else:
                back = k in (pygame.K_LEFTBRACKET, pygame.K_COMMA)
                self.cycle_guard_weapon(mx, my, -1 if back else 1)
        elif k == pygame.K_t:
            mx, my = pygame.mouse.get_pos()
            self.cycle_guard_skill(mx, my, -1 if shift else 1)
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.mode == "guard":
            self.commit_guard()
        elif k == pygame.K_BACKSPACE and self.mode == "guard" and self.guard_wip:
            self.guard_wip.pop()
        elif k in (pygame.K_DELETE, pygame.K_BACKSPACE):
            mx, my = pygame.mouse.get_pos()
            if not self.delete_guard_near(mx, my) and not self.delete_light_near(mx, my):
                self.apply(mx, my, erase=True)
        return True

    def on_mousedown(self, ev):
        mx, my = ev.pos
        if mx < PANEL_W:
            if ev.button == 1:
                self.panel_click(mx, my)
            return
        if ev.button == 1:
            self.painting = True
            self.apply(mx, my)
        elif ev.button == 3:
            if self.mode == "guard":
                self.delete_guard_near(mx, my)   # RMB a route to remove it
            elif self.mode == "light":
                self.delete_light_near(mx, my)
            else:
                self.erasing = True
                self.apply(mx, my, erase=True)
        elif ev.button == 2:
            self.panning = True
            self.pan_from = ev.pos

    def on_motion(self, ev):
        mx, my = ev.pos
        if self.panning or (ev.buttons[0] and pygame.key.get_pressed()[pygame.K_SPACE]):
            dx, dy = ev.rel
            self.cam_x += dx
            self.cam_y += dy
            return
        if self.painting and mx >= PANEL_W and self.mode == "paint":
            self.apply(mx, my)
        elif self.erasing and mx >= PANEL_W:
            self.apply(mx, my, erase=True)

    def on_wheel(self, ev):
        mx, my = pygame.mouse.get_pos()
        if mx < PANEL_W:
            self.panel_scroll = max(0, self.panel_scroll - ev.y * 40)
            return
        if self.mode == "light":
            keys = pygame.key.get_pressed()
            shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            if shift:
                self.tune_light(mx, my, dr=ev.y * 1.0)
            else:
                self.tune_light(mx, my, di=ev.y * 0.25)
            return
        old = self.zoom
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom + ev.y * 2))
        if self.zoom != old:
            # keep the cell under the cursor put
            fx = (mx - self.cam_x) / old
            fy = (my - self.cam_y) / old
            self.cam_x = mx - fx * self.zoom
            self.cam_y = my - fy * self.zoom
            self._rescale()


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    Editor(arg).run()


if __name__ == "__main__":
    main()
