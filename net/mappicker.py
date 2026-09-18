"""
net/mappicker.py — map selection panel for the lobby (pygame).

Host-only control. Shows the maps found by mapcatalog as a scrollable list of
rows: thumbnail (or black if the .png is missing), map name, and total tile
size. Clicking a row sends set_config(map_id=...) so every client's lobby shows
the chosen map. Non-host clients get a read-only view (current selection
highlighted, no click).

Thumbnails are loaded once and cached, and scaled to the row's thumb box.

Usage inside the Lobby class:
    self.picker = MapPicker(client, maps_dir="maps", fonts=fonts,
                            rect=(x, y, w, h))
    # per frame:
    self.picker.draw(surf)
    # per event:
    self.picker.handle_event(ev)      # ignores clicks when not host

The picker reads client.world.map_id each frame to show the live selection, so
it stays correct even when another host (after migration) changes the map.
"""
from __future__ import annotations

import pygame

from .mapcatalog import load_catalog, MapEntry


# palette (kept local so the picker is self-contained; matches lobby.py)
PANEL = (28, 31, 37)
PANEL_LT = (38, 42, 50)
LINE = (52, 57, 66)
TEXT = (222, 226, 232)
TEXT_DIM = (140, 146, 156)
ACCENT = (90, 170, 250)
BLACK = (0, 0, 0)

ROW_H = 56
THUMB_W = 72
THUMB_H = 44


class MapPicker:
    def __init__(self, client, maps_dir, fonts, rect):
        self.cli = client
        self.maps_dir = maps_dir
        self.f, self.f_sm = fonts        # (normal, small)
        self.rect = pygame.Rect(rect)
        self.scroll = 0
        self._thumb_cache: dict[str, pygame.Surface] = {}
        self.entries: list[MapEntry] = []
        self.reload()

    def reload(self):
        """Rescan the maps directory. Call on lobby open or when maps change."""
        self.entries = load_catalog(self.maps_dir)
        self._thumb_cache.clear()

    # ---- thumbnails ----

    def _thumb(self, entry: MapEntry) -> pygame.Surface:
        if entry.map_id in self._thumb_cache:
            return self._thumb_cache[entry.map_id]
        surf = pygame.Surface((THUMB_W, THUMB_H))
        surf.fill(BLACK)
        if entry.thumb_path is not None:
            try:
                img = pygame.image.load(str(entry.thumb_path))
                try:
                    img = img.convert()          # fast blits when display exists
                except pygame.error:
                    pass                         # no display yet; use unconverted
                surf = pygame.transform.smoothscale(img, (THUMB_W, THUMB_H))
            except (pygame.error, OSError):
                surf = pygame.Surface((THUMB_W, THUMB_H))
                surf.fill(BLACK)                 # unreadable -> black
        self._thumb_cache[entry.map_id] = surf
        return surf

    # ---- geometry ----

    def _row_rects(self):
        """Yield (entry, row_rect) for visible rows, clipped to the panel."""
        inner = self.rect.inflate(-8, -8)
        top = inner.y + 26              # reserve header band
        y = top - self.scroll
        for e in self.entries:
            r = pygame.Rect(inner.x, y, inner.w, ROW_H)
            if r.bottom > top and r.top < inner.bottom:
                yield e, r
            y += ROW_H

    @property
    def _max_scroll(self):
        inner_h = self.rect.h - 8 - 26
        total = len(self.entries) * ROW_H
        return max(0, total - inner_h)

    # ---- events ----

    def handle_event(self, ev):
        is_host = self.cli.world.is_host
        if ev.type == pygame.MOUSEWHEEL and self.rect.collidepoint(
                pygame.mouse.get_pos()):
            self.scroll = max(0, min(self._max_scroll,
                                     self.scroll - ev.y * ROW_H))
        elif (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
              and is_host and self.rect.collidepoint(ev.pos)):
            for e, r in self._row_rects():
                if r.collidepoint(ev.pos):
                    self.cli.set_config(map_id=e.map_id)
                    break

    # ---- draw ----

    def draw(self, surf):
        pygame.draw.rect(surf, PANEL, self.rect, border_radius=8)
        pygame.draw.rect(surf, LINE, self.rect, width=1, border_radius=8)
        hdr = self.f_sm.render(
            "MAP" + ("" if self.cli.world.is_host else "  (host picks)"),
            True, TEXT_DIM)
        surf.blit(hdr, (self.rect.x + 12, self.rect.y + 8))

        if not self.entries:
            msg = self.f_sm.render(
                f"no maps found in {self.maps_dir}/", True, TEXT_DIM)
            surf.blit(msg, (self.rect.x + 12, self.rect.y + 34))
            return

        current = self.cli.world.map_id
        prev_clip = surf.get_clip()
        clip = self.rect.inflate(-4, -4)
        clip.y += 24            # don't draw rows over the header
        clip.h -= 24
        surf.set_clip(clip)
        for e, r in self._row_rects():
            selected = (e.map_id == current)
            if selected:
                pygame.draw.rect(surf, PANEL_LT, r, border_radius=6)
                pygame.draw.rect(surf, ACCENT, r, width=2, border_radius=6)
            # thumbnail
            th = self._thumb(e)
            tx, ty = r.x + 6, r.y + (ROW_H - THUMB_H) // 2
            surf.blit(th, (tx, ty))
            pygame.draw.rect(surf, LINE,
                             pygame.Rect(tx, ty, THUMB_W, THUMB_H), width=1)
            # name + size
            nx = tx + THUMB_W + 12
            surf.blit(self.f.render(e.name, True, TEXT), (nx, r.y + 8))
            surf.blit(self.f_sm.render(e.size_label, True, TEXT_DIM),
                      (nx, r.y + 32))
        surf.set_clip(prev_clip)

        # scroll hint
        if self._max_scroll > 0:
            hint = self.f_sm.render("scroll", True, TEXT_DIM)
            surf.blit(hint, (self.rect.right - hint.get_width() - 10,
                             self.rect.y + 8))
