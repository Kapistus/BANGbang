"""
net/mapcatalog.py — discover saved maps for the lobby map picker.

Scans a maps directory for both map formats — .map (the JSON editor format) and
the legacy char-grid .toml — pairs each with a same-named .png thumbnail
(map_1.map -> map_1.png), and reads a display name and size. Missing thumbnail
-> the lobby shows a black image.

Only files that are actually maps are listed. A .toml that declares no grid and
has no .grid sidecar is a definition file, not a map: tiles.toml is the one in
this project, and offering it as something to play on would hand the server a
map it cannot load.

This module does NOT depend on sim/tilemap.py. It reads dimensions defensively
because the exact TOML schema isn't fixed here: it tries, in order,
  1. explicit integer keys: width/height, cols/rows, w/h, size_x/size_y
  2. a [map] or [meta] subtable containing any of those
  3. a character-grid string (grid/map/layout/tiles/data) — counts lines and
     the longest line as rows x cols
If none are found, tile_count is None and the picker shows "?" for size.

If your maps store dimensions under a different key, add it to _DIM_KEYS or
_GRID_KEYS below — that's the only edit needed.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path


# integer dimension keys to probe, as (width_key, height_key) pairs
_DIM_KEYS = [
    ("width", "height"), ("cols", "rows"), ("w", "h"),
    ("size_x", "size_y"), ("columns", "rows"), ("tiles_x", "tiles_y"),
]
# subtables that might hold the dimensions
_SUBTABLES = ["map", "meta", "info", "header", "dimensions"]
# string keys that might hold a character-grid layout
_GRID_KEYS = ["grid", "map", "layout", "tiles", "data", "rows", "cells"]


@dataclass
class MapEntry:
    map_id: str                 # filename stem, e.g. "map_1"
    name: str                   # display name (from the file or prettified stem)
    toml_path: Path             # the map file itself (.map or .toml)
    thumb_path: Path | None     # None if the .png is missing
    cols: int | None
    rows: int | None

    @property
    def tile_count(self) -> int | None:
        if self.cols is None or self.rows is None:
            return None
        return self.cols * self.rows

    @property
    def size_label(self) -> str:
        if self.cols and self.rows:
            return f"{self.cols}x{self.rows} ({self.tile_count} tiles)"
        return "size ?"


def _extract_name(data: dict, stem: str) -> str:
    for k in ("name", "title", "display_name"):
        if isinstance(data.get(k), str) and data[k].strip():
            return data[k].strip()
    for sub in _SUBTABLES:
        t = data.get(sub)
        if isinstance(t, dict):
            for k in ("name", "title", "display_name"):
                if isinstance(t.get(k), str) and t[k].strip():
                    return t[k].strip()
    # prettify the stem: map_1 -> "Map 1"
    return stem.replace("_", " ").replace("-", " ").title()


def _dims_from_int_keys(scope: dict) -> tuple[int | None, int | None]:
    for wk, hk in _DIM_KEYS:
        w, h = scope.get(wk), scope.get(hk)
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            return w, h
    return None, None


def _dims_from_grid(scope: dict) -> tuple[int | None, int | None]:
    for gk in _GRID_KEYS:
        g = scope.get(gk)
        if isinstance(g, str) and "\n" in g:
            lines = [ln for ln in g.splitlines() if ln.strip("\r")]
            if lines:
                return max(len(ln) for ln in lines), len(lines)
        if isinstance(g, list) and g and all(isinstance(r, str) for r in g):
            return max(len(r) for r in g), len(g)
    return None, None


def _dims_from_grid_file(path: Path) -> tuple[int | None, int | None]:
    """Dimensions of a char-grid sidecar: longest line x number of lines."""
    try:
        lines = [ln.rstrip("\r\n") for ln in
                 path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return None, None
    if not lines:
        return None, None
    return max(len(ln) for ln in lines), len(lines)


def _read_dims(data: dict) -> tuple[int | None, int | None]:
    # top level int keys
    c, r = _dims_from_int_keys(data)
    if c:
        return c, r
    # subtables
    for sub in _SUBTABLES:
        t = data.get(sub)
        if isinstance(t, dict):
            c, r = _dims_from_int_keys(t)
            if c:
                return c, r
    # grid at top level
    c, r = _dims_from_grid(data)
    if c:
        return c, r
    # grid inside a subtable
    for sub in _SUBTABLES:
        t = data.get(sub)
        if isinstance(t, dict):
            c, r = _dims_from_grid(t)
            if c:
                return c, r
    return None, None


def _is_playable_toml(path: Path, data: dict) -> bool:
    """A .toml is a map if it carries a grid, names one, or has a .grid sidecar.
    Anything else (tiles.toml) is definitions."""
    if path.with_suffix(".grid").is_file():
        return True
    if any(k in data for k in ("grid", "layout", "map", "sidecar", "grid_file")):
        return True
    for sub_name in _SUBTABLES:
        t = data.get(sub_name)
        if isinstance(t, dict) and any(k in t for k in _GRID_KEYS):
            return True
    return False


def load_catalog(maps_dir: str | Path = "maps") -> list[MapEntry]:
    """Return all playable maps in maps_dir, .map first then .toml, each sorted
    by id. Unreadable or non-map files are skipped (logged to stdout) rather
    than crashing the lobby — or worse, being picked for a match."""
    d = Path(maps_dir)
    entries: list[MapEntry] = []
    if not d.is_dir():
        print(f"[mapcatalog] no maps dir at {d.resolve()}")
        return entries
    seen: set[str] = set()
    for path in sorted(d.glob("*.map")) + sorted(d.glob("*.toml")):
        stem = path.stem
        if stem in seen:
            continue                 # a .map and .toml of the same name: .map wins
        try:
            if path.suffix == ".map":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                size = data.get("size")
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    cols, rows = int(size[0]), int(size[1])
                else:
                    cols, rows = _read_dims(data)
            else:
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                if not _is_playable_toml(path, data):
                    continue         # definitions, not a map
                cols, rows = _read_dims(data)
                if cols is None:
                    # legacy maps keep the grid in a .grid sidecar, so the
                    # dimensions are in that file rather than the TOML
                    cols, rows = _dims_from_grid_file(path.with_suffix(".grid"))
        except (OSError, ValueError, tomllib.TOMLDecodeError) as e:
            print(f"[mapcatalog] skipping {path.name}: {e}")
            continue
        seen.add(stem)
        thumb = path.with_suffix(".png")
        entries.append(MapEntry(
            map_id=stem,
            name=_extract_name(data, stem),
            toml_path=path,
            thumb_path=thumb if thumb.is_file() else None,
            cols=cols, rows=rows,
        ))
    return entries


def find_map(maps_dir: str | Path, map_id: str) -> MapEntry | None:
    for e in load_catalog(maps_dir):
        if e.map_id == map_id:
            return e
    return None
