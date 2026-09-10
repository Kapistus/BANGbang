"""The .map file: a JSON map format for the editor and the game.

Structure (bangbang-map/1):

    {
      "format": "bangbang-map/1",
      "name": "Compound",
      "size": [cols, rows],          # in authoring cells (metres by default)
      "cell_m": 1.0,                 # metres per authoring cell
      "subdiv": 8,                   # sim cells per authoring cell
      "floor":  [["floor", ...], ...],   # rows x cols, every cell filled
      "object": [["", "wall", ...], ...],# rows x cols, "" = nothing on top
      "player_spawn": [x, y],
      "guards": [{"id": "g1", "patrol": [[x, y], ...]}],
      "idle_spots": [{"pos": [x, y], "facing_deg": 0, "tag": "idle"}],
      "lights": [{"pos": [x, y], "radius": 6.0, "intensity": 1.0}]
    }

`load_map` turns one into a game-ready TileMap: the effective tile for a
cell is its object tile if it has one, else its floor tile. The four
booleans and the derived sim values are baked into the same fine-resolution
numpy arrays the legacy loader produces, so the rest of the game does not
care which format a map came from. The raw id grids and the Tileset are
also attached for a future sprite renderer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim.tilemap import (GuardSpec, IdleSpot, Light, Tile, TileMap, _expand,
                         bake_lightmap, compute_roof, darken, lighten,
                         WALL_PEN_COST, WALL_SOUND_COST)
from sim.tileset import Tileset, load_tileset

FORMAT = "bangbang-map/1"
_CHAR_POOL = ("." + "".join(chr(c) for c in range(ord("a"), ord("z") + 1))
              + "".join(chr(c) for c in range(ord("A"), ord("Z") + 1))
              + "0123456789#+='\",;:*/\\|~")


def new_map(cols: int, rows: int, ts: Tileset, name: str = "untitled") -> dict:
    """A blank map document: solid floor, no objects."""
    f = ts.default_floor
    return {
        "format": FORMAT,
        "name": name,
        "size": [cols, rows],
        "cell_m": 1.0,
        "subdiv": 8,
        "floor": [[f] * cols for _ in range(rows)],
        "object": [[""] * cols for _ in range(rows)],
        "player_spawn": [1.5, 1.5],
        "guards": [],
        "idle_spots": [],
        "lights": [],
    }


def save(doc: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(doc)
    doc["format"] = FORMAT
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)


def load_doc(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file (got {doc.get('format')!r})")
    return doc


def _is_special(td) -> bool:
    """door / window / bush keep their own behaviour and ignore Floor/Wall."""
    return bool(td.door or td.glass or td.bush)


def load_map(path: str | Path, tileset: str | Path | None = None) -> TileMap:
    doc = load_doc(path)
    ts = load_tileset(tileset)
    cols, rows = doc["size"]
    subdiv = int(doc.get("subdiv", 8))
    mpc = float(doc.get("cell_m", 1.0))

    floor_ids = np.array(doc["floor"], dtype=object)
    object_ids = np.array([[o or "" for o in row] for row in doc["object"]],
                          dtype=object)

    bm = np.zeros((rows, cols), dtype=bool)
    bs = np.zeros((rows, cols), dtype=bool)
    bb = np.zeros((rows, cols), dtype=bool)
    sc = np.ones((rows, cols), dtype=np.float32)
    fm = np.ones((rows, cols), dtype=np.float32)
    pc = np.zeros((rows, cols), dtype=np.float32)
    gl = np.zeros((rows, cols), dtype=bool)
    bu = np.zeros((rows, cols), dtype=bool)
    en = np.zeros((rows, cols), dtype=bool)      # static enclosing tile (no doors)
    dcz = np.zeros((rows, cols), dtype=bool)     # door tile here

    chars = np.empty((rows, cols), dtype="<U1")
    tiles: dict[str, Tile] = {}
    char_of: dict[str, str] = {}

    for r in range(rows):
        for c in range(cols):
            fid = doc["floor"][r][c]
            ftd = ts[fid] if fid in ts else ts[ts.default_floor]
            oid = doc["object"][r][c]
            otd = ts[oid] if (oid and oid in ts) else None

            if otd is not None and _is_special(otd):
                # door / window / bush: keep the tile's own semantics
                bm[r, c] = not otd.is_walkable
                bs[r, c] = otd.blocks_los
                bb[r, c] = otd.blocks_shots
                sc[r, c] = otd.sound_cost
                fm[r, c] = otd.footstep_mult
                pc[r, c] = otd.pen_cost
                gl[r, c] = otd.glass
                bu[r, c] = otd.bush
                en[r, c] = otd.encloses and not otd.door and not otd.bush
                dcz[r, c] = otd.door
                cell_td, role = otd, "obj"
            elif otd is not None:
                # any tile painted as a WALL
                bm[r, c] = bs[r, c] = bb[r, c] = True
                sc[r, c] = WALL_SOUND_COST
                fm[r, c] = 1.0
                pc[r, c] = WALL_PEN_COST
                en[r, c] = True
                cell_td, role = otd, "wall"
            else:
                # floor: walkable, see-through; keep the tile's step/sound flavour
                sc[r, c] = ftd.sound_cost if ftd.sound_cost else 1.0
                fm[r, c] = ftd.footstep_mult
                cell_td, role = ftd, "floor"

            key = (cell_td.id, role)
            ch = char_of.get(key)
            if ch is None:
                ch = _CHAR_POOL[len(char_of) % len(_CHAR_POOL)]
                char_of[key] = ch
                col = (lighten(cell_td.colour) if role == "wall"
                       else darken(cell_td.colour) if role == "floor"
                       else tuple(cell_td.colour))
                tiles[ch] = Tile(
                    char=ch, name=cell_td.name,
                    blocks_move=bool(bm[r, c]), blocks_sight=bool(bs[r, c]),
                    blocks_bullets=bool(bb[r, c]), sound_cost=float(sc[r, c]),
                    footstep_mult=float(fm[r, c]), pen_cost=float(pc[r, c]),
                    door=bool(dcz[r, c]), glass=bool(gl[r, c]),
                    bush=bool(bu[r, c]), encloses=bool(en[r, c]),
                    colour=col)
            chars[r, c] = ch

    guards = [
        GuardSpec(
            id=g.get("id", f"g{i}"),
            patrol=[tuple(p) for p in g.get("patrol", [])],
            identify_deg=float(g.get("identify_deg", 20.0)),
            recognise_deg=float(g.get("recognise_deg", 100.0)),
            peripheral_deg=float(g.get("peripheral_deg", 170.0)),
            weapon=g.get("weapon", "combat_rifle"),
            skill=g.get("skill", "veteran"),
        )
        for i, g in enumerate(doc.get("guards", []))
    ]
    idle_spots = [
        IdleSpot(
            pos=tuple(s["pos"]),
            facing_deg=float(s.get("facing_deg", 0.0)),
            tag=s.get("tag", "idle"),
            dwell=tuple(s.get("dwell", (4.0, 10.0))),
            distraction=float(s.get("distraction", 0.0)),
        )
        for s in doc.get("idle_spots", [])
    ]
    lights = [
        Light(pos=tuple(l["pos"]), radius=float(l["radius"]),
              intensity=float(l.get("intensity", 1.0)))
        for l in doc.get("lights", [])
    ]

    tm = TileMap(
        name=doc.get("name", Path(path).stem),
        chars=chars,
        tiles=tiles,
        subdiv=subdiv,
        metres_per_char=mpc,
        blocks_move=_expand(bm, subdiv),
        blocks_sight=_expand(bs, subdiv),
        blocks_bullets=_expand(bb, subdiv),
        sound_cost=_expand(sc, subdiv),
        footstep_mult=_expand(fm, subdiv),
        pen_cost=_expand(pc, subdiv),
        glass=_expand(gl, subdiv),
        bush=_expand(bu, subdiv),
        player_spawn=tuple(doc.get("player_spawn", (1.5, 1.5))),
        guards=guards,
        idle_spots=idle_spots,
        lights=lights,
        floor_ids=floor_ids,
        object_ids=object_ids,
        tileset=ts,
        roof_enc=en,
        roof_doorcells=dcz,
    )
    # auto-roof: any coarse cell walled/windowed off from the border in a
    # closed loop. Doors start shut, so they seal here; main.py recomputes
    # on every toggle. None everywhere if nothing is enclosed.
    compute_roof(tm, {})
    tm.lightmap = bake_lightmap(tm)
    return tm
