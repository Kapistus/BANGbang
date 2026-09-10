"""Load assets/tiles/tileset.toml into TileDef records.

One TileDef per palette entry. The four editor booleans (is_see_through,
is_walkable, blocks_los, blocks_shots) are authoritative; sound_cost,
footstep_mult and pen_cost are read if present and otherwise derived from
those booleans so a hand-added tile only needs the four.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "assets" / "tiles" / "tileset.toml"


@dataclass
class TileDef:
    id: str
    name: str
    png: str
    layer: str                 # "floor" | "object"
    is_see_through: bool
    is_walkable: bool
    blocks_los: bool
    blocks_shots: bool
    sound_cost: float
    footstep_mult: float
    pen_cost: float
    door: bool
    glass: bool
    bush: bool
    overlay: bool               # draw ON TOP of characters (foliage canopy etc.)
    encloses: bool              # a full-height barrier that seals a room for roofing
    group: str                  # editor palette group ("" = base); purely UI
    colour: tuple[int, int, int]


def _derive_sound_cost(see_through, walkable, blocks_los, blocks_shots) -> float:
    if walkable:
        return 1.0
    if blocks_los and blocks_shots:
        return 40.0            # solid wall / pillar
    if blocks_shots and not blocks_los:
        return 2.2             # low cover
    if not blocks_shots and blocks_los:
        return 16.0            # thin wall
    return 6.0                 # see-through blocker (window)


def _derive_pen_cost(walkable, blocks_los, blocks_shots) -> float:
    if not blocks_shots:
        return 0.0
    return 3.5 if blocks_los else 1.2


@dataclass
class Tileset:
    tile_px: int
    default_floor: str
    tiles: dict[str, TileDef]
    path: Path

    def __getitem__(self, tid: str) -> TileDef:
        return self.tiles[tid]

    def __contains__(self, tid: str) -> bool:
        return tid in self.tiles

    def by_layer(self, layer: str) -> list[TileDef]:
        return [t for t in self.tiles.values() if t.layer == layer]


def load_tileset(path: str | Path | None = None) -> Tileset:
    path = Path(path) if path else DEFAULT_PATH
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    meta = data.get("meta", {})
    out: dict[str, TileDef] = {}
    for tid, spec in data.get("tiles", {}).items():
        st = bool(spec.get("is_see_through", True))
        wk = bool(spec.get("is_walkable", True))
        bl = bool(spec.get("blocks_los", False))
        bs = bool(spec.get("blocks_shots", False))
        gl = bool(spec.get("glass", False))
        # a full-height barrier: not walkable, and either opaque or a glass pane.
        # low cover (walkable F, see-through, no glass) does NOT enclose.
        enc = (not wk) and ((not st) or gl)
        out[tid] = TileDef(
            id=tid,
            name=spec.get("name", tid.replace("_", " ").title()),
            png=spec.get("png", f"{tid}.png"),
            layer=spec.get("layer", "object"),
            is_see_through=st,
            is_walkable=wk,
            blocks_los=bl,
            blocks_shots=bs,
            sound_cost=float(spec.get("sound_cost",
                                      _derive_sound_cost(st, wk, bl, bs))),
            footstep_mult=float(spec.get("footstep_mult", 1.0)),
            pen_cost=float(spec.get("pen_cost", _derive_pen_cost(wk, bl, bs))),
            door=bool(spec.get("door", False)),
            glass=gl,
            bush=bool(spec.get("bush", tid == "bush")),
            overlay=bool(spec.get("overlay", spec.get("bush", tid == "bush"))),
            encloses=bool(spec.get("encloses", enc)),
            group=str(spec.get("group", "")),
            colour=tuple(spec.get("colour", (180, 180, 180))),
        )
    return Tileset(
        tile_px=int(meta.get("tile_px", 32)),
        default_floor=meta.get("default_floor", "floor"),
        tiles=out,
        path=path,
    )
