"""Map loading: character grid + tile table -> per-property numpy arrays.

Authoring happens at one character per metre. The simulation runs on a finer
grid (subdiv cells per metre) so that doorways are several cells wide and
sound diffraction behaves. Actors live in world metres and sample the fine
arrays; nothing snaps to the grid except the obstacles themselves.

This is the legacy `.grid` + `tiles.toml` loader. New maps use the JSON
`.map` format (sim/mapfile.py, made by editor.py), which builds the same
TileMap. main.py's load_any_map() picks the loader by file extension.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Tile:
    char: str
    name: str
    blocks_move: bool = False
    blocks_sight: bool = False
    blocks_bullets: bool = False
    sound_cost: float = 1.0
    footstep_mult: float = 1.0
    pen_cost: float = 0.0       # penetration budget a bullet spends to pierce it
    door: bool = False
    glass: bool = False         # a bullet passes through but shatters the pane
    bush: bool = False          # walkable concealment: hides a still occupant, blocks sight past
    colour: tuple[int, int, int] = (200, 200, 200)


@dataclass
class IdleSpot:
    pos: tuple[float, float]
    facing_deg: float = 0.0
    tag: str = "idle"
    dwell: tuple[float, float] = (4.0, 10.0)
    distraction: float = 0.0


@dataclass
class GuardSpec:
    id: str
    patrol: list[tuple[float, float]]
    identify_deg: float = 20.0
    recognise_deg: float = 100.0
    peripheral_deg: float = 170.0
    weapon: str = "combat_rifle"      # weapons.ROSTER id; editor can override
    skill: str = "veteran"            # veteran | seasoned | rookie (aim quality)


@dataclass
class Light:
    pos: tuple[float, float]
    radius: float
    intensity: float = 1.0


@dataclass
class TileMap:
    """Coarse authoring grid plus expanded fine-resolution property arrays."""

    name: str
    chars: np.ndarray           # (rows, cols) unicode, one char per metre
    tiles: dict[str, Tile]
    subdiv: int                 # fine cells per authoring char
    metres_per_char: float

    blocks_move: np.ndarray     # bool  (fine)
    blocks_sight: np.ndarray    # bool  (fine)
    blocks_bullets: np.ndarray  # bool  (fine)
    sound_cost: np.ndarray      # f32   (fine)
    footstep_mult: np.ndarray   # f32   (fine)
    pen_cost: np.ndarray        # f32   (fine)  bullet penetration cost
    glass: np.ndarray           # bool  (fine)  shatters when shot through
    bush: np.ndarray            # bool  (fine)  walkable concealment foliage

    player_spawn: tuple[float, float] = (1.5, 1.5)
    guards: list[GuardSpec] = field(default_factory=list)
    idle_spots: list[IdleSpot] = field(default_factory=list)
    lights: list[Light] = field(default_factory=list)

    # set by the JSON map loader (sim/mapfile.py); None for legacy .grid maps.
    # coarse (rows, cols) arrays of tile ids for a sprite renderer.
    floor_ids: "np.ndarray | None" = None
    object_ids: "np.ndarray | None" = None
    tileset: object = None

    @property
    def cells_per_metre(self) -> float:
        return self.subdiv / self.metres_per_char

    @property
    def fine_shape(self) -> tuple[int, int]:
        return self.blocks_move.shape

    @property
    def width_m(self) -> float:
        return self.chars.shape[1] * self.metres_per_char

    @property
    def height_m(self) -> float:
        return self.chars.shape[0] * self.metres_per_char

    def cell_of(self, x_m: float, y_m: float) -> tuple[int, int]:
        """World metres -> fine grid indices (col, row), clamped in bounds."""
        cpm = self.cells_per_metre
        rows, cols = self.fine_shape
        cx = min(max(int(x_m * cpm), 0), cols - 1)
        cy = min(max(int(y_m * cpm), 0), rows - 1)
        return cx, cy

    def solid_at(self, x_m: float, y_m: float) -> bool:
        cx, cy = self.cell_of(x_m, y_m)
        return bool(self.blocks_move[cy, cx])

    def in_bounds(self, x_m: float, y_m: float, radius_m: float = 0.0) -> bool:
        """The point (with an optional body radius) lies inside the map rect."""
        return (radius_m <= x_m <= self.width_m - radius_m
                and radius_m <= y_m <= self.height_m - radius_m)

    def can_stand(self, x_m: float, y_m: float, radius_m: float = 0.28) -> bool:
        """Circle test against blocks_move, sampled at centre plus 4 rim points.
        Also rejects anything that would poke past the map border."""
        if not self.in_bounds(x_m, y_m, radius_m):
            return False
        if self.solid_at(x_m, y_m):
            return False
        r = radius_m
        for dx, dy in ((r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)):
            if self.solid_at(x_m + dx, y_m + dy):
                return False
        return True


def validate_patrols(m: "TileMap", radius: float = 0.30,
                     step: float = 0.10) -> list[str]:
    """Check every patrol segment is actually walkable.

    Waypoints are authored by hand in the sidecar and nothing stops you
    drawing a route straight through a wall, so this samples along each
    segment and reports the first blocked point it finds. Returns a list of
    human-readable problems; empty means the routes are clean.
    """
    problems: list[str] = []
    for g in m.guards:
        pts = g.patrol
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % len(pts)]
            if not m.can_stand(ax, ay, radius):
                problems.append(
                    f"guard {g.id}: waypoint {i} at ({ax:.1f}, {ay:.1f}) "
                    f"is inside geometry")
                continue
            d = math.hypot(bx - ax, by - ay)
            n = max(1, int(d / step))
            for k in range(n + 1):
                t = k / n
                x, y = ax + (bx - ax) * t, ay + (by - ay) * t
                if not m.can_stand(x, y, radius):
                    problems.append(
                        f"guard {g.id}: segment {i}->{(i + 1) % len(pts)} "
                        f"({ax:.1f},{ay:.1f})->({bx:.1f},{by:.1f}) "
                        f"is blocked at ({x:.1f},{y:.1f})")
                    break
    return problems


def validate_spawns(m: "TileMap", radius: float = 0.30) -> list[str]:
    """Check the player spawn and every idle spot are standable."""
    problems: list[str] = []
    if not m.can_stand(*m.player_spawn, radius):
        problems.append(f"player_spawn {m.player_spawn} is inside geometry")
    for i, s in enumerate(m.idle_spots):
        if not m.can_stand(*s.pos, radius):
            problems.append(f"idle_spot {i} ({s.tag}) at {s.pos} is inside geometry")
    return problems


def _expand(coarse: np.ndarray, subdiv: int) -> np.ndarray:
    """Nearest-neighbour upscale of a coarse array to simulation resolution."""
    return np.kron(coarse, np.ones((subdiv, subdiv), dtype=coarse.dtype))


def load_tiles(path: Path) -> tuple[dict[str, Tile], int, float]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    meta = data.get("meta", {})
    subdiv = int(meta.get("subdiv", 4))
    mpc = float(meta.get("metres_per_char", 1.0))

    tiles: dict[str, Tile] = {}
    for ch, spec in data["tiles"].items():
        tiles[ch] = Tile(
            char=ch,
            name=spec.get("name", ch),
            blocks_move=bool(spec.get("blocks_move", False)),
            blocks_sight=bool(spec.get("blocks_sight", False)),
            blocks_bullets=bool(spec.get("blocks_bullets", False)),
            sound_cost=float(spec.get("sound_cost", 1.0)),
            footstep_mult=float(spec.get("footstep_mult", 1.0)),
            pen_cost=float(spec.get("pen_cost", 0.0)),
            door=bool(spec.get("door", False)),
            glass=bool(spec.get("glass", False)),
            bush=bool(spec.get("bush", False)),
            colour=tuple(spec.get("colour", (200, 200, 200))),
        )
    return tiles, subdiv, mpc


def load_grid(path: Path) -> np.ndarray:
    raw = [ln.rstrip("\n") for ln in path.read_text().splitlines()]
    raw = [ln for ln in raw if ln.strip()]
    if not raw:
        raise ValueError(f"{path} is empty")
    widths = {len(ln) for ln in raw}
    if len(widths) != 1:
        counts = {}
        for i, ln in enumerate(raw):
            counts.setdefault(len(ln), []).append(i)
        raise ValueError(
            f"{path}: rows must all be the same width, got {counts}"
        )
    return np.array([list(ln) for ln in raw], dtype="<U1")


def load_map(sidecar: str | Path, tiles_path: str | Path | None = None) -> TileMap:
    sidecar = Path(sidecar)
    root = sidecar.parent
    tiles_path = Path(tiles_path) if tiles_path else root / "tiles.toml"

    tiles, subdiv, mpc = load_tiles(tiles_path)

    with open(sidecar, "rb") as fh:
        meta = tomllib.load(fh)

    subdiv = int(meta.get("subdiv", subdiv))   # per-map override
    chars = load_grid(root / meta["grid"])
    unknown = sorted(set(chars.ravel().tolist()) - set(tiles))
    if unknown:
        raise ValueError(f"{meta['grid']}: characters not in tile table: {unknown}")

    rows, cols = chars.shape
    bm = np.zeros((rows, cols), dtype=bool)
    bs = np.zeros((rows, cols), dtype=bool)
    bb = np.zeros((rows, cols), dtype=bool)
    sc = np.ones((rows, cols), dtype=np.float32)
    fm = np.ones((rows, cols), dtype=np.float32)
    pc = np.zeros((rows, cols), dtype=np.float32)
    gl = np.zeros((rows, cols), dtype=bool)
    bu = np.zeros((rows, cols), dtype=bool)

    for ch, t in tiles.items():
        m = chars == ch
        if not m.any():
            continue
        bm[m] = t.blocks_move
        bs[m] = t.blocks_sight
        bb[m] = t.blocks_bullets
        sc[m] = t.sound_cost
        fm[m] = t.footstep_mult
        pc[m] = t.pen_cost
        gl[m] = t.glass
        bu[m] = t.bush

    return TileMap(
        name=meta.get("name", sidecar.stem),
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
        player_spawn=tuple(meta.get("player_spawn", (1.5, 1.5))),
        guards=[
            GuardSpec(
                id=g.get("id", f"g{i}"),
                patrol=[tuple(p) for p in g.get("patrol", [])],
                identify_deg=float(g.get("identify_deg", 20.0)),
                recognise_deg=float(g.get("recognise_deg", 100.0)),
                peripheral_deg=float(g.get("peripheral_deg", 170.0)),
            )
            for i, g in enumerate(meta.get("guards", []))
        ],
        idle_spots=[
            IdleSpot(
                pos=tuple(s["pos"]),
                facing_deg=float(s.get("facing_deg", 0.0)),
                tag=s.get("tag", "idle"),
                dwell=tuple(s.get("dwell", (4.0, 10.0))),
                distraction=float(s.get("distraction", 0.0)),
            )
            for s in meta.get("idle_spots", [])
        ],
        lights=[
            Light(
                pos=tuple(l["pos"]),
                radius=float(l["radius"]),
                intensity=float(l.get("intensity", 1.0)),
            )
            for l in meta.get("lights", [])
        ],
    )
