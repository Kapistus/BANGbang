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
    door_time: float = 0.35     # seconds the panels take to travel. A powered
                                # door is quick; a blast door is not, and the
                                # wait is the reason to place one
    glass: bool = False         # a bullet passes through but shatters the pane
    encloses: bool = False      # full-height barrier: seals a room for auto-roofing
    colour: tuple[int, int, int] = (200, 200, 200)


@dataclass
class SpawnPoint:
    """Somewhere a multiplayer match can put a player.

    Authored in the map rather than derived from its geometry, because where a
    fight starts is a design decision: a derived spawn knows the ground is
    standable and nothing else, so it will happily drop you in the open at the
    end of a long clean sightline.

    `team` is "" for any player, or "a"/"b" to reserve it for one side in team
    modes. `facing_deg` is which way you are looking when you arrive, so a spawn
    in a corner does not start you staring at the wall."""
    x: float
    y: float
    facing_deg: float = 0.0
    team: str = ""

    @property
    def pos(self) -> tuple:
        return (self.x, self.y)

    def __iter__(self):
        """A spawn point unpacks as the point it is: `x, y = spawn` still
        works, so code that predates the facing and team fields — and anything
        that just wants a position — does not have to care."""
        yield self.x
        yield self.y


def parse_spawn_points(raw) -> list:
    """Read a spawn list, tolerating every shape a map might use: the full
    object form, a bare [x, y] pair, or {x, y}. A malformed entry is skipped
    rather than taken as an error — a typo in one spawn should not stop a map
    loading."""
    out: list[SpawnPoint] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for item in raw:
        try:
            if isinstance(item, SpawnPoint):
                out.append(item)
                continue
            if isinstance(item, dict):
                pos = item.get("pos")
                if pos is None:
                    pos = (item.get("x"), item.get("y"))
                sp = SpawnPoint(float(pos[0]), float(pos[1]),
                                float(item.get("facing_deg", 0.0)),
                                str(item.get("team", "") or "").lower())
            else:
                sp = SpawnPoint(float(item[0]), float(item[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if sp.team not in ("", "a", "b"):
            sp.team = ""
        out.append(sp)
    return out


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
class Item:
    """One entry in an interactable's take-list (or the player's inventory
    once taken). `category` decides what happens on take (main.py):
    weapon/usable/mission actually move to the player; anything else (misc)
    just gets `picked` flagged - lore-wise collected after the mission, not
    mechanically carried yet."""
    id: str
    name: str
    category: str = "misc"    # weapon | usable | mission | misc
    qty: int = 1
    picked: bool = False


def _parse_interactables(raw: "list | None") -> "list[Interactable]":
    """Shared by both loaders: doc['interactables'] (.map) / meta['interactables']
    (.grid sidecar) -> [Interactable]. Missing key = no interactables."""
    out = []
    for i, e in enumerate(raw or []):
        items = [
            Item(id=it.get("id", f"item{k}"), name=it.get("name", "item"),
                category=it.get("category", "misc"),
                qty=int(it.get("qty", 1)), picked=bool(it.get("picked", False)))
            for k, it in enumerate(e.get("items", []))
        ]
        out.append(Interactable(
            id=e.get("id", f"e{i}"), kind=e.get("kind", "npc"),
            pos=tuple(e["pos"]), name=e.get("name", "???"),
            dialog=list(e.get("dialog", [])), items=items))
    return out


@dataclass
class Interactable:
    """An editor-placed NPC / trader / quest object. Which panel opening it
    shows is decided by content, not `kind`: any `items` -> the take-list
    (inventory) panel, else the `dialog` panel. `kind` is authoring/render
    metadata (editor colour, default content shape)."""
    id: str
    kind: str                 # npc | trader | quest_item
    pos: tuple[float, float]
    name: str = "???"
    dialog: list[str] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)


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

    player_spawn: tuple[float, float] = (1.5, 1.5)
    # multiplayer starts. Empty means the map declares none, and the server
    # derives them from the geometry instead (see net/maps.py).
    spawn_points: list = field(default_factory=list)
    guards: list[GuardSpec] = field(default_factory=list)
    idle_spots: list[IdleSpot] = field(default_factory=list)
    lights: list[Light] = field(default_factory=list)
    interactables: list[Interactable] = field(default_factory=list)

    # set by the JSON map loader (sim/mapfile.py); None for legacy .grid maps.
    # coarse (rows, cols) arrays of tile ids for a sprite renderer.
    floor_ids: "np.ndarray | None" = None
    object_ids: "np.ndarray | None" = None
    tileset: object = None

    # Auto-roofing. A coarse cell sealed off from the map border by a closed
    # loop of enclosing tiles (walls, window frames, shut doors) gets a roof
    # that the renderer draws over the interior until you look in through an
    # aperture. `compute_roof` (re)derives the first three from the last two;
    # all None when the map has no fully enclosed area. The sim never reads
    # any of it - render-time only.
    roof: "np.ndarray | None" = None          # bool (fine): cell is under a roof
    roof_bid: "np.ndarray | None" = None      # int  (fine): building id, 0 = none
    roof_rgb: "np.ndarray | None" = None      # uint8 (fine, 3): roof fill colour
    roof_enc: "np.ndarray | None" = None      # bool (coarse): static enclosing tile (no doors)
    roof_doorcells: "np.ndarray | None" = None  # bool (coarse): door tile here

    lightmap: "np.ndarray | None" = None      # fine float 0..1, baked from `lights`

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


def find_doors(m: "TileMap") -> dict:
    """Map coarse (row, col) of every door tile to its open state."""
    out = {}
    for r in range(m.chars.shape[0]):
        for c in range(m.chars.shape[1]):
            tile = m.tiles.get(m.chars[r, c])
            if tile is not None and tile.door:
                out[(r, c)] = False
    return out


def set_door(m: "TileMap", cost, r: int, c: int, is_open: bool) -> None:
    """Rewrite the fine arrays for one door tile. Callers must invalidate the
    visibility cache and any held sound fields afterwards.

    Like glass, a door is map state: opening one changes what can be seen,
    shot and heard through it, so in multiplayer the server owns the toggle and
    every client applies the same one."""
    sub = m.subdiv
    y0, y1 = r * sub, (r + 1) * sub
    x0, x1 = c * sub, (c + 1) * sub
    t = m.tiles[m.chars[r, c]]
    m.blocks_move[y0:y1, x0:x1] = False if is_open else t.blocks_move
    m.blocks_sight[y0:y1, x0:x1] = False if is_open else t.blocks_sight
    m.blocks_bullets[y0:y1, x0:x1] = False if is_open else t.blocks_bullets
    val = 1.0 if is_open else t.sound_cost
    m.sound_cost[y0:y1, x0:x1] = val
    if cost is not None:
        cost[y0:y1, x0:x1] = val


def break_glass_cells(m: "TileMap", fine_cells, cost: "np.ndarray | None" = None,
                      broken: "set | None" = None) -> list:
    """Turn every glass tile named by `fine_cells` into an open hole: bullets
    and sight pass through, the frame still blocks movement, and sound crosses
    almost freely. Returns the COARSE (row, col) cells newly broken.

    This mutates the map, which is why multiplayer has the server do it and
    broadcast the result: a pane broken on one machine and not another would
    give those players different walls to see, shoot and listen through.

    `cost` is the sound-cost array the caller is solving fields against, kept in
    step with the map's own. `broken` is a set of coarse cells already broken,
    so repeated hits on the same pane are cheap and idempotent.
    """
    sub = m.subdiv
    out = []
    for ci, cj in fine_cells:
        r, c = cj // sub, ci // sub
        if broken is not None and (r, c) in broken:
            continue
        tile = m.tiles.get(m.chars[r, c])
        if tile is None or not tile.glass:
            continue
        if broken is not None:
            broken.add((r, c))
        y0, y1, x0, x1 = r * sub, (r + 1) * sub, c * sub, (c + 1) * sub
        m.blocks_bullets[y0:y1, x0:x1] = False
        m.blocks_sight[y0:y1, x0:x1] = False
        m.glass[y0:y1, x0:x1] = False
        m.sound_cost[y0:y1, x0:x1] = BROKEN_GLASS_SOUND_COST
        if cost is not None:
            cost[y0:y1, x0:x1] = BROKEN_GLASS_SOUND_COST
        out.append((int(r), int(c)))
    return out


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
    """Check the player spawn, every idle spot and every multiplayer spawn are
    standable."""
    problems: list[str] = []
    if not m.can_stand(*m.player_spawn, radius):
        problems.append(f"player_spawn {m.player_spawn} is inside geometry")
    for i, s in enumerate(m.idle_spots):
        if not m.can_stand(*s.pos, radius):
            problems.append(f"idle_spot {i} ({s.tag}) at {s.pos} is inside geometry")
    for i, sp in enumerate(m.spawn_points):
        if not m.can_stand(sp.x, sp.y, radius):
            problems.append(f"spawn_point {i} at ({sp.x:.1f}, {sp.y:.1f}) "
                            f"is inside geometry")
    teams = {sp.team for sp in m.spawn_points if sp.team}
    if teams and teams != {"a", "b"}:
        problems.append(f"spawn_points name team(s) {sorted(teams)} but not "
                        f"both sides: team matches would start one side on "
                        f"whatever is left over")
    return problems


def _expand(coarse: np.ndarray, subdiv: int) -> np.ndarray:
    """Nearest-neighbour upscale of a coarse array to simulation resolution."""
    return np.kron(coarse, np.ones((subdiv, subdiv), dtype=coarse.dtype))


ROOF_RGB = (0, 0, 0)             # flat roof fill - opaque black cover
ROOF_MAX_CELLS = 800             # a sealed blob bigger than this (coarse cells) is
                                 # a walled arena / courtyard, not a room - no roof

# Editor Floor / Wall placement roles bake fixed sim properties; the tile is
# just art. For readability the two roles render at different brightness:
# floor cells are darkened, wall cells are lightened toward white.
WALL_SOUND_COST = 40.0
BROKEN_GLASS_SOUND_COST = 1.5   # a shattered pane is barely a barrier at all
WALL_PEN_COST = 3.5
FLOOR_DARKEN = 0.55      # floor cell brightness, fraction of the tile
WALL_LIGHTEN = 0.40      # wall cell blend toward white, 0 = tile .. 1 = white


def darken(colour, f: float = FLOOR_DARKEN) -> tuple:
    return tuple(max(0, min(255, int(round(v * f)))) for v in colour[:3])


def lighten(colour, k: float = WALL_LIGHTEN) -> tuple:
    return tuple(max(0, min(255, int(round(v + (255 - v) * k)))) for v in colour[:3])


def _label_buildings(mask: np.ndarray) -> np.ndarray:
    """4-connected flood fill of a coarse roof mask -> per-cell building id
    (1..N, 0 where there is no roof). One connected roof = one building, so
    the renderer can drop a whole building's roof when the player is inside."""
    from collections import deque

    rows, cols = mask.shape
    bid = np.zeros((rows, cols), dtype=np.int32)
    nxt = 0
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or bid[r, c]:
                continue
            nxt += 1
            dq = deque([(r, c)])
            bid[r, c] = nxt
            while dq:
                y, x = dq.popleft()
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if (0 <= ny < rows and 0 <= nx < cols
                            and mask[ny, nx] and not bid[ny, nx]):
                        bid[ny, nx] = nxt
                        dq.append((ny, nx))
    return bid


def enclosed_mask(enc: np.ndarray) -> np.ndarray:
    """Given a coarse bool of enclosing tiles, return a coarse bool of cells
    that sit inside a closed loop - the interior floor AND the ring around it.

    Flood-fills 'exterior' inward from the map border across every non-
    enclosing cell; whatever it never reaches is walled off. Roof blobs that
    contain no open interior cell (a lone pillar, an unclosed wall stub) are
    dropped so only real rooms roof."""
    rows, cols = enc.shape
    from collections import deque

    exterior = np.zeros((rows, cols), dtype=bool)
    dq: deque = deque()

    def seed(r, c):
        if not enc[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            dq.append((r, c))

    for r in range(rows):
        seed(r, 0)
        seed(r, cols - 1)
    for c in range(cols):
        seed(0, c)
        seed(rows - 1, c)
    while dq:
        y, x = dq.popleft()
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if (0 <= ny < rows and 0 <= nx < cols
                    and not enc[ny, nx] and not exterior[ny, nx]):
                exterior[ny, nx] = True
                dq.append((ny, nx))

    roofed = ~exterior
    if not roofed.any():
        return roofed
    # keep a blob only if it encloses open (non-enclosing) floor AND is small
    # enough to be a room rather than a walled arena
    bid = _label_buildings(roofed)
    interior = roofed & ~enc
    sizes = np.bincount(bid.ravel())
    keep = [int(v) for v in np.unique(bid[interior])
            if v and sizes[v] <= ROOF_MAX_CELLS]
    if not keep:
        return np.zeros((rows, cols), dtype=bool)
    return np.isin(bid, keep)


LIGHT_AMBIENT = 0.15        # brightness of unlit ground (0 = black .. 1 = full)
LIGHT_FALLOFF_POW = 1.6     # >1 = light pools tighter around the source
LIGHT_BLUR = 6             # fine-cell radius the lightmap is softened by


def _blur2d(a: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur - feathers the pool and shadow edges of a light."""
    out = a.astype(np.float32).copy()
    for _ in range(2):                       # two passes ~ a soft gaussian
        src = out.copy()
        for d in range(1, radius + 1):
            out[:, d:] += src[:, :-d]
            out[:, :-d] += src[:, d:]
        out /= (2 * radius + 1)
        src = out.copy()
        for d in range(1, radius + 1):
            out[d:, :] += src[:-d, :]
            out[:-d, :] += src[d:, :]
        out /= (2 * radius + 1)
    return out


def bake_lightmap(m: "TileMap") -> "np.ndarray | None":
    """Fine float array 0..1: an ambient floor plus each Light's radial falloff,
    shadow-cast against blocks_sight so walls throw light shadows, then blurred
    so the pools read soft. Static - baked once at load; the renderer
    multiplies it into the base surface.

    Returns None when the map has no lights, so an unlit map renders at full
    brightness rather than everywhere-ambient."""
    if not m.lights:
        return None
    from sim.vision import shadowcast

    h, w = m.blocks_sight.shape
    lit = np.zeros((h, w), dtype=np.float32)     # the light contribution only
    cpm = m.cells_per_metre
    for lt in m.lights:
        lx = int(round(lt.pos[0] * cpm))
        ly = int(round(lt.pos[1] * cpm))
        if not (0 <= lx < w and 0 <= ly < h):
            continue
        rc = max(1, int(round(lt.radius * cpm)))
        vf = shadowcast(m.blocks_sight, lx, ly, rc)
        fall = np.clip(1.0 - vf.dist / rc, 0.0, 1.0) ** LIGHT_FALLOFF_POW
        add = np.where(vf.visible, fall * float(lt.intensity), 0.0)
        lit[vf.y0:vf.y0 + add.shape[0], vf.x0:vf.x0 + add.shape[1]] += add
    if LIGHT_BLUR > 0:
        lit = _blur2d(lit, LIGHT_BLUR)
    lm = np.clip(LIGHT_AMBIENT + lit, 0.0, 1.0)
    return lm.astype(np.float32)


def compute_roof(m: "TileMap",
                 door_open: "dict[tuple[int, int], bool] | None" = None) -> None:
    """(Re)derive m.roof / m.roof_bid / m.roof_rgb from m.roof_enc plus the
    current door states. Cheap (coarse grid) - safe to call on every door
    toggle. Sets all three to None if nothing is enclosed."""
    enc = m.roof_enc
    if enc is None:
        m.roof = m.roof_bid = m.roof_rgb = None
        return
    mask = enc.copy()
    dc = m.roof_doorcells
    if dc is not None and dc.any():
        opened = door_open or {}
        rows, cols = mask.shape
        for r in range(rows):
            for c in range(cols):
                if dc[r, c] and not opened.get((r, c), False):
                    mask[r, c] = True        # a shut door seals the loop

    roofed = enclosed_mask(mask)
    if not roofed.any():
        m.roof = m.roof_bid = m.roof_rgb = None
        return
    subdiv = m.subdiv
    m.roof = _expand(roofed, subdiv)
    m.roof_bid = _expand(_label_buildings(roofed), subdiv)
    rows, cols = roofed.shape
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
    rgb[roofed] = ROOF_RGB
    m.roof_rgb = np.repeat(np.repeat(rgb, subdiv, axis=0), subdiv, axis=1)


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
            door_time=float(spec.get("door_time", 0.35)),
            glass=bool(spec.get("glass", False)),
            encloses=bool(spec.get(
                "encloses",
                bool(spec.get("blocks_move", False))
                and (bool(spec.get("blocks_sight", False))
                     or bool(spec.get("glass", False))))),
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
    en = np.zeros((rows, cols), dtype=bool)
    dcz = np.zeros((rows, cols), dtype=bool)

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
        en[m] = t.encloses and not t.door
        dcz[m] = t.door

    tm = TileMap(
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
        player_spawn=tuple(meta.get("player_spawn", (1.5, 1.5))),
        spawn_points=parse_spawn_points(meta.get("spawn_points")),
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
        interactables=_parse_interactables(meta.get("interactables", [])),
        roof_enc=en,
        roof_doorcells=dcz,
    )
    compute_roof(tm, {})
    tm.lightmap = bake_lightmap(tm)
    return tm
