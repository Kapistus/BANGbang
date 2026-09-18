"""
net/maps.py — map resolution and multiplayer spawn points.

Shared by the server (which needs geometry to collide against and spawns to
place players on) and by any client that wants the same map the server chose.
Imports sim/, unlike the rest of net/ — this is the one bridge between the
networking layer and the simulation.

A map_id is a bare filename stem as used on the wire ("arena", "compound").
resolve() turns it into a real path, preferring the JSON .map format over the
legacy char-grid + TOML sidecar.

Spawn points come from the map file when it declares them:

    .map (JSON):   "spawn_points": [[x, y], [x, y], ...]
    .toml:         spawn_points = [[x, y], ...]      (top level or under [meta])

When it doesn't — which is every map in this project today — they are derived
geometrically by farthest-point sampling over standable ground, so any map is
playable without being re-authored. Derived spawns are deterministic for a
given map: same map in, same spawns out, on every machine.
"""
from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path

import numpy as np

from sim import mapfile
from sim.tilemap import TileMap, load_map as load_grid_map

DEFAULT_MAPS_DIR = Path(__file__).resolve().parent.parent / "maps"

# Spawns are placed with a more generous radius than the body so nobody
# materialises wedged in a doorway or hugging a wall.
SPAWN_CLEARANCE_M = 0.45
# Sampling step when hunting for candidate ground, in metres.
SAMPLE_STEP_M = 1.0
# Stop adding spawns once the best remaining candidate is closer than this to
# an already-chosen spawn. Small maps simply end up with fewer spawns.
MIN_SEPARATION_M = 5.0
WANT_SPAWNS = 8


class MapNotFound(Exception):
    pass


def resolve(map_id: str, maps_dir: str | Path = DEFAULT_MAPS_DIR) -> Path:
    """map_id -> path. Accepts a bare stem, a filename, or a full path."""
    p = Path(map_id)
    if p.suffix and p.exists():
        return p
    d = Path(maps_dir)
    stem = p.stem or str(map_id)
    for suffix in (".map", ".toml"):
        cand = d / f"{stem}{suffix}"
        if cand.is_file():
            return cand
    available = sorted(q.stem for q in d.glob("*.map")) + \
                sorted(q.stem for q in d.glob("*.toml"))
    raise MapNotFound(
        f"no map {map_id!r} in {d}; available: {', '.join(available) or '(none)'}")


def load(map_id: str, maps_dir: str | Path = DEFAULT_MAPS_DIR) -> TileMap:
    """Load a map by id. Mirrors main.load_any_map's dispatch without dragging
    the whole renderer in as an import."""
    path = resolve(map_id, maps_dir)
    if path.suffix == ".map":
        return mapfile.load_map(path)
    return load_grid_map(path)


# ------------------------------------------------------------------ spawns

def _explicit_spawns(path: Path) -> list[tuple[float, float]]:
    """Spawn points declared by the map file itself, or [] if it declares none.
    Never raises on a malformed entry — it just skips it, because a typo in a
    map should not take the server down."""
    try:
        if path.suffix == ".map":
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        else:
            with open(path, "rb") as f:
                doc = tomllib.load(f)
    except (OSError, ValueError, tomllib.TOMLDecodeError):
        return []
    raw = doc.get("spawn_points")
    if raw is None and isinstance(doc.get("meta"), dict):
        raw = doc["meta"].get("spawn_points")
    if not isinstance(raw, list):
        return []
    out: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, dict):
            item = [item.get("x"), item.get("y")]
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            out.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
    return out


def _candidates(m: TileMap, clearance: float, step: float) -> np.ndarray:
    """Every standable sample point on the map, as an (n, 2) array of metres."""
    xs = np.arange(step * 0.5, m.width_m, step)
    ys = np.arange(step * 0.5, m.height_m, step)
    pts = [(float(x), float(y))
           for y in ys for x in xs
           if m.can_stand(float(x), float(y), clearance)]
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2))


def derive(m: TileMap, want: int = WANT_SPAWNS,
           clearance: float = SPAWN_CLEARANCE_M,
           min_separation: float = MIN_SEPARATION_M,
           step: float = SAMPLE_STEP_M) -> list[tuple[float, float]]:
    """Farthest-point sampling over standable ground: repeatedly take the
    candidate furthest from everything chosen so far. That spreads spawns to
    opposite ends of the map and into separate rooms without knowing anything
    about rooms, and is deterministic — no RNG, so every machine agrees."""
    pts = _candidates(m, clearance, step)
    if len(pts) == 0:
        # Nothing standable at spawn clearance. Fall back to the single-player
        # spawn even if it is tight; better than refusing to start a match.
        return [tuple(float(v) for v in m.player_spawn)]

    # Seed from the authored player spawn when it is usable, so map authors
    # keep some say in where a match begins.
    sx, sy = (float(v) for v in m.player_spawn)
    if m.can_stand(sx, sy, clearance):
        chosen = [(sx, sy)]
    else:
        # otherwise start from the candidate furthest from the map centre
        centre = np.array([m.width_m * 0.5, m.height_m * 0.5])
        chosen = [tuple(pts[int(np.argmax(((pts - centre) ** 2).sum(1)))])]

    d2 = ((pts - np.asarray(chosen[0])) ** 2).sum(1)
    while len(chosen) < want:
        i = int(np.argmax(d2))
        if math.sqrt(float(d2[i])) < min_separation:
            break                      # map is too cramped for more spread
        chosen.append((float(pts[i, 0]), float(pts[i, 1])))
        d2 = np.minimum(d2, ((pts - pts[i]) ** 2).sum(1))
    return chosen


def spawn_points(map_id: str, m: TileMap | None = None,
                 maps_dir: str | Path = DEFAULT_MAPS_DIR,
                 want: int = WANT_SPAWNS) -> list[tuple[float, float]]:
    """Spawn points for a map: whatever the file declares, filtered to those
    that are actually standable, topped up by derivation if there are too few.

    Pass `m` if you have already loaded the map, to avoid loading it twice.
    """
    path = resolve(map_id, maps_dir)
    if m is None:
        m = load(map_id, maps_dir)
    explicit = [p for p in _explicit_spawns(path)
                if m.can_stand(p[0], p[1], SPAWN_CLEARANCE_M)]
    if len(explicit) >= 2:
        return explicit
    derived = derive(m, want=want)
    # keep authored spawns first, then fill with derived ones that aren't
    # sitting on top of them
    out = list(explicit)
    for p in derived:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 1.0 for q in out):
            out.append(p)
    return out


def load_with_spawns(map_id: str, maps_dir: str | Path = DEFAULT_MAPS_DIR
                     ) -> tuple[TileMap, list[tuple[float, float]]]:
    """Load a map and its spawn points in one go. What the server calls."""
    m = load(map_id, maps_dir)
    return m, spawn_points(map_id, m, maps_dir)


def list_map_ids(maps_dir: str | Path = DEFAULT_MAPS_DIR) -> list[str]:
    """Map ids that actually load as maps. Unlike a bare *.toml glob this does
    not offer tiles.toml (a tile definition file) as something to play on."""
    d = Path(maps_dir)
    ids: list[str] = []
    for path in sorted(d.glob("*.map")) + sorted(d.glob("*.toml")):
        stem = path.stem
        if stem in ids:
            continue
        try:
            load(stem, d)
        except Exception:
            continue
        ids.append(stem)
    return ids
