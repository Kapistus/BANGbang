"""
net/maps.py — map resolution and multiplayer spawn points.

Shared by the server (which needs geometry to collide against and spawns to
place players on) and by any client that wants the same map the server chose.
Imports sim/, unlike the rest of net/ — this is the one bridge between the
networking layer and the simulation.

A map_id is a bare filename stem as used on the wire ("arena", "compound").
resolve() turns it into a real path, preferring the JSON .map format over the
legacy char-grid + TOML sidecar.

Spawn points come from the map file when it declares them — placed in the
editor, carried on TileMap.spawn_points, each with a facing and an optional
team tag.

When a map declares none, they are derived geometrically by farthest-point
sampling over standable ground, so any map is playable without being
re-authored. Derived spawns are deterministic for a given map: same map in,
same spawns out, on every machine. What they cannot know is anything about
sightlines — derived spawns know the ground is standable and nothing else —
which is the reason to author them.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from sim import mapfile
from sim.tilemap import SpawnPoint, TileMap, load_map as load_grid_map
from sim.tilemap import parse_spawn_points

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

def as_spawn_list(items) -> list:
    """Normalise whatever a caller hands us into SpawnPoints — plain (x, y)
    tuples included, because tests and older code pin spawns that way."""
    return parse_spawn_points(list(items or []))


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
           step: float = SAMPLE_STEP_M) -> list:
    """Farthest-point sampling over standable ground: repeatedly take the
    candidate furthest from everything chosen so far. That spreads spawns to
    opposite ends of the map and into separate rooms without knowing anything
    about rooms, and is deterministic — no RNG, so every machine agrees."""
    pts = _candidates(m, clearance, step)
    if len(pts) == 0:
        # Nothing standable at spawn clearance. Fall back to the single-player
        # spawn even if it is tight; better than refusing to start a match.
        sx, sy = (float(v) for v in m.player_spawn)
        return [SpawnPoint(sx, sy)]

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
    return [SpawnPoint(x, y) for x, y in chosen]


def spawn_points(map_id: str, m: TileMap | None = None,
                 maps_dir: str | Path = DEFAULT_MAPS_DIR,
                 want: int = WANT_SPAWNS) -> list:
    """Spawn points for a map: whatever the map declares, filtered to those that
    are actually standable, topped up by derivation if there are too few.

    Pass `m` if you have already loaded the map, to avoid loading it twice.
    """
    if m is None:
        m = load(map_id, maps_dir)
    authored = [sp for sp in m.spawn_points
                if m.can_stand(sp.x, sp.y, SPAWN_CLEARANCE_M)]
    if len(authored) >= 2:
        return authored
    # keep authored spawns first, then fill with derived ones that aren't
    # sitting on top of them
    out = list(authored)
    for sp in derive(m, want=want):
        if all(math.hypot(sp.x - q.x, sp.y - q.y) > 1.0 for q in out):
            out.append(sp)
    return out


def for_team(spawns: list, team) -> list:
    """The spawns a player on this team may use.

    A map that tags spawns for a side is making a statement about where that
    side starts, so those are used exclusively when they exist. Untagged spawns
    are the neutral pool; if a map tags none, everyone shares everything."""
    tag = ""
    name = getattr(team, "name", str(team)).lower()
    if name in ("a", "b"):
        tag = name
    if tag:
        mine = [sp for sp in spawns if sp.team == tag]
        if mine:
            return mine
    neutral = [sp for sp in spawns if not sp.team]
    return neutral or list(spawns)


def pick(spawns: list, team=None, away_from=(), rng=None, top: int = 1):
    """One spawn, as far from everyone in `away_from` as the map allows.

    `top` widens the choice to the best N: 1 for somebody joining a match in
    progress, where landing in a firefight is the thing to avoid, and more for
    respawns, where always reappearing in the same corner is an invitation to
    be camped."""
    options = for_team(spawns, team) if team is not None else list(spawns)
    if not options:
        options = list(spawns)
    live = [(float(x), float(y)) for x, y in away_from]
    if not live:
        return (rng or random).choice(options)
    ranked = sorted(options,
                    key=lambda sp: min(math.hypot(sp.x - x, sp.y - y)
                                       for x, y in live),
                    reverse=True)
    return (rng or random).choice(ranked[:max(1, min(top, len(ranked)))])


def load_with_spawns(map_id: str, maps_dir: str | Path = DEFAULT_MAPS_DIR
                     ) -> tuple[TileMap, list[tuple[float, float]]]:
    """Load a map and its spawn points in one go. What the server calls."""
    m = load(map_id, maps_dir)
    return m, spawn_points(map_id, m, maps_dir)


MIN_SPAWNS = 2


def validate(map_id: str, maps_dir: str | Path = DEFAULT_MAPS_DIR) -> list[str]:
    """Why this map cannot be played, or [] if it can.

    The one definition of a playable map, used by the lobby's map list, by the
    server before it starts a match, and by the editor when it saves. A map is
    playable when:

      it loads — a char-grid map with a character the tile table does not
        know raises, and a match started on it crashes every client

      every tile it uses exists — a .map that names a tile the tileset does
        not have still loads, but that cell comes out as bare floor, so a map
        drawn with an old tileset plays with its walls missing

      there are at least two places to spawn once unusable ones are dropped

    Spawn WARNINGS — one authored spawn inside a wall, a map that tags only
    one team — are deliberately not here. The server already drops an unusable
    spawn and plays on with the rest, so those maps are playable; the editor
    and main.py report them through validate_spawns instead.

    Cheap enough to call every time the list is built: tens of milliseconds a
    map.

    Only a bare name for a map sitting directly in `maps_dir` can pass. The
    resolver also takes file paths, which is how a crafted id like
    "../tests/range" or "../somewhere" could otherwise get hosted — and
    the test maps are for tests and trying things by hand, never the lobby."""
    bare = Path(str(map_id))
    if (not map_id or bare.name != str(map_id) or bare.suffix
            or "/" in str(map_id) or "\\" in str(map_id)
            or str(map_id) in (".", "..")):
        return [f"{map_id!r} is not a map in {maps_dir}; only maps directly "
                f"in that folder can be hosted"]
    try:
        m, spawns = load_with_spawns(map_id, maps_dir)
    except Exception as e:                        # noqa: BLE001 — any failure
        return [f"does not load: {e}"]
    problems = []
    ts = getattr(m, "tileset", None)
    fids, oids = getattr(m, "floor_ids", None), getattr(m, "object_ids", None)
    if ts is not None and fids is not None:
        unknown: dict[str, int] = {}
        for grid in (fids, oids):
            if grid is None:
                continue
            for tid in grid.ravel().tolist():
                if tid and tid not in ts:
                    unknown[tid] = unknown.get(tid, 0) + 1
        if unknown:
            shown = ", ".join(sorted(unknown)[:6]) + ("…" if len(unknown) > 6 else "")
            problems.append(f"{sum(unknown.values())} cells use tiles the tileset "
                            f"does not have ({shown}); they would load as bare floor")
    if len(spawns) < MIN_SPAWNS:
        problems.append(f"only {len(spawns)} usable place(s) to spawn; "
                        f"needs {MIN_SPAWNS}")
    return problems


def first_playable(maps_dir: str | Path = DEFAULT_MAPS_DIR,
                   prefer: "str | None" = None) -> "str | None":
    """`prefer` if it is playable, otherwise the first map id that is."""
    if prefer and not validate(prefer, maps_dir):
        return prefer
    for mid in list_map_ids(maps_dir):
        if not validate(mid, maps_dir):
            return mid
    return None


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


# ------------------------------------------------------------------ transfer
# A player who does not have the host's map - or has a different map under the
# same name - gets a copy from the host. These say what "the map" is on disk,
# and give it a fingerprint both sides can compare.

import hashlib
import re

MAP_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
FILE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}\.(map|toml|grid)$")
MAX_MAP_BYTES = 3 * 1024 * 1024     # one wire frame is capped at 4 MiB


def map_files(map_id: str, maps_dir: str | Path = DEFAULT_MAPS_DIR) -> dict:
    """Every file this map needs, name -> bytes. A .map is one file. A
    char-grid map is its sidecar, the grid it names, and the tile table next
    to them, because that table decides what each character is - a copy of a
    grid read against a different table is a different map."""
    path = resolve(map_id, maps_dir)
    files = {path.name: path.read_bytes()}
    if path.suffix == ".toml":
        import tomllib
        meta = tomllib.loads(files[path.name].decode("utf-8"))
        grid = path.parent / str(meta["grid"])
        files[grid.name] = grid.read_bytes()
        tiles = path.parent / "tiles.toml"
        if tiles.is_file():
            files[tiles.name] = tiles.read_bytes()
    return files


def files_sha(files: dict) -> str:
    """One fingerprint for a set of map files, independent of order."""
    h = hashlib.sha256()
    for name in sorted(files):
        data = files[name]
        h.update(name.encode("utf-8") + b"\0")
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def map_sha(map_id: str, maps_dir: str | Path = DEFAULT_MAPS_DIR) -> "str | None":
    """Fingerprint of this map as it is in maps_dir, or None if it is not."""
    try:
        return files_sha(map_files(map_id, maps_dir))
    except Exception:                     # noqa: BLE001 - absent or unreadable
        return None


def save_download(map_id: str, sha: str, files: dict,
                  download_dir: str | Path) -> Path:
    """Write a map received from a host into its own folder under
    download_dir, after checking it is what the host said it was and that
    every name is a plain file name - nothing a host sends can land anywhere
    else. Returns the folder. Raises ValueError on anything wrong."""
    if not MAP_ID_RE.match(str(map_id)):
        raise ValueError(f"refusing map id {map_id!r}")
    if not files or not isinstance(files, dict):
        raise ValueError("no files")
    for name, data in files.items():
        if not isinstance(name, str) or not FILE_NAME_RE.match(name):
            raise ValueError(f"refusing file name {name!r}")
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError(f"{name}: not file data")
    if sum(len(d) for d in files.values()) > MAX_MAP_BYTES:
        raise ValueError("map is too large")
    if files_sha(files) != sha:
        raise ValueError("the files do not match the host's fingerprint")
    stems = {Path(n).stem for n in files if not n.endswith((".grid",))} - {"tiles"}
    if stems != {map_id}:
        raise ValueError(f"files {sorted(files)} are not map {map_id!r}")
    folder = Path(download_dir) / map_id
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.iterdir():            # a previous version of this map
        if old.is_file() and FILE_NAME_RE.match(old.name):
            old.unlink()
    for name, data in files.items():
        (folder / name).write_bytes(bytes(data))
    problems = validate(map_id, folder)
    if problems:
        raise ValueError(f"the downloaded map is not playable: {problems[0]}")
    return folder
