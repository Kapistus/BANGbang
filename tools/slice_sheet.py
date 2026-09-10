"""Slice a sheet of tiles into individual square PNGs.

Two modes:

GRID (default) - the sheet is a clean N x M grid; each cell is auto-cropped
to its non-black content and rescaled to a square.

    python tools/slice_sheet.py SHEET.png --prefix metal_a --out assets/tiles/floor
    python tools/slice_sheet.py SHEET.png --prefix metal_a --rigid   # keep borders
    python tools/slice_sheet.py SHEET.png --prefix metal_a --toml    # print toml

AUTO (--auto) - the sheet has irregular black gutters (no clean grid). Masks
the non-black pixels, seals hairline cracks in the art (--close), labels
connected blobs, keeps the ones near the median blob size, emits them in
reading order (rows top-to-bottom, tiles left-to-right).

    python tools/slice_sheet.py SHEET.png --auto --prefix metal --toml

SETS (--sets) - one master image holding a GRID of sets, separated by wide
black bands that are not uniform. Each set is itself a clean --set-cols x
--set-rows tile grid with no real black between its tiles. The master is
split on the black bands (row/column content profile), each set region is
rescaled to exactly (set-cols*size, set-rows*size) and rigid-sliced.

    python tools/slice_sheet.py MASTER.png --sets --prefix metal --toml
    #  -> metal_setNN.png  (the 16 rescaled 512x512 sheets, a checkpoint)
    #  -> metal_NN_rc.png  (256 tiles, NN = set, rc = row/col)

Tune with --thresh (black cutoff) and --set-gap (min black-band width, px,
that counts as a separator between sets - raise if sets get split, lower if
sets get merged).

Output size --size (default 128, matching tileset.toml [meta] tile_px).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pygame


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def _binary_dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        out[:-1, :] |= out[1:, :]
        out[1:, :] |= out[:-1, :]
        out[:, :-1] |= out[:, 1:]
        out[:, 1:] |= out[:, :-1]
    return out


def _binary_erode(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        out[:-1, :] &= out[1:, :]
        out[1:, :] &= out[:-1, :]
        out[:, :-1] &= out[:, 1:]
        out[:, 1:] &= out[:, :-1]
    return out


def _label(mask: np.ndarray) -> "tuple[np.ndarray, int]":
    """4-connected connected components. Tries scipy, falls back to a BFS."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        return lab, int(n)
    except Exception:
        pass
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    nxt = 0
    stack: list[tuple[int, int]] = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or lab[sy, sx]:
                continue
            nxt += 1
            lab[sy, sx] = nxt
            stack.append((sy, sx))
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if (0 <= ny < h and 0 <= nx < w
                            and mask[ny, nx] and not lab[ny, nx]):
                        lab[ny, nx] = nxt
                        stack.append((ny, nx))
    return lab, nxt


def _order_rows(boxes: list[tuple], row_tol: float) -> list[tuple]:
    """Sort boxes into reading order: rows top-to-bottom, then left-to-right
    within a row (rows detected by y-centre proximity)."""
    if not boxes:
        return []
    heights = [b[3] - b[1] for b in boxes]
    tol = row_tol * float(np.median(heights))
    boxes = sorted(boxes, key=lambda b: b[1])
    rows: list[list[tuple]] = []
    for b in boxes:
        cy = (b[1] + b[3]) / 2
        if rows and cy - (rows[-1][0][1] + rows[-1][0][3]) / 2 <= tol:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for row in rows for b in sorted(row, key=lambda b: b[0])]


def find_tile_boxes(arr: np.ndarray, thresh: float, close_r: int,
                    min_frac: float, max_frac: float,
                    row_tol: float) -> list[tuple]:
    """Reading-ordered (x0,y0,x1,y1) boxes of the non-black blobs in an
    (h,w,3) array, keeping only blobs near the median blob area."""
    mask = _luma(arr) > thresh
    if close_r:
        mask = _binary_erode(_binary_dilate(mask, close_r), close_r)
    lab, n = _label(mask)
    if n == 0:
        return []
    areas = np.bincount(lab.ravel())
    areas[0] = 0
    med = float(np.median(areas[areas > 0]))
    lo, hi = min_frac * med, max_frac * med
    boxes = []
    for i in range(1, n + 1):
        if not (lo <= areas[i] <= hi):
            continue
        ys, xs = np.where(lab == i)
        boxes.append((int(xs.min()), int(ys.min()),
                      int(xs.max()) + 1, int(ys.max()) + 1))
    return _order_rows(boxes, row_tol)


def _crop_square(sheet: "pygame.Surface", box: tuple, size: int) -> "pygame.Surface":
    x0, y0, x1, y1 = box
    cell = sheet.subsurface(pygame.Rect(x0, y0, x1 - x0, y1 - y0)).copy()
    s = max(cell.get_size())
    sq = pygame.Surface((s, s), pygame.SRCALPHA)
    sq.blit(cell, ((s - cell.get_width()) // 2, (s - cell.get_height()) // 2))
    return pygame.transform.smoothscale(sq, (size, size))


def _sheet_array(sheet: "pygame.Surface") -> np.ndarray:
    return pygame.surfarray.array3d(sheet).transpose(1, 0, 2)   # (h, w, 3)


def extract_islands(sheet: "pygame.Surface", thresh: float, close_r: int,
                    min_frac: float, max_frac: float, size: int,
                    row_tol: float) -> list["pygame.Surface"]:
    boxes = find_tile_boxes(_sheet_array(sheet), thresh, close_r,
                            min_frac, max_frac, row_tol)
    return [_crop_square(sheet, b, size) for b in boxes]


def _content_bands(profile: np.ndarray, min_gap: int,
                   min_run: int) -> list[tuple]:
    """1-D bool 'column/row has content' -> list of (start, end) content runs.
    Runs closer together than min_gap are merged (a thin seam is not a set
    separator); runs shorter than min_run are dropped (specks)."""
    idx = np.where(profile)[0]
    if len(idx) == 0:
        return []
    runs = []
    s = p = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v - p > min_gap:
            runs.append((s, p + 1))
            s = v
        p = v
    runs.append((s, p + 1))
    return [(a, b) for a, b in runs if b - a >= min_run]


def extract_sets(sheet: "pygame.Surface", thresh: float, set_gap: int,
                 set_cols: int, set_rows: int, size: int,
                 close_r: int) -> list[tuple]:
    """Master holds a grid of sets separated by wide (irregular) black bands;
    each set is itself a clean set_cols x set_rows tile grid with no real
    internal black. Split the master on the black bands (row/column content
    profile), rescale each set region to exactly (set_cols*size,
    set_rows*size), then rigid-slice that into tiles.

    Returns [(packed_sheet, [tile, ...]), ...] in reading order."""
    mask = _luma(_sheet_array(sheet)) > thresh
    if close_r:
        mask = _binary_erode(_binary_dilate(mask, close_r), close_r)
    xs = _content_bands(mask.any(axis=0), set_gap, size // 2)
    ys = _content_bands(mask.any(axis=1), set_gap, size // 2)
    tw, th = set_cols * size, set_rows * size

    out = []
    for (y0, y1) in ys:
        for (x0, x1) in xs:
            if mask[y0:y1, x0:x1].mean() < 0.05:
                continue
            region = sheet.subsurface(
                pygame.Rect(x0, y0, x1 - x0, y1 - y0)).copy()
            packed = pygame.transform.smoothscale(region, (tw, th))
            tiles = [packed.subsurface(
                        pygame.Rect(c * size, r * size, size, size)).copy()
                     for r in range(set_rows) for c in range(set_cols)]
            out.append((packed, tiles))
    return out


def _content_bbox(cell: np.ndarray, thresh: float, pad: int) -> "tuple | None":
    """(x0, y0, x1, y1) of the non-black region of an (h, w, 3) array, or
    None if the whole cell is below threshold."""
    mask = _luma(cell) > thresh
    if not mask.any():
        return None
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    h, w = mask.shape
    return (max(0, x0 - pad), max(0, y0 - pad),
            min(w, x1 + pad), min(h, y1 + pad))


def slice_sheet(path: pathlib.Path, cols: int, rows: int, size: int,
                prefix: str, out_dir: pathlib.Path, rigid: bool,
                thresh: float, pad: int) -> list[pathlib.Path]:
    if not pygame.get_init():
        pygame.init()
    sheet = pygame.image.load(str(path))
    try:
        sheet = sheet.convert_alpha()
    except pygame.error:
        pygame.display.set_mode((1, 1))
        sheet = sheet.convert_alpha()

    W, H = sheet.get_size()
    cw, ch = W / cols, H / rows
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []

    for r in range(rows):
        for c in range(cols):
            x, y = int(round(c * cw)), int(round(r * ch))
            w, h = int(round((c + 1) * cw)) - x, int(round((r + 1) * ch)) - y
            cell = sheet.subsurface(pygame.Rect(x, y, w, h)).copy()

            if not rigid:
                arr = pygame.surfarray.array3d(cell).transpose(1, 0, 2)
                bb = _content_bbox(arr, thresh, pad)
                if bb is None:
                    print(f"  skip {r}{c}: empty")
                    continue
                bx0, by0, bx1, by1 = bb
                cell = cell.subsurface(
                    pygame.Rect(bx0, by0, bx1 - bx0, by1 - by0)).copy()
                # letterbox to a square so the rescale keeps aspect ratio
                s = max(cell.get_size())
                sq = pygame.Surface((s, s), pygame.SRCALPHA)
                sq.blit(cell, ((s - cell.get_width()) // 2,
                               (s - cell.get_height()) // 2))
                cell = sq

            tile = pygame.transform.smoothscale(cell, (size, size))
            fp = out_dir / f"{prefix}_{r}{c}.png"
            pygame.image.save(tile, str(fp))
            written.append(fp)
    return written


def _mean_colour(fp: pathlib.Path) -> tuple[int, int, int]:
    s = pygame.image.load(str(fp))
    a = pygame.surfarray.array3d(s)
    return tuple(int(v) for v in a.reshape(-1, 3).mean(axis=0))


def _toml_stanzas(paths: list[pathlib.Path], out_dir: pathlib.Path,
                  sheet_root: pathlib.Path, group_of=None) -> str:
    lines = []
    for fp in paths:
        tid = fp.stem
        rel = fp.relative_to(sheet_root).as_posix()
        cr, cg, cb = _mean_colour(fp)
        grp = group_of(fp) if group_of else ""
        lines += [
            f"[tiles.{tid}]",
            f'png = "{rel}"',
            'layer = "floor"',
            f'name = "{tid.replace("_", " ")}"',
            *([f'group = "{grp}"'] if grp else []),
            "is_see_through = true",
            "is_walkable = true",
            "blocks_los = false",
            "blocks_shots = false",
            f"colour = [{cr}, {cg}, {cb}]",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet", type=pathlib.Path)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--size", type=int, default=128, help="output px (tile_px)")
    ap.add_argument("--prefix", default=None, help="file stem (default: sheet name)")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("assets/tiles/floor"))
    ap.add_argument("--rigid", action="store_true",
                    help="keep black borders: plain N-up crop, no autocrop")
    ap.add_argument("--thresh", type=float, default=24.0,
                    help="luma above this counts as content (0-255)")
    ap.add_argument("--pad", type=int, default=1,
                    help="px of margin to keep around detected content")
    ap.add_argument("--toml", action="store_true",
                    help="print tileset.toml stanzas for the sliced tiles")
    ap.add_argument("--auto", action="store_true",
                    help="find tiles as connected non-black blobs (irregular gutters)")
    ap.add_argument("--sets", action="store_true",
                    help="master holds a grid of sets: split on black bands, "
                         "rescale each set, rigid-slice into set-cols x set-rows")
    ap.add_argument("--set-cols", type=int, default=4)
    ap.add_argument("--set-rows", type=int, default=4)
    ap.add_argument("--set-gap", type=int, default=8,
                    help="[sets] min black-band width (px) that separates sets")
    ap.add_argument("--close", type=int, default=2,
                    help="[auto] px radius to bridge hairline cracks in tile art")
    ap.add_argument("--min-frac", type=float, default=0.35,
                    help="[auto] keep blobs at least this fraction of median area")
    ap.add_argument("--max-frac", type=float, default=2.5,
                    help="[auto] keep blobs at most this fraction of median area")
    ap.add_argument("--row-tol", type=float, default=0.6,
                    help="[auto] row grouping tolerance, fraction of median height")
    ap.add_argument("--expect", type=int, default=0,
                    help="warn if the tile count is not this")
    ap.add_argument("--start-index", type=int, default=0,
                    help="[auto] first running index for output names")
    args = ap.parse_args()

    prefix = args.prefix or args.sheet.stem
    root = pathlib.Path("assets/tiles")
    tile_group: dict = {}          # output path -> editor palette group

    def _load(path: pathlib.Path) -> "pygame.Surface":
        if not pygame.get_init():
            pygame.init()
        s = pygame.image.load(str(path))
        try:
            return s.convert_alpha()
        except pygame.error:
            pygame.display.set_mode((1, 1))
            return s.convert_alpha()

    if args.sets:
        sheet = _load(args.sheet)
        groups = extract_sets(sheet, args.thresh, args.set_gap, args.set_cols,
                              args.set_rows, args.size, args.close)
        per = args.set_cols * args.set_rows
        args.out.mkdir(parents=True, exist_ok=True)
        paths = []
        print(f"found {len(groups)} sets ({args.set_cols}x{args.set_rows} tiles each)"
              + ("" if len(groups) else "  <-- raise/lower --set-gap"))
        for si, (packed, tiles) in enumerate(groups):
            pygame.image.save(packed, str(args.out / f"{prefix}_set{si:02d}.png"))
            for k, t in enumerate(tiles):
                r, c = divmod(k, args.set_cols)
                fp = args.out / f"{prefix}_{si:02d}_{r}{c}.png"
                pygame.image.save(t, str(fp))
                paths.append(fp)
                tile_group[fp] = f"{prefix}_{si:02d}"
        print(f"wrote {len(paths)} tiles + {len(groups)} set sheets to {args.out}/")
    elif args.auto:
        sheet = _load(args.sheet)
        tiles = extract_islands(sheet, args.thresh, args.close,
                                args.min_frac, args.max_frac, args.size,
                                args.row_tol)
        args.out.mkdir(parents=True, exist_ok=True)
        paths = []
        for k, t in enumerate(tiles, start=args.start_index):
            fp = args.out / f"{prefix}_{k:03d}.png"
            pygame.image.save(t, str(fp))
            paths.append(fp)
        print(f"found {len(tiles)} tiles, wrote to {args.out}/")
    else:
        paths = slice_sheet(args.sheet, args.cols, args.rows, args.size,
                            prefix, args.out, args.rigid, args.thresh, args.pad)
        print(f"wrote {len(paths)} tiles to {args.out}/")

    if args.expect and len(paths) != args.expect:
        print(f"WARNING: expected {args.expect}, got {len(paths)} - "
              f"adjust --thresh / --close / --min-frac / --max-frac")
    if args.toml and paths:
        print("\n# --- paste into assets/tiles/tileset.toml ---\n")
        print(_toml_stanzas(paths, args.out, root,
                            group_of=tile_group.get if tile_group else None))


if __name__ == "__main__":
    main()
    pygame.quit()
