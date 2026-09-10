"""Regenerate the tile entries in assets/tiles/tileset.toml from the PNG
files on disk, so dropping sliced tiles into a folder is all it takes.

    python tools/sync_tileset.py                 # sync assets/tiles/*.png
    python tools/sync_tileset.py --dry-run       # print, don't write
    python tools/sync_tileset.py --dir assets/tiles/metal --dir assets/tiles/deco

Tiles are pure art now - a stanza is just png / name / colour / group. How a
painted cell behaves (Floor vs Wall) is the editor's toggle, not the tile.

Everything in tileset.toml ABOVE the marker line is left untouched (the
[meta] block and the hand-authored tiles - blank, door, window, bush).
Everything below it is rewritten from the managed folders each run, so
deleting a PNG drops its entry too.

Per file: id = sanitised stem, name = stem with _ as spaces, png = path
relative to assets/tiles, colour = mean of the non-transparent pixels,
group = the stem minus a trailing _<digits> or _<r><c> (so metal_03_12 ->
group "metal_03", metal_007 -> group "metal"), overridable with --group /
--no-group.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pygame

TS_ROOT = ROOT / "assets" / "tiles"          # anchored to the repo, not the CWD
TOML_PATH = TS_ROOT / "tileset.toml"
MARKER = ("# === auto-generated tiles below (tools/sync_tileset.py) — "
          "edits here are overwritten ===")

def _mean_colour(path: pathlib.Path) -> tuple[int, int, int]:
    surf = pygame.image.load(str(path))
    try:
        surf = surf.convert_alpha()
    except pygame.error:
        pygame.display.set_mode((1, 1))
        surf = surf.convert_alpha()
    rgb = pygame.surfarray.array3d(surf).reshape(-1, 3).astype("float64")
    try:
        a = pygame.surfarray.array_alpha(surf).reshape(-1).astype("float64")
    except (ValueError, pygame.error):
        a = None
    if a is not None and a.sum() > 0:
        w = a / a.sum()
        m = (rgb * w[:, None]).sum(axis=0)
    else:
        m = rgb.mean(axis=0)
    return tuple(int(round(v)) for v in m)


def _group_of(stem: str) -> str:
    head, _, tail = stem.rpartition("_")
    return head if (head and tail.isdigit()) else ""


def _sanitise(stem: str) -> str:
    s = re.sub(r"\W+", "_", stem).strip("_")
    return s or "tile"


def _stanza(path: pathlib.Path, group_mode: str,
            fixed_group: str) -> tuple[str, str]:
    stem = path.stem
    tid = _sanitise(stem)
    if group_mode == "none":
        grp = ""
    elif group_mode == "fixed":
        grp = fixed_group
    else:
        grp = _group_of(stem)
    rel = os.path.relpath(path.resolve(), TS_ROOT).replace("\\", "/")
    cr, cg, cb = _mean_colour(path)
    lines = [
        f"[tiles.{tid}]",
        f'png = "{rel}"',
        f'name = "{stem.replace("_", " ")}"',
    ]
    if grp:
        lines.append(f'group = "{grp}"')
    lines += [f"colour = [{cr}, {cg}, {cb}]", ""]
    return tid, "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toml", type=pathlib.Path, default=TOML_PATH)
    ap.add_argument("--dir", type=pathlib.Path, action="append",
                    dest="dirs", metavar="PATH",
                    help="folder of PNGs to manage (repeatable; "
                         "default: assets/tiles)")
    ap.add_argument("--group", default=None,
                    help="force this group on every synced tile")
    ap.add_argument("--no-group", action="store_true",
                    help="do not emit any group line")
    ap.add_argument("--exclude", action="append", default=None,
                    metavar="GLOB",
                    help="skip files matching this glob (repeatable; "
                         "default: *_set*.png)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dirs = args.dirs or [TS_ROOT]
    excludes = args.exclude if args.exclude is not None else ["*_set*.png"]
    group_mode = ("none" if args.no_group
                  else "fixed" if args.group else "auto")

    if not pygame.get_init():
        pygame.init()

    files: list[pathlib.Path] = []
    for d in dirs:
        if not d.exists():
            print(f"warn: {d} does not exist, skipping")
            continue
        for p in sorted(d.rglob("*.png")):
            if any(fnmatch.fnmatch(p.name, g) for g in excludes):
                continue
            files.append(p)
    if not files:
        print("no PNGs found in:", ", ".join(str(d) for d in dirs))
        return

    seen: dict[str, pathlib.Path] = {}
    stanzas: list[str] = []
    for p in files:
        tid, text = _stanza(p, group_mode, args.group or "")
        if tid in seen:
            print(f"warn: duplicate id {tid!r} ({p.name} vs {seen[tid].name}) "
                  f"- last wins")
        seen[tid] = p
        stanzas.append(text)

    old = args.toml.read_text(encoding="utf-8") if args.toml.exists() else ""
    head = old.split(MARKER, 1)[0].rstrip() if MARKER in old else old.rstrip()
    if not head:
        print(f"error: {args.toml} has no content above the marker; refusing "
              f"to clobber the [meta]/hand-authored section")
        return

    # a tile hand-authored above the marker always wins - skip its file
    hand_ids = set(re.findall(r"^\[tiles\.([A-Za-z0-9_]+)\]", head, re.M))
    before = len(stanzas)
    stanzas = [s for s in stanzas
               if re.match(r"\[tiles\.([A-Za-z0-9_]+)\]", s).group(1)
               not in hand_ids]
    if before - len(stanzas):
        print(f"note: {before - len(stanzas)} file(s) skipped - already defined "
              f"by hand above the marker")

    body = f"{head}\n\n{MARKER}\n\n" + "\n".join(stanzas)
    if not body.endswith("\n"):
        body += "\n"

    groups = sorted({_group_of(p.stem) for p in files} - {""})
    summary = (f"{len(files)} tiles, {len(groups)} group(s): "
               f"{', '.join(groups) if groups else '(none)'}")

    if args.dry_run:
        print(body)
        print("\n# " + summary + "  (dry run, nothing written)")
        return

    args.toml.write_text(body, encoding="utf-8")
    print(f"wrote {args.toml}: {summary}")

    try:
        from sim.tileset import load_tileset
        ts = load_tileset(args.toml)
        print(f"validated: {len(ts.tiles)} tiles load OK")
    except Exception as exc:
        print(f"WARNING: tileset.toml failed to load after write: {exc}")


if __name__ == "__main__":
    main()
    pygame.quit()
