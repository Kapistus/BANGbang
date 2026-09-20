"""Generate placeholder tile PNGs from assets/tiles/tileset.toml.

Each tile gets a flat-colour square with a simple motif so the map editor and
game have something to draw. Replace the PNGs with real art at the same size
(meta.tile_px) and this script is no longer needed.

    python tools/gen_sprites.py            # fill in what is MISSING
    python tools/gen_sprites.py --force --only door,blast_door
    python tools/gen_sprites.py --force    # redraw everything (destructive)

Missing-only is the default, and it matters: the editor calls this
automatically whenever any one PNG is absent, so adding a single tile to
tileset.toml used to redraw every sliced tile in the folder as a flat square.
That is how 256 pieces of real art were once replaced by grey. Real art wins
over a placeholder unless somebody explicitly asks otherwise.

    python tools/slice_sheet.py            # is what puts real art there
"""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pygame  # noqa: E402

from sim.tileset import load_tileset  # noqa: E402


def _shade(c, f):
    return tuple(max(0, min(255, int(x * f))) for x in c[:3])


def draw_tile(surf: "pygame.Surface", tid: str, col) -> None:
    px = surf.get_width()
    surf.fill(col)
    dk, lt = _shade(col, 0.72), _shade(col, 1.18)
    q = max(1, px // 4)

    if tid in ("floor", "concrete"):
        pygame.draw.rect(surf, dk, surf.get_rect(), 1)
    elif tid == "dirt":
        pygame.draw.rect(surf, dk, surf.get_rect(), 1)
        random.seed(3)
        for _ in range(px):
            surf.set_at((random.randrange(px), random.randrange(px)),
                        _shade(col, random.uniform(0.8, 1.1)))
    elif tid == "grating":
        for x in range(0, px + 1, max(3, px // 6)):
            pygame.draw.line(surf, dk, (x, 0), (x, px))
        for y in range(0, px + 1, max(3, px // 6)):
            pygame.draw.line(surf, dk, (0, y), (px, y))
    elif tid == "pillar":
        pygame.draw.circle(surf, dk, (px // 2, px // 2), px // 2 - 2)
        pygame.draw.circle(surf, lt, (px // 2, px // 2), px // 2 - 2, 2)
    elif tid == "wall":
        for i, y in enumerate(range(0, px, q)):
            pygame.draw.line(surf, dk, (0, y), (px, y))
            off = 0 if i % 2 == 0 else q * 2
            pygame.draw.line(surf, dk, (off, y), (off, y + q))
            pygame.draw.line(surf, dk, ((off + q * 2) % px, y),
                             ((off + q * 2) % px, y + q))
    elif tid == "thin_wall":
        pygame.draw.rect(surf, dk, (px // 2 - 3, 0, 6, px))
        pygame.draw.rect(surf, lt, (px // 2 - 3, 0, 6, px), 1)
    elif tid == "window":
        # a glazed opening seen from above: frame, four panes, and two glints
        # across the glass, which is what makes it read as glass and not as a
        # tiled floor
        fr = max(3, px // 10)
        bar = max(2, px // 28)
        glass = _shade(col, 1.10)
        frame = _shade(col, 0.42)
        surf.fill(frame)
        pygame.draw.rect(surf, glass, (fr, fr, px - fr * 2, px - fr * 2))
        glint = pygame.Surface((px, px), pygame.SRCALPHA)
        for off, wide, a in ((-px // 5, px // 7, 95), (px // 6, px // 14, 60)):
            pygame.draw.polygon(glint, (255, 255, 255, a), (
                (fr + off, px - fr), (fr + off + wide, px - fr),
                (fr + off + wide + (px - fr * 2), fr), (fr + off + (px - fr * 2), fr)))
        surf.blit(glint.subsurface((fr, fr, px - fr * 2, px - fr * 2)), (fr, fr))
        mid = px // 2
        pygame.draw.rect(surf, frame, (mid - bar // 2, fr, bar, px - fr * 2))
        pygame.draw.rect(surf, frame, (fr, mid - bar // 2, px - fr * 2, bar))
        pygame.draw.rect(surf, _shade(col, 0.30), surf.get_rect(), max(1, px // 32))
        pygame.draw.rect(surf, lt, (fr, fr, px - fr * 2, px - fr * 2), 1)
        pin = max(2, px // 24)
        for cx_, cy_ in ((fr, fr), (px - fr - pin, fr),
                         (fr, px - fr - pin), (px - fr - pin, px - fr - pin)):
            pygame.draw.rect(surf, _shade(col, 0.28), (cx_, cy_, pin, pin))
    elif tid in ("door", "blast_door"):
        # drawn by the renderer itself, shut, so the palette icon is literally
        # what a placed door looks like in the game rather than an impression
        # of one that can drift away from it
        from main import draw_door
        draw_door(surf, surf.get_rect(), 0.0, "h", heavy=(tid == "blast_door"))
    elif tid == "metal_grating":
        # parallel bars with slots between them, and a frame
        bar = max(2, px // 16)
        for y in range(bar * 2, px - bar, bar * 2):
            pygame.draw.rect(surf, dk, (bar, y, px - bar * 2, bar))
        pygame.draw.rect(surf, _shade(col, 0.55), surf.get_rect(), max(2, px // 24))
        pygame.draw.rect(surf, lt, surf.get_rect(), 1)
    elif tid == "metal_grid":
        # a square lattice: holes, not slots
        step = max(4, px // 10)
        for x in range(step, px, step):
            pygame.draw.line(surf, dk, (x, 2), (x, px - 3), max(1, px // 48))
        for y in range(step, px, step):
            pygame.draw.line(surf, dk, (2, y), (px - 3, y), max(1, px // 48))
        pygame.draw.rect(surf, _shade(col, 0.55), surf.get_rect(), max(2, px // 24))
    elif tid == "metal_yellow":
        # hazard stripes, which is what a yellow deck plate is for
        w = max(4, px // 8)
        dark = _shade(col, 0.35)
        for i in range(-px // w, px * 2 // w + 1):
            x = i * w * 2
            pygame.draw.polygon(surf, dark, ((x, 0), (x + w, 0),
                                             (x + w - px, px), (x - px, px)))
        pygame.draw.rect(surf, _shade(col, 0.5), surf.get_rect(), max(2, px // 24))
    elif tid == "low_cover":
        pygame.draw.rect(surf, dk, (3, q + 2, px - 6, px - q - 5))
        pygame.draw.rect(surf, lt, (3, q + 2, px - 6, px - q - 5), 2)
    elif tid == "crate":
        pygame.draw.rect(surf, dk, (3, 3, px - 6, px - 6), 2)
        pygame.draw.line(surf, dk, (3, 3), (px - 3, px - 3), 2)
        pygame.draw.line(surf, dk, (px - 3, 3), (3, px - 3), 2)
    else:
        pygame.draw.rect(surf, dk, surf.get_rect(), 2)


def main(tileset_path: str | None = None, force: bool = False,
         only: "set | None" = None) -> list[str]:
    """Render the placeholder PNGs for tiles that have none.

    Safe to call from an app that already has pygame running: it inits if
    needed and never quits. With `force`, redraws every tile it is allowed to
    touch, which OVERWRITES real art - only ever from an explicit command
    line. `only` limits it to those tile ids, which is how you redraw one
    motif without touching the rest of the folder."""
    if not pygame.get_init():
        pygame.init()
    ts = load_tileset(tileset_path)
    out_dir = ts.path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for td in ts.tiles.values():
        # png in a subfolder = hand-placed real art; never overwrite it.
        if "/" in td.png or "\\" in td.png:
            continue
        if only is not None and td.id not in only:
            continue
        if not force and (out_dir / td.png).exists():
            continue                    # something is already there: leave it
        surf = pygame.Surface((ts.tile_px, ts.tile_px))
        draw_tile(surf, td.id, tuple(td.colour))
        pygame.image.save(surf, str(out_dir / td.png))
        written.append(td.png)
    return written


if __name__ == "__main__":
    rest, only, force = [], None, False
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--force":
            force = True
        elif a == "--only":
            only = {t.strip() for t in next(it, "").split(",") if t.strip()}
        else:
            rest.append(a)
    pygame.init()
    names = main(rest[0] if rest else None, force=force, only=only)
    pygame.quit()
    if names:
        print(f"wrote {len(names)} sprites: {', '.join(names)}")
    else:
        print("nothing missing; pass --force to redraw over existing art")
