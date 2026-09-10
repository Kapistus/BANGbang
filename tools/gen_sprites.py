"""Generate placeholder tile PNGs from assets/tiles/tileset.toml.

Each tile gets a flat-colour square with a simple motif so the map editor
and game have something to draw. Replace the PNGs with real art at the same
size (meta.tile_px) and this script is no longer needed.

    python tools/gen_sprites.py
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
        pygame.draw.rect(surf, dk, surf.get_rect(), 3)
        pygame.draw.line(surf, dk, (px // 2, 0), (px // 2, px), 2)
        pygame.draw.line(surf, dk, (0, px // 2), (px, px // 2), 2)
        pygame.draw.line(surf, lt, (4, 4), (px // 2 - 3, px // 2 - 3), 2)
    elif tid == "door":
        pygame.draw.rect(surf, dk, surf.get_rect(), 2)
        for x in range(6, px, max(6, px // 4)):
            pygame.draw.line(surf, dk, (x, 3), (x, px - 3))
        pygame.draw.circle(surf, lt, (px - 8, px // 2), 2)
    elif tid == "low_cover":
        pygame.draw.rect(surf, dk, (3, q + 2, px - 6, px - q - 5))
        pygame.draw.rect(surf, lt, (3, q + 2, px - 6, px - q - 5), 2)
    elif tid == "crate":
        pygame.draw.rect(surf, dk, (3, 3, px - 6, px - 6), 2)
        pygame.draw.line(surf, dk, (3, 3), (px - 3, px - 3), 2)
        pygame.draw.line(surf, dk, (px - 3, 3), (3, px - 3), 2)
    elif tid == "bush":
        random.seed(hash(tid) & 0xFFFF)
        for _ in range(9):
            r = random.randint(px // 6, px // 4)
            pygame.draw.circle(
                surf, _shade(col, random.uniform(0.8, 1.25)),
                (random.randint(r, px - r), random.randint(r, px - r)), r)
    else:
        pygame.draw.rect(surf, dk, surf.get_rect(), 2)


def main(tileset_path: str | None = None) -> list[str]:
    """Render the placeholder PNGs. Safe to call from an app that already has
    pygame running - it inits if needed and never quits."""
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
        surf = pygame.Surface((ts.tile_px, ts.tile_px))
        draw_tile(surf, td.id, tuple(td.colour))
        pygame.image.save(surf, str(out_dir / td.png))
        written.append(td.png)
    return written


if __name__ == "__main__":
    pygame.init()
    names = main(sys.argv[1] if len(sys.argv) > 1 else None)
    pygame.quit()
    print(f"wrote {len(names)} sprites: {', '.join(names)}")
