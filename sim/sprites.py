"""Character and weapon sprites for the top-down renderer.

Art convention (matches assets/characters and assets/weapons):
  * every sprite is a square 200 px canvas, subject centred, pointing UP
    (toward -y in the image),
  * a weapon sprite is pre-registered to the character canvas: same size,
    grip at the centre, barrel up.

At draw time both the character pose and its weapon are rotated by the same
angle about the shared centre and blitted at the entity's screen position,
so they stay aligned with zero offset maths. Rotation is cached in small
angle steps.

The sim never touches any of this - hit boxes stay the 0.28 m circle.
"""

from __future__ import annotations

import math
from pathlib import Path

import pygame

ASSETS = Path(__file__).resolve().parent.parent / "assets"
CANVAS_PX = 200
ART_PPM = 78.0           # authored pixels per world metre; ~80 px soldier art = ~1 m
SPRITE_STEP = 3          # rotation cache granularity, degrees
MUZZLE_M = 1.10          # fallback muzzle offset (grip -> barrel tip, forward, m)
MUZZLE_RIGHT_M = 0.18    # ...and to the shooter's right (barrel sits off-centre)
                        # both used only for art files with no MUZZLE_PX entry
RECOIL_M = 0.18          # how far the gun sprite kicks back on firing
RECOIL_TIME = 0.09       # seconds for the kick to decay back to rest
SWAP_TIME = 0.55         # weapon-change animation length
FLASH_TIME = 0.05        # muzzle-flash lifetime, seconds (both layers, then core)

# Per-weapon-art muzzle flash placement. Value per layer is
#   (flash image stem, x, y)
# where (x, y) is the pixel position of the flash image's BOTTOM-LEFT corner
# inside the 200 px weapon canvas, top-left origin, art unrotated. The flash
# is pinned there, then the whole gun+flash composite rotates with facing.
MUZZLE_PX = {
    "smg": {
        "big":   ("flash_smg_big", 98, 46),
        "small": ("flash_smg_small", 109, 38),
    },
}

# game weapon id -> art file stem in assets/weapons
WEAPON_ART = {
    "pistol": "smg",
    "smg": "smg",
    "combat_shotgun": "shotgun",
    "flak_cannon": "shotgun",
    "combat_rifle": "battle_rifle",
    "heavy_rifle": "battle_rifle",
    "rail_pistol": "railgun",
    "rail_rifle": "railgun",
    "laser_pistol": "laser_rifle",
    "laser_rifle": "laser_rifle",
    "pulse_carbine": "laser_rifle",
    "plasma_pistol": "laser_rifle",
    "plasma_rifle": "laser_rifle",
    "rocket_launcher": "rocket_launcher",
    "flamethrower": "rocket_launcher",
}
_ID_BY_NAME: dict[str, str] = {}


def weapon_art(weapon_id: str) -> str:
    return WEAPON_ART.get(weapon_id, "battle_rifle")


def swap_frame(progress: float, new_weapon_id: str):
    """(character sprite, weapon sprite | None) during a weapon change, by
    progress 0..1. Sequence: sling -> idle -> sling -> ready + new weapon."""
    if progress < 0.30:
        return "soldier_idle", "weapon_sling"
    if progress < 0.55:
        return "soldier_idle", None
    if progress < 0.80:
        return "soldier_idle", "weapon_sling"
    return "soldier_ready", weapon_art(new_weapon_id)


def weapon_art_for(weapon_obj) -> str:
    """Art stem for a weapons.Weapon instance (guards hold the object, not the id)."""
    if not _ID_BY_NAME:
        from sim import weapons
        _ID_BY_NAME.update({v.name: k for k, v in weapons.ROSTER.items()})
    return weapon_art(_ID_BY_NAME.get(getattr(weapon_obj, "name", ""), ""))


def orient_deg(facing: float) -> float:
    """Degrees for pygame.transform to turn an up-pointing sprite to `facing`
    (world radians, screen y-down)."""
    return -math.degrees(facing) - 90.0


class SpriteBank:
    def __init__(self):
        self.raw: dict[str, pygame.Surface] = {}
        self._cache: dict[tuple[str, int], pygame.Surface] = {}
        self._muzzle: dict[tuple[str, str], tuple] = {}
        self.scale = 1.0
        self.ok = False

    def load(self) -> None:
        try:
            for sub in ("characters", "weapons"):
                for p in sorted((ASSETS / sub).glob("*.png")):
                    self.raw[p.stem] = pygame.image.load(str(p)).convert_alpha()
            self.ok = bool(self.raw)
        except Exception as exc:                       # pragma: no cover
            print("sprite load failed:", exc)
            self.ok = False

    def set_scale(self, scale: float) -> None:
        if abs(scale - self.scale) > 1e-4:
            self.scale = scale
            self._cache.clear()

    def get(self, name: str, facing: float) -> "pygame.Surface | None":
        raw = self.raw.get(name)
        if raw is None:
            return None
        deg = int(round(orient_deg(facing) / SPRITE_STEP) * SPRITE_STEP) % 360
        key = (name, deg)
        surf = self._cache.get(key)
        if surf is None:
            surf = pygame.transform.rotozoom(raw, deg, self.scale)
            self._cache[key] = surf
        return surf

    def blit(self, screen, name: str, cx: float, cy: float, facing: float,
             tint: "tuple[int, int, int] | None" = None,
             alpha: "int | None" = None) -> bool:
        surf = self.get(name, facing)
        if surf is None:
            return False
        if tint is not None or alpha is not None:
            surf = surf.copy()
            if tint is not None:
                surf.fill(tint + (255,), special_flags=pygame.BLEND_RGB_MULT)
            if alpha is not None:
                surf.set_alpha(alpha)
        screen.blit(surf, surf.get_rect(center=(int(cx), int(cy))))
        return True

    def _muzzle_off(self, art: str, layer: str) -> "tuple | None":
        """(raw flash surface, ox, oy) for one muzzle layer, or None.
        ox/oy is the flash centre offset from the weapon-canvas centre, in
        art pixels (+x right, +y down, before rotation)."""
        key = (art, layer)
        got = self._muzzle.get(key)
        if got is not None:
            return got
        spec = MUZZLE_PX.get(art, {}).get(layer)
        if spec is None:
            return None
        stem, x, y = spec
        raw = self.raw.get(stem)
        if raw is None:
            return None
        fw, fh = raw.get_size()
        ox = (x + fw / 2.0) - CANVAS_PX / 2.0
        oy = (y - fh / 2.0) - CANVAS_PX / 2.0
        got = (raw, ox, oy)
        self._muzzle[key] = got
        return got

    def muzzle_world(self, art: str, facing: float) -> "tuple[float, float]":
        """(dx, dy) world-METRE offset from the character centre to the gun
        muzzle, for this facing. Uses the same registered muzzle pixel the
        flash does; falls back to MUZZLE_M straight ahead if the art has none."""
        got = self._muzzle_off(art, "small") or self._muzzle_off(art, "big")
        if got is None:
            s, c = math.sin(facing), math.cos(facing)
            return (c * MUZZLE_M - s * MUZZLE_RIGHT_M,
                    s * MUZZLE_M + c * MUZZLE_RIGHT_M)
        _raw, ox, oy = got
        s, c = math.sin(facing), math.cos(facing)
        return ((-ox * s - oy * c) / ART_PPM, (ox * c - oy * s) / ART_PPM)

    def flash(self, screen, art: str, layer: str, cx: float, cy: float,
              facing: float, roll_deg: float = 0.0, scale_mul: float = 1.0,
              alpha: int = 255) -> bool:
        """Draw one muzzle-flash layer at the weapon's muzzle. `cx, cy` is the
        weapon sprite's blit centre (recoil kick already applied)."""
        got = self._muzzle_off(art, layer)
        if got is None:
            return False
        raw, ox, oy = got
        s, c = math.sin(facing), math.cos(facing)
        dx = (-ox * s - oy * c) * self.scale
        dy = (ox * c - oy * s) * self.scale
        surf = pygame.transform.rotozoom(
            raw, orient_deg(facing) + roll_deg, self.scale * scale_mul)
        if alpha < 255:
            surf.set_alpha(alpha)
        screen.blit(surf, surf.get_rect(center=(int(cx + dx), int(cy + dy))))
        return True
