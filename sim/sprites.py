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

# Player colours. The name is what the art file is suffixed with, the RGB is
# what the lobby swatch shows and what the ring under a body is drawn in.
#
# Colour is art, not a filter: a soldier in red is `soldier_ready_red.png`,
# painted as the sprite should look. Tinting the one khaki sprite was tried and
# looked exactly like what it was — a coloured pane held over a photograph. Any
# pose with no coloured version falls back to the base art, so the palette can
# be filled in one file at a time.
PALETTE = [
    ("red",    (220, 40, 40)),
    ("blue",   (40, 40, 220)),
    ("green",  (40, 200, 60)),
    ("yellow", (230, 210, 40)),
    ("orange", (240, 140, 30)),
    ("purple", (160, 60, 200)),
    ("cyan",   (40, 200, 220)),
    ("white",  (235, 235, 235)),
    ("grey",   (130, 130, 130)),
    ("black",  (20, 20, 20)),
]
SWATCHES = [rgb for _n, rgb in PALETTE]
COLOUR_POSES = ("soldier_ready", "soldier_idle", "soldier_ded")


def colour_name(rgb) -> str:
    """The palette name closest to an arbitrary RGB triple.

    Closest, not exact: a colour arrives over the wire as three numbers and
    nothing guarantees the sender used this palette."""
    r, g, b = (int(c) for c in rgb[:3])
    best, bd = PALETTE[0][0], None
    for name, (cr, cg, cb) in PALETTE:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if bd is None or d < bd:
            best, bd = name, d
    return best


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


def draw_muzzle(bank, screen, art: str, cx: float, cy: float, facing: float,
                ft: float, roll: float, scale: float) -> None:
    """Two-layer muzzle flash at a weapon's muzzle: big + small core early,
    fading core late. `cx, cy` is the weapon sprite's blit centre, `ft` the
    remaining flash time.

    One copy, because single-player and multiplayer must not disagree about
    what a gun going off looks like."""
    if ft <= 0.0 or not bank.ok or art not in MUZZLE_PX:
        return
    half = FLASH_TIME * 0.5
    if ft > half:
        bank.flash(screen, art, "big", cx, cy, facing, roll, scale * 1.15, 255)
        bank.flash(screen, art, "small", cx, cy, facing, -roll * 1.7, scale, 255)
    else:
        bank.flash(screen, art, "small", cx, cy, facing, roll, scale * 0.8,
                   int(210 * ft / half))


# Pickups have no art yet, so they are drawn as markers — but as ONE marker,
# here, rather than three lookalikes in the editor, single-player and the
# networked client.
PICKUP_COLOUR = {"health": (225, 70, 80), "ammo": (235, 190, 60)}


def draw_pickup(screen, cx: float, cy: float, kind: str, ppm: float,
                live: bool = True) -> None:
    """A health or ammo pack at world position (cx, cy) in pixels.

    `live` False draws the empty socket it will come back to, which is the
    information that matters while you wait for it."""
    col = PICKUP_COLOUR.get(kind, (200, 200, 200))
    r = max(5, int(0.30 * ppm))
    box = pygame.Rect(int(cx - r), int(cy - r), r * 2, r * 2)
    if not live:
        pygame.draw.rect(screen, tuple(c // 3 for c in col), box, 1,
                         border_radius=max(2, r // 3))
        return
    pygame.draw.rect(screen, (18, 18, 20), box.inflate(4, 4),
                     border_radius=max(2, r // 3))
    pygame.draw.rect(screen, col, box, border_radius=max(2, r // 3))
    if kind == "health":
        arm = max(1, r // 3)
        pygame.draw.rect(screen, (250, 250, 250),
                         (int(cx - arm // 2 - 1), int(cy - r + arm),
                          arm + 2, (r - arm) * 2))
        pygame.draw.rect(screen, (250, 250, 250),
                         (int(cx - r + arm), int(cy - arm // 2 - 1),
                          (r - arm) * 2, arm + 2))
    else:
        dk = tuple(int(c * 0.45) for c in col)
        for i in (-1, 1):
            pygame.draw.rect(screen, dk,
                             (box.x + 2, int(cy + i * r * 0.42) - 1,
                              box.w - 4, max(1, r // 4)))


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

    def body_art(self, pose: str, colour: "str | None") -> str:
        """`pose` in a player's colour if that art exists, else the base pose."""
        if colour:
            name = f"{pose}_{colour}"
            if name in self.raw:
                return name
        return pose

    def colours(self) -> list[str]:
        """Palette names that have a coloured `soldier_ready` loaded."""
        return [n for n, _rgb in PALETTE if f"soldier_ready_{n}" in self.raw]

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
             alpha: "int | None" = None,
             wash: "tuple[int, int, int] | None" = None) -> bool:
        """Draw a sprite. `tint` multiplies (used for light and shadow), `wash`
        adds (used for player colour).

        The difference matters on this art, which is dark khaki averaging about
        (59, 62, 40): multiplying by a player's colour can only take it further
        down, so saturated colours become silhouettes and the darker swatches
        become holes. Adding lifts the sprite toward the colour instead, keeps
        the art's own shading intact, and separates the palette far better."""
        surf = self.get(name, facing)
        if surf is None:
            return False
        if tint is not None or alpha is not None or wash is not None:
            surf = surf.copy()
            if tint is not None:
                surf.fill(tint + (255,), special_flags=pygame.BLEND_RGB_MULT)
            if wash is not None:
                # alpha component 0: transparent pixels stay transparent
                surf.fill(tuple(wash) + (0,), special_flags=pygame.BLEND_RGB_ADD)
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
