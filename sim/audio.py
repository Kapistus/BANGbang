"""Procedural placeholder sound effects - no asset files.

Every clip is synthesised from noise and a couple of sine bodies with numpy
at import time, then handed to pygame.mixer. They are stand-ins: crunchy,
short, and good enough to feel the game until real audio exists.

Spatial playback piggybacks on the sound-propagation model. When an enemy
event reaches the player, main.py already knows the surviving energy (0..1)
and the bearing it arrived from; those become gain and stereo pan here, so
a shot two rooms away is quiet and off to one side without any extra work.

If there is no audio device, init() returns False and every play() call is
a no-op.
"""

from __future__ import annotations

import random

import numpy as np

try:
    import pygame
except ImportError:                       # pragma: no cover
    pygame = None

SR = 44100
_ready = False
_bank: dict[str, list] = {}


# --- synthesis helpers ------------------------------------------------

def _decay(n: int, tau_s: float, attack_s: float = 0.003) -> np.ndarray:
    t = np.arange(n) / SR
    env = np.exp(-t / max(tau_s, 1e-4))
    a = int(attack_s * SR)
    if a > 1:
        env[:a] *= np.linspace(0.0, 1.0, a)
    return env


def _lp(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def _sine(freq: float, n: int) -> np.ndarray:
    return np.sin(2.0 * np.pi * freq * np.arange(n) / SR)


def _norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) or 1.0
    return x / m * peak


def _gunshot(rng: random.Random, heavy: bool = False) -> np.ndarray:
    dur = rng.uniform(0.15, 0.22) * (1.4 if heavy else 1.0)
    n = int(dur * SR)
    r = np.random.default_rng(rng.randrange(1 << 30))
    body = _lp(r.uniform(-1, 1, n), 3 if heavy else 2) * _decay(n, 0.05 if heavy else 0.035)
    thump = _sine(rng.uniform(55, 85) if heavy else rng.uniform(80, 110), n) * _decay(n, 0.03)
    x = np.tanh((body * 1.3 + thump * 0.8) * 2.2)
    return _norm(x)


def _boom(rng: random.Random) -> np.ndarray:
    n = int(0.55 * SR)
    r = np.random.default_rng(rng.randrange(1 << 30))
    blast = _lp(r.uniform(-1, 1, n), 8) * _decay(n, 0.16)
    rumble = _sine(rng.uniform(38, 48), n) * _decay(n, 0.22)
    return _norm(np.tanh((blast * 1.5 + rumble) * 1.8))


def _railgun(rng: random.Random, light: bool = False) -> np.ndarray:
    """Rail discharge: an instant crack, a harsh electric arc buzz sweeping
    down, a bright spark fizz and a short metallic coil ring-out. No powder
    body - the point is that it is electric and sharp, not a gunshot."""
    n = int((0.10 if light else 0.14) * SR)
    r = np.random.default_rng(rng.randrange(1 << 30))
    t = np.arange(n) / SR

    crack = r.uniform(-1, 1, n) * _decay(n, 0.0030, 0.0003)       # instant transient

    sweep = np.linspace(rng.uniform(3400, 4400) * (0.9 if light else 1.0),
                        rng.uniform(520, 760), n)
    buzz = np.sign(np.sin(2 * np.pi * np.cumsum(sweep) / SR))     # square: rich harmonics
    am = 0.55 + 0.45 * np.sin(2 * np.pi * rng.uniform(85, 150) * t)
    arc = buzz * am * _decay(n, 0.011, 0.0006)

    ring = np.zeros(n)
    for fr in (rng.uniform(2500, 3100), rng.uniform(4400, 5400)):
        ring += _sine(fr, n) * _decay(n, rng.uniform(0.018, 0.032)) \
                * rng.uniform(0.12, 0.20)

    hiss = r.uniform(-1, 1, n)
    hiss = (hiss - _lp(hiss, 5)) * _decay(n, 0.006) * 0.5         # bright spark fizz

    x = crack * 1.6 + arc * (0.62 if light else 0.82) + ring + hiss
    return _norm(np.tanh(x * 2.4), 0.82 if light else 0.92)       # hard clip = edge


def _railcharge(rng: random.Random) -> np.ndarray:
    """A 3 s electric spool-up: the railgun character stretched out, pitch
    rising from a low hum to a high whine that peaks exactly at full charge."""
    dur = 3.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    p = (t / dur) ** 1.35                            # eased 0..1, arrives at the top
    f = 70.0 + p * (1500.0 - 70.0)
    ph = 2 * np.pi * np.cumsum(f) / SR
    # a mellow buzz: mostly sine with a little square edge (harsh square alone
    # reads as loud); modest fizz
    hum = 0.65 * np.sin(ph) + 0.35 * np.sign(np.sin(ph))
    hum2 = 0.28 * np.sin(1.5 * ph)                   # a fifth up, shimmer
    r = np.random.default_rng(rng.randrange(1 << 30))
    fizz = (r.uniform(-1, 1, n) - _lp(r.uniform(-1, 1, n), 6)) * 0.05 * p
    env = 0.10 + 0.70 * p ** 1.15                    # swells in, tops out gentler
    return _norm(np.tanh((hum * 0.7 + hum2 + fizz) * env * 1.5), 0.42)


def _shotgun(rng: random.Random) -> np.ndarray:
    """A wide, deep blast: mid boom + low punch + a short spray of pellet hiss.
    Boomier and lower than the generic gunshot, shorter than an explosion."""
    n = int(0.28 * SR)
    r = np.random.default_rng(rng.randrange(1 << 30))
    blast = _lp(r.uniform(-1, 1, n), 5) * _decay(n, 0.085)
    punch = _sine(rng.uniform(58, 78), n) * _decay(n, 0.05)
    spray = _lp(r.uniform(-1, 1, n), 2) * _decay(n, 0.02) * 0.35
    return _norm(np.tanh((blast * 1.6 + punch * 1.1 + spray) * 2.0))


def _zap(rng: random.Random) -> np.ndarray:
    n = int(0.13 * SR)
    t = np.arange(n) / SR
    f = np.linspace(rng.uniform(1100, 1400), rng.uniform(280, 360), n)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR)
    r = np.random.default_rng(rng.randrange(1 << 30))
    x = (tone * 0.8 + r.uniform(-1, 1, n) * 0.2) * _decay(n, 0.05)
    return _norm(x)


def _laser(rng: random.Random) -> np.ndarray:
    n = int(0.10 * SR)
    f = np.linspace(rng.uniform(1900, 2200), rng.uniform(700, 900), n)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * _decay(n, 0.035)
    return _norm(x)


def _plasma(rng: random.Random) -> np.ndarray:
    n = int(0.30 * SR)
    r = np.random.default_rng(rng.randrange(1 << 30))
    core = _lp(r.uniform(-1, 1, n), 5) * _decay(n, 0.11)
    low = _sine(rng.uniform(100, 130), n) * _decay(n, 0.14)
    return _norm(np.tanh((core + low) * 1.6))


def _footstep(rng: random.Random) -> np.ndarray:
    n = int(0.075 * SR)
    r = np.random.default_rng(rng.randrange(1 << 30))
    # heavy low-pass on the noise + a sub-bass thump = a low, dull footfall
    thud = _lp(r.uniform(-1, 1, n), 16) * _decay(n, 0.022)
    body = _sine(rng.uniform(46, 58), n) * _decay(n, 0.032) * 0.55
    return _norm(thud + body, 0.38)


def _clicks(rng: random.Random, k: int, spread: float, hi: bool) -> np.ndarray:
    gap = int(rng.uniform(0.035, 0.06) * SR)
    seg = int(0.02 * SR)
    n = gap * (k - 1) + seg + 4
    out = np.zeros(n)
    for i in range(k):
        r = np.random.default_rng(rng.randrange(1 << 30))
        c = r.uniform(-1, 1, seg) * _decay(seg, 0.004)
        if not hi:
            c = _lp(c, 3)
        s = i * gap
        out[s:s + seg] += c * rng.uniform(0.7, 1.0)
    return _norm(out, 0.7)


def _knock(rng: random.Random) -> np.ndarray:
    n = int(0.13 * SR)
    r = np.random.default_rng(rng.randrange(1 << 30))
    x = (_sine(rng.uniform(150, 200), n) * 0.7
         + _lp(r.uniform(-1, 1, n), 5) * 0.5) * _decay(n, 0.03)
    return _norm(x)


def _sweep(f0: float, f1: float, n: int) -> np.ndarray:
    """A sine gliding from f0 to f1 over n samples — a servo spinning up or
    winding down."""
    t = np.arange(n) / SR
    f = np.linspace(f0, f1, n)
    return np.sin(2.0 * np.pi * np.cumsum(f) / SR)


def _slide(rng: random.Random, dur: float, heavy: bool) -> np.ndarray:
    """A powered door moving: the pneumatic release, the servo carrying the
    panels, and the clunk of them arriving home.

    The old clip was a wooden thud plus a rumble, which is what a door on
    hinges sounds like. These are panels on rails, so the body of the sound is
    the travel itself — and a blast door's travel is long enough to hear all
    the way through."""
    n = int(dur * SR)
    r = np.random.default_rng(rng.randrange(1 << 30))

    # release: a sharp hiss of pressure at the start
    hiss_n = min(n, int(0.16 * SR))
    hiss = np.zeros(n)
    hiss[:hiss_n] = (r.uniform(-1, 1, hiss_n) - _lp(r.uniform(-1, 1, hiss_n), 14)
                     ) * _decay(hiss_n, 0.05 if not heavy else 0.09)

    # travel: a servo whine under a bed of rolling noise, both fading in and
    # out so the clip has a middle rather than two ends
    body = np.zeros(n)
    t0, t1 = int(n * 0.06), int(n * 0.93)
    tn = max(1, t1 - t0)
    ramp = np.sin(np.linspace(0.0, np.pi, tn)) ** 0.6
    lo = rng.uniform(58, 74) if heavy else rng.uniform(150, 185)
    servo = _sweep(lo * 0.82, lo, tn) * (0.55 if heavy else 0.4)
    roll = _lp(r.uniform(-1, 1, tn), 26 if heavy else 12) * (0.5 if heavy else 0.3)
    body[t0:t1] = (servo + roll) * ramp

    # arrival: the panels seating in the jamb
    clunk = np.zeros(n)
    cn = min(n, int(0.12 * SR))
    if cn > 8:
        seat = (_sine(rng.uniform(52, 68) if heavy else rng.uniform(96, 124), cn)
                * _decay(cn, 0.035) * (1.0 if heavy else 0.7))
        clunk[n - cn:] = seat

    return _norm(np.tanh((hiss * 0.5 + body + clunk) * 1.3))


def _door(rng: random.Random) -> np.ndarray:
    return _slide(rng, 0.42, heavy=False)


def _door_heavy(rng: random.Random) -> np.ndarray:
    # as long as the panels take: a blast door you can hear finish is a blast
    # door you can count down
    return _slide(rng, 5.0, heavy=True)


def _glass(rng: random.Random) -> np.ndarray:
    n = int(0.35 * SR)
    r = np.random.default_rng(rng.randrange(1 << 30))
    x = r.uniform(-1, 1, n) * _decay(n, 0.045)
    for _ in range(rng.randint(4, 7)):           # ringing shards
        f = rng.uniform(2600, 6200)
        s = rng.randint(0, n // 2)
        seg = n - s
        x[s:] += _sine(f, seg) * _decay(seg, rng.uniform(0.04, 0.1)) * rng.uniform(0.1, 0.3)
    return _norm(np.tanh(x * 1.4))


def _dry(rng: random.Random) -> np.ndarray:
    n = int(0.018 * SR)
    r = np.random.default_rng(rng.randrange(1 << 30))
    return _norm(_lp(r.uniform(-1, 1, n), 2) * _decay(n, 0.004), 0.6)


def _whoosh(rng: random.Random) -> np.ndarray:
    """A blade through air: filtered noise that swells as the arm comes round
    and cuts off at the end of the swing. No impact and no body - what you
    hear is the blade missing, which is most of the time."""
    n = int(rng.uniform(0.16, 0.22) * SR)
    r = np.random.default_rng(rng.randrange(1 << 30))
    t = np.linspace(0.0, 1.0, n)
    swell = np.sin(np.pi * t ** 0.75) ** 2        # in and out, peak mid-swing
    air = _lp(r.uniform(-1, 1, n), 9) * swell
    edge = _lp(r.uniform(-1, 1, n), 3) * swell * (t ** 2)   # brighter as it
    return _norm(air + edge * 0.4, 0.55)                    # passes you


def _throw(rng: random.Random) -> np.ndarray:
    """A grenade leaving a hand: cloth, and a short low rush of air. The bang
    is three seconds away and is a different sound entirely."""
    n = int(0.18 * SR)
    r = np.random.default_rng(rng.randrange(1 << 30))
    air = _lp(r.uniform(-1, 1, n), 14) * _decay(n, 0.06, attack_s=0.02)
    cloth = _lp(r.uniform(-1, 1, n), 5) * _decay(n, 0.015)
    return _norm(air + cloth * 0.5, 0.4)


# name -> (generator, how many random variants to pre-render)
_RECIPES = {
    "gunshot":       (lambda rng: _gunshot(rng, False), 4),
    "gunshot_heavy": (lambda rng: _gunshot(rng, True), 3),
    "railgun":       (lambda rng: _railgun(rng, False), 3),
    "railgun_light": (lambda rng: _railgun(rng, True), 3),
    "railcharge":    (_railcharge, 2),
    "shotgun":       (_shotgun, 3),
    "boom":          (_boom, 2),
    "zap":           (_zap, 3),
    "laser":         (_laser, 3),
    "plasma":        (_plasma, 2),
    "footstep":      (_footstep, 5),
    "reload":        (lambda rng: _clicks(rng, 3, 0.05, True), 3),
    "magazine":      (lambda rng: _clicks(rng, 2, 0.05, False), 3),
    "knock":         (_knock, 2),
    "door":          (_door, 2),
    "door_heavy":    (_door_heavy, 1),
    "glass":         (_glass, 3),
    "dryfire":       (_dry, 2),
    "whoosh":        (_whoosh, 3),
    "throw":         (_throw, 2),
}


def _to_sound(mono: np.ndarray):
    stereo = np.repeat((np.clip(mono, -1, 1) * 32767).astype(np.int16)[:, None], 2, axis=1)
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def init(seed: int = 1234) -> bool:
    """Bring up the mixer and render the clip bank. False if no audio device."""
    global _ready
    if _ready:
        return True
    if pygame is None:
        return False
    try:
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init(SR, -16, 2, 512)
            except pygame.error:
                pygame.mixer.init()
        pygame.mixer.set_num_channels(24)
    except Exception:
        return False
    rng = random.Random(seed)
    for name, (gen, count) in _RECIPES.items():
        try:
            _bank[name] = [_to_sound(gen(rng)) for _ in range(count)]
        except Exception:
            _bank[name] = []
    _ready = bool(_bank)
    return _ready


# Per-clip playback trim, applied on top of whatever gain the caller asks for,
# so a clip is quieter everywhere it plays - your own and other people's, in
# single-player and multiplayer - without touching how far the sound carries
# in the world (that is perception's reach, and it is what guards hear).
CLIP_GAIN = {
    "whoosh": 0.8,
    "throw": 0.6,
    "footstep": 0.75,
    "door": 0.5,
    "door_heavy": 0.375,   # 0.75, then halved with the quick door
}


def play(name: str, gain: float = 1.0, pan: float = 0.0) -> None:
    """Play a clip. `pan` is -1 (left) .. +1 (right); `gain` 0 .. 1."""
    gain *= CLIP_GAIN.get(name, 1.0)
    if not _ready or gain <= 0.02:
        return
    variants = _bank.get(name)
    if not variants:
        return
    ch = random.choice(variants).play()
    if ch is None:
        return
    g = max(0.0, min(1.0, gain))
    p = max(-1.0, min(1.0, pan)) * 0.85
    ch.set_volume(g * min(1.0, 1.0 - p), g * min(1.0, 1.0 + p))


_FIRE_BY_CATEGORY = {
    "ballistic": "gunshot", "heavy": "gunshot_heavy",
    "energy": "zap", "laser": "laser", "plasma": "plasma",
}

# per-weapon overrides where the category clip is uncharacteristic
_FIRE_BY_NAME = {
    "rail rifle": "railgun",
    "rail pistol": "railgun_light",
    "combat shotgun": "shotgun",
    "rocket launcher": "boom",      # the launch itself is the loud part
}


def fire_clip(weapon) -> str:
    """What pulling this trigger sounds like - not what the round does later.

    A weapon that puts something into the air is heard twice: once when it
    fires and once when the round goes off, and those are different sounds. A
    grenade leaving a hand is cloth and air; the bang belongs to the fuse.
    """
    cat = getattr(weapon, "category", "")
    if getattr(weapon, "melee_range", 0.0) > 0.0:
        return "whoosh"
    if getattr(weapon, "fuse_s", 0.0) > 0.0:
        return "throw"
    name = getattr(weapon, "name", "")
    if name in _FIRE_BY_NAME:
        return _FIRE_BY_NAME[name]
    # an explosive that goes off at the muzzle booms as it fires; one that
    # flies there fires with its own launch sound and booms on arrival
    if (getattr(weapon, "blast_r", 0.0) > 0.0
            and getattr(weapon, "projectile_speed", 0.0) <= 0.0
            and cat not in ("energy", "plasma", "laser")):
        return "boom"
    return _FIRE_BY_CATEGORY.get(cat, "gunshot")


def play_fire(weapon, gain: float = 1.0, pan: float = 0.0) -> None:
    play(fire_clip(weapon), gain, pan)


def play_channel(name: str, gain: float = 1.0, pan: float = 0.0):
    """Like play() but returns the pygame Channel so the caller can stop /
    fade it (used for the hold-to-charge rail spool-up). None if no audio."""
    gain *= CLIP_GAIN.get(name, 1.0)
    if not _ready or gain <= 0.02:
        return None
    variants = _bank.get(name)
    if not variants:
        return None
    ch = random.choice(variants).play()
    if ch is None:
        return None
    g = max(0.0, min(1.0, gain))
    p = max(-1.0, min(1.0, pan)) * 0.85
    ch.set_volume(g * min(1.0, 1.0 - p), g * min(1.0, 1.0 + p))
    return ch


# label prefix (from ActiveSound.label, e.g. "g1/fire") -> clip for enemy events
_ENEMY_CLIP = {"fire": "gunshot", "step": "footstep", "reload": "magazine",
               "blast": "boom", "throw": "throw", "melee": "whoosh"}


def enemy_clip(label: str) -> str:
    return _ENEMY_CLIP.get(label.rsplit("/", 1)[-1], "footstep")
