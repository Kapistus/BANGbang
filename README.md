# BANGbang

A top-down stealth game built in Python with pygame. Move a player through indoor environments from point A to point B while evading guards. The game is built around three mechanical pillars: sound propagation physics, line-of-sight fog of war, and enemy AI.

## Features

- **Sound propagation** — an Eikonal field solver models how sound travels through geometry. Guards and the player perceive sound as arriving from the route it traveled, not the true source, enabling misdirection. A two-layer model combines an omnidirectional ping ("something happened") with a propagating field for directional information.
- **Fog of war** — recursive shadowcasting with three-band vision cones (identify / recognise / peripheral) and remembered map state.
- **Enemy AI** — awareness is a leaky-bucket scalar with position estimates rather than binary detection. Guards patrol, emit footsteps, and react to perceived sound.
- **Fragile combat** — player health is low, but a meaningful combat option exists (e.g. a suppressed pistol with limited reach).
- **Maps** — defined via a character grid plus TOML tile definitions. Two maps ship: `arena.toml` (30×18m test space) and `compound.toml` (56×34m multi-room building with windows, thin walls, and gratings).

## Requirements

- **Python 3.14.0** (developed on Windows using the `py` launcher)
- Dependencies:
  - `pygame`
  - `numpy`
  - `numba` — JIT backend for the Eikonal solver

> Note: `scikit-fmm` is **not** used, as it is unavailable on CPython 3.14 (cp314). The Eikonal solver uses a numba JIT backend instead (~26× faster than pure Python, with disk-cached compilation).

## Installation

```bash
# Clone the repository
git clone <REPO_URL>
cd bangbang

# (Recommended) create and activate a virtual environment
py -3.14 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
py -m pip install -r requirements.txt
```

If there is no `requirements.txt`, install directly:

```bash
py -m pip install pygame numpy numba
```

## Running

```bash
py main.py
```

This launches the debug renderer, which includes ripple visualization, fog of war with memory, guard patrolling, enemy footstep emission, arrival-arc sound cues, weapon firing, door interaction, and a frame profiler.

> On first run, numba compiles the solver and caches it to disk; the initial launch will be slower than subsequent ones.

## Project structure
