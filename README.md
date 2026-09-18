# BANGbang

Top-down stealth/action prototype in Python + pygame. Built around three mechanical pillars: sound propagation (an Eikonal field solver), line-of-sight fog of war with memory, and enemy AI.

## Features

- **Sound propagation** — an Eikonal field solver models how sound travels through geometry; perception uses arrival-time gradients so sounds arriving around corners appear to come from the route they travelled.
- **Fog of war** — recursive shadowcasting with three-band vision cones and remembered map state.
- **Enemy AI** — guards patrol, emit footsteps, and accumulate leaky-bucket awareness rather than binary detection.
- **Fragile combat** — low player health with meaningful weapon variety and trade-offs.
- **Maps** — character-grid + TOML tiles; example maps ship with the repo.

Status: **prototype**. Single-player works. Multiplayer works: two or more machines on a LAN share a map and fight on it, with the server authoritative over movement, shooting and damage, and the sound-propagation model running on every client. Guards, doors and mid-match joining are not networked yet; see [Work in progress](#work-in-progress).

---

## Requirements

Python 3.11+ (development tested on Python 3.14 on Windows; the `py` launcher is used on Windows). Dependencies include `numpy`, `pygame` and `msgpack`. `numba` is optional but strongly recommended as a JIT backend for the Eikonal solver (~26× speedup; a pistol field drops from ~115 ms to ~4 ms). `scikit-fmm` is not used here (Windows wheels only and slower).

Installation (recommended using a venv):

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

Check the sound backend you actually got:

```
py -c "from sim import sound; print(sound.backend_report())"
```

---

## Running

```
py main.py                      # defaults to maps/arena.toml
```

`main.py` is the debug renderer: walk a map, emit sounds, watch the field propagate, fight guards. Two map formats load — `.map` is the JSON editor format (`sim/mapfile.py`), anything else is treated as the legacy char-grid + TOML sidecar.

Controls in brief: WASD + mouse, shift run, ctrl crawl, Mouse1 fire, R reload, B fire mode, F interact, Space knock, wheel/1-7 weapon, V guard cones, F1-F11 overlays, Esc quit. The full list is the module docstring at the top of `main.py`.

Map editor:

```
py editor.py maps/foo.map
```

---

## Multiplayer

### Play

```
py mp_client.py                  # start screen: host here, or join
py mp_client.py --host           # host on this machine straight away
py mp_client.py 192.168.1.42     # join that host
py mp_client.py 192.168.1.42:47999
py mp_client.py --host --window 900x700     # two clients on one screen
```

`mp_client.py` is the networked game: start screen, lobby, then the match. Hosting runs the server inside the same process and connects your own client to it over loopback, so the host sits in the lobby like everyone else — the address other machines need is printed on the host screen and shown in the lobby.

In the lobby everyone sets name, colour, team (in team mode) and ready; the host also picks the map, the mode and the match length. The match starts when everyone is ready, or when the host forces it. Minimum two players.

In the match:

```
WASD / arrows   move        shift run        ctrl crawl
mouse           aim         left mouse fire (hold for auto, or to charge rail)
1-7 / wheel     weapon      r reload         b fire mode
f               open or close a door you are standing next to
tab             scoreboard  esc quit
```

The full seven-weapon loadout works — pistol, combat rifle, SMG, combat shotgun, rail rifle, laser rifle, rocket launcher — with magazines, reloads, fire modes, rail charge-up and rocket flight time. Damage runs through the same shields → armor → health model as single-player, bullets spend the same penetration budget against cover, and glass shatters for everyone at once.

Doors work: opening one changes what can be seen, shot and heard through it for everyone, and it will not close on a body standing in the leaf. Other players are hidden by fog of war exactly like anything else — if you cannot see into a room, you cannot see who is in it, and their tracers and muzzle flashes are hidden with them.

What you hear of another player arrives as an arc at your own position, pointing the way the sound travelled. Arrivals from the same direction refresh one arc rather than stacking: a sprinter emits five footsteps a second, and one arc per step rings you completely and tells you nothing.

### Testing on one machine

Two instances side by side, which is also the minimum player count:

```
py mp_client.py --host --window 900x700
py mp_client.py 127.0.0.1 --window 900x700
```

Only the focused window takes keyboard input; the other keeps rendering and its player stands still, which is correct — the server holds position when no input command arrives rather than inventing one.

### How it works

An authoritative server runs in a daemon thread inside the host process. TCP, one length-prefixed msgpack frame per message (`net/protocol.py`), default port **47801**, protocol version 3 — a mismatched client is rejected outright.

The server ticks at 30 Hz and broadcasts snapshots at 20 Hz. Input is a queue of fixed-size commands, one per tick, and the server applies at most one per tick, so simulated time can never run ahead of real time and a client that floods inputs gains no speed. It moves players with `sim/movement.py` and resolves shots with `sim/ballistics.py` — the same functions single-player uses.

The client predicts its own movement with that same function, keeps each command until a snapshot says the server has applied it, then takes the server's position as truth and replays whatever it had not yet seen. Because both sides step identical code by identical amounts, the correction is normally under a millimetre. Shooting is not predicted: the trigger goes to the server and the shot comes back, which on a LAN is imperceptible and keeps hit registration honest.

**Sound** is the one thing the server does not decide. It broadcasts what happened and how loud — a footstep every stride, at the energy your stance earns; a shot at the weapon's reach; a magazine, a dry click, a breaking pane. Each client then solves its own Eikonal field and asks `sim/perception.py` what reached it: how much energy survived the route, how long it took, and which direction it seems to come from, which is the negative gradient of arrival time rather than the straight line to the source. So a shot round a corner arrives late, quieter, and from the corner. A modified client could ignore all of that, but fog of war is already client-side, so it could already see through walls — this is the same trust model, and it keeps a 4 ms (or 115 ms without numba) solve off the tick loop.

Spawn points come from `net/maps.py`: whatever the map file declares (`spawn_points` in a `.map`, or `spawn_points = [[x, y], ...]` in a `.toml`), and otherwise derived by farthest-point sampling over standable ground, which is deterministic — every machine derives the same set.

LAN only. There is no matchmaking, no NAT punch-through, no relay. If the host machine has a firewall, allow inbound TCP on 47801.

### Headless harnesses

`run_host.py` and `run_client.py` are the no-pygame harnesses: they connect, ready up, run a match and print the scoreboard, but send no input, so nobody moves or shoots. To play, use `mp_client.py`.

```
py run_host.py --mode team --duration 300 --map compound
py run_client.py 192.168.178.26 --name Bob --colour blue
```

Console commands (`ready`, `unready`, `team a|b`, `start`, `end`, `score`, `quit`) are **ignored on Windows**: stdin polling uses POSIX-only `select()`. The connection and the match run fine regardless.

### Tests

```
py -m net.smoke_test        # lobby -> match -> kill -> respawn -> end -> lobby
py -m net.movement_test     # speeds, stance, input clamping, walls
py -m net.combat_test       # firing, damage, mags, cover, kills, glass,
                            #   doors, sound
py mp_smoke_test.py         # the real client, headless, for six seconds
py mp_smoke_test.py vessel_interior
```

All four are headless. `mp_smoke_test.py` drives the actual `MatchView` with scripted input against a dummy video driver, so it exercises rendering, prediction, shooting and the sound path without opening a window.

---

## Layout

```
main.py             debug renderer / single-player entry point
mp_client.py        networked client: host or join, lobby, match
editor.py           map editor (.map JSON format)
lobby.py            pygame start screen and lobby (host and join)
run_host.py         headless host harness
run_client.py       headless client harness
mp_smoke_test.py    headless test of the networked client
net/
  protocol.py       wire format, enums, message tags, input bits, kill verbs
  server.py         authoritative server: threads, ticks, movement, scoring
  client.py         client socket, world mirror, event queue
  maps.py           map id -> file, map loading, spawn point derivation
  combat_test.py    headless combat test
  mapcatalog.py     scans maps/ for playable maps (both formats)
  mappicker.py      host-only map selection panel
  smoke_test.py     headless netcode test
  movement_test.py  headless movement test
sim/
  movement.py       speeds, collision, input clamping — one definition, shared
  perception.py     loudness, and what a listener makes of a sound — shared
  sound.py          Eikonal solver, separate attenuation + travel-time fields
  vision.py         recursive shadowcasting, three-band vision cones
  ai.py             guard state machine (idle / patrol / search / combat)
  tilemap.py        char grid + TOML tiles -> per-property numpy arrays
  mapfile.py        .map JSON format
  ballistics.py     penetration, per-tile pen_cost budgets
  combat.py         shields -> armor -> health damage model
  weapons.py        weapon roster and fire modes
  audio.py          procedural placeholder SFX (no asset files)
  sprites.py        sprite loading
  tileset.py        tileset definitions
maps/               .map, .toml + .grid, tiles.toml
assets/             tiles, characters, weapons, source sheets
tools/              sprite slicing, tileset sync, sheet generation
mechanics/          legacy design docs, keybindings, older asset sets
```

---

## Work in progress

(abridged)

- No guards in multiplayer yet; guards would need server-side simulation.
- No mid-match join or spectating; latecomers are rejected with "match in progress".
- No stamina over the network; single-player drains/regenerates stamina.
- Missing visual and feedback polish compared to single-player (animations, lighting, etc.).
- Fog rendering costs can be high on large maps; consider blurring only the camera window.
- Sound is emitted authoritatively but heard client-side — deliberate trade-off for performance and simplicity.

---

For full details, controls, and the longer WIP list see the project files and module docstrings.
