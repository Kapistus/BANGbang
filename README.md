# BANGbang

Top-down stealth/action prototype in Python + pygame. The three mechanical
pillars are sound propagation (an Eikonal field, not a radius check), line-of-
sight fog of war with memory, and enemy AI that hears the same field the player
does.

Status: **prototype**. Single-player works. Multiplayer works: two or more
machines on a LAN share a map and fight on it, with the server authoritative
over movement, shooting and damage, and the sound-propagation model running on
every client. Guards, doors and mid-match joining are not networked yet; see
[Work in progress](#work-in-progress).

---

## Requirements

Python 3.11+ (`tomllib` is stdlib from 3.11). Development is on Python 3.14 on
Windows, where the launcher is `py`.

```
pip install -r requirements.txt
```

`numpy`, `pygame` and `msgpack` are required. `numba` is optional but strongly
recommended: it JIT-compiles the sound solver (~26x; a pistol field drops from
~115 ms to ~4 ms) and falls back cleanly if absent. `scikit-fmm` is commented
out — Windows wheels only, nothing for cp314, and slower here anyway.

Check the sound backend you actually got:

```
py -c "from sim import sound; print(sound.backend_report())"
```

---

## Single player

```
py main.py                      # defaults to maps/arena.toml
py main.py maps/vessel_interior.map
```

`main.py` is the debug renderer: walk a map, emit sounds, watch the field
propagate, fight guards. Two map formats load — `.map` is the JSON editor
format (`sim/mapfile.py`), anything else is treated as the legacy char-grid +
TOML sidecar.

Controls in brief: WASD + mouse, shift run, ctrl crawl, Mouse1 fire, R reload,
B fire mode, F interact, Space knock, wheel/1-7 weapon, V guard cones, F1-F11
overlays, Esc quit. The full list is the module docstring at the top of
`main.py`.

Map editor:

```
py editor.py maps/foo.map
```

Every editor hotkey is a single unshifted key on the number row or the top
letter row, so none of them need AltGr on a Nordic layout. Modes are
`q` paint, `e` entity, `r` light, `t` guard, `y` player spawn, `u` multiplayer
spawn. `w` swaps placing tiles as Floor or Wall, `i` toggles the grid, `o` the
roof preview, `p` pages the palette. `1` `2` and `3` `4` adjust whatever is
under the cursor — team tag and facing on a spawn, radius and intensity on a
lamp, kind on an entity, weapon on a guard — with shift taking the coarse step
on a lamp. `5` `6` cycle a guard's skill, `7` clears every guard, `8` saves
(shift = save as), `9` loads, `0` starts a new map. Enter, Backspace, Del,
arrows, space-drag and Esc are unchanged.

`e` enters entity mode and `1` `2` cycle the kind. Alongside npc, trader and
quest_item there are **health** and **ammo** packs — those two place straight
away with no prompts, since there is nothing to author on one, and they draw
with a ring at the radius a player has to reach to take it.

`u` enters multiplayer spawn mode: click to place a start point, the wheel (or
`3` `4`) turns it, `1` `2` tag it for team A, team B or anyone, right-click or
Del removes it. Spawns draw at the character's real collision size with a facing
line, and `main.py` warns on load about any that sit inside geometry — or about
a map that tags one team but not the other.

Two door tiles are in the palette — **Door (0.35s)** and **Blast door (5s)** —
and they replace the single door that used to be there. Set either into a wall
run: the panels retract into whatever is solid beside them, so a door with
walls to its left and right slides sideways and one with walls above and below
slides up and down. The editor turns the icon to match, so you can see which
way a door will open as you place it. A doorway with nothing either side still
works; it just picks an axis.

The same two tiles exist in the legacy char-grid format as `+` and `B` in
`maps/tiles.toml`. Adding a third kind is a tile entry with a different
`door_time`, not a code change.

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

`mp_client.py` is the networked game: start screen, lobby, then the match.
Hosting runs the server inside the same process and connects your own client to
it over loopback, so the host sits in the lobby like everyone else — the address
other machines need is printed on the host screen and shown in the lobby.

In the lobby everyone sets name, colour, team (in team mode) and ready; the host
also picks the map, the mode and the match length. The match starts when
everyone is ready, or when the host forces it. Minimum two players.

You can arrive late. Connect while a match is running and you drop straight into
it, at the spawn point furthest from anyone still fighting, with the doors and
broken windows as they stand rather than as the map file has them. Esc steps you
out again without dropping your connection: you land back in the lobby, your
score stays on the board, and a JOIN MATCH button puts you back in. In team mode
a latecomer goes to the thinner side. A match everybody walks out of ends by
itself rather than running on empty.

In the match:

```
WASD / arrows   move        shift run        ctrl crawl
mouse           aim         left mouse fire (hold for auto, or to charge rail)
1-7 / wheel     weapon      r reload         b fire mode
f               open or close a door you are standing next to
l               flashlight
tab             scoreboard  esc step out to the lobby (still connected)
```

The full seven-weapon loadout works — pistol, combat rifle, SMG, combat shotgun,
rail rifle, laser rifle, rocket launcher — with magazines, reloads, fire modes,
rail charge-up and rocket flight time. Ammo is **capped**: every gun carries
three to six spare magazines (the rocket launcher, three rockets ever), so a
firefight you cannot walk away from is one you have to finish. Damage runs through the same
shields → armor → health model as single-player, bullets spend the same
penetration budget against cover, and glass shatters for everyone at once.

Muzzle flashes are the single-player ones: two sprite layers pinned to the
barrel with a random roll and scale per shot, the gun kicking back and settling
over `RECOIL_TIME`, and a real light cast from the muzzle that briefly shows the
room — orange for slugthrowers, blue for rail, blue-green for plasma, cyan for
laser. A rocket detonating gets a bigger, longer one. The light is shadow-cast
from the flash, so it stops at walls, and it lifts the fog only while it burns:
a room you saw for a twelfth of a second by somebody's gunfire is not a room you
remember.

Players wear the colour they picked in the lobby as **art**: `soldier_ready` in
red is `assets/characters/soldier_ready_red.png`. A pose with no coloured
version falls back to the base sprite, so the palette can be filled one file at
a time (see *Colour art* under Work in progress). There is a ring of the same
colour at their feet either way — in an unlit room a body is a silhouette and
the ring is all there is to read. The gun stays gunmetal: a red rifle reads as a
toy.

Light works as it does in single-player, out of `sim/lighting.py`, which both
modes now call. What you can see is gated on what is lit: a room with no lamp
and no torch reveals nothing, so darkness is cover. Your flashlight (`l`) is a
16° beam cast from your own position, and everyone else's is cast from theirs —
you see the room their beam lights and the sweeping cone itself, drawn in a cool
white so it reads as somebody else's. Which cuts both ways: the lamp is also the
clearest thing on the map about where you are. A map with no lights baked into
it is dark, full stop — light is placed, not assumed, so an unlit map needs a
flashlight to play at all.

Health and ammo packs are placed in the editor and taken by walking over one.
A health pack is worth 35, an ammo pack tops up **every carried weapon by 30%
of its own cap**. A pack that would do nothing is not consumed — walk over a
health pack at full health and it stays there for whoever needs it. In
multiplayer the server decides who reached it first, and a taken pack comes
back after 30 seconds (health) or 20 (ammo), which turns the pack spots into
map control.

Doors are two steel panels that part along the wall and retract into the jambs.
There are two kinds, and the difference is time: a powered door (`+`) is open
in about a third of a second, and a blast door (`B`) takes five. Neither is
passable, transparent or quiet until the panels are fully home, in either
direction — a door part-way open is not a gap to squeeze through, and one
part-way shut is not a gap to dive back out of. That is the whole design of the
slow one: five seconds standing in the open, watching an amber lamp go yellow,
is a cost somebody can make you pay. A door already moving ignores the key,
so there is no cancelling it once it starts.

Opening one changes what can be seen, shot and heard through it for everyone,
and it will not close on a body standing in the doorway. Other players are
hidden by fog of war exactly like anything else — if you cannot see into a room,
you cannot see who is in it, and their tracers and muzzle flashes are hidden with
them. What their gunfire LIGHTS is still visible, which is usually the more
useful tell.

Each source keeps one reusable propagation field per kind of sound, so a weapon
on full auto costs about one field solve rather than one per round — measured at
56 shots per 2 solves — and a source that moves more than a metre gets a fresh
one. What you hear of another player arrives as an arc at your own position,
pointing the way the sound travelled. Arrivals from the same direction refresh one arc
rather than stacking: a sprinter emits five footsteps a second, and one arc per
step rings you completely and tells you nothing.

### Testing on one machine

Two instances side by side, which is also the minimum player count:

```
py mp_client.py --host --window 900x700
py mp_client.py 127.0.0.1 --window 900x700
```

Only the focused window takes keyboard input; the other keeps rendering and its
player stands still, which is correct — the server holds position when no input
command arrives rather than inventing one.

### How it works

An authoritative server runs in a daemon thread inside the host process. TCP,
one length-prefixed msgpack frame per message (`net/protocol.py`), default port
**47801**, protocol version 3 — a mismatched client is rejected outright.

The server ticks at 30 Hz and broadcasts snapshots at 20 Hz. Every connection
has an outbox drained by its own thread, so the tick loop never blocks on a
socket: one player whose connection backs up costs themselves their backlog and
nobody else their frame rate. Under pressure a queue sheds what is cheapest to
lose — a snapshot is a whole picture of the world, so only the newest is ever
queued, then tracers and sound events go, and state (lobby, kills, doors, glass)
is delivered or the client is cut off. Input is a queue of
fixed-size commands, one per tick, and the server applies at most one per tick,
so simulated time can never run ahead of real time and a client that floods
inputs gains no speed. It moves players with `sim/movement.py` and resolves
shots with `sim/ballistics.py` — the same functions single-player uses.

The client predicts its own movement with that same function, keeps each command
until a snapshot says the server has applied it, then takes the server's position
as truth and replays whatever it had not yet seen. Because both sides step
identical code by identical amounts, the correction is normally under a
millimetre. Shooting is not predicted: the trigger goes to the server and the
shot comes back, which on a LAN is imperceptible and keeps hit registration
honest.

**Sound** is the one thing the server does not decide. It broadcasts what
happened and how loud — a footstep every stride, at the energy your stance
earns; a shot at the weapon's reach; a magazine, a dry click, a breaking pane.
Each client then solves its own Eikonal field and asks `sim/perception.py` what
reached it: how much energy survived the route, how long it took, and which
direction it seems to come from, which is the negative gradient of arrival time
rather than the straight line to the source. So a shot round a corner arrives
late, quieter, and from the corner. A modified client could ignore all of that,
but fog of war is already client-side, so it could already see through walls —
this is the same trust model, and it keeps a 4 ms (or 115 ms without numba)
solve off the tick loop.

Spawn points are map objects. Place them in the editor (`m`), each with a facing
and an optional team tag; team modes then start each side on its own spawns, and
you arrive looking the way the spawn says rather than always due north. A map
that declares none falls back to farthest-point sampling over standable ground,
which is deterministic — every machine derives the same set — but knows only
that the ground is standable, not what can see it. That is the reason to place
them.

LAN only. There is no matchmaking, no NAT punch-through, no relay. If the host
machine has a firewall, allow inbound TCP on 47801.

### Headless harnesses

`run_host.py` and `run_client.py` are the no-pygame harnesses: they connect,
ready up, run a match and print the scoreboard, but send no input, so nobody
moves or shoots. To play, use `mp_client.py`.

```
py run_host.py --mode team --duration 300 --map compound
py run_client.py 192.168.178.26 --name Bob --colour blue
```

Console commands (`ready`, `unready`, `team a|b`, `start`, `end`, `score`,
`quit`) are **ignored on Windows**: stdin polling uses POSIX-only `select()`.
The connection and the match run fine regardless.

### Tile art

`assets/tiles/*.png` is real art, sliced from the sheets in
`assets/tiles/tilesets/` by `tools/slice_sheet.py`. `tools/gen_sprites.py`
fills in **missing** PNGs with flat placeholder motifs and leaves everything
else alone — it will not redraw art that is already there unless you ask:

```
py tools/gen_sprites.py                        # fill in what is missing
py tools/gen_sprites.py --force --only door    # redraw just these
py tools/gen_sprites.py --force                # redraw everything (destructive)
```

That default matters. The editor calls the generator automatically whenever any
one PNG is absent, so before this it only took adding a tile to `tileset.toml`
to flatten every sliced tile in the folder into a grey square.

The two door icons are drawn by the renderer itself (`main.draw_door`, shut),
so the palette icon is literally what a placed door looks like in game rather
than an impression of one that can drift.

`metal_grating`, `metal_grid` and `metal_yellow` are currently generated
motifs — bars, a lattice and hazard stripes — standing in for sliced art that
was lost. Re-slice them from a sheet and they stop being placeholders; nothing
else needs to change.

### Tests

```
py -m net.smoke_test        # lobby -> match -> kill -> respawn -> end -> lobby
py -m net.movement_test     # speeds, stance, input clamping, walls
py -m net.combat_test       # firing, damage, mags, cover, kills, glass,
                            #   doors, sound
py -m net.backpressure_test # a client that stops reading must not stall the
                            #   server for everyone else
py -m net.rejoin_test       # joining, leaving and rejoining a running match
py -m net.spawn_points_test # authored spawns: format, teams, facing, fallback
py -m net.doors_test        # slide axis, travel time, a moving door is a wall
py -m net.pickups_test      # ammo caps, the 30% rule, respawns, who got there
                            #   first
py mp_smoke_test.py         # the real client, headless, for six seconds
py mp_smoke_test.py vessel_interior
py editor_test.py           # the editor opens every map and draws every mode
```

All ten are headless. `mp_smoke_test.py` drives the actual `MatchView` with
scripted input against a dummy video driver, so it exercises rendering,
prediction, shooting, muzzle flashes and the sound path without opening a
window. It then runs four more passes: the lobby's late-join path; field
reuse — that a stationary shooter costs one solve, that a moving one does not
keep reusing a stale field, that footsteps never inherit a gunshot's loudness,
and that opening a door throws every cached field away; colour — that every
lobby swatch has a sprite set of its own and that a missing one falls back; and
lighting — that an unlit room hides a player, that a beam reveals only what it
lights, that another player's torch reaches you, and that a muzzle flash lights
a room without writing it into fog memory.

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
editor_test.py      headless test that the editor draws and edits
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
  spawn_points_test.py  authored spawns, teams, facing
  rejoin_test.py    joining, leaving and rejoining
  backpressure_test.py  a client that stops reading
sim/
  movement.py       speeds, collision, input clamping — one definition, shared
  perception.py     loudness, and what a listener makes of a sound — shared
  lighting.py       beams, point lights, and the light gate on sight — shared
  doors.py          sliding-door state machine: axis, travel time, arrival
  pickups.py        health/ammo packs: what they give, and when they return
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

### Multiplayer — what is still missing

1. **No guards, no AI.** Matches are player-versus-player only. Guards would
   need to run server-side and ship in snapshots, and they already hear the same
   sound field, so most of the work is netcode rather than behaviour.
2. **No spectating.** You can join, leave and rejoin a match, but not watch one
   you are not playing in — stepping out puts you on the lobby screen, not a
   camera. Worth deciding whether a spectator should see through the fog of war
   before building it, since in a stealth game that leaks positions to anyone
   sitting in the same room.
3. **No stamina.** Sprinting is unlimited over the network; single-player drains
   and regenerates it.
4. **No interactables beyond doors.** Single-player has traders, corpses, quest
   objects and an inventory behind the same key; multiplayer has doors only.
5. **Missing feedback the single-player renderer has:** blood hit-reactions,
   the weapon-swap animation frames, the rail spool-up sound while charging,
   and the slung/raise weapon states. Muzzle flashes, recoil, the flashlight
   and the shadow model are done.
7. **No ripple view.** Single-player can show the sound wavefront itself
   (F11 cycles arcs / full ripples / off, and ripples are its default);
   multiplayer draws the arrival arcs only.
8. **Fog costs more than it should on large maps.** The veil is rebuilt and
   blurred across the whole map array every frame; on the 80x52 m vessel map that
   is the dominant frame cost. Single-player has the same pattern. Blur only the
   camera window.
9. **Sound is emitted authoritatively but heard client-side**, which is a
   deliberate trade (see above) and the place to revisit first if this ever
   needs to be cheat-resistant.
10. **If the host quits, the match ends for everyone.** The server lives inside
    the host's process. The `is_host` flag migrates between players, but it
    cannot move the server, so nobody can take over; `net/smoke_test.py`'s
    migration check passes only because the test itself owns the server.

### Colour art

Players wear their colour as painted art rather than a filter over one sprite.
The loader globs every PNG under `assets/characters`, so a coloured set is a
drop-in: no code change, no registration.

Name a file `<pose>_<colour>.png` in `assets/characters/`, where `<pose>` is one
of `soldier_ready`, `soldier_idle`, `soldier_ded` and `<colour>` is one of the
ten palette names in `sim/sprites.py` — `red`, `blue`, `green`, `yellow`,
`orange`, `purple`, `cyan`, `white`, `grey`, `black`. Same 200 px canvas, same
centre, same up-facing orientation as the base art; the game only swaps which
file it draws.

Anything missing falls back to the base sprite, so a palette half-painted is
fine and one file is enough to see it working. `python mp_smoke_test.py` prints
which coloured sets it found. `sprites.PALETTE` is also what the lobby swatches
are built from, so adding a colour there adds it to the lobby — but a swatch
with no art is a soldier in khaki wearing a coloured ring.

### Map picker

9. No `.png` thumbnails exist in `maps/`, so every picker row renders black.
   `net/mapcatalog.py` pairs `map_1.map` with `map_1.png` when one is there.

### Art and animation

10. Sprite layering not built: the plan is a legs layer driven by movement
    direction, torso + weapon following mouse aim, weapon anchored to a per-frame
    torso grip point. Multiplayer draws the same `soldier_ready` pose
    single-player does, with a coloured ring so players stay tellable apart.
11. Locomotion unresolved. Procedural bob-on-idle is preferred over generated
    walk cycles — AI motion frames come out inconsistent, and DAIN-style
    interpolation was assessed and deprioritized.
12. Rocket backblast draws a flat placeholder cone; the three backblast sprites
    aren't blitted yet (`main.py`, TODO near line 2012).
13. All sound effects are procedural placeholders in `sim/audio.py`.

### Housekeeping

14. **Not a git repository.** No version control on the project folder.
15. `sim/` has no direct test coverage; the four headless tests cover the net
    path and reach into `sim/` through it.
16. `maps/` mixes formats and scratch files (`untitled.map`, `untitled2.map`,
    `arena_1x.*`), and they all show up in the map picker.
17. `mechanics/` is imported design material from an older project (some files
    dated 2018) kept as reference — the weapon roster and the
    shield/armor/accuracy model were ported from it into `sim/weapons.py` and
    `sim/combat.py`. It is not live code.
