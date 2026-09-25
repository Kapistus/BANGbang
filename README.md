# BANGbang

Top-down stealth/action prototype in Python + pygame. The three mechanical
pillars are sound propagation (an Eikonal field, not a radius check), line-of-
sight fog of war with memory, and enemy AI that hears the same field the player
does.

Status: **prototype**. Single-player works. Multiplayer works: two or more
machines on a LAN share a map and fight on it, with the server authoritative
over movement, shooting and damage, and the sound-propagation model running on
every client. Doors, pickups and joining a match already in progress are
networked; guards are not, so multiplayer is player-versus-player only. See
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
B fire mode, F interact, E knock, wheel/1-7 weapon, V guard cones, F1-F11
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

Saving runs the same check the lobby uses, and says whether the map will be
listed. A map that fails is still saved — you never lose work — but it's marked
**NOT PLAYABLE** in the status bar until it saves clean, and the full list of
problems goes to the console.

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

The window sizes itself to the display: the match view opens at the desktop
size less a margin for the title bar and the taskbar (`view_cap` in `main.py`,
about 1860x960 on a 1920x1080 screen, and the full width of anything larger),
falling back to 1860x776 if the desktop size cannot be read. The view is a
camera onto a world that is usually bigger than it at a fixed 48 pixels per
metre, so a taller window is about five more metres of map rather than a
bigger picture of the same one; `--window WxH` overrides it. The lobby is its
own 1280x800 window, which is what gives the class cards 200 px each.

`mp_client.py` is the networked game: start screen, lobby, then the match.
Hosting runs the server inside the same process and connects your own client to
it over loopback, so the host sits in the lobby like everyone else — the address
other machines need is printed on the host screen and shown in the lobby.

In the lobby everyone sets name, colour, team (in team mode) and ready; the host
also picks the map, the mode and the match length. The map list only offers
**playable** maps — ones that load, use only tiles the tileset has, and have at
least two usable spawns. Anything else is left out and the reason printed to the
console. If the host's map isn't playable (the default is `arena`), the server
switches to the first one that is and the lobby shows which; a match never
starts on a map the clients can't load. The match starts when
everyone is ready, or when the host forces it. Minimum two players.

Only the host needs the map. The lobby carries a fingerprint of the host's
map files, and a player whose copy is missing, or different under the same
name, downloads the host's copy. Downloads go into `maps/downloaded/<map>/`, so
they never overwrite a map of your own. A copy downloaded once is used again as
long as the host's map hasn't changed. While a player is downloading, their
ready dot in the lobby is hollow and readying up starts nothing; joining a
match in progress waits for the download too. The server only sends the map
the host has selected, and the client writes only plain `.map`, `.toml` and
`.grid` files into its own download folder, after checking them against the
fingerprint.

You can arrive late. Connect while a match is running and you drop straight into
it, at the spawn point furthest from anyone still fighting, with the doors and
broken windows as they stand rather than as the map file has them (after
choosing a class - see below). Leave match in the Esc menu steps you
out again without dropping your connection: you land back in the lobby, your
score stays on the board, and a JOIN MATCH button puts you back in. In team mode
a latecomer goes to the thinner side. A match everybody walks out of ends by
itself rather than running on empty.

In the match:

```
WASD / arrows   move        shift run        ctrl crawl
mouse           aim         left mouse fire (hold for auto, or to charge rail)
1-3 / wheel     weapon      r reload         b fire mode
f               open or close a door you are standing next to
e               knock on the wall (single-player has this on e too now)
space           your class ability
l               flashlight
tab             scoreboard  esc menu: resume, class, leave match
```

Sprinting runs on fuel, and both modes run the same model (`sim/movement.py`):
about six seconds of running empties the tank, standing still refills it in
five and walking in fourteen, and bottoming it out locks the sprint until it is
back to a quarter — which is what stops a player sprinting on fumes. The server
owns the number in multiplayer and ships it in the snapshot, so the bar under
your shields is the server's figure and not a guess; your own client predicts
it with the same function, so the bar does not jump.

In the Esc menu, up/down (or w/s) moves between Resume, Class and Leave match;
left/right, a/d or the mouse wheel steps through the classes; enter or a click
picks; Esc closes it. While it is open you stand still and don't fire, but the
match carries on around you.

### Classes

Every player picks a class. A new connection is asked first, with the five
side by side; after that the class row in the lobby (under the player list)
or the Esc menu changes it. A pick takes effect at your **next spawn**: the
next match start, or your next respawn mid-match. Until then you play the
class you spawned as.

```
class          health  shields  armour  speed     weapons
Commando         98      71      20%    4.0 m/s   pistol, combat rifle, SMG
Heavy support   129      65      30%    3.0 m/s   pistol, flak cannon, rocket launcher
Medic            94      75      20%    4.0 m/s   pistol, SMG, combat shotgun
Tech             61      98      20%    3.5 m/s   pistol, rail rifle, plasma rifle
Saboteur         88      56      10%    5.0 m/s   combat knife, frag grenade
```

Every number there is a rounded one, and the game uses what the card shows:
health and shields to the whole point, speed to the nearest half a metre per
second. The design numbers behind them stay in `sim/classes.py` — the class
reads them back through `hp`, `sh` and `spd` — because a card reading 4.2 m/s
next to 4.1 m/s is a difference nobody can feel. Speed is a multiplier on
every stance, against the Commando's 4.0: the Saboteur runs at 7.5 m/s, the
Commando at 6.0, the Heavy at 4.5.

Each class has one ability on **space**, and thirty seconds between uses,
counted from the moment the last one ended:

- **Commando — Blitz** (6 s): double speed, and an empty gun reloads itself
  the instant it runs dry, out of the ammo you are actually carrying. With an
  empty reserve it stays empty.
- **Heavy support — Brace** (6 s): half the damage taken, but only while you
  stand still. Walk and it is off until you stop again.
- **Medic — Field dressing** (3 s): channeled, like a reload. Heals half your
  health, paid out as it goes, and moving, firing or being hit stops it —
  what it has already given stays given.
- **Tech — Echo ping** (3 s): opens the fog around you for three seconds,
  through walls, showing the layout and anyone standing in it. When it fades
  those cells stay remembered, like anywhere else you have seen.
- **Saboteur — Vanish** (4 s): your feet make no sound, nobody draws you past
  three metres, and you move at double speed. Firing or swinging ends it on
  the spot. On your own screen you are drawn blue and see-through for as long
  as it runs — nobody else sees that, and it is the only cue on the body
  itself that the ability is still on. It is for
  crossing ground and getting behind someone, not for winning a fight you are
  already in — and it is the answer to the Tech's ping.

The server runs all of it. The client predicts the Commando's speed and draws
the Tech's ping; everything else — the heal, the mitigation, the cooldown — is
the server's, so a modified client gains nothing.

The stats come from the class sheets in `mechanics/Classes`, scaled to the
prototype Commando and then by `TOUGHNESS` (1.25) for the reaction-time
rebalance below; the abilities are newer than those sheets. Both live in
`sim/classes.py`. Single-player's Commando is deliberately not scaled: it
still carries the 78 health and 57 shields it always did.

Each class draws its own sprite set from `assets/characters`: a class with
`art="medic"` looks for `medic_ready` / `medic_idle` and falls back to its own
idle before it falls back to the khaki soldier, so a class with only one pose
keeps its own armour rather than changing body mid-fight. The Saboteur still
points at `unnamed_idle.png`; rename the file and the class's `art` field
together when it gets a name of its own.

### Reaction time

A firefight has to last long enough to answer. Point blank with every shot
hitting — the fastest any of it can happen — it now runs:

```
weapon          Commando  Heavy  Medic   Tech  Saboteur
pistol             1.92s  3.69s  1.92s  1.92s  1.54s
combat rifle       1.50s  2.00s  1.50s  1.33s  1.17s
SMG                1.08s  1.46s  1.08s  1.00s  0.85s
combat shotgun     1 shell at point blank (a Heavy survives it, just)
heavy rifle        2.22s  3.33s  2.22s  2.22s  1.11s
rail pistol        3.12s  3.75s  3.12s  3.12s  2.50s
pulse carbine      2.00s  3.00s  2.00s  1.50s  1.50s
```

What moved: classes carry 25% more health and shields; the SMG lost a little
damage; the pistol gained damage and then a faster trigger — 2.6 pulls a
second, so eight rounds go out in three seconds and a sidearm is a weapon
again; and
every gun except the two rails shoots 33% wider than its accuracy rating alone
would give (`SPREAD_WIDEN` in `sim/weapons.py`), which is what turns distance
into time. The rails are exempt — the slow, deliberate shot is the point of
them.

### The Saboteur's kit

He carries no gun at all: a blade, three grenades, and the speed to get to
where those are the right answer. Two things exist only on this class.

The **combat knife** has no projectile: a 1.6 m reach and a 50° arc, and it
will not swing through a wall. From the
front it takes two swings, three on a Heavy; from behind (more than 100° off
the way they are facing) it does 4.5x damage, which kills any class outright.
It carries no ammo and never runs out.

The **frag grenade** runs a three-second fuse that starts when the pin comes
out, not when it lands — so you cook it. Hold fire and the same ring the rail
weapons draw fills at your cursor, amber then red; let go and it flies at
16 m/s with whatever is left of the fuse:

- a quick throw lands and sits there for most of three seconds, which is
  plenty of time for the room to leave
- cooked halfway, it goes off about a second after it leaves your hand
- let go too late and it goes off in the air between you and them
- hold it the whole three seconds and it goes off in your hand, which kills
  you

It blasts 3.5 m, traced through cover like any other explosion, so a shut door
stops it. One that lands on somebody kills them where they stand; a metre out
it still takes most of a Commando, and at three it is a bad wound rather than
a death.

Three grenades, and no reload: the belt is the magazine. There is nothing to
reload from and an ammo pack has nothing to give it, so the only question is
whether this is worth one of the three.

It is quiet going out, too: the throw carries about a sixth as far as the
explosion, so what gives you away is the bang, not the arm.

On everyone else's screen a thrown grenade is the round itself and nothing
else: no tracer is sent for anything that travels — a rocket, a flak shell or
a grenade — because a line drawn to where it will land arrives before the
round does, gives the throw away, and reads as a streak out of the barrel.
`tests/weapons_test.py` checks each of them on the wire. The throw
message carries the fuse, so every client draws the round lying there with a
quickening blink on the same clock, and the blast arrives as its own message
when the fuse ends — one explosion, from the server, not one per client.

The shotgun went the other way on purpose. It throws ten pellets, and what
decides a fight is how many of them are still on a person:

```
range   pellets on the body   shells to kill (Commando / Heavy)
 1 m          10.0                   1 / 2
 3 m           6.9                   2 / 2
 5 m           4.6                   2 / 3
 8 m           3.1                   3 / 4
12 m           2.2                   5 / 6
```

At arm's length it is the one gun you cannot react to. Across a room or down a
hallway — five to eight metres — half the pattern still lands and it kills in
two or three shells, which is a rifle's pace. Past twelve it is the wrong gun:
the pattern is wider than a person. Its numbers say the same: a 20 m effective
range (the shortest of any gun), a 12° cone, and 0.92 accuracy.

`tests/balance_test.py` pins all of it: nothing but that one shell kills in
under half a second. Speed scales every stance, so a Heavy walks, runs and
crawls at 0.75 of the Commando's pace and the Saboteur at 1.25 of it.
Single-player still plays the Commando with the full kit.

### The heavy and energy weapons

Three weapons put something into the air rather than a line on the map, and
`tests/weapons_test.py` pins each one.

The **rocket launcher** flies: 28 m/s, so twenty metres of hallway is most of
a second in which the man you aimed at can be somewhere else. It goes off
where it arrives, never where it left, and it is traced through cover like any
other blast.

The **flak cannon** fires a shell, not a burst. It travels at 20 m/s and
carries to whatever stops it — a wall, a door, a person — and only comes apart
in mid-air if it has flown ten metres without finding anything. Where it stops
it throws thirty-six pellets outward through the full circle, each one an
ordinary shot with no penetration at all, carrying two and a half metres; a
wall, a door or a corner stops them dead.

It is a weapon for the length of a room and no further. Against a Commando, a
shell that lands on somebody kills them; a metre from the burst takes half of
what they have, two metres a quarter, and by three there is nothing left of
the ring. The burst is drawn as short stubs of spark leaving the shell rather
than as thirty-six full-length pellet tracers, which read as a weapon that
fires lines.

The **plasma rifle** carries five bolts and no spare cells, and makes itself a
new one every five seconds whether it is in your hands or slung. There is
nothing to reload and an ammo pack has nothing to give it, so the only
question it ever asks is whether to spend the round you have; both modes draw
the wait as a bar where the reserve count would be.

Ammo is **capped** everywhere else: every gun carries
three to six spare magazines (the rocket launcher, three rockets ever), so a
firefight you cannot walk away from is one you have to finish. The two
exceptions are the ones with no reload at all — the plasma rifle, which makes
its own, and the grenade belt, which does not — and `tests/pickups_test.py`
holds all three rules apart. Damage runs through the same
shields → armor → health model as single-player, bullets spend the same
penetration budget against cover, and glass shatters for everyone at once.

Explosions respect cover too. A blast is traced from where it goes off to each
body in range and pays for the cover in between the way a bullet does, out of
a small budget (`BLAST_PEN` = 1.0 in `sim/ballistics.py`): glass and thin walls
let it through, while low cover, doors, blast doors and walls stop it outright. A
rocket or bolt that hits a wall goes off on the near face of it, so the
side it hit takes the blast and the far side doesn't. The rail pistol punches
through a shut blast door, but only at full charge (`pen_charged` = 3.3 against
the door's 3.2). It never gets through a full wall (3.5).

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
in about a third of a second, and a blast door (`B`) takes five. Neither can be
walked through, and neither lets sound past, until the panels are fully home.
While the panels move — opening or closing — there's a slit between them,
widening as they part and narrowing as they meet, and you can see through it,
as can whoever is on the far side. On a blast door you can also shoot through
it. That's the point of the slow one: five seconds in which each side can see
and fire on the other through the gap, and nobody can get through it.

A blast door is committed: once it starts, `f` does nothing until it's idle
again. A powered door is quick enough to change your mind about — press `f`
mid-travel and it turns round from wherever the panels are.

Opening one changes what can be seen, shot and heard through it for everyone,
and it will not close on a body standing in the doorway. Other players are
hidden by fog of war exactly like anything else — if you cannot see into a room,
you cannot see who is in it, and their tracers and muzzle flashes are hidden with
them. What their gunfire LIGHTS is still visible, which is usually the more
useful tell.

The fog itself is built only where it is seen. Each frame the veil is made,
blurred and drawn for the fine cells under the camera plus a margin wide enough
for the blur to read from, rather than for the whole map; and since the veil is
black, it goes on as a multiply rather than as a per-pixel-alpha layer. On the
80x52 m vessel map that took the fog from 4.0 ms a frame to 1.7 ms, and the
whole frame from 7.5 ms to 5.1 ms (single-player: 8.7 ms to 6.2 ms). The window
is `fog_windows` and the veil `fog_veil`, both in `main.py` and both used by the
multiplayer client too, with `tests/fog_window_test.py` checking that a window
gives the same veil the whole map would.

Each source keeps one reusable propagation field per kind of sound, so a weapon
on full auto costs about one field solve rather than one per round — measured at
56 shots per 2 solves — and a source that moves more than a metre gets a fresh
one. What you hear of another player arrives as an arc at your own position,
pointing the way the sound travelled. Arrivals from the same direction refresh
one arc rather than stacking: a sprinter emits five footsteps a second, and
one arc per step rings you completely and tells you nothing.

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
**47801**, protocol version 13 — a mismatched client is rejected outright.

The server ticks at 30 Hz and broadcasts snapshots at 30 Hz. Clients draw
remote players 70 ms in the past (`INTERP_DELAY`), which is two snapshots of
buffer: enough that one late packet is covered, and no more. Measured on a
loopback match, a remote player's position lags the server's by a median of
78 ms, worst 102 ms — down from 89/144 ms at 20 Hz and 100 ms of interpolation.
Dropping the buffer below two snapshots measures faster still (70 ms) but
freezes and snaps the moment a packet is late, so it is not the default.

Every connection has an outbox drained by its own thread, so the tick loop
never blocks on a socket: one player whose connection backs up costs
themselves their backlog and
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

### Test maps

The test maps live in `tests/` with the tests. They are never offered in the
multiplayer lobby, and the server refuses them even by a crafted id; open them
in the editor or play them in single-player:

```
py main.py tests/range.map             # the shooting range every test uses
py editor.py tests/not_playable.map
```

`range` is one map with a corner for each thing the tests check. The top
strip has a powered door in a vertical wall (the first door on the map) and a
window. Under it is a powered door in a horizontal wall, then an open hall
with two of the four spawns. Below the hall a blast door in a horizontal
wall leads to four cover lanes, each with its cover in column 8: window,
door, blast door, wall. A health pack and an ammo pack sit in the lower
right. `tests/range_spots.py` names every one of these positions, and the
tests check the map against it first, so if an edit moves something a test
needs, the test fails and names what moved.

Two things are not on the range. `not_playable` uses a tile the tileset
doesn't have, so it can't share a map with anything playable; save it in
the editor to see the NOT PLAYABLE warning. The editor has no thin wall or
low cover tile, so `blast_cover_test` checks those two on a small char grid.

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

All tests live in `tests/` and run from the project root:

```
py -m tests.smoke_test        # lobby -> match -> kill -> respawn -> end -> lobby
py -m tests.movement_test     # speeds, stance, input clamping, walls
py -m tests.combat_test       # firing, damage, mags, cover, kills, glass,
                              #   doors, sound
py -m tests.backpressure_test # a client that stops reading must not stall the
                              #   server for everyone else
py -m tests.rejoin_test       # joining, leaving and rejoining a running match
py -m tests.spawn_points_test # authored spawns: format, teams, facing, fallback
py -m tests.doors_test        # slide axis, travel time, a moving door is a wall
py -m tests.pickups_test      # ammo caps, the 30% rule, respawns, who got there
                              #   first
py -m tests.map_validity_test # nothing unplayable is listed, hosted or started
py -m tests.map_download_test # a player without the host's map gets a copy
py -m tests.classes_test      # classes: stats, kits, speed, the esc menu, respawn
py -m tests.fog_window_test   # the camera-window veil matches a whole-map one
py -m tests.lobby_layout_test # nothing is drawn off the lobby or off a card
py -m tests.abilities_test    # the class abilities, their cooldown, knocking
py -m tests.balance_test      # time to kill, the shotgun's one shell, the spread
py -m tests.saboteur_test     # the knife, the grenade's fuse, Vanish and how
                              #   it looks, class art
py -m tests.stamina_test      # sprint fuel over the wire, and a thrown round
py -m tests.weapons_test      # the rocket's flight, the flak ring, plasma
                              #   cells, one blast per round, what each is
                              #   heard as, and that nothing thrown draws a
                              #   line
py -m tests.blast_cover_test  # what blasts and a charged rail shot get through
py -m tests.sp_downed_test    # a downed player cannot move, aim, act or heal
py -m tests.mp_smoke_test     # the real client, headless, for six seconds
py -m tests.mp_smoke_test vessel_interior
py -m tests.editor_test       # the editor opens every test map, every mode
```

All twenty-two are headless. `mp_smoke_test.py` drives the actual `MatchView` with
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
tests/              every test, headless; python -m tests.<name>
                    range.map + range_spots.py: the map they share
net/
  protocol.py       wire format, enums, message tags, input bits, kill verbs
  server.py         authoritative server: threads, ticks, movement, scoring
  client.py         client socket, world mirror, event queue
  maps.py           map id -> file, map loading, spawn point derivation
  mapcatalog.py     scans maps/ for playable maps (both formats)
  mappicker.py      host-only map selection panel
sim/
  movement.py       speeds, collision, input clamping — one definition, shared
  perception.py     loudness, and what a listener makes of a sound — shared
  lighting.py       beams, point lights, and the light gate on sight — shared
  classes.py        multiplayer classes: stats, kits, the body each spawns in
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
3. **No interactables beyond doors.** Single-player has traders, corpses, quest
   objects and an inventory behind the same key; multiplayer has doors only.
4. **Missing feedback the single-player renderer has:** blood hit-reactions,
   the weapon-swap animation frames, and the slung/raise weapon states. Muzzle
   flashes, recoil, the flashlight, the shadow model and the rail spool-up
   while charging are done.
5. **No ripple view.** Single-player can show the sound wavefront itself
   (F11 cycles arcs / full ripples / off, and ripples are its default);
   multiplayer draws the arrival arcs only.
6. **Sound is emitted authoritatively but heard client-side**, which is a
   deliberate trade (see above) and the place to revisit first if this ever
   needs to be cheat-resistant.
7. **If the host quits, the match ends for everyone.** The server lives inside
   the host's process. The `is_host` flag migrates between players, but it
   cannot move the server, so nobody can take over; `tests/smoke_test.py`'s
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
fine and one file is enough to see it working. `python -m tests.mp_smoke_test`
prints which coloured sets it found. `sprites.PALETTE` is also what the lobby
swatches are built from, so adding a colour there adds it to the lobby — but
a swatch with no art is a soldier in khaki wearing a coloured ring.

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
13. All sound effects are procedural placeholders in `sim/audio.py`. What a
    weapon is heard as is decided by what it does rather than by its
    category: a blade is a whoosh, a grenade leaving a hand is cloth and air
    carrying about a sixth as far as the blast will, a shell or a rocket is a
    launch, and the bang belongs to the round going off — which is its own
    event, at the other end of the flight.

### Housekeeping

14. `sim/` is covered mostly through the net path rather than directly.
    `balance_test`, `blast_cover_test`, `movement_test` and `weapons_test` do
    call into it on their own, but most of what `sim/` does is exercised by
    driving a server and a client at it.
15. `maps/` mixes formats and scratch files (`untitled.map`, `untitled2.map`,
    `assault.map`, `demo.map`, `arena_1x.*`). The picker only lists the ones
    that load against the current tileset — six of the eleven, as it stands —
    and prints the reason for each of the rest to the console, so the scratch
    files are noise in the folder rather than in the lobby.
16. `mechanics/` is imported design material from an older project (some files
    dated 2018) kept as reference — the weapon roster and the
    shield/armor/accuracy model were ported from it into `sim/weapons.py` and
    `sim/combat.py`. It is not live code.
