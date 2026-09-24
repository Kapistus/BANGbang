"""
net/protocol.py — shared wire protocol for BANGbang multiplayer.

Transport: TCP stream, length-prefixed (4-byte big-endian) msgpack frames.
No dependency on the sim; both server and client import only this.

Install: pip install msgpack
"""
from __future__ import annotations

import socket
from enum import IntEnum

import msgpack

DEFAULT_PORT = 47801
# 2: input gained run/crawl bits and the server became authoritative over
#    movement, so a v1 client would walk at the wrong speed and desync.
# 3: weapons. Input carries the selected weapon and fire mode; the server
#    resolves shots and broadcasts them, along with sound events, glass it
#    shattered, and each player's health, shields and ammo.
# 4: joining, leaving and rejoining a match in progress. A connection is no
#    longer the same thing as playing, so players carry a "playing" flag and a
#    latecomer is caught up on the map state that has drifted from the file.
# 5: spawn points are map objects with a facing and an optional team, so
#    match_start says which way you are looking when you arrive.
# 6: flashlights. An input bit toggles one, and snapshots carry whose is lit,
#    because a beam is something everyone else can see coming.
# 9: map transfer. The lobby and match_start carry the map's fingerprint, and a
#    player without that exact map asks the host for a copy.
# 10: player classes. A player picks one in the lobby or from the in-match
#     menu; it takes effect at their next spawn. The lobby roster carries each
#     player's pick and snapshots the class they are playing right now, which
#     sets their speed and weapons.
# 11: class abilities and knocking. Two more input bits (space = ability,
#     e = knock), and snapshots carry each player's ability state so a client
#     can predict a Commando's sprint and draw a Tech's ping.
# 12: the Saboteur. A melee swing has no projectile to trace, so it has its
#     own message, and snapshots carry who is currently unseen.
PROTOCOL_VERSION = 13


# ---------------------------------------------------------------- enums

class GameMode(IntEnum):
    FFA = 0          # free-for-all deathmatch
    TEAM = 1         # team deathmatch


class ServerState(IntEnum):
    LOBBY = 0
    MATCH = 1
    END_COUNTDOWN = 2


class Team(IntEnum):
    """FFA players all sit on NONE. Team modes use A/B (extend as needed)."""
    NONE = 0
    A = 1
    B = 2


# ---------------------------------------------------------------- message types
# Kept as short string tags under key "t". Cheap to read in logs, still compact.

# client -> server
C_JOIN = "join"              # {name, colour, version}
C_SET_NAME = "set_name"      # {name}
C_SET_COLOUR = "set_colour"  # {colour}
C_SET_TEAM = "set_team"      # {team}          (team modes only)
C_SET_READY = "set_ready"    # {ready}
C_SET_CLASS = "set_class"    # {cls}  sim/classes.py key; applies at the next
                             #   spawn, so it can be sent from the lobby or
                             #   mid-match
C_SET_CONFIG = "set_config"  # host only: {mode?, duration_s?, map_id?}
C_JOIN_MATCH = "join_match"   # drop me into the match that is already running
C_LEAVE_MATCH = "leave_match"  # take me out of it, but keep my connection
C_START = "start"            # host only: force-start ignoring un-ready? (see server)
C_END_MATCH = "end_match"    # host only
C_MAP_REQ = "map_req"        # {map_id}   send me the map you have selected
C_MAP_HAVE = "map_have"      # {map_id, sha}  I have this map, this version:
                             #   until the server hears it, readying does not
                             #   start a match
C_INPUT = "input"            # {seq, mx, my, aim, buttons, wep, mode} in MATCH
                             #   wep  = index into the player's loadout
                             #   mode = index into that weapon's fire modes

# server -> client
S_WELCOME = "welcome"        # {your_id, is_host}
S_REJECT = "reject"          # {reason}
S_LOBBY = "lobby"            # {state, mode, duration_s, map_id, map_sha,
                             #  host_id, players:[...]}  map_sha fingerprints
                             #  the map's files; a player whose copy differs,
                             #  or who has none, downloads the host's
S_MAP_DATA = "map_data"      # {map_id, sha, files: {name: bytes}} or
                             #   {map_id, error}: the host's copy of its map
S_MATCH_START = "match_start"  # {mode, duration_s, map_id, map_sha, spawn:{x,y,aim},
                               #  team, players:[...]}
S_SNAPSHOT = "snapshot"      # {tick, time_left, players:[{id,x,y,aim,alive,
                             #   respawn_in, seq, hp, sh, wep, mag, rl, pl}]}
                             #   pl = playing: connected players sitting out a
                             #   match have no position worth drawing
                             #   hp/sh = health and shields, 0..1 of maximum
                             #   wep   = weapon index, mag = rounds in it
                             #   rl    = reloading (seconds left, 0 = not)
                             #   fl    = flashlight lit: everyone can see a
                             #           beam, so everyone is told about it
                             #   cl    = class they are playing now (their
                             #           lobby roster entry has the one they
                             #           have picked for next time)
                             #   ab    = seconds of their ability left, 0 = not
                             #           using it. Everyone is told, because an
                             #           ability is something you can see
                             #           somebody doing.
                             #   acd   = YOUR ability's cooldown, seconds left
                             #   vn    = vanished: drawn only from close up
                             #   rc    = how far along the next round a
                             #           self-charging magazine is making,
                             #           0..1 (0 = not charging one)
S_SHOT = "shot"              # {id, x, y, heading, wep, charge, segs, impact,
                             #   blast, travel, fuse}  one trigger pull, for
                             #   tracers, muzzle flash and the sound it makes.
                             #   A round that travels sends no segs - the round
                             #   itself is the visual - and carries travel (its
                             #   speed) plus fuse, the seconds it still has to
                             #   run once thrown. It goes off on its own later
                             #   message, so a fused round is drawn, never
                             #   detonated, by this one.
S_SOUND = "sound"            # {x, y, energy, clip, label, id, stance}  something
                             #   audible happened here; the client's own
                             #   propagation field decides what it can hear
S_MAP_STATE = "map_state"    # {doors: [[r, c, open, left], ...],
                             #    glass: [[r, c], ...],
                             #    pickups: [[id, live, left], ...]}  `left` is seconds of
                             #   travel still to run, so a latecomer picks up a
                             #   blast door already four seconds into its move
                             #   everything about the map that has drifted from
                             #   the file since the match began. Sent to a
                             #   latecomer, who would otherwise be shooting at
                             #   windows that are no longer there.
S_MELEE = "melee"            # {id, heading, reach, hit}  a blade swung, and
                             #   whose id it landed on (0 = thin air). No
                             #   projectile to trace, so clients draw the arc
                             #   from this.
S_ABILITY = "ability"        # {id, ab, dur}  somebody used their class
                             #   ability: which one, and how long it runs.
                             #   Sent to everyone - an ability is something you
                             #   can see a player doing.
S_PICKUP = "pickup"          # {pid, live, by, dur}  a health or ammo pack was
                             #   taken, or came back. The server decides who
                             #   reached it first; clients only draw it.
S_DOOR = "door"              # {r, c, open, dur, id, blocked}  a door moved, or
                             #   someone tried and couldn't. `dur` is the travel
                             #   still to run: the whole door for a fresh start,
                             #   less for a quick door that was turned round.
                             #   someone tried and couldn't. Doors change
                             #   sight, sound and bullets, so like glass the
                             #   server owns the toggle and everyone applies it.
S_GLASS = "glass"            # {cells: [[cx, cy], ...]}  panes that shattered.
                             #   Glass changes the map — sound, sight and
                             #   bullets all pass once it breaks — so the
                             #   server owns it and everyone applies the same
                             #   break, rather than each client deciding.
S_KILL = "kill"              # {killer, victim, verb, wep}
S_SCORE = "score"            # {mode, scores:[{id,name,team,kills}], team_scores:{...}}
S_MATCH_END = "match_end"    # {}
S_END_COUNT = "end_count"    # {n}             5..0 back to lobby


# ---------------------------------------------------------------- input buttons
# Bitmask packed into C_INPUT "buttons".

BTN_FIRE = 1 << 0         # held, not tapped: the server decides refire rate,
                          # full-auto streaming, and when a charged rail
                          # weapon lets go
BTN_INTERACT = 1 << 1
BTN_RELOAD = 1 << 2
BTN_RUN = 1 << 3          # shift: sprint
BTN_CRAWL = 1 << 4        # ctrl: crawl. Wins if both are held.
BTN_LIGHT = 1 << 5        # flashlight, toggled on the press
BTN_ABILITY = 1 << 6      # space: the class ability, on the press
BTN_KNOCK = 1 << 7        # e: knock on the wall you are standing at
# reserve more bits as the sim grows


def stance_of(buttons: int) -> str:
    """Movement stance encoded in an input bitmask. Crawl beats run so a
    client mashing both can't pick the fast one by accident."""
    if buttons & BTN_CRAWL:
        return "crawl"
    if buttons & BTN_RUN:
        return "run"
    return "walk"


# ---------------------------------------------------------------- framing

def send_msg(sock: socket.socket, obj: dict) -> None:
    """Serialize and send one length-prefixed frame. Raises on socket error."""
    data = msgpack.packb(obj, use_bin_type=True)
    sock.sendall(len(data).to_bytes(4, "big") + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes, or None if the peer closed cleanly mid-stream."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> dict | None:
    """Blocking read of one frame. None means the connection closed."""
    hdr = _recv_exact(sock, 4)
    if hdr is None:
        return None
    n = int.from_bytes(hdr, "big")
    if n <= 0 or n > 4 * 1024 * 1024:      # 4 MiB sanity cap
        raise ValueError(f"bad frame length {n}")
    body = _recv_exact(sock, n)
    if body is None:
        return None
    return msgpack.unpackb(body, raw=False)


# ---------------------------------------------------------------- kill verbs

KILL_VERBS = [
    "obliterated", "vaporized", "sent to respawn", "deleted",
    "put down", "clapped", "unalived", "folded", "dumpstered",
    "yeeted", "turned off", "canceled", "sat down", "ratio'd",
    "made a ghost of", "speedran the death of", "introduced to the floor",
    "logged out", "reset", "returned to sender",
]
