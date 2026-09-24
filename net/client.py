"""
net/client.py — BANGbang multiplayer client.

Wraps a TCP connection to GameServer. A background thread reads frames and
updates a thread-safe local mirror of world state; the pygame main loop reads
that mirror each frame and calls send_input() each tick.

Typical use in main.py:

    cli = GameClient("192.168.1.42")     # host IP; "127.0.0.1" for the host itself
    cli.connect(name="Kapistus", colour=(220,40,40))
    ...
    # each frame:
    for ev in cli.drain_events():        # kill feed, match_start, etc.
        handle(ev)
    world = cli.world                     # snapshot mirror, read freely
    if world.state == ServerState.MATCH:
        cli.send_input(mx, my, aim, buttons)
    render(world)

All mirror reads are cheap copies of plain data; the receive thread only ever
mutates under self._lock.
"""
from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field

from . import protocol as P
from .protocol import GameMode, ServerState, Team

# Where a client looks for the host's map, and where it keeps copies it had to
# download. None = maps/ and maps/downloaded/ in the project. The tests point
# both somewhere of their own, so a test run never writes into maps/.
MAPS_DIR = None
DOWNLOAD_DIR = None


@dataclass
class RemotePlayer:
    id: int
    name: str = "?"
    colour: tuple = (200, 200, 200)
    team: Team = Team.NONE
    ready: bool = False
    is_host: bool = False
    playing: bool = False       # in the current match, as opposed to sitting out
    has_map: bool = True        # False while their client fetches the map
    cls: str = "commando"       # the class they have picked (lobby roster)
    cls_now: str = "commando"   # the class they are playing (snapshots): what
                                #   their speed and weapons come from
    vanished: bool = False      # a Saboteur nobody can see past a few metres
    stamina: float = 1.0        # sprint fuel, 0..1
    ability_t: float = 0.0      # seconds of their class ability still running
    ability_cd: float = 0.0     # (you only) seconds until you can use it again
    # match state (interpolation buffer)
    x: float = 0.0
    y: float = 0.0
    aim: float = 0.0
    alive: bool = False
    respawn_in: float = 0.0
    kills: int = 0
    acked_seq: int = -1         # last input of THEIRS the server has applied
    # combat state, as of the last snapshot
    hp: float = 1.0             # health, 0..1 of maximum
    sh: float = 0.0             # shields, 0..1 of maximum
    wep: int = 0                # weapon index in their loadout
    mag: int = 0                # rounds left in it
    recharge: float = 0.0       # 0..1 toward the next self-made round
    reload_t: float = 0.0       # seconds of reload remaining, 0 = not
    flashlight: bool = False    # their beam is lit
    charge: float = 0.0         # rail spool-up, seconds
    # previous snapshot for interpolation
    prev_x: float = 0.0
    prev_y: float = 0.0
    prev_aim: float = 0.0
    snap_t: float = 0.0
    prev_snap_t: float = 0.0

    def render_pos(self, now: float, interp_delay: float = 0.1):
        """Linear interpolation between the last two snapshots for smoothing.
        interp_delay renders slightly in the past so we always have two
        bracketing samples. On LAN 100ms is safe; lower if you prefer."""
        target = now - interp_delay
        span = self.snap_t - self.prev_snap_t
        if span <= 0:
            return self.x, self.y, self.aim
        f = (target - self.prev_snap_t) / span
        f = max(0.0, min(1.0, f))
        return (
            self.prev_x + (self.x - self.prev_x) * f,
            self.prev_y + (self.y - self.prev_y) * f,
            self.prev_aim + (self.aim - self.prev_aim) * f,
        )


@dataclass
class KillFeedEntry:
    killer: str
    victim: str
    verb: str
    at: float                       # monotonic time received; for fade-out
    weapon: str = ""


@dataclass
class World:
    """Client-side mirror. Read from the main loop; written by recv thread."""
    state: ServerState = ServerState.LOBBY
    mode: GameMode = GameMode.FFA
    duration_s: int = 300
    map_id: str = "arena"
    map_sha: str = ""           # fingerprint of the host's copy of the map
    # where this machine has that exact map: its own maps folder, a copy
    # downloaded from a host earlier, or None while it is still being fetched
    map_dir: str | None = None
    map_status: str = ""        # "", "downloading", "ready", "failed: why"
    host_id: int | None = None
    my_id: int | None = None
    is_host: bool = False
    playing: bool = False       # are WE in the match, or watching from the lobby
    my_spawn: tuple | None = None
    my_spawn_aim: float | None = None    # which way the spawn point faces
    my_team: Team = Team.NONE
    time_left: float = 0.0
    end_count: int | None = None
    players: dict[int, RemotePlayer] = field(default_factory=dict)
    # the map as it stands now, for a client that arrived after it changed.
    # Held on the world rather than only fired as an event, because it lands
    # while the lobby screen is draining events and the match render that needs
    # it does not exist yet.
    map_doors: list = field(default_factory=list)  # (row, col, is_open, left)
    map_pickups: list = field(default_factory=list)   # (id, live, left)
    map_glass: list = field(default_factory=list)    # (row, col) broken
    killfeed: list[KillFeedEntry] = field(default_factory=list)
    team_scores: dict[int, int] = field(default_factory=dict)


class GameClient:
    def __init__(self, host: str, port: int = P.DEFAULT_PORT):
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None
        self.world = World()
        self._lock = threading.RLock()
        self._events: list[dict] = []          # high-level events for main loop
        self._seq = 0
        self._running = False
        self._recv_thread: threading.Thread | None = None
        self.connected = False
        self.reject_reason: str | None = None
        self._send_lock = threading.Lock()   # the receive thread sends too
        self._map_key: tuple | None = None   # (map_id, sha) last looked for
        self._join_pending = False
        from . import maps as netmaps
        from pathlib import Path
        self.maps_dir = Path(MAPS_DIR or netmaps.DEFAULT_MAPS_DIR)
        self.download_dir = Path(DOWNLOAD_DIR or (netmaps.DEFAULT_MAPS_DIR
                                                  / "downloaded"))

    # ---------------------------------------------------------------- connect

    def connect(self, name: str, colour: tuple, timeout: float = 5.0) -> bool:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        try:
            self.sock.connect((self.host, self.port))
        except OSError as e:
            self.reject_reason = f"connect failed: {e}"
            return False
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(None)
        P.send_msg(self.sock, {"t": P.C_JOIN, "version": P.PROTOCOL_VERSION,
                               "name": name, "colour": list(colour)})
        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        # wait briefly for welcome/reject
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self.world.my_id is not None:
                    self.connected = True
                    return True
                if self.reject_reason:
                    return False
            time.sleep(0.02)
        self.reject_reason = "no welcome from server"
        return False

    def disconnect(self) -> None:
        self._running = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass

    # ---------------------------------------------------------------- outbound

    def _send(self, obj: dict) -> None:
        if not self.sock:
            return
        try:
            with self._send_lock:
                P.send_msg(self.sock, obj)
        except OSError:
            self.connected = False

    def set_name(self, name: str) -> None:
        self._send({"t": P.C_SET_NAME, "name": name[:20]})

    def set_colour(self, colour: tuple) -> None:
        self._send({"t": P.C_SET_COLOUR, "colour": list(colour)})

    def set_team(self, team: Team) -> None:
        self._send({"t": P.C_SET_TEAM, "team": int(team)})

    def set_class(self, cls: str) -> None:
        """Pick a class (sim/classes.py key). It takes effect at your next
        spawn: the start of the next match, or your next respawn."""
        with self._lock:
            me = self.world.players.get(self.world.my_id)
            if me is not None:
                me.cls = cls          # show it at once; the server confirms
        self._send({"t": P.C_SET_CLASS, "cls": cls})

    def set_ready(self, ready: bool) -> None:
        self._send({"t": P.C_SET_READY, "ready": ready})

    def set_config(self, mode=None, duration_s=None, map_id=None) -> None:
        """Host only. Server ignores if sender isn't host."""
        msg = {"t": P.C_SET_CONFIG}
        if mode is not None:
            msg["mode"] = int(mode)
        if duration_s is not None:
            msg["duration_s"] = int(duration_s)
        if map_id is not None:
            msg["map_id"] = map_id
        self._send(msg)

    def join_match(self) -> None:
        """Ask to be dropped into the match already in progress. Without the
        map yet, the request waits and goes out the moment it arrives."""
        with self._lock:
            if self.world.map_dir is None:
                self._join_pending = True
                return
        self._send({"t": P.C_JOIN_MATCH})

    def wait_for_map(self, timeout: float = 20.0) -> str | None:
        """Block until this client has the selected map; its folder, or None
        if it could not be had in time (see world.map_status for why)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                w = self.world
                if w.map_dir is not None:
                    return w.map_dir
                if w.map_status.startswith("failed"):
                    return None
            time.sleep(0.02)
        return None

    def leave_match(self) -> None:
        """Step out of the match but stay connected, so you can rejoin it."""
        self._send({"t": P.C_LEAVE_MATCH})

    def force_start(self) -> None:
        self._send({"t": P.C_START})

    def end_match(self) -> None:
        self._send({"t": P.C_END_MATCH})

    def send_input(self, mx: float, my: float, aim: float, buttons: int,
                   wep: int = 0, mode: int = 0, aim_dist: float = 10.0) -> int:
        """Send one input command and return the sequence number it went out
        with. A predicting client keeps that number against its own predicted
        position, so when a snapshot says which input the server has applied it
        knows exactly which prediction to check."""
        self._seq += 1
        self._send({"t": P.C_INPUT, "seq": self._seq, "mx": mx, "my": my,
                    "aim": aim, "buttons": buttons, "wep": wep, "mode": mode,
                    "aimd": aim_dist})
        return self._seq

    # ---------------------------------------------------------------- events

    def drain_events(self) -> list[dict]:
        """Pop queued high-level events (match_start, kill, match_end,
        end_count). Call once per frame in the main loop."""
        with self._lock:
            out = self._events
            self._events = []
            return out

    def scoreboard(self) -> list[tuple[str, int]]:
        """(name, kills) sorted descending by kills. For the Tab overlay."""
        with self._lock:
            rows = [(p.name, p.kills) for p in self.world.players.values()]
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows

    # ---------------------------------------------------------------- recv loop

    def _recv_loop(self) -> None:
        while self._running:
            try:
                msg = P.recv_msg(self.sock)
            except OSError as e:
                self.reject_reason = self.reject_reason or f"connection lost: {e}"
                break
            except ValueError as e:
                # A frame we could not decode. Swallowing this silently ends
                # the receive thread and the client simply stops hearing from
                # the server with no explanation at all — which is exactly how
                # an int-keyed team_scores dict hid for as long as it did.
                self.reject_reason = f"bad frame from server: {e}"
                print(f"[client] {self.reject_reason}")
                break
            if msg is None:
                break
            self._apply(msg)
        self.connected = False
        with self._lock:
            self._events.append({"t": "disconnected"})

    # ---------------------------------------------------------------- the map

    def _check_map(self, map_id: str, sha: str) -> None:
        """Do we have the host's map, this exact version? Our own maps folder
        first, then a copy downloaded earlier; if neither, ask the host.
        Called from the receive thread, outside the world lock, because it
        touches the disk."""
        key = (map_id, sha)
        if key == self._map_key:
            return
        self._map_key = key
        from . import maps as netmaps
        found = None
        if not sha:
            # a host that could not fingerprint its map: all we can do is use
            # ours if we have one by that name
            if netmaps.map_sha(map_id, self.maps_dir) is not None:
                found = str(self.maps_dir)
            with self._lock:
                self.world.map_dir = found
                self.world.map_status = "ready" if found else \
                    "failed: the host did not offer this map for download"
            return
        for d in (self.maps_dir, self.download_dir / map_id):
            if netmaps.map_sha(map_id, d) == sha:
                found = str(d)
                break
        with self._lock:
            w = self.world
            w.map_dir = found
            w.map_status = "ready" if found else "downloading"
        if found:
            self._have_map(map_id, sha)
        else:
            print(f"[client] no copy of map {map_id!r} matching the host's; "
                  f"downloading it")
            self._send({"t": P.C_MAP_REQ, "map_id": map_id})

    def _have_map(self, map_id: str, sha: str) -> None:
        self._send({"t": P.C_MAP_HAVE, "map_id": map_id, "sha": sha})
        with self._lock:
            self._events.append({"t": "map_ready", "map_id": map_id})
            join = self._join_pending
            self._join_pending = False
        if join:
            self._send({"t": P.C_JOIN_MATCH})

    def _on_map_data(self, msg: dict) -> None:
        from . import maps as netmaps
        map_id = str(msg.get("map_id", ""))
        with self._lock:
            current = (self.world.map_id, self.world.map_sha)
        if map_id != current[0] or msg.get("sha", current[1]) != current[1]:
            return                      # the host has moved on since we asked
        if "error" in msg:
            why = f"failed: {msg['error']}"
        else:
            try:
                folder = netmaps.save_download(map_id, current[1],
                                               msg.get("files"),
                                               self.download_dir)
            except (ValueError, OSError) as e:
                why = f"failed: {e}"
            else:
                with self._lock:
                    self.world.map_dir = str(folder)
                    self.world.map_status = "ready"
                print(f"[client] map {map_id!r} downloaded to {folder}")
                self._have_map(map_id, current[1])
                return
        print(f"[client] map {map_id!r}: {why}")
        with self._lock:
            self.world.map_status = why
            self._events.append({"t": "map_failed", "map_id": map_id,
                                 "reason": why})

    def _apply(self, msg: dict) -> None:
        t = msg.get("t")
        if t == P.S_MAP_DATA:
            self._on_map_data(msg)
            return
        self._apply_world(msg)
        if t in (P.S_LOBBY, P.S_MATCH_START):
            self._check_map(str(msg.get("map_id", "")),
                            str(msg.get("map_sha", "")))

    def _apply_world(self, msg: dict) -> None:
        t = msg.get("t")
        now = time.monotonic()
        with self._lock:
            w = self.world
            if t == P.S_WELCOME:
                w.my_id = msg["your_id"]
                w.is_host = msg["is_host"]

            elif t == P.S_REJECT:
                self.reject_reason = msg.get("reason", "rejected")
                self._events.append({"t": "reject", "reason": self.reject_reason})

            elif t == P.S_LOBBY:
                w.state = ServerState(msg["state"])
                w.mode = GameMode(msg["mode"])
                w.duration_s = msg["duration_s"]
                w.map_id = msg["map_id"]
                w.map_sha = msg.get("map_sha", "")
                w.host_id = msg["host_id"]
                seen = set()
                for pd in msg["players"]:
                    pid = pd["id"]
                    seen.add(pid)
                    p = w.players.get(pid) or RemotePlayer(id=pid)
                    p.name = pd["name"]
                    p.colour = tuple(pd["colour"])
                    p.team = Team(pd["team"])
                    p.ready = pd["ready"]
                    p.is_host = pd["is_host"]
                    p.playing = pd.get("playing", False)
                    p.has_map = pd.get("has_map", True)
                    p.cls = pd.get("cls", p.cls)
                    if pid == w.my_id:
                        w.playing = p.playing
                    w.players[pid] = p
                for pid in list(w.players):
                    if pid not in seen:
                        del w.players[pid]
                w.is_host = (w.my_id == w.host_id)

            elif t == P.S_MATCH_START:
                w.state = ServerState.MATCH
                w.mode = GameMode(msg["mode"])
                w.duration_s = msg["duration_s"]
                w.map_id = msg["map_id"]
                w.map_sha = msg.get("map_sha", "")
                w.my_spawn = (msg["spawn"]["x"], msg["spawn"]["y"])
                w.my_spawn_aim = msg["spawn"].get("aim")
                w.my_team = Team(msg["team"])
                w.playing = True
                w.end_count = None
                w.killfeed.clear()
                for pd in msg["players"]:
                    pid = pd["id"]
                    p = w.players.get(pid) or RemotePlayer(id=pid)
                    p.name = pd["name"]
                    p.colour = tuple(pd["colour"])
                    p.team = Team(pd["team"])
                    p.kills = 0
                    p.alive = True
                    w.players[pid] = p
                self._events.append({"t": "match_start"})

            elif t == P.S_SNAPSHOT:
                w.time_left = msg["time_left"]
                for pd in msg["players"]:
                    pid = pd["id"]
                    p = w.players.get(pid)
                    if p is None:
                        p = RemotePlayer(id=pid)
                        w.players[pid] = p
                    p.prev_x, p.prev_y, p.prev_aim = p.x, p.y, p.aim
                    p.prev_snap_t = p.snap_t
                    p.x, p.y, p.aim = pd["x"], pd["y"], pd["aim"]
                    p.acked_seq = pd.get("seq", -1)
                    p.alive = pd["alive"]
                    p.respawn_in = pd["respawn_in"]
                    p.playing = pd.get("pl", True)
                    if pid == w.my_id:
                        w.playing = p.playing
                    p.hp = pd.get("hp", 1.0)
                    p.sh = pd.get("sh", 0.0)
                    p.wep = pd.get("wep", 0)
                    p.mag = pd.get("mag", 0)
                    p.reload_t = pd.get("rl", 0.0)
                    p.recharge = pd.get("rc", 0.0)
                    p.flashlight = pd.get("fl", False)
                    p.cls_now = pd.get("cl", p.cls_now)
                    p.vanished = pd.get("vn", False)
                    p.stamina = pd.get("st", 1.0)
                    p.ability_t = pd.get("ab", 0.0)
                    p.ability_cd = pd.get("acd", 0.0)
                    p.charge = pd.get("chg", 0.0)
                    p.snap_t = now

            elif t == P.S_MELEE:
                self._events.append({"t": "melee", "id": msg["id"],
                                     "heading": msg.get("heading", 0.0),
                                     "reach": msg.get("reach", 1.6),
                                     "hit": msg.get("hit", 0)})

            elif t == P.S_ABILITY:
                self._events.append({"t": "ability", "id": msg["id"],
                                     "ab": msg.get("ab", ""),
                                     "dur": msg.get("dur", 0.0)})

            elif t == P.S_KILL:
                killer = w.players.get(msg["killer"])
                victim = w.players.get(msg["victim"])
                entry = KillFeedEntry(
                    killer=killer.name if killer else "?",
                    victim=victim.name if victim else "?",
                    verb=msg["verb"], at=now, weapon=msg.get("wep", ""))
                w.killfeed.append(entry)
                w.killfeed = w.killfeed[-6:]        # keep last 6
                self._events.append({"t": "kill", "entry": entry})

            elif t == P.S_SHOT:
                # somebody fired: tracers, muzzle flash, and the noise it made
                self._events.append({
                    "t": "shot", "id": msg["id"],
                    "x": msg["x"], "y": msg["y"],
                    "heading": msg.get("heading", 0.0),
                    "wep": msg.get("wep", ""),
                    "charge": msg.get("charge", 0.0),
                    "segs": msg.get("segs", []),
                    "impact": tuple(msg.get("impact", (msg["x"], msg["y"]))),
                    "blast": msg.get("blast", 0.0),
                    "travel": msg.get("travel", 0.0),
                    # a fused round lies where it lands until it goes off; the
                    # drawing side runs the server's clock rather than guessing
                    "fuse": msg.get("fuse", 0.0),
                    "at": now,
                })

            elif t == P.S_SOUND:
                # where a noise was made and how far it carries. What THIS
                # player can hear of it is the client's own business: it solves
                # the propagation field and asks sim/perception.py.
                self._events.append({
                    "t": "sound", "x": msg["x"], "y": msg["y"],
                    "energy": msg["energy"], "clip": msg.get("clip", ""),
                    "label": msg.get("label", ""), "id": msg.get("id", 0),
                    # walk / run / crawl, so your own steps play at the volume
                    # the stance earns rather than one flat level
                    "stance": msg.get("stance", ""),
                    "at": now,
                })

            elif t == P.S_MAP_STATE:
                # joining late: the doors and windows as they stand now, not as
                # the map file has them
                w.map_doors = [(int(row[0]), int(row[1]), bool(row[2]),
                                float(row[3]) if len(row) > 3 else 0.0)
                               for row in msg.get("doors", [])]
                w.map_glass = [(int(r), int(c)) for r, c in msg.get("glass", [])]
                w.map_pickups = [(str(row[0]), bool(row[1]),
                                  float(row[2]) if len(row) > 2 else 0.0)
                                 for row in msg.get("pickups", [])]
                self._events.append({"t": "map_state", "doors": w.map_doors,
                                     "glass": w.map_glass,
                                     "pickups": w.map_pickups})

            elif t == P.S_PICKUP:
                # a pack was taken or came back. The server owns this
                # completely: there is nothing to predict and nothing to
                # argue with.
                self._events.append({
                    "t": "pickup", "pid": str(msg.get("pid", "")),
                    "live": bool(msg.get("live", True)),
                    "by": msg.get("by", 0),
                    "dur": float(msg.get("dur", 0.0)),
                })

            elif t == P.S_DOOR:
                # a door moved, or somebody tried and was refused. Every client
                # applies the same toggle, or their walls stop matching.
                self._events.append({
                    "t": "door", "r": msg["r"], "c": msg["c"],
                    "open": bool(msg.get("open", False)),
                    "dur": float(msg.get("dur", 0.0)),
                    "id": msg.get("id", 0),
                    "blocked": bool(msg.get("blocked", False)),
                })

            elif t == P.S_GLASS:
                self._events.append({
                    "t": "glass",
                    "cells": [tuple(c) for c in msg.get("cells", [])],
                })

            elif t == P.S_SCORE:
                w.mode = GameMode(msg["mode"])
                for sd in msg["scores"]:
                    p = w.players.get(sd["id"])
                    if p:
                        p.kills = sd["kills"]
                        p.team = Team(sd["team"])
                w.team_scores = {int(k): v for k, v in msg["team_scores"].items()}

            elif t == P.S_MATCH_END:
                self._events.append({"t": "match_end"})

            elif t == P.S_END_COUNT:
                w.end_count = msg["n"]
                if msg["n"] == 0:
                    w.state = ServerState.LOBBY
                self._events.append({"t": "end_count", "n": msg["n"]})
