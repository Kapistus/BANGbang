"""
net/server.py — authoritative BANGbang game server.

Runs in a daemon thread inside the host process. Owns all match state.
Clients send inputs; server integrates, resolves shots, broadcasts snapshots.

Movement is authoritative: the server loads the same map the clients render
and moves every player through it with sim/movement.py, the identical code
path single-player uses. Spawn points come from net/maps.py.

Still to fill in (marked SIM HOOK):
  - resolve_shots(now)     : run weapon/hit logic, call self.kill() on hits

Usage from host:
    srv = GameServer(map_id="arena")   # spawns derived from the map
    srv.start()                        # spawns listener + tick threads
    ip = srv.lan_ip()                  # show this on the host lobby screen
    ...
    srv.stop()
"""
from __future__ import annotations

import dataclasses
import math
import random
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from sim import ballistics, classes, combat, movement, perception, weapons
from sim.doors import DoorSet
from sim.pickups import PickupSet
from sim import pickups as pk
from sim.tilemap import break_glass_cells, set_door, set_door_gap

from . import maps as netmaps
from . import protocol as P
from .protocol import GameMode, ServerState, Team

TICK_HZ = 30
TICK_DT = 1.0 / TICK_HZ
SNAPSHOT_HZ = 30                      # broadcast rate (<= TICK_HZ)
RESPAWN_DELAY = 10.0
END_COUNTDOWN_FROM = 5
MIN_DURATION = 120
MAX_DURATION = 3600
MIN_PLAYERS_TO_START = 2

# Input is a queue of fixed-size commands, one per tick, not a "latest value"
# that the tick loop samples. That is what lets a client predict its own
# movement exactly: both sides apply the same commands for the same TICK_DT, in
# the same order. The server takes at most one per tick, so a client that sends
# faster than the tick rate gains no speed — its surplus is simply dropped.
INPUT_QUEUE_MAX = 4

# Sending. The tick loop must never block on a socket: one player whose
# connection backs up — wifi hiccup, a laptop going to sleep, a client paused in
# a debugger — would otherwise freeze the simulation for everybody else. Each
# connection has an outbox drained by its own thread, and the tick loop only
# ever appends to it.
SEND_QUEUE_SOFT = 48      # past this, shed: stale snapshots first, then effects
SEND_QUEUE_HARD = 256     # past this the client is not reading at all: cut it
SENDER_POLL_S = 0.5       # how often an idle sender wakes to check for shutdown

# Only the newest of these matters — an old snapshot is worthless once a newer
# one exists, so the queue keeps one.
COALESCE_TYPES = frozenset({P.S_SNAPSHOT})
# Losing one of these costs a tracer, a footstep or a muzzle flash. Everything
# NOT listed here is state — lobby rosters, match start, kills, scores, doors,
# glass — and is delivered or the connection is dropped, because a client that
# misses a door opening is playing a different game from everyone else.
DROPPABLE_TYPES = frozenset({P.S_SHOT, P.S_SOUND})

# Weapon handling. These mirror main.py, which is the reference implementation
# of how the guns feel; the server owns them because a client that decided its
# own rate of fire would be deciding how much damage it deals.
RAIL_IDS = ("rail_rifle", "rail_pistol")   # hold-to-charge weapons
RAIL_CHARGE_MAX = 3.0                      # seconds of hold for a full charge
RAIL_CHARGE_BOOST = 1.0                    # +100% damage at full charge
SWAP_TIME = 0.55                           # mirrors sprites.SWAP_TIME, which is
                                           # not imported here: it needs pygame
MUZZLE_M = weapons.MUZZLE_M                # shot origin ahead of the body
INTERACT_RANGE = 1.6                       # mirrors main.py: how far you reach
DOOR_SOUND_M = perception.KNOCK_REACH_M * 0.7
ABILITY_SOUND_M = perception.KNOCK_REACH_M * 0.5   # a special is audible, quietly


@dataclass
class NetPlayer:
    id: int
    sock: socket.socket
    addr: tuple
    name: str = "Player"
    colour: tuple = (200, 200, 200)
    team: Team = Team.NONE
    ready: bool = False
    is_host: bool = False
    # Connected is not the same as playing. A latecomer sits here until they
    # are dropped in, and someone who leaves a match sits here again rather
    # than losing their connection.
    playing: bool = False
    # match state
    x: float = 0.0
    y: float = 0.0
    aim: float = 0.0
    alive: bool = False
    respawn_at: float | None = None
    kills: int = 0
    hp: float = 100.0
    # combat: the single-player firing model, made authoritative. `body` is a
    # combat.Combatant, so ballistics.fire_shot and the shields/armor/health
    # model work on players exactly as they already do on guards.
    body: object = None
    loadout: list = field(default_factory=list)
    wi: int = 0                 # index into loadout
    mags: list = field(default_factory=list)
    reserves: list = field(default_factory=list)
    fmode: list = field(default_factory=list)   # fire mode index, per weapon
    charges: list = field(default_factory=list)  # self-recharging magazines:
                                # seconds accumulated toward the next round,
                                # one per weapon slot (sim/weapons.py)
    fire_cd: float = 0.0        # seconds until this weapon can fire again
    reload_t: float = 0.0
    swap_t: float = 0.0
    charging: bool = False      # rail weapon spooling up
    charge: float = 0.0         # seconds held so far
    flashlight: bool = False    # lit? everyone can see a beam, so it is shared
    firing_was: bool = False    # last tick's trigger, for press/release edges
    reload_was: bool = False
    interact_was: bool = False
    light_was: bool = False
    ability_was: bool = False
    knock_was: bool = False
    # the class ability: when it ends, when it may be used again, and how much
    # health a channeled heal still owes
    ability_until: float = 0.0
    ability_ready_at: float = 0.0
    heal_left: float = 0.0
    step_dist: float = 0.0      # metres since this player's last footstep
    stamina: float = 1.0        # sprint fuel, 0..1 (sim/movement.py)
    sprint_locked: bool = False  # bottomed out: no sprinting until it recovers
    # (map_id, sha) of the map this player's client says it has. Readying up
    # starts nothing until it matches the selected map: a player still
    # downloading it would be dropped into a match they cannot draw.
    has_map: tuple = ("", "")
    # the class they are playing (what their body and loadout were built
    # from) and the one they have picked, which takes over at the next spawn
    cls: str = classes.DEFAULT
    cls_next: str = classes.DEFAULT

    # input: queued commands, plus the last one applied (what the sim reads)
    inputs: deque = field(default_factory=deque)
    mx: float = 0.0
    my: float = 0.0
    aim_dist: float = 10.0      # how far away they are aiming: spread scales
    wep_want: int = 0           # weapon the client is asking to hold
    mode_want: int = 0          # fire mode it is asking for
    buttons: int = 0
    last_seq: int = -1          # highest seq RECEIVED (dedupe/reorder guard)
    acked_seq: int = -1         # highest seq APPLIED (what snapshots report)

    # outbound: appended by the tick loop, drained by this player's own thread
    out: deque = field(default_factory=deque)
    dropped_msgs: int = 0       # shed under backpressure, for diagnostics
    _out_cv: threading.Condition = field(default_factory=threading.Condition)
    _sender: object = None
    _dead: bool = False

    def send(self, obj: dict) -> bool:
        """Queue a message for this player. Never touches the socket, so a
        stalled connection costs this player their backlog and nobody else
        their frame rate. False means the connection is finished."""
        with self._out_cv:
            if self._dead:
                return False
            if obj.get("t") in COALESCE_TYPES:
                # A snapshot is a complete picture of the world, so an older one
                # queued behind it is worthless — it never waits in line. This
                # has to happen on every append, not only once the queue is
                # deep: while a stalled sender sits in sendall the outbox looks
                # empty, and stale snapshots would pile up under the cap.
                for i in range(len(self.out) - 1, -1, -1):
                    if self.out[i].get("t") == obj["t"]:
                        del self.out[i]
                        self.dropped_msgs += 1
            self.out.append(obj)
            if len(self.out) > SEND_QUEUE_SOFT:
                self._shed_locked()
            if len(self.out) > SEND_QUEUE_HARD:
                # nothing is being read at the far end; further patience just
                # costs memory
                self._dead = True
                self._out_cv.notify_all()
                return False
            self._out_cv.notify()
        return True

    def _shed_locked(self) -> None:
        """Make room by dropping effects, oldest first. Snapshots have already
        coalesced to one on the way in."""
        i = 0
        while len(self.out) > SEND_QUEUE_SOFT and i < len(self.out):
            if self.out[i].get("t") in DROPPABLE_TYPES:
                del self.out[i]
                self.dropped_msgs += 1
            else:
                i += 1                          # state: must be delivered

    def start_sender(self) -> None:
        self._sender = threading.Thread(target=self._send_loop, daemon=True)
        self._sender.start()

    def _send_loop(self) -> None:
        """Blocking sends live here, where blocking only affects this player."""
        while True:
            with self._out_cv:
                while not self.out and not self._dead:
                    self._out_cv.wait(SENDER_POLL_S)
                if self._dead and not self.out:
                    break
                batch = list(self.out)
                self.out.clear()
            try:
                for obj in batch:
                    P.send_msg(self.sock, obj)
            except OSError:
                with self._out_cv:
                    self._dead = True
                break
        try:
            self.sock.close()       # unblocks this player's receive thread too
        except OSError:
            pass

    def close(self) -> None:
        with self._out_cv:
            self._dead = True
            self.out.clear()
            self._out_cv.notify_all()


class GameServer:
    def __init__(
        self,
        spawn_points: list[tuple[float, float]] | None = None,
        host: str = "0.0.0.0",
        port: int = P.DEFAULT_PORT,
        mode: GameMode = GameMode.FFA,
        duration_s: int = 300,
        map_id: str = "arena",
        maps_dir=None,
    ):
        """spawn_points=None (the normal case) derives them from the map. Pass
        a list to pin them — tests do this to place players deliberately; pinned
        spawns survive a map change, derived ones follow it."""
        self.spawn_points = netmaps.as_spawn_list(spawn_points)
        self._spawns_pinned = bool(spawn_points)
        self.maps_dir = maps_dir or netmaps.DEFAULT_MAPS_DIR
        self.map = None                    # TileMap: geometry players collide with
        self._map_loaded_id: str | None = None
        self._map_dirty = False
        # the loaded map's files and their fingerprint, served to any player
        # who does not have this exact map
        self.map_sha = ""
        self._map_files: dict = {}
        self.host = host
        self.port = port
        self.mode = mode
        self.duration_s = max(MIN_DURATION, min(MAX_DURATION, duration_s))
        self.map_id = map_id

        self.state = ServerState.LOBBY
        self.players: dict[int, NetPlayer] = {}
        self.host_id: int | None = None
        self._next_id = 1
        self._lock = threading.RLock()          # guards players + state

        self.match_end_time = 0.0
        self.end_count = END_COUNTDOWN_FROM
        self._end_next = 0.0
        self._last_snapshot = 0.0

        self._running = False
        self._srv_sock: socket.socket | None = None
        self._threads: list[threading.Thread] = []

        self._rng = random.Random()
        self.doors = DoorSet()             # every doorway and what it is doing
        self.packs = PickupSet()           # health and ammo on the map
        self._broken_glass: set = set()    # coarse cells already shattered
        self._projectiles: list = []       # rockets in flight

        self._load_map()

    # ---------------------------------------------------------------- map

    def _load_map(self) -> bool:
        """Load the geometry the sim collides against, plus the spawn points
        that go with it.

        Only a PLAYABLE map is ever loaded (net.maps.validate). This used to
        fall back to running the match "without collision" when a map failed —
        but the clients load the same map to draw it, so that fallback started
        a match every client immediately crashed out of. Now a map that fails
        keeps the one already loaded, or, if there is none, switches to the
        first playable map in the folder, and the lobby shows which."""
        map_id = self.map_id
        problem = None
        if map_id != self._map_loaded_id:
            # a new map: validate it first. A reload of the one we already
            # have (to reset doors and glass between matches) skips this.
            found = netmaps.validate(map_id, self.maps_dir)
            problem = found[0] if found else None
        if problem is None:
            try:
                m, spawns = netmaps.load_with_spawns(map_id, self.maps_dir)
            except Exception as e:                    # noqa: BLE001
                problem = f"does not load: {e}"
        if problem is not None:
            if self.map is not None and self._map_loaded_id:
                print(f"[server] map {map_id!r} is not playable ({problem}); "
                      f"staying on {self._map_loaded_id!r}")
                self.map_id = self._map_loaded_id
                return False
            fallback = netmaps.first_playable(self.maps_dir)
            if fallback is None or fallback == map_id:
                print(f"[server] map {map_id!r} is not playable ({problem}), "
                      f"and no map in {self.maps_dir} is: refusing to start "
                      f"a match")
                with self._lock:
                    self.map = None
                    self._map_loaded_id = None
                    self.map_sha = ""
                    self._map_files = {}
                return False
            print(f"[server] map {map_id!r} is not playable ({problem}); "
                  f"using {fallback!r} instead")
            self.map_id = fallback
            return self._load_map()
        try:
            files = netmaps.map_files(map_id, self.maps_dir)
        except Exception as e:                        # noqa: BLE001
            print(f"[server] map {map_id!r} loads but its files cannot be "
                  f"read to share: {e}")
            files = {}
        with self._lock:
            self._map_files = files
            self.map_sha = netmaps.files_sha(files) if files else ""
            self.map = m
            self.doors = DoorSet(m)
            self.packs = PickupSet(m)
            self._map_loaded_id = map_id
            if not self._spawns_pinned:
                self.spawn_points = spawns
            n = len(self.spawn_points)
            authored = sum(1 for sp in self.spawn_points if sp in m.spawn_points)
        print(f"[server] map {map_id}: {m.name} "
              f"{m.width_m:.0f}x{m.height_m:.0f}m, {n} spawn points "
              f"({'authored' if authored else 'derived'})")
        return True

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._running = True
        self._srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv_sock.bind((self.host, self.port))
        self._srv_sock.listen(16)
        t_listen = threading.Thread(target=self._accept_loop, daemon=True)
        t_tick = threading.Thread(target=self._tick_loop, daemon=True)
        t_listen.start()
        t_tick.start()
        self._threads = [t_listen, t_tick]

    def stop(self) -> None:
        self._running = False
        with self._lock:
            for p in list(self.players.values()):
                p.close()
                try:
                    p.sock.close()
                except OSError:
                    pass
            self.players.clear()
        if self._srv_sock:
            try:
                self._srv_sock.close()
            except OSError:
                pass

    @staticmethod
    def lan_ip() -> str:
        """Best-effort local LAN IP for display on the host lobby screen."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))          # no packet sent, just picks iface
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"
        finally:
            s.close()

    # ---------------------------------------------------------------- accept / recv

    def _accept_loop(self) -> None:
        while self._running:
            try:
                sock, addr = self._srv_sock.accept()
            except OSError:
                break
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._client_thread, args=(sock, addr),
                             daemon=True).start()

    def _client_thread(self, sock: socket.socket, addr: tuple) -> None:
        pid: int | None = None
        try:
            while self._running:
                msg = P.recv_msg(sock)
                if msg is None:
                    break
                t = msg.get("t")
                if pid is None:
                    if t != P.C_JOIN:
                        P.send_msg(sock, {"t": P.S_REJECT, "reason": "expected join"})
                        break
                    pid = self._handle_join(sock, addr, msg)
                    if pid is None:
                        break
                else:
                    self._handle_msg(pid, msg)
        except (OSError, ValueError):
            pass
        finally:
            if pid is not None:
                self._drop_player(pid)
            try:
                sock.close()
            except OSError:
                pass

    # ---------------------------------------------------------------- join / leave

    def _handle_join(self, sock, addr, msg) -> int | None:
        if msg.get("version") != P.PROTOCOL_VERSION:
            P.send_msg(sock, {"t": P.S_REJECT, "reason": "version mismatch"})
            return None
        with self._lock:
            pid = self._next_id
            self._next_id += 1
            is_host = self.host_id is None
            if is_host:
                self.host_id = pid
            p = NetPlayer(
                id=pid, sock=sock, addr=addr,
                name=str(msg.get("name", f"Player{pid}"))[:20],
                colour=tuple(msg.get("colour", (200, 200, 200))),
                is_host=is_host,
            )
            self.players[pid] = p
            p.start_sender()
            p.send({"t": P.S_WELCOME, "your_id": pid, "is_host": is_host})
            mid_match = self.state != ServerState.LOBBY
            if mid_match:
                # Catch them up on everything about the map that has drifted
                # from the file: a door someone opened, a window someone shot
                # out. Without this they would be walking into walls that are
                # no longer there and shooting at glass that already broke.
                p.send(self._map_state_locked())
        self._broadcast_lobby()
        with self._lock:
            self._broadcast_score()
        return pid

    def _map_state_locked(self) -> dict:
        # caller holds _lock
        return {"t": P.S_MAP_STATE,
                "doors": self.doors.wire(),
                "pickups": self.packs.wire(),
                "glass": [[r, c] for (r, c) in sorted(self._broken_glass)]}

    def _remove_players_locked(self, pids: list[int]) -> bool:
        """Remove players by id and migrate host if needed. Returns True if
        anything changed. Caller holds _lock."""
        changed = False
        host_gone = False
        for pid in pids:
            p = self.players.pop(pid, None)
            if p is not None:
                p.close()
                changed = True
                if p.is_host:
                    host_gone = True
        if host_gone:
            self.host_id = None
            for q in sorted(self.players.values(), key=lambda z: z.id):
                q.is_host = True
                self.host_id = q.id
                break
        return changed

    def _drop_player(self, pid: int) -> None:
        with self._lock:
            changed = self._remove_players_locked([pid])
        if changed:
            self._broadcast_lobby()

    # ---------------------------------------------------------------- message handling

    def _handle_msg(self, pid: int, msg: dict) -> None:
        t = msg.get("t")
        joined = left = False
        with self._lock:
            p = self.players.get(pid)
            if p is None:
                return

            if t == P.C_INPUT:
                if self.state == ServerState.MATCH and p.alive:
                    seq = int(msg.get("seq", 0))
                    if seq > p.last_seq:             # drop stale/reordered
                        p.last_seq = seq
                        p.inputs.append((seq,
                                         float(msg.get("mx", 0.0)),
                                         float(msg.get("my", 0.0)),
                                         float(msg.get("aim", 0.0)),
                                         int(msg.get("buttons", 0)),
                                         int(msg.get("wep", p.wep_want)),
                                         int(msg.get("mode", p.mode_want)),
                                         float(msg.get("aimd", p.aim_dist))))
                        # A backlog means the client is running ahead of us (or
                        # trying to). Keep the newest and drop the rest: it is
                        # corrected by the next snapshot, and nobody gets to
                        # bank extra movement by flooding.
                        while len(p.inputs) > INPUT_QUEUE_MAX:
                            p.inputs.popleft()
                return

            if t == P.C_MAP_REQ:
                self._send_map(p, str(msg.get("map_id", "")))
                return
            if t == P.C_SET_CLASS:
                want = str(msg.get("cls", ""))
                if want in classes.CLASSES and want != p.cls_next:
                    p.cls_next = want
                    self._broadcast_lobby_locked()
                return
            if t == P.C_MAP_HAVE:
                p.has_map = (str(msg.get("map_id", "")), str(msg.get("sha", "")))
                if self.state != ServerState.LOBBY:
                    self._broadcast_lobby_locked()
                    return
                # in the lobby: fall through to the broadcast, and to the
                # autostart this may have been the last thing waiting for

            if self.state != ServerState.LOBBY:
                # lobby-only settings ignored mid-match, except host end_match
                # and a player coming or going
                if t == P.C_END_MATCH and p.is_host and self.state == ServerState.MATCH:
                    self._begin_end_countdown()
                elif t == P.C_JOIN_MATCH and self.state == ServerState.MATCH:
                    if not p.playing:
                        self._join_match_locked(p)
                        joined = True
                elif t == P.C_LEAVE_MATCH and p.playing:
                    self._leave_match_locked(p)
                    left = True
                if joined or left:
                    self._broadcast_lobby_locked()
                    self._broadcast_score()
                    if left and not any(q.playing for q in self.players.values()):
                        # everyone walked out; no sense running an empty match
                        self._begin_end_countdown()
                return

            # ---- lobby settings ----
            if t == P.C_SET_NAME:
                p.name = str(msg.get("name", p.name))[:20]
            elif t == P.C_SET_COLOUR:
                p.colour = tuple(msg.get("colour", p.colour))
            elif t == P.C_SET_TEAM and self.mode == GameMode.TEAM:
                try:
                    p.team = Team(int(msg.get("team", Team.A)))
                except ValueError:
                    p.team = Team.A
            elif t == P.C_SET_READY:
                p.ready = bool(msg.get("ready", False))
            elif t == P.C_SET_CONFIG and p.is_host:
                if "mode" in msg:
                    self.mode = GameMode(int(msg["mode"]))
                    if self.mode == GameMode.FFA:
                        for q in self.players.values():
                            q.team = Team.NONE
                    else:
                        for q in self.players.values():
                            if q.team == Team.NONE:
                                q.team = Team.A
                if "duration_s" in msg:
                    self.duration_s = max(MIN_DURATION,
                                          min(MAX_DURATION, int(msg["duration_s"])))
                if "map_id" in msg:
                    new_id = str(msg["map_id"])
                    if new_id != self.map_id:
                        # Accepted provisionally: _load_map reverts it if the
                        # map turns out not to load, and the lobby broadcast
                        # below therefore never advertises a map nobody has.
                        self.map_id = new_id
                        self._map_dirty = True
            elif t == P.C_START and p.is_host:
                self._try_start(force=True)

        if self._map_dirty:
            # loading touches the disk, so it happens outside the lock the
            # block above held
            self._map_dirty = False
            self._load_map()

        self._broadcast_lobby()
        # auto-start when everyone readies
        self._maybe_autostart()

    # ---------------------------------------------------------------- start / end

    def _maybe_autostart(self) -> None:
        with self._lock:
            if self.state != ServerState.LOBBY:
                return
            ps = list(self.players.values())
            if len(ps) >= MIN_PLAYERS_TO_START and all(
                    q.ready and self._has_map(q) for q in ps):
                self._try_start(force=False)

    def _try_start(self, force: bool) -> None:
        # caller holds _lock
        ps = list(self.players.values())
        if len(ps) < MIN_PLAYERS_TO_START:
            return
        if not force and not all(q.ready for q in ps):
            return
        if self.map is None or self._map_loaded_id != self.map_id:
            self._load_map()          # _lock is an RLock; re-entry is fine
        if self.map is None:
            # nothing playable to put anyone on: starting anyway is exactly
            # what used to close every client's window
            print("[server] not starting: no playable map loaded")
            return
        # one spawn each where possible, from each player's own pool in team
        # modes, and never two people materialising on the same spot
        chosen, used = [], []
        for q in ps:
            team = q.team if self.mode == GameMode.TEAM else None
            pool = netmaps.for_team(self.spawn_points, team)
            free = [sp for sp in pool if sp not in used] or pool
            sp = random.choice(free)
            used.append(sp)
            chosen.append(sp)
        if len(set(id(sp) for sp in chosen)) < len(ps):
            print(f"[server] WARNING: {len(self.spawn_points)} spawns < "
                  f"{len(ps)} players; spawns will repeat")
        now = time.monotonic()
        if self._broken_glass or any(self.doors.values()) \
                or self.doors.moving_keys():
            # a new match gets its windows and doors back as the map authored
            # them
            self._broken_glass.clear()
            self._load_map()
        self._projectiles.clear()
        for q, sp in zip(ps, chosen):
            q.x, q.y = float(sp.x), float(sp.y)
            q.aim = math.radians(sp.facing_deg)
            q.alive = True
            q.respawn_at = None
            q.kills = 0
            q.ready = False
            q.firing_was = False
            q.playing = True
            q.inputs.clear()
            self._spawn_body(q)
        self.state = ServerState.MATCH
        self.match_end_time = now + self.duration_s
        # per-player match_start (carries their own spawn + team)
        roster = [{"id": q.id, "name": q.name, "colour": list(q.colour),
                   "team": int(q.team)} for q in ps]
        for q in ps:
            q.send({
                "t": P.S_MATCH_START,
                "mode": int(self.mode),
                "duration_s": self.duration_s,
                "map_id": self.map_id,
                "map_sha": self.map_sha,
                "spawn": {"x": q.x, "y": q.y, "aim": round(q.aim, 3)},
                "team": int(q.team),
                "players": roster,
            })
        self._broadcast_score()

    def _has_map(self, p: NetPlayer) -> bool:
        """Does this player's client have the selected map, this version?"""
        return bool(self.map_sha) and p.has_map == (self.map_id, self.map_sha)

    def _send_map(self, p: NetPlayer, map_id: str) -> None:
        """Hand a player the selected map. Only that one: the server never
        reads a file a client names, just the map the host already chose and
        validated. Caller holds _lock."""
        if map_id != self.map_id or not self._map_files:
            p.send({"t": P.S_MAP_DATA, "map_id": map_id,
                    "error": f"the host is not on map {map_id!r}"})
            return
        size = sum(len(d) for d in self._map_files.values())
        if size > netmaps.MAX_MAP_BYTES:
            p.send({"t": P.S_MAP_DATA, "map_id": map_id,
                    "error": f"map is {size // 1024} KiB, too large to send"})
            return
        print(f"[server] sending map {map_id!r} ({size // 1024} KiB) to "
              f"{p.name}")
        p.send({"t": P.S_MAP_DATA, "map_id": map_id, "sha": self.map_sha,
                "files": dict(self._map_files)})

    def _spawn_for(self, p: NetPlayer, top: int = 1):
        """A spawn point for this player: from their side's pool in team modes,
        and as far from everyone still fighting as the map allows.

        Dropping a latecomer into a firefight — or worse, into somebody's line
        of fire while they are mid-burst — is what makes joining late feel
        unfair."""
        team = p.team if self.mode == GameMode.TEAM else None
        away = [(q.x, q.y) for q in self.players.values()
                if q.playing and q.alive and q.id != p.id]
        return netmaps.pick(self.spawn_points, team=team, away_from=away,
                            rng=self._rng, top=top)

    def _join_match_locked(self, p: NetPlayer) -> None:
        """Drop a connected player into the match that is already running."""
        if self.mode == GameMode.TEAM:
            # put them on the thinner side, whatever they picked in the lobby
            counts = {Team.A: 0, Team.B: 0}
            for q in self.players.values():
                if q.playing and q.team in counts:
                    counts[q.team] += 1
            p.team = Team.A if counts[Team.A] <= counts[Team.B] else Team.B
        sp = self._spawn_for(p)
        p.x, p.y = float(sp.x), float(sp.y)
        p.aim = math.radians(sp.facing_deg)
        p.playing = True
        p.alive = True
        p.respawn_at = None
        p.firing_was = False
        p.inputs.clear()
        self._spawn_body(p)
        roster = [{"id": q.id, "name": q.name, "colour": list(q.colour),
                   "team": int(q.team)}
                  for q in self.players.values() if q.playing]
        # the map may have moved on since they last saw it — or since they
        # connected, if they have been sitting in the lobby a while
        p.send(self._map_state_locked())
        p.send({
            "t": P.S_MATCH_START,
            "mode": int(self.mode),
            "duration_s": self.duration_s,
            "map_id": self.map_id,
            "map_sha": self.map_sha,
            "spawn": {"x": p.x, "y": p.y, "aim": round(p.aim, 3)},
            "team": int(p.team),
            "players": roster,
        })

    def _leave_match_locked(self, p: NetPlayer) -> None:
        """Take a player out of the match without dropping their connection.
        Their score stays on the board until the match ends."""
        p.playing = False
        p.alive = False
        p.respawn_at = None
        p.body = None
        p.inputs.clear()
        p.buttons = 0
        p.charging = False
        p.charge = 0.0

    def _begin_end_countdown(self) -> None:
        # caller holds _lock
        self.state = ServerState.END_COUNTDOWN
        self.end_count = END_COUNTDOWN_FROM
        self._end_next = time.monotonic()
        self._broadcast({"t": P.S_MATCH_END})

    def _reset_to_lobby(self) -> None:
        # caller holds _lock
        self.state = ServerState.LOBBY
        for q in self.players.values():
            q.alive = False
            q.playing = False
            q.respawn_at = None
            q.ready = False
            q.kills = 0

    # ---------------------------------------------------------------- tick loop

    def _tick_loop(self) -> None:
        next_t = time.monotonic()
        snap_interval = 1.0 / SNAPSHOT_HZ
        while self._running:
            now = time.monotonic()
            with self._lock:
                if self.state == ServerState.MATCH:
                    self._tick_match(now)
                    if now - self._last_snapshot >= snap_interval:
                        self._last_snapshot = now
                        self._broadcast_snapshot(now)
                elif self.state == ServerState.END_COUNTDOWN:
                    self._tick_end_countdown(now)
                elif self.state == ServerState.LOBBY:
                    # idle liveness: periodic no-op broadcast reaps dead sockets
                    # (send failure drops the player), which drives host migration.
                    if now - self._last_snapshot >= 1.0:
                        self._last_snapshot = now
                        self._reap_dead()
            next_t += TICK_DT
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()          # fell behind; resync

    def _step_doors(self, dt: float, now: float) -> None:
        """Move every door that is travelling, apply the ones that arrive, and
        write the slit between any moving panels into the map.

        A moving door is never walkable: `set_door` opens the doorway at the
        end of the travel and nowhere else. Its slit lets sight through, and on
        a blast door bullets — which is what makes those five seconds a
        firefight rather than a wait."""
        for (r, c), is_open, changed in self.doors.step(dt):
            if changed:
                set_door(self.map, None, r, c, is_open)
            if self.doors.door((r, c)).heavy:
                # the panels seating: a second, quieter cue that it is done
                self._sound(c + 0.5, r + 0.5, DOOR_SOUND_M * 0.6, "door",
                            0, label="door")
        # the slit between moving panels. The server needs it for BULLETS: on
        # a blast door, a shot aimed through the gap has to get through, and
        # the server is the one that decides what a shot hits
        if self.map is not None:
            for (r, c), gap in self.doors.gap_changes(self.map.subdiv):
                d = self.doors.door((r, c))
                set_door_gap(self.map, r, c, gap, d.axis,
                             bullets=d.shoot_through)

    def _step_pickups(self, dt: float, now: float) -> None:
        """Respawn the packs whose time is up, then hand out any a live player
        is standing on.

        The server is the only thing that decides this. Two players reaching a
        pack on the same tick is not a tie — the first one this loop reaches
        takes it, and `PickupSet.take` returns False for the second."""
        for pid in self.packs.step(dt):
            self._broadcast({"t": P.S_PICKUP, "pid": pid, "live": True,
                             "by": 0, "dur": 0.0})
        if not self.packs.packs:
            return
        for p in self.players.values():
            if not (p.playing and p.alive and p.body is not None):
                continue
            got = self.packs.at(p.x, p.y)
            if got is None:
                continue
            if not self._apply_pack(p, got):
                continue                  # full up: leave it for somebody else
            if not self.packs.take(got.id):
                continue                  # somebody beat them to it this tick
            self._broadcast({"t": P.S_PICKUP, "pid": got.id, "live": False,
                             "by": p.id, "dur": round(got.respawn_s, 2)})
            self._sound(got.x, got.y, perception.MAGDROP_REACH_M, "reload",
                        p.id, label=f"{p.id}/reload")

    def _apply_pack(self, p: NetPlayer, got) -> bool:
        """Give a player a pack. False if it would do nothing."""
        if got.kind == pk.HEALTH:
            gain = pk.health_gain(p.body)
            if gain is None:
                return False
            p.body.health = gain
            p.hp = gain
            return True
        gain = pk.ammo_gain(p.loadout, p.reserves, weapons.ROSTER)
        if gain is None:
            return False
        p.reserves = gain
        return True

    def _tick_match(self, now: float) -> None:
        self._step_doors(TICK_DT, now)
        self._step_pickups(TICK_DT, now)
        for p in self.players.values():
            if not p.playing:
                continue                  # connected, but sitting this one out
            if not p.alive:
                if p.respawn_at is not None and now >= p.respawn_at:
                    self._respawn(p)
            else:
                # exactly one command per tick: simulated time can never run
                # ahead of real time, whatever a client sends
                if p.inputs:
                    (seq, mx, my, aim, buttons,
                     wep, mode, aimd) = p.inputs.popleft()
                    p.mx, p.my, p.aim, p.buttons = mx, my, aim, buttons
                    p.wep_want, p.mode_want, p.aim_dist = wep, mode, max(0.3, aimd)
                    p.acked_seq = seq
                    before = (p.x, p.y)
                    self.integrate(p, TICK_DT)
                    self._footsteps(p, before, now)
                # no command waiting: the player holds position. Nothing is
                # invented on their behalf, so a stalled client cannot drift.
        for p in self.players.values():
            if p.alive:
                self._tick_ability(p, TICK_DT, now)
                self._weapon_tick(p, TICK_DT, now)
        self.resolve_shots(now)
        if now >= self.match_end_time:
            self._begin_end_countdown()

    def _tick_end_countdown(self, now: float) -> None:
        if now >= self._end_next:
            self._broadcast({"t": P.S_END_COUNT, "n": self.end_count})
            self.end_count -= 1
            self._end_next = now + 1.0
            if self.end_count < 0:
                self._reset_to_lobby()
                self._broadcast_lobby()
                self._broadcast_score()

    # ---------------------------------------------------------------- sim hooks

    def integrate(self, p: NetPlayer, dt: float) -> None:
        """Authoritative movement. The client's input vector is clamped to the
        unit disc (so a diagonal isn't 1.41x faster, and a hacked client can't
        just send 5.0), scaled by the speed its stance allows, and collided
        against the map per-axis so bodies slide along walls.

        This runs sim/movement.py — the same function single-player uses — so a
        client predicting its own movement arrives at the same position the
        server does, and nothing rubber-bands."""
        stance = P.stance_of(p.buttons)
        moving = (p.mx, p.my) != (0.0, 0.0)
        # sprint fuel, drained here so a client cannot run for ever by simply
        # holding the bit down
        stance = movement.allowed_stance(stance, p.stamina, p.sprint_locked)
        p.stamina, p.sprint_locked = movement.step_stamina(
            p.stamina, p.sprint_locked, stance, moving, dt)
        if self.map is None:                # map failed to load: move freely
            mx, my = movement.clamp_input(p.mx, p.my)
            v = (movement.speed_of(stance) * classes.get(p.cls).speed_mult
                 * self._ability_speed(p) * dt)
            p.x += mx * v
            p.y += my * v
            return
        p.x, p.y = movement.step(
            self.map, p.x, p.y, p.mx, p.my, stance, dt,
            speed_mult=classes.get(p.cls).speed_mult * self._ability_speed(p))

    # ------------------------------------------------------------ weapons

    def _weapon(self, p: NetPlayer):
        return weapons.ROSTER[p.loadout[p.wi]]

    def _fire_mode(self, p: NetPlayer, w) -> str:
        modes = w.fire_modes()
        return modes[min(p.fmode[p.wi], len(modes) - 1)]

    def _reset_loadout(self, p: NetPlayer) -> None:
        p.loadout = list(classes.get(p.cls).loadout)
        p.mags = [weapons.ROSTER[k].mag for k in p.loadout]
        p.reserves = [weapons.ROSTER[k].reserve for k in p.loadout]
        p.fmode = [0] * len(p.loadout)
        p.charges = [0.0] * len(p.loadout)
        p.wi = p.wep_want = 0
        p.fire_cd = p.reload_t = p.swap_t = 0.0
        p.charging = False
        p.charge = 0.0
        p.step_dist = 0.0

    def _spawn_body(self, p: NetPlayer) -> None:
        """Give a player a fresh Combatant. Damage lands on this, not on the
        NetPlayer, so every weapon in sim/ works on players untouched.

        This is where a class picked since the last spawn takes over."""
        p.cls = classes.get(p.cls_next).key
        p.body = classes.make_body(p.cls, p.x, p.y)
        p.ability_until = 0.0
        p.ability_ready_at = 0.0
        p.heal_left = 0.0
        p.ability_was = p.knock_was = False
        p.stamina, p.sprint_locked = 1.0, False
        p.body.faction = "player"
        p.body.net_id = p.id
        p.hp = p.body.max_health
        self._reset_loadout(p)

    def _sound(self, x: float, y: float, reach_m: float, clip: str,
               src_id: int = 0, label: str | None = None,
               stance: str = "") -> None:
        """Tell everyone something audible happened here.

        Emission is authoritative — the server decides who made a noise and how
        loud — but audibility is not: each client solves its own propagation
        field and works out what it can hear and from which direction. That
        keeps the expensive solve off the tick loop, at the cost of trusting
        clients not to listen to what they shouldn't."""
        self._broadcast({"t": P.S_SOUND, "x": round(x, 2), "y": round(y, 2),
                         "energy": float(reach_m), "clip": clip,
                         "label": label or clip, "id": src_id,
                         "stance": stance})

    def _footsteps(self, p: NetPlayer, before: tuple, now: float) -> None:
        """A footstep every stride, as loud as the stance earns. Crawling
        carries barely two metres; a run carries eleven, and further on metal
        grating, which is what makes the floor you choose matter."""
        if self.map is None:
            return
        moved = math.hypot(p.x - before[0], p.y - before[1])
        if moved <= 0.0:
            return
        stance = P.stance_of(p.buttons)
        p.step_dist += moved
        stride = perception.STRIDE.get(stance, 0.7)
        if p.step_dist < stride:
            return
        p.step_dist = 0.0
        if self._vanished(p):
            return                      # a vanished Saboteur crosses in silence
        cx, cy = self.map.cell_of(p.x, p.y)
        mult = float(self.map.footstep_mult[cy, cx])
        self._sound(p.x, p.y, perception.FOOTSTEP_REACH_M[stance] * mult,
                    "footstep", p.id, label=f"{p.id}/step", stance=stance)

    def _weapon_tick(self, p: NetPlayer, dt: float, now: float) -> None:
        """Timers: cooldown, reload, weapon swap, shield regeneration."""
        if p.body is None:
            return
        p.body.tick(dt, now)
        p.fire_cd = max(0.0, p.fire_cd - dt)
        p.swap_t = max(0.0, p.swap_t - dt)
        # a weapon that makes its own ammunition does it slung as well as held
        if weapons.recharge_step(p.loadout, p.mags, p.charges, dt):
            self._sound(p.x, p.y, perception.MAGDROP_REACH_M * 0.35,
                        "reload", p.id, label=f"{p.id}/reload")

        # the client asks for a weapon; the server decides when it is in hand
        if p.wep_want != p.wi and p.swap_t <= 0.0 \
                and 0 <= p.wep_want < len(p.loadout):
            p.wi = p.wep_want
            p.reload_t = 0.0
            p.swap_t = SWAP_TIME
            p.charging = False
            p.charge = 0.0
        w = self._weapon(p)
        modes = w.fire_modes()
        p.fmode[p.wi] = min(max(p.mode_want, 0), len(modes) - 1)

        # reload: on the button's edge, so holding R doesn't loop
        want_reload = bool(p.buttons & P.BTN_RELOAD)
        if want_reload and not p.reload_was and p.reload_t <= 0.0 \
                and p.swap_t <= 0.0 and p.mags[p.wi] < w.mag \
                and p.reserves[p.wi] != 0:
            p.reload_t = w.reload_step
            self._sound(p.x, p.y, perception.MAGDROP_REACH_M, "reload", p.id,
                        label=f"{p.id}/reload")
        p.reload_was = want_reload

        if p.reload_t > 0.0:
            p.reload_t = max(0.0, p.reload_t - dt)
            if p.reload_t <= 0.0:
                self._finish_reload(p, w)

        want_light = bool(p.buttons & P.BTN_LIGHT)
        if want_light and not p.light_was:
            p.flashlight = not p.flashlight
        p.light_was = want_light

        want_use = bool(p.buttons & P.BTN_INTERACT)
        if want_use and not p.interact_was:
            self._interact(p, now)
        p.interact_was = want_use

        want_knock = bool(p.buttons & P.BTN_KNOCK)
        if want_knock and not p.knock_was:
            self._sound(p.x, p.y, perception.KNOCK_REACH_M, "knock", p.id,
                        label=f"{p.id}/knock")
        p.knock_was = want_knock

        want_ability = bool(p.buttons & P.BTN_ABILITY)
        if want_ability and not p.ability_was:
            self._start_ability(p, now)
        p.ability_was = want_ability

    # ------------------------------------------------------------ abilities

    def _start_ability(self, p: NetPlayer, now: float) -> None:
        """One special per class, on a shared cooldown. Everything it does is
        decided here; clients only predict and draw it."""
        if now < p.ability_ready_at or p.ability_until > now:
            return
        ab = classes.ability_of(p.cls)
        p.ability_until = now + ab.duration
        p.ability_ready_at = p.ability_until + classes.ABILITY_COOLDOWN_S
        if ab.key == "heal" and p.body is not None:
            p.heal_left = p.body.max_health * classes.HEAL_FRACTION
        elif ab.key == "brace" and p.body is not None:
            p.body.incoming_mult = classes.BRACE_MITIGATION
        self._broadcast({"t": P.S_ABILITY, "id": p.id, "ab": ab.key,
                         "dur": round(ab.duration, 2)})
        self._sound(p.x, p.y, ABILITY_SOUND_M, "ability", p.id,
                    label=f"{p.id}/ability")

    def _end_ability(self, p: NetPlayer, now: float) -> None:
        if p.ability_until <= 0.0:
            return
        p.ability_until = 0.0
        p.heal_left = 0.0
        if p.body is not None:
            p.body.incoming_mult = 1.0
        # the cooldown runs from the end, so a cancelled channel does not get
        # to start again straight away
        p.ability_ready_at = max(p.ability_ready_at,
                                 now + classes.ABILITY_COOLDOWN_S)

    def _tick_ability(self, p: NetPlayer, dt: float, now: float) -> None:
        """Advance whatever this player's ability is doing this tick."""
        if p.ability_until <= 0.0:
            return
        ab = classes.ability_of(p.cls)
        moving = (p.mx, p.my) != (0.0, 0.0)
        if ab.key == "heal":
            hit = p.body is not None and now - p.body.last_hit_t < dt * 1.5
            if moving or bool(p.buttons & P.BTN_FIRE) or hit:
                self._end_ability(p, now)          # like a reload: interrupted
                return
            if p.body is not None and p.heal_left > 0.0:
                # paid out as it goes, so an interrupted heal still counted
                give = min(p.heal_left, p.body.max_health
                           * classes.HEAL_FRACTION * dt / ab.duration)
                p.body.health = min(p.body.max_health, p.body.health + give)
                p.heal_left -= give
        elif ab.key == "brace" and p.body is not None:
            # bracing is standing your ground: the moment you move, it is off
            p.body.incoming_mult = (1.0 if moving
                                    else classes.BRACE_MITIGATION)
        if now >= p.ability_until:
            self._end_ability(p, now)

    def _vanished(self, p: NetPlayer) -> bool:
        """Is this player unseen and silent right now?"""
        return (p.ability_until > 0.0
                and classes.ability_of(p.cls).key == "vanish")

    def _ability_speed(self, p: NetPlayer) -> float:
        return classes.ability_speed(p.cls, p.ability_until > 0.0)

    def _interact(self, p: NetPlayer, now: float) -> None:
        """Open or close the nearest door. Same rules as single-player, with
        one addition: you cannot close a door on somebody else either."""
        if self.map is None or not self.doors:
            return
        best, bd = None, INTERACT_RANGE
        for (r, c) in self.doors:
            d = math.hypot(c + 0.5 - p.x, r + 0.5 - p.y)
            if d < bd:
                best, bd = (r, c), d
        if best is None:
            return
        door = self.doors.door(best)
        if door.moving and not door.reversible:
            return                        # once a blast door starts, it finishes
        # a quick door mid-travel turns round, so flip where it is HEADING
        want_open = not (door.target if door.moving else door.is_open)
        if not want_open:
            # closing: nobody may be standing in the leaf, including the person
            # pulling it shut
            br, bc = best
            for q in self.players.values():
                if not q.alive:
                    continue
                nx = min(max(q.x, bc), bc + 1.0)
                ny = min(max(q.y, br), br + 1.0)
                if math.hypot(q.x - nx, q.y - ny) < movement.BODY_R + 0.05:
                    self._broadcast({"t": P.S_DOOR, "r": br, "c": bc,
                                     "open": True, "id": p.id,
                                     "blocked": True})
                    return
        moved, changed = self.doors.begin(best, want_open)
        if moved is None:
            return
        if changed:
            # sealing: the gap closes the moment the panels move
            set_door(self.map, None, best[0], best[1], moved.is_open)
        # The panels start moving NOW; the wall stops being a wall when they
        # arrive (_step_doors). Clients are told how long the travel takes and
        # run the same clock, so nobody has to be told again when it lands.
        # `dur` is the travel LEFT, which for a fresh start is the whole door
        # and for a turned-round one is only the part it had already covered
        self._broadcast({"t": P.S_DOOR, "r": best[0], "c": best[1],
                         "open": want_open, "dur": round(moved.left, 3),
                         "id": p.id, "blocked": False})
        clip = "door_heavy" if door.heavy else "door"
        self._sound(best[1] + 0.5, best[0] + 0.5, DOOR_SOUND_M, clip, p.id,
                    label=f"{p.id}/{clip}")

    def _finish_reload(self, p: NetPlayer, w) -> None:
        if w.shell_reload:
            p.mags[p.wi] += 1                      # one shell at a time
            if p.reserves[p.wi] > 0:
                p.reserves[p.wi] -= 1
            if p.mags[p.wi] < w.mag and p.reserves[p.wi] != 0:
                p.reload_t = w.shell_reload_s      # keep loading
            return
        need = w.mag - p.mags[p.wi]
        if p.reserves[p.wi] < 0:                   # unlimited reserve
            p.mags[p.wi] = w.mag
        else:
            take = min(need, p.reserves[p.wi])
            p.mags[p.wi] += take
            p.reserves[p.wi] -= take

    # ------------------------------------------------------------ shooting

    def resolve_shots(self, now: float) -> None:
        """Every trigger held this tick, resolved authoritatively."""
        if self.map is None:
            return
        live = [q for q in self.players.values() if q.alive and q.body]
        for q in live:
            q.body.x, q.body.y = q.x, q.y      # bodies follow the sim
        for p in live:
            self._trigger(p, live, now)
        self._advance_projectiles(live, now)

    def _trigger(self, p: NetPlayer, live: list, now: float) -> None:
        w = self._weapon(p)
        held = bool(p.buttons & P.BTN_FIRE)
        pressed = held and not p.firing_was
        released = p.firing_was and not held
        p.firing_was = held
        blocked = p.reload_t > 0.0 or p.swap_t > 0.0 or p.fire_cd > 0.0

        if w.is_cooked:
            # a grenade cooks while you hold it: the fuse runs from the moment
            # the pin comes out, in your hand as readily as in the air
            if held and not blocked and p.mags[p.wi] > 0:
                p.charging = True
                p.charge += TICK_DT
                if p.charge >= w.fuse_s:
                    self._cook_off(p, live, w, now)
            elif released:
                cooked = p.charge if p.charging else 0.0
                fire = p.charging and not blocked and p.mags[p.wi] > 0
                p.charging = False
                p.charge = 0.0
                if fire:
                    self._fire(p, live, w, now, charge=cooked)
            return

        if p.loadout[p.wi] in RAIL_IDS:
            # rail weapons spool up while held and go off on release, so the
            # shot you take is the one you decided to stop charging
            if held and not blocked and p.mags[p.wi] > 0:
                p.charging = True
                p.charge = min(RAIL_CHARGE_MAX, p.charge + TICK_DT)
            elif released:
                frac = min(1.0, p.charge / RAIL_CHARGE_MAX) if p.charging else 0.0
                fire = p.charging and not blocked and p.mags[p.wi] > 0
                p.charging = False
                p.charge = 0.0
                if fire:
                    self._fire(p, live, w, now, charge=frac)
            return

        if blocked:
            return
        if p.mags[p.wi] <= 0:
            if (p.ability_until > 0.0
                    and classes.ability_of(p.cls).key == "blitz"
                    and p.reserves[p.wi] != 0 and p.reload_t <= 0.0):
                # blitz: the magazine comes back the instant it runs dry, out
                # of the ammo you are actually carrying
                self._finish_reload(p, w)
                self._sound(p.x, p.y, perception.MAGDROP_REACH_M, "reload",
                            p.id, label=f"{p.id}/reload")
                return
            if pressed:                    # the click that tells you it's empty
                self._sound(p.x, p.y, perception.DRYFIRE_REACH_M, "dryfire",
                            p.id, label=f"{p.id}/dry")
            return
        auto = self._fire_mode(p, w) == "auto"
        if auto and held:
            self._fire(p, live, w, now, auto=True)
        elif pressed:
            self._fire(p, live, w, now)

    def _fire(self, p: NetPlayer, live: list, w, now: float,
              charge: float = 0.0, auto: bool = False) -> None:
        """One trigger pull: rounds x pellets, each with its own spread, damage
        resolved by the same ballistics the single-player game uses."""
        if charge > 0.0 and not w.is_cooked:
            mult = 1.0 + RAIL_CHARGE_BOOST * charge
            w = dataclasses.replace(w, dmg_lo=w.dmg_lo * mult,
                                    dmg_hi=w.dmg_hi * mult)
            if charge >= 0.999 and w.pen_charged > 0.0:
                w = dataclasses.replace(w, pen=w.pen_charged)
        if self._vanished(p):
            self._end_ability(p, now)   # a shot or a swing gives you away
        if w.is_melee:
            self._swing(p, live, w, now)
            return
        p.fire_cd = w.auto_refire if auto else w.burst_time
        rounds = 1 if auto else w.burst
        acc = w.auto_accuracy if auto else None
        moving = (p.mx, p.my) != (0.0, 0.0)
        travels = w.blast_r > 0.0 and w.projectile_speed > 0.0

        others = [q.body for q in live if q.id != p.id]
        by_body = {id(q.body): q for q in live}
        was_alive = {id(b): b.alive for b in others}

        m = self.map
        ox, oy = self._muzzle(p)
        aim_d = max(0.3, p.aim_dist)

        segs, impact, shattered, fired = [], (ox, oy), [], 0
        thrown_fuse = 0.0
        for _ in range(rounds):
            if p.mags[p.wi] <= 0:
                break
            p.mags[p.wi] -= 1
            fired += 1
            for _pellet in range(w.pellets):
                hd = (p.aim + weapons.pellet_offset(w, self._rng)
                      + weapons.jitter(w, aim_d, moving, self._rng, acc))
                sh = ballistics.fire_shot(m, m.blocks_bullets, m.pen_cost,
                                          m.glass, (ox, oy), hd, w, others,
                                          self._rng, now,
                                          apply_damage=not travels)
                segs.extend(sh.segments)
                impact = sh.impact
                if sh.shattered:
                    shattered.extend(sh.shattered)
                if travels:
                    d = math.hypot(sh.impact[0] - ox, sh.impact[1] - oy)
                    if 0.0 < w.range_m < d:
                        # nothing stopped it: a shell that outruns its own
                        # range goes off there rather than carrying on down
                        # the hall to whatever wall is at the end of it
                        f = w.range_m / d
                        sh = dataclasses.replace(
                            sh,
                            impact=(ox + (sh.impact[0] - ox) * f,
                                    oy + (sh.impact[1] - oy) * f),
                            blast_at=(ox + (sh.blast_at[0] - ox) * f,
                                      oy + (sh.blast_at[1] - oy) * f))
                        d = w.range_m
                    # only a fused round runs a clock. A rocket or a plasma
                    # bolt has no fuse at all and goes off where it lands:
                    # giving it one detonates it in the shooter's face.
                    fuse = max(0.05, w.fuse_s - charge) if w.fuse_s > 0.0 else 0.0
                    self._projectiles.append({
                        # the roster key, not w.name: the client looks the
                        # weapon up by key to light and sound the blast
                        "owner": p.id, "w": w, "wid": p.loadout[p.wi],
                        "x": ox, "y": oy,
                        "ix": sh.impact[0], "iy": sh.impact[1],
                        "bx": sh.blast_at[0], "by": sh.blast_at[1],
                        "dist": d, "flown": 0.0,
                        # what is left of the fuse after the cooking
                        "fuse": fuse})
                    thrown_fuse = fuse
                elif w.blast_r > 0.0:
                    # the shooter is not immune to their own blast
                    ballistics.blast(sh.blast_at, w.blast_r, w,
                                     others + [p.body], self._rng, now, m=m)
        if not fired:
            return

        self._apply_glass(shattered, now)
        self._collect_kills(p, others, was_alive, by_body, w, now)

        self._broadcast({
            "t": P.S_SHOT, "id": p.id, "x": round(ox, 2), "y": round(oy, 2),
            "heading": round(p.aim, 3), "wep": p.loadout[p.wi],
            "charge": round(charge, 2),
            # a thrown or flying round IS the visual; a tracer line to
            # where it will land gives the throw away and looks like a shot
            "segs": [] if travels else
                    [[round(a[0], 2), round(a[1], 2),
                      round(b[0], 2), round(b[1], 2)] for a, b, _k in segs[:24]],
            "impact": [round(impact[0], 2), round(impact[1], 2)],
            "blast": w.blast_r if not travels else 0.0,
            "travel": w.projectile_speed if travels else 0.0,
            # a fused round keeps being drawn where it lands until it goes
            # off; the client runs the same clock rather than guessing
            "fuse": round(thrown_fuse, 2) if thrown_fuse else 0.0,
        })
        self._sound(ox, oy, w.sound_reach_m, "fire", p.id,
                    label=f"{p.id}/fire")

    def _swing(self, p: NetPlayer, live: list, w, now: float) -> None:
        """A blade: no projectile, no spread. Whoever is inside the reach and
        the arc takes it, and takes triple if you are behind them."""
        p.fire_cd = w.burst_time
        others = [q for q in live if q.id != p.id and q.body is not None]
        by_body = {id(q.body): q for q in others}
        was_alive = {id(q.body): q.body.alive for q in others}
        hit = None
        best = w.melee_range + movement.BODY_R
        for q in others:
            d = math.hypot(q.x - p.x, q.y - p.y)
            if d > best:
                continue
            off = abs((math.atan2(q.y - p.y, q.x - p.x) - p.aim + math.pi)
                      % (2 * math.pi) - math.pi)
            if off > math.radians(w.melee_arc_deg):
                continue
            if self.map is not None:
                cx, cy = self.map.cell_of((p.x + q.x) / 2, (p.y + q.y) / 2)
                if self.map.blocks_bullets[cy, cx]:
                    continue              # not through a wall
            best, hit = d, q
        if hit is not None:
            # behind them: the angle between where they face and where the
            # blade comes from
            from_behind = abs(
                (math.atan2(p.y - hit.y, p.x - hit.x) - hit.aim + math.pi)
                % (2 * math.pi) - math.pi) > math.radians(100.0)
            dmg = weapons.roll_damage(w, self._rng) * (w.backstab if from_behind
                                                       else 1.0)
            hit.body.take(dmg, now, w.shield_mult, w.health_mult, src=(p.x, p.y))
            self._collect_kills(p, [q.body for q in others], was_alive,
                                by_body, w, now)
        self._broadcast({"t": P.S_MELEE, "id": p.id,
                         "heading": round(p.aim, 3),
                         "reach": w.melee_range,
                         "hit": hit.id if hit is not None else 0})
        self._sound(p.x, p.y, w.sound_reach_m, "melee", p.id,
                    label=f"{p.id}/melee")

    def _cook_off(self, p: NetPlayer, live: list, w, now: float) -> None:
        """Held too long. It goes off where it is: in their hand."""
        p.charging = False
        p.charge = 0.0
        p.mags[p.wi] = max(0, p.mags[p.wi] - 1)
        p.fire_cd = w.burst_time
        bodies = [q.body for q in live if q.body]
        by_body = {id(q.body): q for q in live}
        was_alive = {id(b): b.alive for b in bodies}
        ballistics.blast((p.x, p.y), w.blast_r, w, bodies, self._rng, now,
                         m=self.map)
        self._collect_kills(p, bodies, was_alive, by_body, w, now)
        self._broadcast({"t": P.S_SHOT, "id": p.id, "x": round(p.x, 2),
                         "y": round(p.y, 2), "heading": round(p.aim, 3),
                         "wep": p.loadout[p.wi], "charge": 0.0, "segs": [],
                         "impact": [round(p.x, 2), round(p.y, 2)],
                         "blast": w.blast_r, "travel": 0.0})
        self._sound(p.x, p.y, w.sound_reach_m, "fire", p.id,
                    label=f"{p.id}/fire")

    def _muzzle(self, p: NetPlayer) -> tuple[float, float]:
        """Where the shot starts: the muzzle, a bit ahead of the body — unless
        getting there would cross cover.

        The offset is over a metre, which is wider than some walls. Taking it
        on trust lets a player standing against a thin wall fire from the far
        side of it, so the path from body to muzzle is checked the same way the
        bullet's path will be, and a blocked one falls back to firing from the
        body centre."""
        m = self.map
        ox = p.x + math.cos(p.aim) * MUZZLE_M
        oy = p.y + math.sin(p.aim) * MUZZLE_M
        steps = max(2, int(MUZZLE_M * m.cells_per_metre))
        for i in range(1, steps + 1):
            f = i / steps
            cx, cy = m.cell_of(p.x + (ox - p.x) * f, p.y + (oy - p.y) * f)
            if m.blocks_bullets[cy, cx]:
                return p.x, p.y
        return ox, oy

    def _apply_glass(self, cells, now: float) -> None:
        """Break panes once, on the server, and tell everyone which. Glass that
        broke on one machine only would leave players with different walls to
        see and shoot through."""
        if not cells or self.map is None:
            return
        broke = break_glass_cells(self.map, cells, broken=self._broken_glass)
        if not broke:
            return
        self._broadcast({"t": P.S_GLASS, "cells": [[r, c] for r, c in broke]})
        for r, c in broke:
            self._sound(c + 0.5, r + 0.5, perception.GLASS_BREAK_REACH_M,
                        "glass")

    def _collect_kills(self, killer: NetPlayer, bodies: list, was_alive: dict,
                       by_body: dict, w, now: float) -> None:
        """Anything that was alive before this shot and isn't now died to it."""
        for b in bodies:
            if was_alive.get(id(b)) and not b.alive:
                victim = by_body.get(id(b))
                if victim is not None:
                    self.kill(killer, victim, weapon=getattr(w, "name", None))

    def _advance_projectiles(self, live: list, now: float) -> None:
        """Rockets fly rather than arrive: a launcher fired across a room gives
        its target most of a second to not be standing there."""
        if not self._projectiles:
            return
        still = []
        for pr in self._projectiles:
            if pr.get("fuse", 0.0) > 0.0:
                # a fuse runs wherever the thing is: in the air, or on the
                # floor where it landed. Let one go too late and it goes off
                # between you and whatever you threw it at.
                pr["fuse"] -= TICK_DT
                flying = pr["flown"] < pr["dist"]
                if flying:
                    pr["flown"] = min(pr["dist"],
                                      pr["flown"] + pr["w"].projectile_speed * TICK_DT)
                if pr["fuse"] > 0.0:
                    still.append(pr)
                    continue
                if pr["flown"] < pr["dist"]:
                    # it never got there: work out where it actually is
                    f = pr["flown"] / max(pr["dist"], 1e-6)
                    pr["bx"] = pr["x"] + (pr["bx"] - pr["x"]) * f
                    pr["by"] = pr["y"] + (pr["by"] - pr["y"]) * f
                    pr["ix"] = pr["x"] + (pr["ix"] - pr["x"]) * f
                    pr["iy"] = pr["y"] + (pr["iy"] - pr["y"]) * f
            elif pr["flown"] < pr["dist"]:
                pr["flown"] += pr["w"].projectile_speed * TICK_DT
                if pr["flown"] < pr["dist"]:
                    still.append(pr)
                    continue
            shooter = self.players.get(pr["owner"])
            bodies = [q.body for q in live if q.body]
            by_body = {id(q.body): q for q in live}
            was_alive = {id(b): b.alive for b in bodies}
            segs = []
            if pr["w"].burst_pellets > 0:
                # a flak shell: the blast for whoever it hit, then the ring
                segs = ballistics.burst((pr["bx"], pr["by"]), pr["w"],
                                        bodies, self._rng, now, m=self.map)
            else:
                ballistics.blast((pr["bx"], pr["by"]), pr["w"].blast_r, pr["w"],
                                 bodies, self._rng, now, m=self.map)
            if shooter is not None:
                self._collect_kills(shooter, bodies, was_alive, by_body,
                                    pr["w"], now)
            self._broadcast({"t": P.S_SHOT, "id": pr["owner"],
                             "x": round(pr["ix"], 2), "y": round(pr["iy"], 2),
                             "heading": 0.0, "wep": pr["wid"],
                             "charge": 0.0,
                             "segs": [[round(a[0], 2), round(a[1], 2),
                                       round(b[0], 2), round(b[1], 2)]
                                      for a, b, _k in segs[:48]],
                             "impact": [round(pr["ix"], 2), round(pr["iy"], 2)],
                             "blast": pr["w"].blast_r, "travel": 0.0})
            self._sound(pr["ix"], pr["iy"], pr["w"].sound_reach_m, "fire",
                        pr["owner"], label=f"{pr['owner']}/fire")
        self._projectiles = still

    # ---------------------------------------------------------------- kill / respawn

    def kill(self, killer: NetPlayer, victim: NetPlayer,
             weapon: str | None = None) -> None:
        """Call from resolve_shots. Handles corpse-farming guard, scoring, feed."""
        if not victim.alive:
            return
        # friendly-fire: score nothing in team mode on same team (still dies)
        friendly = (self.mode == GameMode.TEAM
                    and killer.team == victim.team and killer.id != victim.id)
        victim.alive = False
        victim.respawn_at = time.monotonic() + RESPAWN_DELAY
        victim.hp = 0.0
        if killer.id != victim.id and not friendly:
            killer.kills += 1
        if victim.body is not None:
            victim.body.alive = False
        verb = random.choice(P.KILL_VERBS)
        self._broadcast({"t": P.S_KILL, "killer": killer.id,
                         "victim": victim.id, "verb": verb,
                         "wep": weapon or ""})
        self._broadcast_score()

    def _respawn(self, p: NetPlayer) -> None:
        p.inputs.clear()          # commands aimed at where they used to be
        # a few options rather than the single furthest one: always coming back
        # to the same corner is an invitation to be camped
        sp = self._spawn_for(p, top=3)
        p.x, p.y = float(sp.x), float(sp.y)
        p.aim = math.radians(sp.facing_deg)
        p.alive = True
        p.respawn_at = None
        p.buttons = 0
        p.firing_was = False
        self._spawn_body(p)

    # ---------------------------------------------------------------- broadcast

    def _reap_dead(self) -> None:
        """Ping each client; drop any whose send fails. Runs while idle in
        lobby so a departed host triggers migration even with no traffic."""
        # caller holds _lock
        dead = [p.id for p in self.players.values()
                if not p.send({"t": "ping"})]
        if self._remove_players_locked(dead):
            self._broadcast_lobby_locked()

    def _broadcast_lobby_locked(self) -> None:
        roster = [{"id": p.id, "name": p.name, "colour": list(p.colour),
                   "team": int(p.team), "ready": p.ready,
                   "is_host": p.is_host, "playing": p.playing,
                   "has_map": self._has_map(p), "cls": p.cls_next}
                  for p in self.players.values()]
        self._broadcast({
            "t": P.S_LOBBY, "state": int(self.state), "mode": int(self.mode),
            "duration_s": self.duration_s, "map_id": self.map_id,
            "map_sha": self.map_sha,
            "host_id": self.host_id, "players": roster,
        })

    def _broadcast(self, obj: dict) -> None:
        # caller holds _lock
        dead = [p.id for p in self.players.values() if not p.send(obj)]
        if dead:
            self._remove_players_locked(dead)

    def _broadcast_lobby(self) -> None:
        with self._lock:
            roster = [{"id": p.id, "name": p.name, "colour": list(p.colour),
                       "team": int(p.team), "ready": p.ready,
                       "is_host": p.is_host, "playing": p.playing,
                       "has_map": self._has_map(p), "cls": p.cls_next}
                      for p in self.players.values()]
            self._broadcast({
                "t": P.S_LOBBY,
                "state": int(self.state),
                "mode": int(self.mode),
                "duration_s": self.duration_s,
                "map_id": self.map_id,
                "map_sha": self.map_sha,
                "host_id": self.host_id,
                "players": roster,
            })

    def _broadcast_snapshot(self, now: float) -> None:
        # caller holds _lock
        entries = []
        for p in self.players.values():
            entries.append({
                "id": p.id, "x": round(p.x, 3), "y": round(p.y, 3),
                "aim": round(p.aim, 3), "alive": p.alive,
                # the last input applied to this player: a client uses its own
                # number to know which of its predictions the server has seen
                "seq": p.acked_seq,
                "hp": (round(p.body.health / p.body.max_health, 3)
                       if p.body else 0.0),
                "sh": (round(p.body.shields / p.body.max_shields, 3)
                       if p.body and p.body.max_shields > 0 else 0.0),
                "pl": p.playing,
                "fl": p.flashlight,
                "cl": p.cls,
                "vn": self._vanished(p),
                "st": round(p.stamina, 2),
                "ab": round(max(0.0, p.ability_until - now), 2),
                "acd": round(max(0.0, p.ability_ready_at - now), 1),
                "wep": p.wi,
                "mag": p.mags[p.wi] if p.mags else 0,
                "rl": round(p.reload_t, 2),
                # how far along the next self-made round is, for the bar the
                # plasma rifle draws where a reserve count would be
                "rc": (round(weapons.recharge_frac(
                    p.loadout[p.wi], p.mags[p.wi], p.charges[p.wi]), 2)
                    if p.loadout and p.charges else 0.0),
                "chg": round(p.charge, 2),
                "respawn_in": (round(max(0.0, p.respawn_at - now), 1)
                               if p.respawn_at else 0.0),
            })
        self._broadcast({
            "t": P.S_SNAPSHOT,
            "time_left": max(0.0, round(self.match_end_time - now, 1)),
            "players": entries,
        })

    def _broadcast_score(self) -> None:
        # caller holds _lock
        scores = [{"id": p.id, "name": p.name, "team": int(p.team),
                   "kills": p.kills} for p in self.players.values()]
        # Keys are strings on purpose. msgpack refuses integer map keys on
        # unpack (strict_map_key), so a team-numbered dict here raises a
        # ValueError inside the client's receive loop, which quietly ends that
        # thread — the client stops receiving ANYTHING and just sits there.
        # This only ever fired in team modes, which is why it went unnoticed.
        team_scores: dict[str, int] = {}
        if self.mode == GameMode.TEAM:
            for p in self.players.values():
                key = str(int(p.team))
                team_scores[key] = team_scores.get(key, 0) + p.kills
        self._broadcast({"t": P.S_SCORE, "mode": int(self.mode),
                         "scores": scores, "team_scores": team_scores})
