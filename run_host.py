"""
run_host.py — boot a BANGbang multiplayer host.

Run from the project root (the folder that contains the net/ folder).

USAGE:
    python run_host.py [--name NAME] [--colour COLOUR] [--mode ffa|team]
                       [--duration SECONDS] [--map MAP] [--port PORT]

All arguments are optional and can go in any order. With none, you host an
FFA match on the arena map. The console prints a LAN IP — the other machines
type that into run_client.py to join.

EXAMPLES:
    python run_host.py
    python run_host.py --name Kapistus --colour red
    python run_host.py --mode team --duration 300 --map compound

COLOUR accepts a name (red, blue, green, yellow, orange, purple, cyan,
white, black, grey) OR three comma-separated numbers "r,g,b" (quote it if you
put spaces between the numbers: --colour "40, 40, 220").

Starts the authoritative server AND connects this machine's own client (the
host plays too). HEADLESS harness for testing the netcode — no pygame, so
nothing is drawn and this client sends no input: it verifies lobby, match,
scoreboard and end flow over a real socket.

To actually PLAY, run mp_client.py instead — same server, with a window.

NOTE (Windows): the connection and match run fine, but typed console commands
(ready/start/end) are IGNORED on Windows because stdin polling uses select(),
which is POSIX-only for keyboards. On macOS/Linux the commands work. The real
controls come from pygame once GameServer/GameClient are wired into main.py.
"""
from __future__ import annotations

import argparse
import sys
import time

from net import GameServer, GameClient
from net.protocol import GameMode, ServerState, DEFAULT_PORT


NAMED_COLOURS = {
    "red": (220, 40, 40), "blue": (40, 40, 220), "green": (40, 200, 60),
    "yellow": (230, 210, 40), "orange": (240, 140, 30),
    "purple": (160, 60, 200), "cyan": (40, 200, 220),
    "white": (235, 235, 235), "black": (20, 20, 20), "grey": (130, 130, 130),
    "gray": (130, 130, 130),
}


def parse_colour(s: str) -> tuple[int, int, int]:
    """Accept a colour name or 'r,g,b'. Raise ValueError with a clear message."""
    key = s.strip().lower()
    if key in NAMED_COLOURS:
        return NAMED_COLOURS[key]
    parts = [p for p in key.replace(" ", "").split(",") if p != ""]
    if len(parts) != 3:
        raise ValueError(
            f"colour must be a name ({', '.join(sorted(NAMED_COLOURS))}) "
            f"or three numbers 'r,g,b' with no spaces — got {s!r}")
    try:
        rgb = tuple(int(p) for p in parts)
    except ValueError:
        raise ValueError(f"colour numbers must be integers — got {s!r}")
    if not all(0 <= c <= 255 for c in rgb):
        raise ValueError(f"colour values must be 0-255 — got {s!r}")
    return rgb  # type: ignore[return-value]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Host a BANGbang multiplayer match.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example: python run_host.py --mode team --map compound")
    ap.add_argument("--name", default="Host", help="your display name")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"listen port (default {DEFAULT_PORT})")
    ap.add_argument("--mode", choices=["ffa", "team"], default="ffa",
                    help="ffa = free-for-all, team = team deathmatch")
    ap.add_argument("--duration", type=int, default=300,
                    help="match length in seconds (clamped 120-3600)")
    ap.add_argument("--map", dest="map_id", default="arena",
                    help="map id (e.g. arena, compound)")
    ap.add_argument("--colour", default="red",
                    help="colour name (red, blue, ...) or 'r,g,b'")
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        colour = parse_colour(args.colour)
    except ValueError as e:
        print("error:", e)
        sys.exit(2)
    mode = GameMode.TEAM if args.mode == "team" else GameMode.FFA

    # Spawn points come from the map itself (net/maps.py): whatever the file
    # declares, otherwise spread across its standable ground.
    srv = GameServer(port=args.port, mode=mode,
                     duration_s=args.duration, map_id=args.map_id)
    srv.start()
    lan_ip = srv.lan_ip()
    port_arg = "" if args.port == DEFAULT_PORT else f" {args.port}"
    print("=" * 60)
    print(f" BANGbang host up.  mode={args.mode}  map={args.map_id}")
    print(f" Others on your network join with:")
    print(f"   python run_client.py {lan_ip}{port_arg} --name YOURNAME")
    print("=" * 60)

    # host's own client connects to loopback
    cli = GameClient("127.0.0.1", args.port)
    if not cli.connect(name=args.name, colour=colour):
        print("host client failed to connect:", cli.reject_reason)
        srv.stop()
        return
    print(f"[{args.name}] connected as id={cli.world.my_id} "
          f"(host={cli.world.is_host})")
    print("Type 'ready' to ready up, 'start' to force-start, "
          "'end' to end match, 'quit' to exit.\n")

    try:
        _loop(cli, srv, args)
    except KeyboardInterrupt:
        pass
    finally:
        cli.disconnect()
        srv.stop()
        print("\nhost shut down.")


def _loop(cli: GameClient, srv: GameServer, args):
    """Minimal REPL + event pump so you can drive the match from the console."""
    import sys
    import select

    last_print = 0.0
    while True:
        # pump network events
        for ev in cli.drain_events():
            t = ev["t"]
            if t == "kill":
                e = ev["entry"]
                print(f"  KILL: {e.killer} {e.verb} {e.victim}")
            elif t == "match_start":
                print("  >>> MATCH STARTED. spawn:", cli.world.my_spawn)
            elif t == "match_end":
                print("  >>> MATCH ENDED")
            elif t == "end_count":
                print(f"  ...back to lobby in {ev['n']}")
            elif t == "reject":
                print("  REJECTED:", ev["reason"])
            elif t == "disconnected":
                print("  disconnected from server")
                return

        # periodic status line during a match
        now = time.monotonic()
        if cli.world.state == ServerState.MATCH and now - last_print > 2.0:
            last_print = now
            sb = cli.scoreboard()
            print(f"  [t-{cli.world.time_left:.0f}s] " +
                  " | ".join(f"{n}:{k}" for n, k in sb))

        # non-blocking stdin read (POSIX). On Windows this select won't work on
        # stdin; fall back to a plain blocking input if needed.
        cmd = _poll_stdin()
        if cmd is not None:
            cmd = cmd.strip().lower()
            if cmd == "quit":
                return
            elif cmd == "ready":
                cli.set_ready(True)
                print("  readied.")
            elif cmd == "unready":
                cli.set_ready(False)
            elif cmd == "start":
                cli.force_start()
            elif cmd == "end":
                cli.end_match()
            elif cmd == "score":
                for n, k in cli.scoreboard():
                    print(f"    {n} - {k} kills")
            elif cmd:
                print("  commands: ready unready start end score quit")

        time.sleep(0.03)


def _poll_stdin():
    """Return a line if one is waiting on stdin, else None. POSIX select;
    on Windows, degrade to None (use Ctrl-C to quit, or run run_client)."""
    import sys
    try:
        import select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.readline()
    except (OSError, ValueError):
        pass
    return None


if __name__ == "__main__":
    main()
