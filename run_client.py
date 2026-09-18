"""
run_client.py — join a BANGbang multiplayer host from another machine.

Run from the project root (the folder that contains the net/ folder).

USAGE:
    python run_client.py <host_ip> [port] [--name NAME] [--colour COLOUR]

The host_ip is the address shown on the host's console when it starts.
The port is optional; leave it out unless the host changed it.
--name and --colour are optional and can go in any order AFTER the ip.

EXAMPLES:
    python run_client.py 192.168.178.26
    python run_client.py 192.168.178.26 --name Kapistus --colour red
    python run_client.py 192.168.178.26 47801 --name Bob --colour 40,40,220

COLOUR accepts a name (red, blue, green, yellow, orange, purple, cyan,
white, black, grey) OR three comma-separated numbers "r,g,b".
    good:  --colour red        --colour 40,40,220
    bad:   --colour 40, 40, 220     (unquoted spaces split into extra args)
    if you want spaces, quote it:   --colour "40, 40, 220"
Do NOT put extra words after the options (e.g. a stray "host" or port at the
end) — the ip and port come first, everything else is --name/--colour only.

Headless netcode harness (no pygame). Console commands work on macOS/Linux;
on Windows the connection runs but typed commands are ignored (see note in
run_host.py). Host-only commands (start/end) do nothing unless you're promoted.
"""
from __future__ import annotations

import argparse
import sys
import time

from net import GameClient
from net.protocol import ServerState, DEFAULT_PORT


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
        description="Join a BANGbang multiplayer host.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example: python run_client.py 192.168.178.26 --name Kapistus "
               "--colour red")
    ap.add_argument("host", help="host LAN IP shown on the host console "
                                 "(e.g. 192.168.178.26)")
    ap.add_argument("port", nargs="?", type=int, default=DEFAULT_PORT,
                    help=f"host port (default {DEFAULT_PORT}; omit unless changed)")
    ap.add_argument("--name", default="Player", help="your display name")
    ap.add_argument("--colour", default="blue",
                    help="colour name (red, blue, ...) or 'r,g,b'")
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        colour = parse_colour(args.colour)
    except ValueError as e:
        print("error:", e)
        sys.exit(2)

    cli = GameClient(args.host, args.port)
    print(f"connecting to {args.host}:{args.port} ...")
    if not cli.connect(name=args.name, colour=colour):
        print("failed:", cli.reject_reason)
        return
    print(f"[{args.name}] connected as id={cli.world.my_id} "
          f"(host={cli.world.is_host})")
    print("commands: ready unready team a|b start end score quit\n")

    try:
        _loop(cli)
    except KeyboardInterrupt:
        pass
    finally:
        cli.disconnect()
        print("\ndisconnected.")


def _loop(cli: GameClient):
    from net.protocol import Team
    last_print = 0.0
    while True:
        for ev in cli.drain_events():
            t = ev["t"]
            if t == "kill":
                e = ev["entry"]
                print(f"  KILL: {e.killer} {e.verb} {e.victim}")
            elif t == "match_start":
                print("  >>> MATCH STARTED. spawn:", cli.world.my_spawn,
                      "team:", cli.world.my_team.name)
            elif t == "match_end":
                print("  >>> MATCH ENDED")
            elif t == "end_count":
                print(f"  ...back to lobby in {ev['n']}")
            elif t == "reject":
                print("  REJECTED:", ev["reason"])
            elif t == "disconnected":
                print("  server closed the connection")
                return

        now = time.monotonic()
        if cli.world.state == ServerState.MATCH and now - last_print > 2.0:
            last_print = now
            sb = cli.scoreboard()
            print(f"  [t-{cli.world.time_left:.0f}s] " +
                  " | ".join(f"{n}:{k}" for n, k in sb))

        cmd = _poll_stdin()
        if cmd is not None:
            parts = cmd.strip().lower().split()
            if not parts:
                pass
            elif parts[0] == "quit":
                return
            elif parts[0] == "ready":
                cli.set_ready(True); print("  readied.")
            elif parts[0] == "unready":
                cli.set_ready(False)
            elif parts[0] == "team" and len(parts) > 1:
                cli.set_team(Team.A if parts[1] == "a" else Team.B)
            elif parts[0] == "start":
                cli.force_start()
            elif parts[0] == "end":
                cli.end_match()
            elif parts[0] == "score":
                for n, k in cli.scoreboard():
                    print(f"    {n} - {k} kills")
            else:
                print("  commands: ready unready team a|b start end score quit")

        time.sleep(0.03)


def _poll_stdin():
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
