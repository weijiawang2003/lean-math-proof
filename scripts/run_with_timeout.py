"""Portable per-process timeout (macOS lacks coreutils `timeout`).

Usage: python3 scripts/run_with_timeout.py <seconds> <cmd> [args...]
Runs the command in its own process group; on timeout, SIGTERM then SIGKILL
the whole group. Exit code is the child's, or 124 on timeout.
"""
import os
import signal
import subprocess
import sys
import time

def main():
    secs = float(sys.argv[1])
    cmd = sys.argv[2:]
    if not cmd:
        print("no command", file=sys.stderr)
        sys.exit(2)
    p = subprocess.Popen(cmd, start_new_session=True)
    try:
        p.wait(timeout=secs)
        sys.exit(p.returncode)
    except subprocess.TimeoutExpired:
        pgid = os.getpgid(p.pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(5)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        print(f"[run_with_timeout] killed after {secs}s: {' '.join(cmd)}",
              file=sys.stderr)
        sys.exit(124)

if __name__ == "__main__":
    main()
