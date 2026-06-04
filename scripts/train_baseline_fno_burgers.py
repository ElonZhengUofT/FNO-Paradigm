from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--help", action="store_true")
    args, unknown = parser.parse_known_args()

    cmd = [sys.executable, "scripts/compare_burgers.py", "--experiment", "baseline", *unknown]
    if args.help:
        cmd.append("--help")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
