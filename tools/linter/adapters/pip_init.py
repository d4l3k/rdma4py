"""Install pinned Lintrunner dependencies with uv or pip."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="pip initializer")
    parser.add_argument("packages", nargs="+")
    parser.add_argument("--dry-run")
    args = parser.parse_args()

    for package in args.packages:
        if "==" not in package:
            raise RuntimeError(f"linter dependency must be pinned: {package}")

    if args.dry_run == "1":
        print("Would install: " + " ".join(args.packages))
        return

    env = {
        **os.environ,
        "UV_PYTHON": sys.executable,
        "UV_PYTHON_DOWNLOADS": "never",
    }
    uv = shutil.which("uv")
    in_environment = bool(
        os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    )
    if uv and in_environment:
        command = [uv, "pip", "install", *args.packages]
    else:
        command = [sys.executable, "-m", "pip", "install"]
        if not in_environment:
            command.append("--user")
        command.extend(args.packages)

    logging.info("running %s", " ".join(command))
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
