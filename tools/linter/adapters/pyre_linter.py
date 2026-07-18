"""Lintrunner adapter for a repository-wide Pyre check."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import NamedTuple


class LintSeverity(str, Enum):
    ERROR = "error"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def main() -> None:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()

    process = subprocess.run(
        ["pyre", "--output=json", "check"],
        capture_output=True,
        text=True,
    )
    try:
        results = json.loads(process.stdout or "[]")
    except json.JSONDecodeError:
        results = []

    if process.returncode not in (0, 1) and not results:
        message = LintMessage(
            path=None,
            line=None,
            char=None,
            code="PYRE",
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description=process.stderr.strip() or "pyre failed",
        )
        print(json.dumps(message._asdict()))
        return

    selected = set()
    for filename in args.filenames:
        selected.add(os.path.normpath(filename))
        selected.add(os.path.normpath(os.path.relpath(filename)))

    for result in results:
        path = os.path.normpath(result["path"])
        if path not in selected:
            continue
        message = LintMessage(
            path=str(Path(path)),
            line=result["line"],
            char=result["column"],
            code="PYRE",
            severity=LintSeverity.ERROR,
            name=result["name"],
            original=None,
            replacement=None,
            description=result["description"],
        )
        print(json.dumps(message._asdict()), flush=True)


if __name__ == "__main__":
    main()
