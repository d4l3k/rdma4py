"""Lintrunner adapter for Ruff diagnostics."""

from __future__ import annotations

import argparse
import json
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
        ["ruff", "check", "--output-format=json", *args.filenames],
        capture_output=True,
        text=True,
    )
    if process.returncode not in (0, 1):
        message = LintMessage(
            path=None,
            line=None,
            char=None,
            code="RUFF",
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description=process.stderr.strip() or "ruff failed",
        )
        print(json.dumps(message._asdict()))
        return

    for result in json.loads(process.stdout or "[]"):
        filename = Path(result["filename"])
        try:
            filename = filename.relative_to(Path.cwd())
        except ValueError:
            pass
        message = LintMessage(
            path=str(filename),
            line=result["location"]["row"],
            char=result["location"]["column"],
            code="RUFF",
            severity=LintSeverity.ERROR,
            name=result["code"],
            original=None,
            replacement=None,
            description=result["message"],
        )
        print(json.dumps(message._asdict()), flush=True)


if __name__ == "__main__":
    main()
