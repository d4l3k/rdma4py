"""Lintrunner adapter for ufmt using Black and usort."""

from __future__ import annotations

import argparse
import json
from enum import Enum
from pathlib import Path
from typing import NamedTuple


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


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


def check_file(filename: str) -> list[LintMessage]:
    from ufmt import Result, ufmt_file

    path = Path(filename)
    try:
        original = path.read_text(encoding="utf-8")
        result: Result = ufmt_file(path, dry_run=True, return_content=True)
        if result.error:
            raise result.error
        replacement = result.after.decode("utf-8") if result.after else original
    except Exception as error:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="UFMT",
                severity=LintSeverity.ERROR,
                name="format-error",
                original=None,
                replacement=None,
                description=f"ufmt failed: {error}",
            )
        ]

    if original == replacement:
        return []
    return [
        LintMessage(
            path=filename,
            line=1,
            char=1,
            code="UFMT",
            severity=LintSeverity.WARNING,
            name="format",
            original=original,
            replacement=replacement,
            description="Run `lintrunner -a` to apply formatting changes.",
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()
    for filename in args.filenames:
        for message in check_file(filename):
            print(json.dumps(message._asdict()), flush=True)


if __name__ == "__main__":
    main()
