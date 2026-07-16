"""Build the ibverbs Cython extension.

The extension is compiled against the system ``libibverbs`` (rdma-core). The
required compile/link flags are discovered with ``pkg-config`` when available,
falling back to a plain ``-libverbs`` link.
"""

from __future__ import annotations

import shlex
import subprocess

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:  # pragma: no cover - build-time only
    raise SystemExit(
        "Cython is required to build ibverbs. Install it with `pip install Cython`."
    ) from exc


def _pkg_config(*args: str) -> list[str]:
    try:
        out = subprocess.check_output(["pkg-config", *args], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return shlex.split(out)


def _ext_kwargs() -> dict:
    cflags = _pkg_config("--cflags", "libibverbs")
    libs = _pkg_config("--libs", "libibverbs")
    if not libs:
        # pkg-config unavailable (no -devel .pc file); link directly.
        libs = ["-libverbs"]
    include_dirs, extra_compile = [], []
    for flag in cflags:
        (include_dirs.append(flag[2:]) if flag.startswith("-I") else extra_compile.append(flag))
    libraries, library_dirs, extra_link = [], [], []
    for flag in libs:
        if flag.startswith("-l"):
            libraries.append(flag[2:])
        elif flag.startswith("-L"):
            library_dirs.append(flag[2:])
        else:
            extra_link.append(flag)
    if "ibverbs" not in libraries:
        libraries.append("ibverbs")
    return {
        "include_dirs": include_dirs,
        "libraries": libraries,
        "library_dirs": library_dirs,
        "extra_compile_args": extra_compile,
        "extra_link_args": extra_link,
    }


extensions = [
    Extension("ibverbs._ibverbs", ["src/ibverbs/_ibverbs.pyx"], **_ext_kwargs()),
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level="3",
        compiler_directives={"embedsignature": True, "binding": True},
    ),
)
