"""Build the efa Cython extension.

The extension is compiled against the ``rdma-core`` headers (for struct layouts
and the ``static inline`` data-path verbs / extended work-request API) but is
**not** linked against ``libibverbs`` or ``libefa``: the exported symbols are
resolved at import time with ``dlopen``/``dlsym`` (see ``_efa.pyx``). The
header include path is discovered with ``pkg-config`` when available.

It is built against the CPython **Limited API** (abi3), so a single wheel works
across CPython 3.9+.
"""

from __future__ import annotations

import shlex
import subprocess

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:  # pragma: no cover - build-time only
    raise SystemExit(
        "Cython is required to build efa. Install it with `pip install Cython`."
    ) from exc

# CPython Limited API floor == requires-python floor (3.9).
LIMITED_API_VERSION = "0x03090000"


def _pkg_config(*args: str) -> list[str]:
    try:
        out = subprocess.check_output(["pkg-config", *args], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return shlex.split(out)


def _include_dirs() -> list[str]:
    dirs = []
    for flag in _pkg_config("--cflags-only-I", "libefa", "libibverbs"):
        if flag.startswith("-I") and flag[2:] not in dirs:
            dirs.append(flag[2:])
    return dirs


extensions = [
    Extension(
        "efa._efa",
        ["src/efa/_efa.pyx"],
        include_dirs=_include_dirs(),
        # No libibverbs/libefa link: they are dlopen'd at runtime. libdl
        # provides dlopen/dlsym (a no-op stub on glibc >= 2.34).
        libraries=["dl"],
        define_macros=[("Py_LIMITED_API", LIMITED_API_VERSION)],
        py_limited_api=True,
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level="3",
        compiler_directives={
            "annotation_typing": False,
            "embedsignature": True,
            "binding": True,
        },
    ),
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
