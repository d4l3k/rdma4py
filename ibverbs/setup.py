"""Build the ibverbs Cython extension.

The extension is compiled against the ``rdma-core`` headers (for struct layouts
and the ``static inline`` data-path verbs) but is **not** linked against
``libibverbs``: the exported verbs are resolved at import time with
``dlopen``/``dlsym`` (see ``_ibverbs.pyx``). The header include path is
discovered with ``pkg-config`` when available.

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
        "Cython is required to build ibverbs. Install it with `pip install Cython`."
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
    for flag in _pkg_config("--cflags-only-I", "libibverbs"):
        if flag.startswith("-I"):
            dirs.append(flag[2:])
    return dirs


extensions = [
    Extension(
        "ibverbs._ibverbs",
        ["src/ibverbs/_ibverbs.pyx"],
        include_dirs=_include_dirs(),
        # No libibverbs link: it is dlopen'd at runtime. libdl provides
        # dlopen/dlsym (a no-op stub on glibc >= 2.34, required on older).
        libraries=["dl"],
        define_macros=[("Py_LIMITED_API", LIMITED_API_VERSION)],
        py_limited_api=True,
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level="3",
        compiler_directives={"embedsignature": True, "binding": True},
    ),
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
