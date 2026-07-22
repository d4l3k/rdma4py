"""Build the GPUNetIO device wrapper as linkable NVPTX LLVM bitcode."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


def _normalize_arch(arch: str | int) -> str:
    value = str(arch).lower().replace("compute_", "").replace("sm_", "")
    if not value.isdigit() or int(value) < 80:
        raise ValueError("GPUNetIO requires an NVIDIA GPU architecture of sm_80+")
    return "sm_" + value


def _find_cuda(cuda_home=None) -> Path:
    candidates = [cuda_home, os.environ.get("CUDA_HOME"), "/usr/local/cuda"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "include" / "cuda.h").is_file():
            return Path(candidate)
    raise FileNotFoundError(
        "CUDA headers were not found; set CUDA_HOME or pass cuda_home="
    )


def _find_doca_include(doca_include=None) -> Path:
    candidates = [
        doca_include,
        os.environ.get("DOCA_INCLUDE"),
        "/opt/mellanox/doca/include",
    ]
    try:
        flags = subprocess.check_output(
            ["pkg-config", "--cflags-only-I", "doca-gpunetio"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        flags = ""
    candidates.extend(x[2:] for x in shlex.split(flags) if x.startswith("-I"))
    for candidate in candidates:
        if (
            candidate
            and (Path(candidate) / "doca_gpunetio_dev_verbs_onesided.cuh").is_file()
        ):
            return Path(candidate)
    raise FileNotFoundError(
        "DOCA GPUNetIO development headers were not found; install "
        "doca-sdk-gpunetio-devel or pass doca_include="
    )


def cache_bitcode_path(arch: str | int = "sm_90") -> Path:
    """Return the per-user cache path for bitcode built for ``arch``."""
    arch = _normalize_arch(arch)
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "rdma4py" / "gpunetio" / arch / "device.bc"


def bitcode_path(path=None, *, arch: str | int = "sm_90") -> Path:
    """Resolve an existing GPUNetIO bitcode file.

    ``path`` takes precedence over ``RDMA4PY_GPUNETIO_BITCODE`` and the
    per-user build cache. Use :func:`build_bitcode` to create the cache file.
    """
    candidate = path or os.environ.get("RDMA4PY_GPUNETIO_BITCODE")
    resolved = Path(candidate) if candidate else cache_bitcode_path(arch)
    if not resolved.is_file():
        raise FileNotFoundError(
            "%s does not exist; call ibverbs.gpunetio.build_bitcode() first" % resolved
        )
    return resolved.resolve()


def build_bitcode(
    output=None,
    *,
    arch: str | int = "sm_90",
    doca_include=None,
    cuda_home=None,
    clang=None,
    force: bool = False,
) -> Path:
    """Compile the device wrapper for Triton and CuTe DSL.

    The output is architecture-specific because DOCA selects memory-ordering
    instructions from ``__CUDA_ARCH__``. ``clang++`` must support CUDA device
    compilation and the installed CUDA toolkit.
    """
    arch = _normalize_arch(arch)
    output = Path(output) if output else cache_bitcode_path(arch)
    if output.exists() and not force:
        return output.resolve()

    cuda = _find_cuda(cuda_home)
    doca = _find_doca_include(doca_include)
    compiler = clang or os.environ.get("CLANG") or shutil.which("clang++")
    if not compiler:
        raise FileNotFoundError("clang++ was not found; pass clang=")

    source = Path(__file__).with_name("_device.cu")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="device-", suffix=".bc", dir=output.parent, delete=False
    ) as tmp:
        temporary = Path(tmp.name)
    command = [
        str(compiler),
        "-x",
        "cuda",
        "--cuda-device-only",
        "--cuda-path=" + str(cuda),
        "--cuda-gpu-arch=" + arch,
        "--no-cuda-version-check",
        "-std=gnu++17",
        "-O3",
        "-fno-exceptions",
        "-fgpu-flush-denormals-to-zero",
        "-emit-llvm",
        "-c",
        str(source),
        "-I" + str(doca),
        "-o",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output.resolve()


__all__ = ["bitcode_path", "build_bitcode", "cache_bitcode_path"]
