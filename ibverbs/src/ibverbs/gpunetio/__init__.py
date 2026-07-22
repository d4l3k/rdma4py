"""Optional GPUNetIO support for GPU-initiated mlx5 communication.

The base :mod:`ibverbs` package still has no CUDA or DOCA dependency. This
module loads DOCA only when :class:`DeviceQP` is exported, while the
framework-specific :mod:`.triton` and :mod:`.cutedsl` modules remain optional.
"""

from ._build import bitcode_path, build_bitcode, cache_bitcode_path
from ._runtime import DeviceQP, GPUNetIOError

__all__ = [
    "DeviceQP",
    "GPUNetIOError",
    "bitcode_path",
    "build_bitcode",
    "cache_bitcode_path",
]
