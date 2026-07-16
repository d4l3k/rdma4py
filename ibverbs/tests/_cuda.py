"""Minimal CUDA VMM + dma-buf helper for the GPUDirect tests.

We use the CUDA driver VMM API (``cuMemCreate`` / ``cuMemMap``) rather than the
torch caching allocator because only VMM allocations can be exported as a
dma-buf fd via ``cuMemGetHandleForAddressRange`` — the fd libibverbs needs for
``ibv_reg_dmabuf_mr``. This keeps the binding itself free of any CUDA linkage;
CUDA lives entirely in the test scaffolding.
"""

from __future__ import annotations

import ctypes

_DMA_BUF_FD = 1                 # CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD
_TYPE_PINNED = 1                # CU_MEM_ALLOCATION_TYPE_PINNED
_LOC_DEVICE = 1                 # CU_MEM_LOCATION_TYPE_DEVICE
_ACCESS_RW = 3                  # CU_MEM_ACCESS_FLAGS_PROT_READWRITE
_GRANULARITY_MIN = 0            # CU_MEM_ALLOC_GRANULARITY_MINIMUM


class _Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _Prop(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Loc),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _Loc), ("flags", ctypes.c_int)]


class CudaUnavailable(RuntimeError):
    pass


class CudaVMM:
    """A tiny CUDA driver wrapper exposing dma-buf-backed device buffers."""

    def __init__(self, device=0):
        try:
            self.cuda = ctypes.CDLL("libcuda.so.1")
        except OSError as exc:
            raise CudaUnavailable("libcuda.so.1 not found") from exc
        self.device = device
        self._buffers = []
        self._check(self.cuda.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self._check(self.cuda.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
        self._dev = dev
        ctx = ctypes.c_void_p()
        self._check(self.cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev),
                    "cuDevicePrimaryCtxRetain")
        self._check(self.cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    def _check(self, rc, what):
        if rc != 0:
            msg = ctypes.c_char_p()
            self.cuda.cuGetErrorString(rc, ctypes.byref(msg))
            raise CudaUnavailable("%s failed rc=%d (%s)" % (
                what, rc, msg.value.decode() if msg.value else "?"))

    def alloc(self, size):
        """Allocate a VMM device buffer; return ``(ptr, size, dmabuf_fd)``."""
        prop = _Prop()
        prop.type = _TYPE_PINNED
        prop.location.type = _LOC_DEVICE
        prop.location.id = self.device
        prop.allocFlags.gpuDirectRDMACapable = 1
        gran = ctypes.c_size_t()
        self._check(self.cuda.cuMemGetAllocationGranularity(
            ctypes.byref(gran), ctypes.byref(prop), _GRANULARITY_MIN),
            "cuMemGetAllocationGranularity")
        g = gran.value
        size = ((size + g - 1) // g) * g
        handle = ctypes.c_ulonglong()
        self._check(self.cuda.cuMemCreate(
            ctypes.byref(handle), ctypes.c_size_t(size), ctypes.byref(prop), 0),
            "cuMemCreate")
        ptr = ctypes.c_ulonglong()
        self._check(self.cuda.cuMemAddressReserve(
            ctypes.byref(ptr), ctypes.c_size_t(size), 0, 0, 0),
            "cuMemAddressReserve")
        self._check(self.cuda.cuMemMap(ptr, ctypes.c_size_t(size), 0, handle, 0),
                    "cuMemMap")
        ad = _AccessDesc()
        ad.location.type = _LOC_DEVICE
        ad.location.id = self.device
        ad.flags = _ACCESS_RW
        self._check(self.cuda.cuMemSetAccess(
            ptr, ctypes.c_size_t(size), ctypes.byref(ad), 1), "cuMemSetAccess")
        fd = ctypes.c_int(-1)
        self._check(self.cuda.cuMemGetHandleForAddressRange(
            ctypes.byref(fd), ptr, ctypes.c_size_t(size), _DMA_BUF_FD, 0),
            "cuMemGetHandleForAddressRange")
        self._buffers.append(ptr.value)
        return ptr.value, size, fd.value

    def memset_host_to_device(self, ptr, data: bytes):
        buf = (ctypes.c_char * len(data)).from_buffer_copy(data)
        self._check(self.cuda.cuMemcpyHtoD_v2(
            ctypes.c_ulonglong(ptr), buf, ctypes.c_size_t(len(data))), "cuMemcpyHtoD")

    def device_to_host(self, ptr, n) -> bytes:
        buf = (ctypes.c_char * n)()
        self._check(self.cuda.cuMemcpyDtoH_v2(
            buf, ctypes.c_ulonglong(ptr), ctypes.c_size_t(n)), "cuMemcpyDtoH")
        return bytes(buf)
