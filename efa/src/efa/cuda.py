"""Optional CUDA GPUDirect helpers for registering GPU tensors as memory regions.

This module lazily ``dlopen``s ``libcuda`` and imports **no** torch and links
**no** CUDA at build time, so it adds no required dependency. It is duck-typed:
anything exposing ``data_ptr()`` (e.g. a torch CUDA tensor) works.

For torch, allocate with the VMM allocator so the memory can be exported as a
dma-buf fd (the fd libibverbs needs for ``ibv_reg_dmabuf_mr``)::

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Typical use::

    import efa, efa.cuda, torch
    t = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    gmr = efa.cuda.register_tensor(pd, t, efa.AccessFlags.LOCAL_WRITE)
    qp.post_send(efa.SendWR(sg_list=[gmr.sge()], dest=peer, ...))

Before the NIC reads a tensor, synchronize the CUDA stream that produced it.
After an inbound completion, call :func:`flush_gpudirect_writes` before CUDA
work consumes the destination.
"""

from __future__ import annotations

import ctypes
import os

from . import _efa  # pyre-ignore[21]: Implemented by the Cython extension.

_PAGE = os.sysconf("SC_PAGESIZE")
_DMA_BUF_FD = 1  # CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD
_FLUSH_TARGET_CURRENT_CTX = 0
_FLUSH_TO_OWNER = 100
_FLUSH_TO_ALL_DEVICES = 200
_lib = None


def _cuda():
    global _lib
    if _lib is None:
        try:
            _lib = ctypes.CDLL("libcuda.so.1")
        except OSError as exc:  # pragma: no cover - depends on host
            raise RuntimeError(
                "libcuda.so.1 not found; CUDA GPUDirect helpers unavailable"
            ) from exc
        _lib.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        _lib.cuGetErrorString.restype = ctypes.c_int
        _lib.cuMemGetHandleForAddressRange.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_ulonglong,
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_ulonglong,
        ]
        _lib.cuMemGetHandleForAddressRange.restype = ctypes.c_int
        flush = getattr(_lib, "cuFlushGPUDirectRDMAWrites", None)
        if flush is not None:
            flush.argtypes = [ctypes.c_int, ctypes.c_int]
            flush.restype = ctypes.c_int
    return _lib


def _cuda_err(rc: int) -> str:
    s = ctypes.c_char_p()
    _cuda().cuGetErrorString(rc, ctypes.byref(s))
    return s.value.decode() if s.value else f"cuda error {rc}"


class GpuMR:
    """A registered GPU memory region plus the info needed to address it.

    ``ibv_mr.addr`` is not meaningful for dma-buf MRs, so this wraps the real
    device virtual address. Use :meth:`sge` for the local scatter/gather list
    and :attr:`addr` / :attr:`rkey` for the remote side of an RDMA op.
    """

    def __init__(self, mr, addr: int, length: int, tensor=None):
        self.mr = mr
        self.addr = int(addr)
        self.length = int(length)
        self._tensor = tensor

    @property
    def tensor(self):
        """The tensor retained as this registration's allocation owner."""
        return self._tensor

    @property
    def closed(self) -> bool:
        """Whether the underlying memory region has been deregistered."""
        return self.mr.closed

    @property
    def lkey(self) -> int:
        """Return the local key used in scatter/gather entries."""
        return self.mr.lkey

    @property
    def rkey(self) -> int:
        """Return the remote key shared with an RDMA peer."""
        return self.mr.rkey

    def sge(self, length=None, offset=0):
        """Build an SGE for a byte range of this GPU memory region."""
        offset = int(offset)
        if offset < 0 or offset > self.length:
            raise ValueError("SGE offset is outside the GPU memory region")
        if length is None:
            length = self.length - offset
        length = int(length)
        if length < 0 or length > self.length - offset:
            raise ValueError("SGE range exceeds the GPU memory region")
        return _efa.SGE(self.addr, length, lkey=self.lkey, offset=offset)._keepalive(
            self
        )

    def close(self):
        """Deregister the memory region and release the retained tensor."""
        self.mr.close()
        self._tensor = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        if self.closed:
            return "GpuMR(closed=True)"
        return "GpuMR(addr=0x%x, length=%d, lkey=0x%x, rkey=0x%x)" % (
            self.addr,
            self.length,
            self.lkey,
            self.rkey,
        )


def dmabuf_fd(ptr: int, length: int) -> int:
    """Export a dma-buf fd for the device VA range ``[ptr, ptr+length)``.

    ``ptr`` and ``length`` must be page-aligned (:func:`register_tensor`
    handles alignment). The caller owns and must ``os.close`` the returned fd.
    """
    ptr = int(ptr)
    length = int(length)
    if ptr < 0 or ptr % _PAGE:
        raise ValueError("ptr must be a non-negative, page-aligned address")
    if length <= 0 or length % _PAGE:
        raise ValueError("length must be positive and page-aligned")
    lib = _cuda()
    fd = ctypes.c_int(-1)
    rc = lib.cuMemGetHandleForAddressRange(
        ctypes.byref(fd),
        ctypes.c_ulonglong(ptr),
        ctypes.c_size_t(length),
        _DMA_BUF_FD,
        ctypes.c_ulonglong(0),
    )
    if rc != 0:
        raise RuntimeError(
            "cuMemGetHandleForAddressRange failed: %s. The allocation must be "
            "VMM-backed (for torch set "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)." % _cuda_err(rc)
        )
    return fd.value


def tensor_ptr_len(tensor):
    """Return ``(device_addr, nbytes)`` for a tensor-like object."""
    from .helpers import _require_contiguous

    _require_contiguous(tensor)
    if not hasattr(tensor, "data_ptr"):
        raise TypeError("expected a tensor exposing data_ptr(); got %r" % type(tensor))
    ptr = int(tensor.data_ptr())
    if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
        n = int(tensor.numel()) * int(tensor.element_size())
    elif hasattr(tensor, "nbytes"):
        n = int(tensor.nbytes)
    else:
        raise TypeError("cannot determine byte length of %r" % type(tensor))
    return ptr, n


def flush_gpudirect_writes(*, all_devices: bool = False) -> None:
    """Make completed inbound GPUDirect RDMA writes visible to CUDA.

    Call this after the relevant receive/completion or application-level
    remote-write notification, before launching CUDA work that consumes the
    destination tensor. The current thread must have the destination tensor's
    CUDA context active. By default CUDA flushes writes only to the owning
    device; set ``all_devices=True`` to flush visibility to every GPU in the
    current context's scope.
    """
    flush = getattr(_cuda(), "cuFlushGPUDirectRDMAWrites", None)
    if flush is None:
        raise RuntimeError(
            "cuFlushGPUDirectRDMAWrites is unavailable; CUDA 11.3 or newer is required"
        )
    scope = _FLUSH_TO_ALL_DEVICES if all_devices else _FLUSH_TO_OWNER
    rc = flush(_FLUSH_TARGET_CURRENT_CTX, scope)
    if rc != 0:
        raise RuntimeError("cuFlushGPUDirectRDMAWrites failed: %s" % _cuda_err(rc))


def register_tensor(pd, tensor, access) -> GpuMR:
    """Register a CUDA ``tensor`` for RDMA and return a :class:`GpuMR`.

    Prefers the dma-buf path (no kernel module needed) and transparently
    handles page alignment. Falls back to the raw ``reg_mr`` (nvidia_peermem)
    path when dma-buf export or registration is unavailable. The returned
    object retains ``tensor`` so its allocation cannot be recycled while the
    MR is registered.

    ``pd`` is the owning protection domain and ``access`` is an ORed
    :class:`~efa.enums.AccessFlags` mask.
    """
    is_cuda = getattr(tensor, "is_cuda", None)
    if is_cuda is not None and not bool(is_cuda):
        raise ValueError(
            "register_tensor requires a CUDA tensor; use efa.reg_tensor "
            "for host tensors"
        )
    ptr, n = tensor_ptr_len(tensor)
    if n <= 0:
        raise ValueError("cannot register an empty tensor")
    base = ptr - (ptr % _PAGE)
    off = ptr - base
    alen = ((off + n + _PAGE - 1) // _PAGE) * _PAGE
    fd = None
    try:
        fd = dmabuf_fd(base, alen)
        mr = pd.reg_dmabuf_mr(off, n, ptr, fd, access)
    except (RuntimeError, OSError) as dmabuf_err:
        try:
            mr = pd.reg_mr(ptr, n, access)
        except Exception as regmr_err:
            raise RuntimeError(
                "GPUDirect registration failed. dma-buf path: %s; "
                "reg_mr (nvidia_peermem) path: %s" % (dmabuf_err, regmr_err)
            ) from regmr_err
    finally:
        if fd is not None:
            os.close(fd)
    return GpuMR(mr, ptr, n, tensor=tensor)


__all__ = [
    "GpuMR",
    "dmabuf_fd",
    "flush_gpudirect_writes",
    "tensor_ptr_len",
    "register_tensor",
]
