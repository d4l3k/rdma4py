"""Optional CUDA GPUDirect helpers for registering GPU tensors as memory regions.

This module lazily ``dlopen``s ``libcuda`` and imports **no** torch and links
**no** CUDA at build time, so it adds no required dependency. It is duck-typed:
anything exposing ``data_ptr()`` (e.g. a torch CUDA tensor) works.

For torch, allocate with the VMM allocator so the memory can be exported as a
dma-buf fd (the fd libibverbs needs for ``ibv_reg_dmabuf_mr``)::

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Typical use::

    import ibverbs, ibverbs.cuda, torch
    t = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    gmr = ibverbs.cuda.register_tensor(pd, t, ibverbs.AccessFlags.LOCAL_WRITE)
    qp.post_send(ibverbs.SendWR(sg_list=[gmr.sge()], ...))
"""

from __future__ import annotations

import ctypes
import os

from . import _ibverbs

_PAGE = 4096
_DMA_BUF_FD = 1  # CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD
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

    def __init__(self, mr, addr: int, length: int):
        self.mr = mr
        self.addr = int(addr)
        self.length = int(length)

    @property
    def lkey(self) -> int:
        return self.mr.lkey

    @property
    def rkey(self) -> int:
        return self.mr.rkey

    def sge(self, length=None, offset=0):
        return _ibverbs.SGE(self.addr + offset,
                            length if length is not None else self.length,
                            lkey=self.lkey)

    def close(self):
        self.mr.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return "GpuMR(addr=0x%x, length=%d, lkey=0x%x, rkey=0x%x)" % (
            self.addr, self.length, self.lkey, self.rkey)


def dmabuf_fd(ptr: int, length: int) -> int:
    """Export a dma-buf fd for the device VA range ``[ptr, ptr+length)``.

    ``ptr`` and ``length`` must be page-aligned (:func:`register_tensor`
    handles alignment). The caller owns and must ``os.close`` the returned fd.
    """
    lib = _cuda()
    fd = ctypes.c_int(-1)
    rc = lib.cuMemGetHandleForAddressRange(
        ctypes.byref(fd), ctypes.c_ulonglong(int(ptr)),
        ctypes.c_size_t(int(length)), _DMA_BUF_FD, ctypes.c_ulonglong(0))
    if rc != 0:
        raise RuntimeError(
            "cuMemGetHandleForAddressRange failed: %s. The allocation must be "
            "VMM-backed (for torch set "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)." % _cuda_err(rc))
    return fd.value


def tensor_ptr_len(tensor):
    """Return ``(device_addr, nbytes)`` for a tensor-like object."""
    if not hasattr(tensor, "data_ptr"):
        raise TypeError("expected a tensor exposing data_ptr(); got %r"
                        % type(tensor))
    ptr = int(tensor.data_ptr())
    if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
        n = int(tensor.numel()) * int(tensor.element_size())
    elif hasattr(tensor, "nbytes"):
        n = int(tensor.nbytes)
    else:
        raise TypeError("cannot determine byte length of %r" % type(tensor))
    return ptr, n


def register_tensor(pd, tensor, access) -> GpuMR:
    """Register a CUDA ``tensor`` for RDMA and return a :class:`GpuMR`.

    Prefers the dma-buf path (no kernel module needed) and transparently
    handles page alignment. Falls back to the raw ``reg_mr`` (nvidia_peermem)
    path when dma-buf export is unavailable.
    """
    ptr, n = tensor_ptr_len(tensor)
    base = ptr - (ptr % _PAGE)
    off = ptr - base
    alen = ((off + n + _PAGE - 1) // _PAGE) * _PAGE
    try:
        fd = dmabuf_fd(base, alen)
    except RuntimeError as dmabuf_err:
        try:
            return GpuMR(pd.reg_mr(ptr, n, access), ptr, n)
        except Exception as regmr_err:
            raise RuntimeError(
                "GPUDirect registration failed. dma-buf export: %s; "
                "reg_mr (nvidia_peermem) path: %s" % (dmabuf_err, regmr_err)
            ) from regmr_err
    try:
        mr = pd.reg_dmabuf_mr(off, n, ptr, fd, access)
    finally:
        os.close(fd)
    return GpuMR(mr, ptr, n)


__all__ = ["GpuMR", "dmabuf_fd", "tensor_ptr_len", "register_tensor"]
