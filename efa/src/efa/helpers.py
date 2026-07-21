"""Thin, optional helpers for the common EFA/SRD workflow.

These build on the raw verbs in :mod:`efa._efa`. The raw ``QP.modify`` /
``QP.to_init`` / ``QP.to_rtr`` / ``QP.to_rts`` remain available for callers
that want full control.

SRD is connectionless: unlike RC there is no per-QP connection handshake.
Bootstrap is a one-shot exchange of :class:`EndpointInfo` (GID + QP number +
qkey) out-of-band, e.g. over a TCP socket or a torch distributed store,
after which :meth:`EndpointInfo.peer` gives a :class:`Peer` that any number
of sends can target.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass
class EndpointInfo:
    """The minimum a peer needs to send to an SRD/UD queue pair.

    Exchange this out-of-band (e.g. over a TCP socket or a distributed
    key-value store) during setup. :meth:`to_bytes` / :meth:`from_bytes` give
    a fixed 24-byte wire layout so both ends agree regardless of platform
    endianness.
    """

    gid: bytes
    qp_num: int
    qkey: int

    # ! = network byte order; 16s gid, I qp_num, I qkey
    _STRUCT = struct.Struct("!16sII")

    def __post_init__(self) -> None:
        self.gid = bytes(self.gid)
        self.qp_num = int(self.qp_num)
        self.qkey = int(self.qkey)
        if len(self.gid) != 16:
            raise ValueError("gid must be exactly 16 bytes")
        if not 0 <= self.qp_num <= 0xFFFFFF:
            raise ValueError("qp_num must be a 24-bit value")
        if not 0 <= self.qkey <= 0xFFFFFFFF:
            raise ValueError("qkey must be a 32-bit value")

    def to_bytes(self) -> bytes:
        """Serialize this endpoint to its fixed 24-byte network format."""
        return self._STRUCT.pack(self.gid, self.qp_num, self.qkey)

    @classmethod
    def from_bytes(cls, data: bytes) -> "EndpointInfo":
        """Parse a fixed 24-byte endpoint description."""
        data = bytes(data)
        if len(data) != cls._STRUCT.size:
            raise ValueError(
                "EndpointInfo payload must be exactly %d bytes (got %d)"
                % (cls._STRUCT.size, len(data))
            )
        gid, qp_num, qkey = cls._STRUCT.unpack(data)
        return cls(gid=gid, qp_num=qp_num, qkey=qkey)

    def peer(self, pd, sgid_index: int = 0, port: int = 1) -> "Peer":
        """Create the :class:`Peer` (AH + addressing) for this endpoint.

        ``pd`` owns the new address handle; ``sgid_index`` and ``port`` select
        the local source GID and physical port used to reach the peer.
        """
        ah = pd.create_ah(self.gid, sgid_index=sgid_index, port_num=port)
        return Peer(ah=ah, qp_num=self.qp_num, qkey=self.qkey)


class Peer:
    """A resolved remote endpoint: an AH plus QP number and qkey.

    Pass as ``dest=`` to :class:`~efa._efa.SendWR`. Owns the AH; ``close()``
    (or use as a context manager) destroys it.
    """

    def __init__(self, ah, qp_num: int, qkey: int):
        self.ah = ah
        self.qp_num = int(qp_num)
        self.qkey = int(qkey)

    def close(self) -> None:
        """Destroy the owned address handle, if it is still open."""
        if self.ah is not None:
            self.ah.close()
            self.ah = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return "Peer(qp_num=%d, qkey=0x%x)" % (self.qp_num, self.qkey)


def local_endpoint_info(
    qp, qkey, *, gid=None, gid_index: int = 0, port: int = 1
) -> EndpointInfo:
    """Build an :class:`EndpointInfo` describing ``qp`` for sending to a peer.

    ``qkey`` must match what the QP was prepared with. The GID is queried
    from the QP's device unless passed explicitly. ``gid_index`` and ``port``
    select that query when ``gid`` is omitted.
    """
    if gid is None:
        gid = qp.pd.context.query_gid(port, gid_index)
    raw = gid.raw if hasattr(gid, "raw") else bytes(gid)
    return EndpointInfo(gid=raw, qp_num=qp.qp_num, qkey=qkey)


def _require_contiguous(tensor) -> None:
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if callable(is_contiguous) and not is_contiguous():
        raise ValueError("tensor must be contiguous")

    flags = getattr(tensor, "flags", None)
    if flags is not None and not bool(getattr(flags, "c_contiguous", False)):
        raise ValueError("array must be C-contiguous")


def tensor_addr_len(tensor):
    """Return ``(addr, nbytes)`` for a tensor-like object (torch or numpy).

    Works with a torch tensor (``data_ptr()`` + ``numel()`` * ``element_size()``)
    or a numpy array (``ctypes.data`` + ``nbytes``). No import of either library
    is required; the object is duck-typed.
    """
    _require_contiguous(tensor)
    if hasattr(tensor, "data_ptr"):  # torch.Tensor
        addr = int(tensor.data_ptr())
        n = int(tensor.numel()) * int(tensor.element_size())
    elif hasattr(tensor, "ctypes") and hasattr(tensor, "nbytes"):  # numpy.ndarray
        addr = int(tensor.ctypes.data)
        n = int(tensor.nbytes)
    else:
        raise TypeError("unsupported tensor type: %r" % type(tensor))
    return addr, n


def reg_tensor(pd, tensor, access):
    """Register a host (CPU) tensor's memory as an MR via ``reg_mr``.

    Convenience for a contiguous torch CPU tensor or numpy array. For CUDA
    tensors use :func:`efa.cuda.register_tensor` (dma-buf/GPUDirect) instead.
    ``pd`` is the owning protection domain and ``access`` is an ORed
    :class:`~efa.enums.AccessFlags` mask.
    """
    if bool(getattr(tensor, "is_cuda", False)):
        raise ValueError(
            "reg_tensor only accepts host tensors; use "
            "efa.cuda.register_tensor for CUDA tensors"
        )
    addr, n = tensor_addr_len(tensor)
    if n <= 0:
        raise ValueError("cannot register an empty tensor")
    return pd.reg_mr(addr, n, access)._keepalive(tensor)


def _chunked_rdma_wrs(
    opcode, mr, dest, remote_addr, rkey, length, offset, chunk, wr_id
):
    from . import _efa  # pyre-ignore[21]: Implemented by the Cython extension.
    from .enums import SendFlags

    if length is None:
        length = mr.length - offset
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    if offset < 0 or length <= 0 or offset + length > mr.length:
        raise ValueError("range exceeds the memory region")
    wrs = []
    pos = 0
    while pos < length:
        n = min(chunk, length - pos)
        # EFA requires every send WR to be SIGNALED (unsignaled posts are
        # rejected at wr_complete), so each chunk produces one completion.
        wrs.append(
            _efa.SendWR(
                wr_id=wr_id + len(wrs),
                sg_list=[_efa.SGE(mr.addr + offset + pos, n, lkey=mr.lkey)],
                opcode=opcode,
                send_flags=SendFlags.SIGNALED,
                remote_addr=int(remote_addr) + pos,
                rkey=int(rkey),
                dest=dest,
            )
        )
        pos += n
    return wrs


def write_wrs(
    mr,
    dest,
    remote_addr,
    rkey,
    *,
    length=None,
    offset: int = 0,
    chunk: int = 1 << 30,
    wr_id: int = 0,
):
    """Build the :class:`~efa._efa.SendWR` list for a (possibly huge) RDMA write.

    EFA caps a single RDMA operation at
    :attr:`~efa._efa.EfaDeviceAttr.max_rdma_size` (1 GiB on current
    hardware), so a tensor-sized transfer may need several work requests.
    This slices ``mr[offset:offset+length]`` into ``chunk``-sized writes
    targeting ``remote_addr`` at matching offsets. Every WR is SIGNALED (an
    EFA requirement), so expect ``len(wrs)`` completions; they complete in
    posting order, and SRD completes a WR only once it is fully delivered.

    ``mr`` may be an :class:`~efa._efa.MR` or a GpuMR-like object exposing
    ``addr``/``length``/``lkey``. Returns a list of SendWRs to pass to
    :meth:`~efa._efa.QP.post_send`; ``wr_id`` numbers them sequentially.

    ``dest`` names the remote EFA peer; ``remote_addr`` and ``rkey`` select
    its memory region. ``length`` defaults to the local MR remainder after
    ``offset``. ``chunk`` is the maximum bytes per WR.
    """
    from .enums import WROpcode

    return _chunked_rdma_wrs(
        WROpcode.RDMA_WRITE, mr, dest, remote_addr, rkey, length, offset, chunk, wr_id
    )


def read_wrs(
    mr,
    dest,
    remote_addr,
    rkey,
    *,
    length=None,
    offset: int = 0,
    chunk: int = 1 << 30,
    wr_id: int = 0,
):
    """Build the work requests for an RDMA read into ``mr``.

    ``dest`` names the remote EFA peer; ``remote_addr`` and ``rkey`` select
    its memory region. ``length`` defaults to the local MR remainder after
    ``offset``. ``chunk`` is the maximum bytes per WR, and ``wr_id`` seeds the
    sequential request identifiers.
    """
    from .enums import WROpcode

    return _chunked_rdma_wrs(
        WROpcode.RDMA_READ, mr, dest, remote_addr, rkey, length, offset, chunk, wr_id
    )
