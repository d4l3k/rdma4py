"""Thin, optional helpers for the common RC connection dance.

These build on the raw verbs in :mod:`ibverbs._ibverbs`. The raw
``QP.modify`` / ``QP.to_init`` / ``QP.to_rtr`` / ``QP.to_rts`` remain available
for callers that want full control.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass
class QPInfo:
    """The minimum a peer needs to connect an RC/UD queue pair to this one.

    Exchange this out-of-band (e.g. over a TCP socket) during connection
    setup. :meth:`to_bytes` / :meth:`from_bytes` give a fixed 28-byte wire
    layout so both ends agree regardless of platform endianness.
    """

    qp_num: int
    psn: int
    lid: int
    gid: bytes
    port: int
    mtu: int

    # !  = network byte order; I qp_num, I psn, H lid, 16s gid, B port, B mtu
    _STRUCT = struct.Struct("!IIH16sBB")

    def __post_init__(self) -> None:
        self.qp_num = int(self.qp_num)
        self.psn = int(self.psn)
        self.lid = int(self.lid)
        self.port = int(self.port)
        self.mtu = int(self.mtu)
        self.gid = bytes(self.gid)
        if len(self.gid) != 16:
            raise ValueError("gid must be exactly 16 bytes")
        if not 0 <= self.qp_num <= 0xFFFFFF:
            raise ValueError("qp_num must be a 24-bit value")
        if not 0 <= self.psn <= 0xFFFFFF:
            raise ValueError("psn must be a 24-bit value")
        if not 0 <= self.lid <= 0xFFFF:
            raise ValueError("lid must be a 16-bit value")
        if not 1 <= self.port <= 0xFF:
            raise ValueError("port must be between 1 and 255")
        if not 1 <= self.mtu <= 5:
            raise ValueError("mtu must be a valid ibv_mtu value (1 through 5)")

    def to_bytes(self) -> bytes:
        return self._STRUCT.pack(
            self.qp_num, self.psn, self.lid, self.gid, self.port, self.mtu
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "QPInfo":
        data = bytes(data)
        if len(data) != cls._STRUCT.size:
            raise ValueError(
                "QPInfo payload must be exactly %d bytes (got %d)"
                % (cls._STRUCT.size, len(data))
            )
        qp_num, psn, lid, gid, port, mtu = cls._STRUCT.unpack(data)
        return cls(qp_num=qp_num, psn=psn, lid=lid, gid=gid, port=port, mtu=mtu)


def _require_contiguous(tensor) -> None:
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if callable(is_contiguous) and not is_contiguous():
        raise ValueError("tensor must be contiguous")

    flags = getattr(tensor, "flags", None)
    if flags is not None and not bool(getattr(flags, "c_contiguous", False)):
        raise ValueError("array must be C-contiguous")


def tensor_addr_len(tensor):
    """Return ``(addr, nbytes)`` for a tensor-like object (torch or numpy).

    Works with a torch tensor (``data_ptr()`` + ``numel()`` × ``element_size()``)
    or a numpy array (``ctypes.data`` + ``nbytes``). No import of either library
    is required — the object is duck-typed.
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
    tensors use :func:`ibverbs.cuda.register_tensor` (dma-buf/GPUDirect) instead.
    """
    if bool(getattr(tensor, "is_cuda", False)):
        raise ValueError(
            "reg_tensor only accepts host tensors; use "
            "ibverbs.cuda.register_tensor for CUDA tensors"
        )
    addr, n = tensor_addr_len(tensor)
    if n <= 0:
        raise ValueError("cannot register an empty tensor")
    return pd.reg_mr(addr, n, access)._keepalive(tensor)


def local_qp_info(qp, port_attr, gid, *, port: int, psn: int = 0) -> QPInfo:
    """Build a :class:`QPInfo` describing ``qp`` for sending to a peer.

    ``port_attr`` comes from :meth:`Context.query_port`, ``gid`` from
    :meth:`Context.query_gid` (its ``.raw`` bytes are used).
    """
    raw = gid.raw if hasattr(gid, "raw") else bytes(gid)
    return QPInfo(
        qp_num=qp.qp_num,
        psn=psn,
        lid=port_attr.lid,
        gid=raw,
        port=port,
        mtu=port_attr.active_mtu,
    )


def connect_rc(
    qp,
    remote: QPInfo,
    *,
    port: int,
    sgid_index: int,
    access: int,
    local_psn: int = 0,
    mtu: int | None = None,
    **rts_kwargs,
) -> None:
    """Drive an RC ``qp`` all the way from RESET to RTS against ``remote``.

    Equivalent to calling :meth:`QP.to_init`, :meth:`QP.to_rtr` and
    :meth:`QP.to_rts` in sequence with sensible defaults. Unless explicitly
    overridden, the path MTU is negotiated as the smaller of the local and
    remote active MTUs.
    """
    if mtu is None:
        local_mtu = qp.pd.context.query_port(port).active_mtu
        mtu = min(int(local_mtu), int(remote.mtu))
    qp.to_init(port, access)
    qp.to_rtr(remote, sgid_index=sgid_index, mtu=mtu)
    qp.to_rts(local_psn, **rts_kwargs)
