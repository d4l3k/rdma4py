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
    setup. :meth:`to_bytes` / :meth:`from_bytes` give a fixed 34-byte wire
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
        self.gid = bytes(self.gid)
        if len(self.gid) != 16:
            raise ValueError("gid must be exactly 16 bytes")

    def to_bytes(self) -> bytes:
        return self._STRUCT.pack(
            self.qp_num, self.psn, self.lid, self.gid, self.port, self.mtu
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "QPInfo":
        qp_num, psn, lid, gid, port, mtu = cls._STRUCT.unpack(
            data[: cls._STRUCT.size]
        )
        return cls(qp_num=qp_num, psn=psn, lid=lid, gid=gid, port=port, mtu=mtu)


def tensor_addr_len(tensor):
    """Return ``(addr, nbytes)`` for a tensor-like object (torch or numpy).

    Works with a torch tensor (``data_ptr()`` + ``numel()`` × ``element_size()``)
    or a numpy array (``ctypes.data`` + ``nbytes``). No import of either library
    is required — the object is duck-typed.
    """
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
    addr, n = tensor_addr_len(tensor)
    return pd.reg_mr(addr, n, access)


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
    :meth:`QP.to_rts` in sequence with sensible defaults.
    """
    qp.to_init(port, access)
    qp.to_rtr(remote, sgid_index=sgid_index, mtu=mtu)
    qp.to_rts(local_psn, **rts_kwargs)
