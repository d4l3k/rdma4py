"""efa: low-level Pythonic bindings for AWS EFA (SRD RDMA).

Quickstart (SRD is connectionless -- no per-QP handshake, just exchange
:class:`EndpointInfo` once)::

    import efa

    dev = efa.get_efa_device_list()[0]
    ctx = dev.open()
    pd = ctx.alloc_pd()
    cq = ctx.create_cq(256)
    qp = pd.create_qp(efa.QPInitAttr(send_cq=cq, recv_cq=cq)).prepare(qkey=0x1234)

    info = efa.local_endpoint_info(qp, qkey=0x1234)
    # ... exchange info.to_bytes() with the peer out-of-band ...
    peer = efa.EndpointInfo.from_bytes(remote_bytes).peer(pd)

    mr = efa.reg_tensor(pd, tensor, efa.AccessFlags.LOCAL_WRITE)
    qp.post_send(efa.SendWR(sg_list=[mr.sge()], opcode=efa.WROpcode.SEND,
                            send_flags=efa.SendFlags.SIGNALED, dest=peer))

The compiled fast paths live in :mod:`efa._efa`; enums in :mod:`efa.enums`;
address-exchange and tensor helpers in :mod:`efa.helpers`.
"""

from __future__ import annotations

from ._efa import (  # pyre-ignore[21]: Implemented by the Cython extension.
    AH,
    AHAttr,
    AsyncEvent,
    CompChannel,
    CompletionError,
    Context,
    CQ,
    CQAttr,
    CQEx,
    Device,
    DeviceAttr,
    EfaDeviceAttr,
    EfaError,
    get_device_list,
    get_efa_device_list,
    Gid,
    MR,
    MRAttr,
    PD,
    PortAttr,
    QP,
    QPCap,
    QPInitAttr,
    RecvWR,
    SendWR,
    SGE,
    WC,
    WQAttr,
)
from ._version import __version__
from .enums import (
    AccessFlags,
    CreateCQWCFlags,
    EfaDeviceCaps,
    MTU,
    PortState,
    QPAttrMask,
    QPExSendOpsFlags,
    QPState,
    QPType,
    SendFlags,
    WCFlags,
    WCOpcode,
    WCStatus,
    WROpcode,
)
from .helpers import (
    EndpointInfo,
    local_endpoint_info,
    Peer,
    read_wrs,
    reg_tensor,
    tensor_addr_len,
    write_wrs,
)

__all__ = [
    "__version__",
    # handles / core
    "AH",
    "CQ",
    "CQAttr",
    "CQEx",
    "MR",
    "PD",
    "QP",
    "AHAttr",
    "AsyncEvent",
    "CompChannel",
    "CompletionError",
    "Context",
    "Device",
    "DeviceAttr",
    "EfaDeviceAttr",
    "EfaError",
    "Gid",
    "MRAttr",
    "PortAttr",
    "QPCap",
    "QPInitAttr",
    "RecvWR",
    "SendWR",
    "SGE",
    "WC",
    "WQAttr",
    "get_device_list",
    "get_efa_device_list",
    # enums
    "AccessFlags",
    "CreateCQWCFlags",
    "EfaDeviceCaps",
    "MTU",
    "PortState",
    "QPAttrMask",
    "QPExSendOpsFlags",
    "QPState",
    "QPType",
    "SendFlags",
    "WCFlags",
    "WCOpcode",
    "WCStatus",
    "WROpcode",
    # helpers
    "EndpointInfo",
    "Peer",
    "local_endpoint_info",
    "read_wrs",
    "reg_tensor",
    "tensor_addr_len",
    "write_wrs",
]

# `efa.cuda` (optional GPUDirect helpers) is intentionally NOT imported here
# so that plain `import efa` never dlopens libcuda; import it explicitly with
# `import efa.cuda` when you need it.
