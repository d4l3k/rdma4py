"""ibverbs: low-level Pythonic bindings for libibverbs (RDMA).

Quickstart::

    import ibverbs as ib

    dev = ib.get_device_list()[0]
    ctx = dev.open()
    pd = ctx.alloc_pd()
    cq = ctx.create_cq(16)
    mr = pd.reg_mr(buf_addr, nbytes,
                   ib.AccessFlags.LOCAL_WRITE | ib.AccessFlags.REMOTE_WRITE)
    qp = pd.create_qp(ib.QPInitAttr(send_cq=cq, recv_cq=cq))

The compiled fast paths live in :mod:`ibverbs._ibverbs`; enums in
:mod:`ibverbs.enums`; thin RC helpers in :mod:`ibverbs.helpers`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from ._ibverbs import (
    AH,
    CQ,
    MR,
    PD,
    QP,
    SRQ,
    AHAttr,
    AsyncEvent,
    CompChannel,
    Context,
    Device,
    DeviceAttr,
    Gid,
    PortAttr,
    QPCap,
    QPInitAttr,
    RecvWR,
    SendWR,
    SGE,
    VerbsError,
    WC,
    get_device_list,
)
from .enums import (
    AccessFlags,
    MTU,
    NodeType,
    PortState,
    QPAttrMask,
    QPState,
    QPType,
    SendFlags,
    WCFlags,
    WCOpcode,
    WCStatus,
    WROpcode,
)
from .helpers import (
    QPInfo,
    connect_rc,
    local_qp_info,
    reg_tensor,
    tensor_addr_len,
)

__all__ = [
    "__version__",
    # handles / core
    "AH", "CQ", "MR", "PD", "QP", "SRQ", "AHAttr", "AsyncEvent", "CompChannel",
    "Context", "Device", "DeviceAttr", "Gid", "PortAttr", "QPCap", "QPInitAttr",
    "RecvWR", "SendWR", "SGE", "VerbsError", "WC", "get_device_list",
    # enums
    "AccessFlags", "MTU", "NodeType", "PortState", "QPAttrMask", "QPState",
    "QPType", "SendFlags", "WCFlags", "WCOpcode", "WCStatus", "WROpcode",
    # helpers
    "QPInfo", "connect_rc", "local_qp_info", "reg_tensor", "tensor_addr_len",
]

# `ibverbs.cuda` (optional GPUDirect helpers) is intentionally NOT imported
# here so that plain `import ibverbs` never dlopens libcuda; import it
# explicitly with `import ibverbs.cuda` when you need it.
