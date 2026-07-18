"""Pythonic enums mirroring the libibverbs / libefa ABI constants.

These values are part of the stable rdma-core ABI. They are declared here as
:class:`enum.IntEnum` / :class:`enum.IntFlag` so they interoperate with plain
ints everywhere the low-level API accepts a flag or code.
"""

from __future__ import annotations

from enum import IntEnum, IntFlag

# rdma-core marks a range of access flags as "optional"; RELAXED_ORDERING is
# the first of them (IBV_ACCESS_OPTIONAL_FIRST == 1 << 20).
_ACCESS_OPTIONAL_FIRST = 1 << 20


class AccessFlags(IntFlag):
    """Memory-region access permissions (``IBV_ACCESS_*``)."""

    LOCAL_WRITE = 1
    REMOTE_WRITE = 1 << 1
    REMOTE_READ = 1 << 2
    RELAXED_ORDERING = _ACCESS_OPTIONAL_FIRST


class QPType(IntEnum):
    """EFA queue-pair type.

    ``SRD`` (Scalable Reliable Datagram -- reliable, connectionless,
    out-of-order; EFA's native transport) or ``UD`` (unreliable datagram).
    SRD maps onto ``IBV_QPT_DRIVER`` and is created through the EFA provider.
    """

    UD = 4
    SRD = 0xFF


class QPState(IntEnum):
    """Queue-pair state (``IBV_QPS_*``)."""

    RESET = 0
    INIT = 1
    RTR = 2
    RTS = 3
    SQD = 4
    SQE = 5
    ERR = 6
    UNKNOWN = 7


class QPAttrMask(IntFlag):
    """Bitmask selecting which fields ``modify_qp`` applies (``IBV_QP_*``)."""

    STATE = 1 << 0
    CUR_STATE = 1 << 1
    PKEY_INDEX = 1 << 4
    PORT = 1 << 5
    QKEY = 1 << 6
    RNR_RETRY = 1 << 11
    RQ_PSN = 1 << 12
    SQ_PSN = 1 << 16


class QPExSendOpsFlags(IntFlag):
    """Extended-QP send operations (``IBV_QP_EX_WITH_*``).

    Passed as ``QPInitAttr.send_ops_flags``; the created QP can only post
    the opcodes enabled here.
    """

    RDMA_WRITE = 1 << 0
    RDMA_WRITE_WITH_IMM = 1 << 1
    SEND = 1 << 2
    SEND_WITH_IMM = 1 << 3
    RDMA_READ = 1 << 4


class WROpcode(IntEnum):
    """Send work-request opcode (``IBV_WR_*``, the subset EFA supports)."""

    RDMA_WRITE = 0
    RDMA_WRITE_WITH_IMM = 1
    SEND = 2
    SEND_WITH_IMM = 3
    RDMA_READ = 4


class SendFlags(IntFlag):
    """Send work-request flags (``IBV_SEND_*``)."""

    FENCE = 1 << 0
    SIGNALED = 1 << 1
    SOLICITED = 1 << 2
    INLINE = 1 << 3


class WCStatus(IntEnum):
    """Work-completion status (``IBV_WC_*``)."""

    SUCCESS = 0
    LOC_LEN_ERR = 1
    LOC_QP_OP_ERR = 2
    LOC_EEC_OP_ERR = 3
    LOC_PROT_ERR = 4
    WR_FLUSH_ERR = 5
    MW_BIND_ERR = 6
    BAD_RESP_ERR = 7
    LOC_ACCESS_ERR = 8
    REM_INV_REQ_ERR = 9
    REM_ACCESS_ERR = 10
    REM_OP_ERR = 11
    RETRY_EXC_ERR = 12
    RNR_RETRY_EXC_ERR = 13
    LOC_RDD_VIOL_ERR = 14
    REM_INV_RD_REQ_ERR = 15
    REM_ABORT_ERR = 16
    INV_EECN_ERR = 17
    INV_EEC_STATE_ERR = 18
    FATAL_ERR = 19
    RESP_TIMEOUT_ERR = 20
    GENERAL_ERR = 21


class WCOpcode(IntEnum):
    """Work-completion opcode (``IBV_WC_*``)."""

    SEND = 0
    RDMA_WRITE = 1
    RDMA_READ = 2
    RECV = 1 << 7
    RECV_RDMA_WITH_IMM = (1 << 7) + 1


class WCFlags(IntFlag):
    """Work-completion flags (``IBV_WC_*`` bit flags)."""

    GRH = 1 << 0
    WITH_IMM = 1 << 1


class CreateCQWCFlags(IntFlag):
    """Extended-CQ per-completion fields (``IBV_WC_EX_WITH_*``).

    ``STANDARD`` matches what classic ``ibv_poll_cq`` reports and is the
    default for :meth:`~efa._efa.Context.create_cq_ex`.
    """

    BYTE_LEN = 1 << 0
    IMM = 1 << 1
    QP_NUM = 1 << 2
    SRC_QP = 1 << 3
    SLID = 1 << 4
    SL = 1 << 5
    DLID_PATH_BITS = 1 << 6
    STANDARD = 0x7F


class EfaDeviceCaps(IntFlag):
    """EFA device capability bits (``EFADV_DEVICE_ATTR_CAPS_*``).

    Test against :attr:`~efa._efa.EfaDeviceAttr.device_caps`.
    """

    RDMA_READ = 1 << 0
    RNR_RETRY = 1 << 1
    CQ_WITH_SGID = 1 << 2
    RDMA_WRITE = 1 << 3
    UNSOLICITED_WRITE_RECV = 1 << 4
    CQ_WITH_EXT_MEM_DMABUF = 1 << 5


class MTU(IntEnum):
    """Path MTU (``IBV_MTU_*``)."""

    MTU_256 = 1
    MTU_512 = 2
    MTU_1024 = 3
    MTU_2048 = 4
    MTU_4096 = 5


class PortState(IntEnum):
    """Port state (``IBV_PORT_*``)."""

    NOP = 0
    DOWN = 1
    INIT = 2
    ARMED = 3
    ACTIVE = 4
    ACTIVE_DEFER = 5


__all__ = [
    "AccessFlags",
    "QPType",
    "QPState",
    "QPAttrMask",
    "QPExSendOpsFlags",
    "WROpcode",
    "SendFlags",
    "WCStatus",
    "WCOpcode",
    "WCFlags",
    "CreateCQWCFlags",
    "EfaDeviceCaps",
    "MTU",
    "PortState",
]
