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

    #: Permit the local HCA to write the registered memory.
    LOCAL_WRITE = 1
    #: Permit a remote peer to write the region; also requires ``LOCAL_WRITE``.
    REMOTE_WRITE = 1 << 1
    #: Permit a remote peer to read the region.
    REMOTE_READ = 1 << 2
    #: Allow relaxed inbound write ordering for higher performance.
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

    #: Apply ``qp_state``.
    STATE = 1 << 0
    #: Apply ``cur_qp_state`` as a transition precondition.
    CUR_STATE = 1 << 1
    #: Apply the partition-key table index.
    PKEY_INDEX = 1 << 4
    #: Apply the local physical port.
    PORT = 1 << 5
    #: Apply the datagram queue key.
    QKEY = 1 << 6
    #: Apply the receiver-not-ready retry count.
    RNR_RETRY = 1 << 11
    #: Apply the receive packet sequence number.
    RQ_PSN = 1 << 12
    #: Apply the send packet sequence number.
    SQ_PSN = 1 << 16


class QPExSendOpsFlags(IntFlag):
    """Extended-QP send operations (``IBV_QP_EX_WITH_*``).

    Passed as ``QPInitAttr.send_ops_flags``; the created QP can only post
    the opcodes enabled here.
    """

    #: Enable RDMA Write work requests.
    RDMA_WRITE = 1 << 0
    #: Enable RDMA Write with Immediate work requests.
    RDMA_WRITE_WITH_IMM = 1 << 1
    #: Enable Send work requests.
    SEND = 1 << 2
    #: Enable Send with Immediate work requests.
    SEND_WITH_IMM = 1 << 3
    #: Enable RDMA Read work requests.
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

    #: Fence this request behind earlier RDMA Read or atomic operations.
    FENCE = 1 << 0
    #: Generate a send completion for this request.
    SIGNALED = 1 << 1
    #: Request a solicited receive event at the peer.
    SOLICITED = 1 << 2
    #: Copy the payload into the work request instead of retaining its SGEs.
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

    #: A global route header precedes the received payload.
    GRH = 1 << 0
    #: The completion contains valid immediate data.
    WITH_IMM = 1 << 1


class CreateCQWCFlags(IntFlag):
    """Extended-CQ per-completion fields (``IBV_WC_EX_WITH_*``).

    ``STANDARD`` matches what classic ``ibv_poll_cq`` reports and is the
    default for :meth:`~efa._efa.Context.create_cq_ex`.
    """

    #: Request the completed byte count.
    BYTE_LEN = 1 << 0
    #: Request immediate data.
    IMM = 1 << 1
    #: Request the local queue-pair number.
    QP_NUM = 1 << 2
    #: Request the source queue-pair number.
    SRC_QP = 1 << 3
    #: Request the source LID.
    SLID = 1 << 4
    #: Request the service level.
    SL = 1 << 5
    #: Request destination LID path bits.
    DLID_PATH_BITS = 1 << 6
    #: Request every field reported by a classic completion queue.
    STANDARD = 0x7F


class EfaDeviceCaps(IntFlag):
    """EFA device capability bits (``EFADV_DEVICE_ATTR_CAPS_*``).

    Test against :attr:`~efa._efa.EfaDeviceAttr.device_caps`.
    """

    #: The provider supports SRD RDMA Read.
    RDMA_READ = 1 << 0
    #: The provider supports configurable RNR retry.
    RNR_RETRY = 1 << 1
    #: Extended completion queues can report sender GIDs.
    CQ_WITH_SGID = 1 << 2
    #: The provider supports SRD RDMA Write.
    RDMA_WRITE = 1 << 3
    #: QPs can receive unsolicited RDMA Write with Immediate.
    UNSOLICITED_WRITE_RECV = 1 << 4
    #: Extended completion-queue memory can be supplied through dma-buf.
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
