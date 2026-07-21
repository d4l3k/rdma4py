"""Pythonic enums mirroring the libibverbs ABI constants.

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
    """Memory-region / QP access permissions (``IBV_ACCESS_*``)."""

    #: Permit the local HCA to write the registered memory.
    LOCAL_WRITE = 1
    #: Permit a remote peer to write the region; also requires ``LOCAL_WRITE``.
    REMOTE_WRITE = 1 << 1
    #: Permit a remote peer to read the region.
    REMOTE_READ = 1 << 2
    #: Permit remote atomic operations; also requires ``LOCAL_WRITE``.
    REMOTE_ATOMIC = 1 << 3
    #: Permit binding this region into a memory window.
    MW_BIND = 1 << 4
    #: Interpret SGE addresses as offsets from the MR base address.
    ZERO_BASED = 1 << 5
    #: Register the region with on-demand paging.
    ON_DEMAND = 1 << 6
    #: Assert that the on-demand-paging region is backed by huge pages.
    HUGETLB = 1 << 7
    #: Permit remote flushes to global visibility.
    FLUSH_GLOBAL = 1 << 8
    #: Permit remote flushes to persistent storage.
    FLUSH_PERSISTENT = 1 << 9
    #: Allow relaxed inbound write ordering for higher performance.
    RELAXED_ORDERING = _ACCESS_OPTIONAL_FIRST


class QPType(IntEnum):
    """Queue-pair transport service type (``IBV_QPT_*``)."""

    RC = 2
    UC = 3
    UD = 4
    RAW_PACKET = 8
    XRC_SEND = 9
    XRC_RECV = 10
    DRIVER = 0xFF


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
    #: Apply send-queue-drained asynchronous notification.
    EN_SQD_ASYNC_NOTIFY = 1 << 2
    #: Apply QP access permissions.
    ACCESS_FLAGS = 1 << 3
    #: Apply the partition-key table index.
    PKEY_INDEX = 1 << 4
    #: Apply the local physical port.
    PORT = 1 << 5
    #: Apply the datagram queue key.
    QKEY = 1 << 6
    #: Apply the address vector.
    AV = 1 << 7
    #: Apply the path MTU.
    PATH_MTU = 1 << 8
    #: Apply the local ACK timeout exponent.
    TIMEOUT = 1 << 9
    #: Apply the transport retry count.
    RETRY_CNT = 1 << 10
    #: Apply the receiver-not-ready retry count.
    RNR_RETRY = 1 << 11
    #: Apply the receive packet sequence number.
    RQ_PSN = 1 << 12
    #: Apply the initiator RDMA Read/atomic depth.
    MAX_QP_RD_ATOMIC = 1 << 13
    #: Apply the alternate path.
    ALT_PATH = 1 << 14
    #: Apply the minimum receiver-not-ready timer.
    MIN_RNR_TIMER = 1 << 15
    #: Apply the send packet sequence number.
    SQ_PSN = 1 << 16
    #: Apply the responder RDMA Read/atomic depth.
    MAX_DEST_RD_ATOMIC = 1 << 17
    #: Apply the path-migration state.
    PATH_MIG_STATE = 1 << 18
    #: Apply queue-pair capacity limits.
    CAP = 1 << 19
    #: Apply the destination queue-pair number.
    DEST_QPN = 1 << 20
    #: Apply the packet pacing rate limit.
    RATE_LIMIT = 1 << 25


class WROpcode(IntEnum):
    """Send work-request opcode (``IBV_WR_*``)."""

    RDMA_WRITE = 0
    RDMA_WRITE_WITH_IMM = 1
    SEND = 2
    SEND_WITH_IMM = 3
    RDMA_READ = 4
    ATOMIC_CMP_AND_SWP = 5
    ATOMIC_FETCH_AND_ADD = 6
    LOCAL_INV = 7
    BIND_MW = 8
    SEND_WITH_INV = 9
    TSO = 10


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
    #: Ask a raw-packet QP to calculate the IP checksum.
    IP_CSUM = 1 << 4


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
    TM_ERR = 22
    TM_RNDV_INCOMPLETE = 23


class WCOpcode(IntEnum):
    """Work-completion opcode (``IBV_WC_*``)."""

    SEND = 0
    RDMA_WRITE = 1
    RDMA_READ = 2
    COMP_SWAP = 3
    FETCH_ADD = 4
    BIND_MW = 5
    LOCAL_INV = 6
    TSO = 7
    RECV = 1 << 7
    RECV_RDMA_WITH_IMM = (1 << 7) + 1


class WCFlags(IntFlag):
    """Work-completion flags (``IBV_WC_*`` bit flags)."""

    #: A global route header precedes the received payload.
    GRH = 1 << 0
    #: The completion contains valid immediate data.
    WITH_IMM = 1 << 1
    #: The received IP checksum was verified successfully.
    IP_CSUM_OK = 1 << 2
    #: The completion contains a valid invalidated rkey.
    WITH_INV = 1 << 3
    #: A tag-matching synchronization operation completed.
    TM_SYNC_REQ = 1 << 4
    #: A tag-matching receive matched.
    TM_MATCH = 1 << 5
    #: Tag-matching completion data is valid.
    TM_DATA_VALID = 1 << 6


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


class NodeType(IntEnum):
    """Device node type (``IBV_NODE_*``)."""

    UNKNOWN = -1
    CA = 1
    SWITCH = 2
    ROUTER = 3
    RNIC = 4
    USNIC = 5
    USNIC_UDP = 6
    UNSPECIFIED = 7


__all__ = [
    "AccessFlags",
    "QPType",
    "QPState",
    "QPAttrMask",
    "WROpcode",
    "SendFlags",
    "WCStatus",
    "WCOpcode",
    "WCFlags",
    "MTU",
    "PortState",
    "NodeType",
]
