"""Enum values must match the libibverbs ABI constants exactly."""

from ibverbs import enums


def test_access_flags():
    assert enums.AccessFlags.LOCAL_WRITE == 1
    assert enums.AccessFlags.REMOTE_WRITE == 2
    assert enums.AccessFlags.REMOTE_READ == 4
    assert enums.AccessFlags.REMOTE_ATOMIC == 8
    assert enums.AccessFlags.RELAXED_ORDERING == (1 << 20)
    # IntFlag composes:
    combo = enums.AccessFlags.LOCAL_WRITE | enums.AccessFlags.REMOTE_WRITE
    assert int(combo) == 3


def test_wr_opcodes():
    assert enums.WROpcode.RDMA_WRITE == 0
    assert enums.WROpcode.RDMA_WRITE_WITH_IMM == 1
    assert enums.WROpcode.SEND == 2
    assert enums.WROpcode.SEND_WITH_IMM == 3
    assert enums.WROpcode.RDMA_READ == 4
    assert enums.WROpcode.ATOMIC_CMP_AND_SWP == 5
    assert enums.WROpcode.ATOMIC_FETCH_AND_ADD == 6


def test_send_flags():
    assert enums.SendFlags.FENCE == 1
    assert enums.SendFlags.SIGNALED == 2
    assert enums.SendFlags.SOLICITED == 4
    assert enums.SendFlags.INLINE == 8


def test_qp_state_and_type():
    assert enums.QPState.RESET == 0
    assert enums.QPState.INIT == 1
    assert enums.QPState.RTR == 2
    assert enums.QPState.RTS == 3
    assert enums.QPType.RC == 2
    assert enums.QPType.UC == 3
    assert enums.QPType.UD == 4


def test_wc_status_and_opcode():
    assert enums.WCStatus.SUCCESS == 0
    assert enums.WCStatus.LOC_LEN_ERR == 1
    assert enums.WCOpcode.SEND == 0
    assert enums.WCOpcode.RDMA_WRITE == 1
    assert enums.WCOpcode.RECV == (1 << 7)


def test_qp_attr_mask():
    assert enums.QPAttrMask.STATE == 1
    assert enums.QPAttrMask.PORT == (1 << 5)
    assert enums.QPAttrMask.DEST_QPN == (1 << 20)


def test_mtu_and_port_state():
    assert enums.MTU.MTU_1024 == 3
    assert enums.MTU.MTU_4096 == 5
    assert enums.PortState.ACTIVE == 4
