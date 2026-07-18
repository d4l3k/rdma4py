"""Enum values must match the libibverbs/libefa ABI constants exactly."""

from efa import enums


def test_access_flags():
    assert enums.AccessFlags.LOCAL_WRITE == 1
    assert enums.AccessFlags.REMOTE_WRITE == 2
    assert enums.AccessFlags.REMOTE_READ == 4
    assert enums.AccessFlags.RELAXED_ORDERING == (1 << 20)
    combo = enums.AccessFlags.LOCAL_WRITE | enums.AccessFlags.REMOTE_WRITE
    assert int(combo) == 3


def test_qp_types():
    assert enums.QPType.UD == 4
    assert enums.QPType.SRD == 0xFF  # tag mapped to IBV_QPT_DRIVER at create


def test_wr_opcodes():
    assert enums.WROpcode.RDMA_WRITE == 0
    assert enums.WROpcode.RDMA_WRITE_WITH_IMM == 1
    assert enums.WROpcode.SEND == 2
    assert enums.WROpcode.SEND_WITH_IMM == 3
    assert enums.WROpcode.RDMA_READ == 4


def test_send_flags():
    assert enums.SendFlags.FENCE == 1
    assert enums.SendFlags.SIGNALED == 2
    assert enums.SendFlags.SOLICITED == 4
    assert enums.SendFlags.INLINE == 8


def test_qp_state():
    assert enums.QPState.RESET == 0
    assert enums.QPState.INIT == 1
    assert enums.QPState.RTR == 2
    assert enums.QPState.RTS == 3


def test_qp_attr_mask():
    assert enums.QPAttrMask.STATE == 1
    assert enums.QPAttrMask.PKEY_INDEX == (1 << 4)
    assert enums.QPAttrMask.PORT == (1 << 5)
    assert enums.QPAttrMask.QKEY == (1 << 6)
    assert enums.QPAttrMask.RNR_RETRY == (1 << 11)
    assert enums.QPAttrMask.SQ_PSN == (1 << 16)


def test_qp_ex_send_ops_flags():
    assert enums.QPExSendOpsFlags.RDMA_WRITE == 1
    assert enums.QPExSendOpsFlags.RDMA_WRITE_WITH_IMM == 2
    assert enums.QPExSendOpsFlags.SEND == 4
    assert enums.QPExSendOpsFlags.SEND_WITH_IMM == 8
    assert enums.QPExSendOpsFlags.RDMA_READ == 16


def test_wc_status_and_opcode():
    assert enums.WCStatus.SUCCESS == 0
    assert enums.WCStatus.LOC_LEN_ERR == 1
    assert enums.WCStatus.RNR_RETRY_EXC_ERR == 13
    assert enums.WCOpcode.SEND == 0
    assert enums.WCOpcode.RDMA_WRITE == 1
    assert enums.WCOpcode.RECV == (1 << 7)
    assert enums.WCOpcode.RECV_RDMA_WITH_IMM == (1 << 7) + 1


def test_wc_flags():
    assert enums.WCFlags.GRH == 1
    assert enums.WCFlags.WITH_IMM == 2


def test_efa_device_caps():
    assert enums.EfaDeviceCaps.RDMA_READ == 1
    assert enums.EfaDeviceCaps.RNR_RETRY == 2
    assert enums.EfaDeviceCaps.CQ_WITH_SGID == 4
    assert enums.EfaDeviceCaps.RDMA_WRITE == 8
    assert enums.EfaDeviceCaps.UNSOLICITED_WRITE_RECV == 16
    assert enums.EfaDeviceCaps.CQ_WITH_EXT_MEM_DMABUF == 32


def test_create_cq_wc_flags():
    assert enums.CreateCQWCFlags.BYTE_LEN == 1
    assert enums.CreateCQWCFlags.STANDARD == 0x7F


def test_port_state_and_mtu():
    assert enums.PortState.ACTIVE == 4
    assert enums.MTU.MTU_4096 == 5
