"""Queue-pair creation, state machine, and validation on real hardware."""

from __future__ import annotations

import efa
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def cq(ctx):
    cq = ctx.create_cq(64)
    yield cq
    cq.close()


def test_create_srd_qp(pd, cq):
    qp = pd.create_qp(efa.QPInitAttr(send_cq=cq, recv_cq=cq))
    assert qp.qp_type == efa.QPType.SRD
    assert qp.qp_num >= 0
    assert qp.state == efa.QPState.RESET
    assert "SRD" in repr(qp)
    qp.close()
    with pytest.raises(efa.EfaError):
        _ = qp.qp_num


def test_create_ud_qp(pd, cq):
    qp = pd.create_qp(efa.QPInitAttr(send_cq=cq, recv_cq=cq, qp_type=efa.QPType.UD))
    assert qp.qp_type == efa.QPType.UD
    qp.close()


def test_rejects_unknown_qp_type(pd, cq):
    with pytest.raises(ValueError):
        pd.create_qp(efa.QPInitAttr(send_cq=cq, recv_cq=cq, qp_type=2))  # RC


def test_state_machine_stepwise(pd, cq):
    qp = pd.create_qp(efa.QPInitAttr(send_cq=cq, recv_cq=cq))
    qp.to_init(qkey=0xABCD)
    assert qp.state == efa.QPState.INIT
    qp.to_rtr()
    assert qp.state == efa.QPState.RTR
    qp.to_rts(psn=7)
    assert qp.state == efa.QPState.RTS
    attrs, cap = qp.query()
    assert attrs["qp_state"] == efa.QPState.RTS
    assert attrs["qkey"] == 0xABCD
    assert cap.max_send_wr >= 1
    qp.close()


def test_prepare_shortcut(pd, cq):
    qp = pd.create_qp(efa.QPInitAttr(send_cq=cq, recv_cq=cq))
    assert qp.prepare(qkey=0x1111) is qp
    assert qp.state == efa.QPState.RTS
    qp.close()


def test_prepare_with_rnr_retry(pd, cq, efa_caps):
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.RNR_RETRY)
    qp = pd.create_qp(efa.QPInitAttr(send_cq=cq, recv_cq=cq))
    qp.prepare(qkey=0x2222, rnr_retry=3)
    attrs, _ = qp.query()
    assert attrs["rnr_retry"] == 3
    qp.close()


def test_modify_raw(pd, cq):
    qp = pd.create_qp(efa.QPInitAttr(send_cq=cq, recv_cq=cq))
    qp.modify(
        efa.QPAttrMask.STATE
        | efa.QPAttrMask.PKEY_INDEX
        | efa.QPAttrMask.PORT
        | efa.QPAttrMask.QKEY,
        qp_state=efa.QPState.INIT,
        pkey_index=0,
        port_num=1,
        qkey=0x77,
    )
    assert qp.state == efa.QPState.INIT
    with pytest.raises(TypeError):
        qp.modify(efa.QPAttrMask.STATE, bogus_field=1)
    qp.close()


def test_send_ops_flags_restriction(pd, cq):
    """A QP created with SEND-only ops must reject RDMA posts at wr time."""
    qp = pd.create_qp(
        efa.QPInitAttr(
            send_cq=cq,
            recv_cq=cq,
            send_ops_flags=efa.QPExSendOpsFlags.SEND
            | efa.QPExSendOpsFlags.SEND_WITH_IMM,
        )
    ).prepare(qkey=0x3333)
    peer = efa.local_endpoint_info(qp, qkey=0x3333).peer(pd)
    with pytest.raises((efa.EfaError, ValueError)):
        qp.post_send(
            efa.SendWR(
                wr_id=1,
                sg_list=[efa.SGE(0x1000, 8, lkey=1)],
                opcode=efa.WROpcode.RDMA_WRITE,
                send_flags=efa.SendFlags.SIGNALED,
                remote_addr=0x2000,
                rkey=1,
                dest=peer,
            )
        )
    peer.close()
    qp.close()


def test_send_requires_destination(pd, cq):
    qp = pd.create_qp(efa.QPInitAttr(send_cq=cq, recv_cq=cq)).prepare(qkey=1)
    with pytest.raises(ValueError, match="connectionless"):
        qp.post_send(
            efa.SendWR(
                wr_id=1,
                sg_list=[efa.SGE(0x1000, 8, lkey=1)],
                opcode=efa.WROpcode.SEND,
                send_flags=efa.SendFlags.SIGNALED,
            )
        )
    qp.close()


def test_unsupported_opcode_rejected(pd, cq):
    qp = pd.create_qp(efa.QPInitAttr(send_cq=cq, recv_cq=cq)).prepare(qkey=1)
    ah = pd.create_ah(qp.pd.context.query_gid())
    with pytest.raises(ValueError, match="EFA supports"):
        qp.post_send(
            efa.SendWR(
                wr_id=1,
                sg_list=[efa.SGE(0x1000, 8, lkey=1)],
                opcode=99,
                send_flags=efa.SendFlags.SIGNALED,
                ah=ah,
                remote_qpn=qp.qp_num,
                remote_qkey=1,
            )
        )
    ah.close()
    qp.close()


def test_query_wqs(pd, cq):
    qp = pd.create_qp(efa.QPInitAttr(send_cq=cq, recv_cq=cq))
    try:
        sq, rq = qp.query_wqs()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert sq.num_entries > 0
    assert rq.num_entries > 0
    assert sq.entry_size > 0
    qp.close()


def test_cq_from_other_context_rejected(efa_devices, pd):
    if len(efa_devices) < 2:
        pytest.skip("needs two EFA devices")
    other = efa_devices[1].open()
    other_cq = other.create_cq(16)
    with pytest.raises(ValueError):
        pd.create_qp(efa.QPInitAttr(send_cq=other_cq, recv_cq=other_cq))
    other_cq.close()
    other.close()
