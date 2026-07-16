"""Queue-pair creation, state transitions, SRQ, and address handles."""

from __future__ import annotations

import numpy as np
import pytest

import ibverbs as ib


@pytest.fixture()
def pd(ctx):
    p = ctx.alloc_pd()
    yield p
    p.close()


def test_create_rc_qp(ctx, pd):
    cq = ctx.create_cq(16)
    qp = pd.create_qp(ib.QPInitAttr(send_cq=cq, recv_cq=cq, qp_type=ib.QPType.RC))
    assert qp.qp_num > 0
    assert qp.qp_type == ib.QPType.RC
    assert qp.state == ib.QPState.RESET
    qp.close()
    cq.close()


def test_qp_to_init(ctx, pd):
    cq = ctx.create_cq(16)
    qp = pd.create_qp(ib.QPInitAttr(send_cq=cq, recv_cq=cq, qp_type=ib.QPType.RC))
    qp.to_init(1, ib.AccessFlags.LOCAL_WRITE | ib.AccessFlags.REMOTE_WRITE)
    assert qp.state == ib.QPState.INIT
    qp.close()
    cq.close()


def test_qp_query_reports_caps(ctx, pd):
    cq = ctx.create_cq(16)
    qp = pd.create_qp(ib.QPInitAttr(send_cq=cq, recv_cq=cq, qp_type=ib.QPType.RC,
                                    max_send_wr=64, max_recv_wr=64))
    attrs, cap = qp.query()
    assert cap.max_send_wr >= 64
    assert cap.max_recv_wr >= 64
    assert attrs["qp_state"] == ib.QPState.RESET
    qp.close()
    cq.close()


def test_post_recv_on_init_qp(ctx, pd):
    cq = ctx.create_cq(16)
    qp = pd.create_qp(ib.QPInitAttr(send_cq=cq, recv_cq=cq, qp_type=ib.QPType.RC))
    qp.to_init(1, ib.AccessFlags.LOCAL_WRITE)
    buf = np.zeros(1024, dtype=np.uint8)
    mr = pd.reg_mr(buf.ctypes.data, buf.nbytes, ib.AccessFlags.LOCAL_WRITE)
    qp.post_recv(ib.RecvWR(wr_id=1, sg_list=[ib.SGE(mr, 1024)]))  # must not raise
    mr.close()
    qp.close()
    cq.close()


def test_create_srq_and_post(ctx, pd):
    srq = pd.create_srq(max_wr=32, max_sge=1)
    info = srq.query()
    assert info["max_wr"] >= 32
    buf = np.zeros(1024, dtype=np.uint8)
    mr = pd.reg_mr(buf.ctypes.data, buf.nbytes, ib.AccessFlags.LOCAL_WRITE)
    srq.post_recv(ib.RecvWR(wr_id=1, sg_list=[ib.SGE(mr, 1024)]))
    mr.close()
    srq.close()


def test_create_qp_with_srq(ctx, pd):
    cq = ctx.create_cq(16)
    srq = pd.create_srq(max_wr=32)
    qp = pd.create_qp(ib.QPInitAttr(send_cq=cq, recv_cq=cq, qp_type=ib.QPType.RC,
                                    srq=srq))
    assert qp.srq is srq
    qp.close()
    srq.close()
    cq.close()


def test_create_ah(ctx, pd, dev_name, first_active):
    from conftest import find_roce_gid

    _, port = first_active
    gid_index, gid = find_roce_gid(ctx, dev_name, port)
    attr = ib.AHAttr(dgid=gid.raw, sgid_index=gid_index, port_num=port,
                     is_global=1, hop_limit=1)
    ah = pd.create_ah(attr)
    assert ah is not None
    ah.close()


def test_sge_from_mr_and_from_addr(pd):
    buf = np.zeros(2048, dtype=np.uint8)
    mr = pd.reg_mr(buf.ctypes.data, buf.nbytes, ib.AccessFlags.LOCAL_WRITE)
    sge_mr = ib.SGE(mr, 512, offset=64)
    assert sge_mr.addr == mr.addr + 64
    assert sge_mr.length == 512
    assert sge_mr.lkey == mr.lkey
    sge_addr = ib.SGE(0x1000, 128, lkey=0x42)
    assert sge_addr.addr == 0x1000
    assert sge_addr.lkey == 0x42
    mr.close()
