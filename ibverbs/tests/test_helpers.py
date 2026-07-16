"""QPInfo serialization and RC connection helpers."""

from __future__ import annotations

import ibverbs as ib
from ibverbs.helpers import QPInfo


def test_qpinfo_roundtrip():
    gid = bytes(range(16))
    info = QPInfo(qp_num=0x123456, psn=0xABCDEF, lid=42, gid=gid, port=1, mtu=5)
    wire = info.to_bytes()
    assert len(wire) == QPInfo._STRUCT.size
    back = QPInfo.from_bytes(wire)
    assert back == info
    assert back.gid == gid


def test_qpinfo_rejects_bad_gid():
    import pytest

    with pytest.raises(ValueError):
        QPInfo(qp_num=1, psn=1, lid=1, gid=b"short", port=1, mtu=5)


def test_local_qp_info_builds_from_port_and_gid(ctx):
    pd = ctx.alloc_pd()
    cq = ctx.create_cq(8)
    qp = pd.create_qp(ib.QPInitAttr(send_cq=cq, recv_cq=cq, qp_type=ib.QPType.RC))
    pa = ctx.query_port(1)
    gid = ctx.query_gid(1, 0)
    info = ib.local_qp_info(qp, pa, gid, port=1, psn=7)
    assert info.qp_num == qp.qp_num
    assert info.psn == 7
    assert info.gid == gid.raw
    assert info.mtu == pa.active_mtu
    qp.close()
    cq.close()
    pd.close()
