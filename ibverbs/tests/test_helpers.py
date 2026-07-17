"""QPInfo serialization and RC connection helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

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
    with pytest.raises(ValueError):
        QPInfo(qp_num=1, psn=1, lid=1, gid=b"short", port=1, mtu=5)


@pytest.mark.parametrize("size", [QPInfo._STRUCT.size - 1, QPInfo._STRUCT.size + 1])
def test_qpinfo_rejects_wrong_wire_size(size):
    with pytest.raises(ValueError, match="exactly 28 bytes"):
        QPInfo.from_bytes(b"\0" * size)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qp_num", 1 << 24),
        ("psn", -1),
        ("lid", 1 << 16),
        ("port", 0),
        ("mtu", 6),
    ],
)
def test_qpinfo_rejects_out_of_range_fields(field, value):
    values = dict(qp_num=1, psn=1, lid=1, gid=bytes(16), port=1, mtu=5)
    values[field] = value
    with pytest.raises(ValueError):
        QPInfo(**values)


def test_tensor_addr_len_rejects_noncontiguous_array():
    tensor = np.zeros((4, 4), dtype=np.float32)[:, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        ib.tensor_addr_len(tensor)


def test_reg_tensor_rejects_cuda_tensor():
    tensor = SimpleNamespace(is_cuda=True)
    with pytest.raises(ValueError, match="cuda.register_tensor"):
        ib.reg_tensor(None, tensor, ib.AccessFlags.LOCAL_WRITE)


def test_connect_rc_negotiates_minimum_path_mtu():
    calls = []

    class QP:
        pd = SimpleNamespace(
            context=SimpleNamespace(
                query_port=lambda port: SimpleNamespace(active_mtu=3)
            )
        )

        def to_init(self, *args):
            calls.append(("init", args))

        def to_rtr(self, remote, **kwargs):
            calls.append(("rtr", kwargs))

        def to_rts(self, *args, **kwargs):
            calls.append(("rts", args, kwargs))

    remote = QPInfo(qp_num=1, psn=2, lid=0, gid=bytes(16), port=1, mtu=5)
    ib.connect_rc(
        QP(), remote, port=1, sgid_index=0,
        access=ib.AccessFlags.LOCAL_WRITE,
    )
    assert calls[1] == ("rtr", {"sgid_index": 0, "mtu": 3})


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
