"""EndpointInfo serialization and pure-Python helper behavior (no hardware)."""

from __future__ import annotations

from types import SimpleNamespace

import efa
import numpy as np
import pytest
from efa.helpers import EndpointInfo


def test_endpoint_info_roundtrip():
    gid = bytes(range(16))
    info = EndpointInfo(gid=gid, qp_num=0x123456, qkey=0xDEADBEEF)
    wire = info.to_bytes()
    assert len(wire) == EndpointInfo._STRUCT.size == 24
    back = EndpointInfo.from_bytes(wire)
    assert back == info
    assert back.gid == gid


def test_endpoint_info_rejects_bad_gid():
    with pytest.raises(ValueError):
        EndpointInfo(gid=b"short", qp_num=1, qkey=1)


@pytest.mark.parametrize(
    "size", [EndpointInfo._STRUCT.size - 1, EndpointInfo._STRUCT.size + 1]
)
def test_endpoint_info_rejects_wrong_wire_size(size):
    with pytest.raises(ValueError, match="exactly 24 bytes"):
        EndpointInfo.from_bytes(b"\0" * size)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qp_num", 1 << 24),
        ("qp_num", -1),
        ("qkey", 1 << 32),
        ("qkey", -1),
    ],
)
def test_endpoint_info_rejects_out_of_range_fields(field, value):
    values = dict(gid=bytes(16), qp_num=1, qkey=1)
    values[field] = value
    with pytest.raises(ValueError):
        EndpointInfo(**values)


def test_tensor_addr_len_rejects_noncontiguous_array():
    tensor = np.zeros((4, 4), dtype=np.float32)[:, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        efa.tensor_addr_len(tensor)


def test_tensor_addr_len_numpy():
    arr = np.zeros(64, dtype=np.uint16)
    addr, n = efa.tensor_addr_len(arr)
    assert addr == arr.ctypes.data
    assert n == 128


def test_reg_tensor_rejects_cuda_tensor():
    tensor = SimpleNamespace(is_cuda=True)
    with pytest.raises(ValueError, match="cuda.register_tensor"):
        efa.reg_tensor(None, tensor, efa.AccessFlags.LOCAL_WRITE)


class _FakeMR:
    """Duck-typed MR for exercising the WR builders without hardware."""

    def __init__(self, addr=0x1000, length=1 << 20, lkey=0xAB):
        self.addr = addr
        self.length = length
        self.lkey = lkey


def test_write_wrs_chunks_and_signals_every_wr():
    mr = _FakeMR(length=250_000)
    wrs = efa.write_wrs(mr, None, 0x9000, 0xCD, chunk=100_000)
    assert len(wrs) == 3
    assert [w.sg_list[0].length for w in wrs] == [100_000, 100_000, 50_000]
    # chunks walk the local and remote ranges in lockstep
    assert [w.sg_list[0].addr for w in wrs] == [
        0x1000,
        0x1000 + 100_000,
        0x1000 + 200_000,
    ]
    assert [w.remote_addr for w in wrs] == [0x9000, 0x9000 + 100_000, 0x9000 + 200_000]
    # EFA rejects unsignaled send WRs, so every chunk must be SIGNALED
    assert all(w.send_flags & efa.SendFlags.SIGNALED for w in wrs)
    assert all(w.opcode == efa.WROpcode.RDMA_WRITE for w in wrs)
    assert all(w.rkey == 0xCD for w in wrs)
    assert [w.wr_id for w in wrs] == [0, 1, 2]


def test_read_wrs_respects_offset_and_length():
    mr = _FakeMR(length=1000)
    wrs = efa.read_wrs(mr, None, 0x9000, 1, offset=100, length=800, chunk=500)
    assert len(wrs) == 2
    assert wrs[0].sg_list[0].addr == mr.addr + 100
    assert wrs[0].opcode == efa.WROpcode.RDMA_READ
    assert sum(w.sg_list[0].length for w in wrs) == 800


def test_write_wrs_rejects_bad_ranges():
    mr = _FakeMR(length=1000)
    with pytest.raises(ValueError):
        efa.write_wrs(mr, None, 0, 0, offset=500, length=600)
    with pytest.raises(ValueError):
        efa.write_wrs(mr, None, 0, 0, chunk=0)


def test_peer_close_is_idempotent():
    closed = []
    ah = SimpleNamespace(close=lambda: closed.append(1))
    peer = efa.Peer(ah=ah, qp_num=5, qkey=6)
    with peer:
        pass
    peer.close()
    assert closed == [1]
    assert peer.ah is None
