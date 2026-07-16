"""Single-NIC RC loopback: exercise every send opcode against real hardware."""

from __future__ import annotations

import pytest

import ibverbs as ib
from _rc import HostBuffer, make_connected_pair

pytestmark = pytest.mark.integration


@pytest.fixture()
def pair(ctx, dev_name, first_active):
    from conftest import find_roce_gid

    _, port = first_active
    gid_index, gid = find_roce_gid(ctx, dev_name, port)
    pd = ctx.alloc_pd()
    a, b = make_connected_pair(ctx, pd, dev_name, port, gid_index, gid)
    yield a, b, pd
    a.close()
    b.close()
    pd.close()


def test_send_recv(pair):
    a, b, pd = pair
    src = HostBuffer(pd, 256, fill=0)
    dst = HostBuffer(pd, 256, fill=0)
    payload = b"hello rdma send/recv" + bytes(range(32))
    src.set_bytes(payload)

    b.qp.post_recv(ib.RecvWR(wr_id=1, sg_list=[dst.sge(len(payload))]))
    a.qp.post_send(ib.SendWR(wr_id=2, sg_list=[src.sge(len(payload))],
                             opcode=ib.WROpcode.SEND,
                             send_flags=ib.SendFlags.SIGNALED))

    send_wc = a.poll_one()
    recv_wc = b.poll_one()
    assert send_wc.status == ib.WCStatus.SUCCESS, send_wc
    assert recv_wc.status == ib.WCStatus.SUCCESS, recv_wc
    assert recv_wc.byte_len == len(payload)
    assert dst.get_bytes(len(payload)) == payload

    src.close()
    dst.close()


def test_rdma_write(pair):
    a, b, pd = pair
    src = HostBuffer(pd, 512, fill=0)
    remote = HostBuffer(pd, 512, fill=0)
    payload = bytes(range(200))
    src.set_bytes(payload)

    a.qp.post_send(ib.SendWR(
        wr_id=10, sg_list=[src.sge(len(payload))], opcode=ib.WROpcode.RDMA_WRITE,
        send_flags=ib.SendFlags.SIGNALED, remote_addr=remote.addr,
        rkey=remote.rkey))
    wc = a.poll_one()
    assert wc.status == ib.WCStatus.SUCCESS, wc
    assert remote.get_bytes(len(payload)) == payload

    src.close()
    remote.close()


def test_rdma_write_with_imm(pair):
    a, b, pd = pair
    src = HostBuffer(pd, 128, fill=0)
    remote = HostBuffer(pd, 128, fill=0)
    payload = b"immediate-data-write" + bytes(range(16))
    src.set_bytes(payload)

    # The receiver must have a recv posted to consume the immediate.
    b.qp.post_recv(ib.RecvWR(wr_id=20, sg_list=[remote.sge(len(payload))]))
    a.qp.post_send(ib.SendWR(
        wr_id=21, sg_list=[src.sge(len(payload))],
        opcode=ib.WROpcode.RDMA_WRITE_WITH_IMM,
        send_flags=ib.SendFlags.SIGNALED, remote_addr=remote.addr,
        rkey=remote.rkey, imm_data=0xDEADBEEF))

    send_wc = a.poll_one()
    recv_wc = b.poll_one()
    assert send_wc.status == ib.WCStatus.SUCCESS, send_wc
    assert recv_wc.status == ib.WCStatus.SUCCESS, recv_wc
    assert recv_wc.imm_data == 0xDEADBEEF
    assert recv_wc.wc_flags & ib.WCFlags.WITH_IMM
    assert remote.get_bytes(len(payload)) == payload

    src.close()
    remote.close()


def test_rdma_read(pair):
    a, b, pd = pair
    local = HostBuffer(pd, 512, fill=0)
    remote = HostBuffer(pd, 512, fill=0)
    payload = bytes((i * 7) & 0xFF for i in range(300))
    remote.set_bytes(payload)

    a.qp.post_send(ib.SendWR(
        wr_id=30, sg_list=[local.sge(len(payload))], opcode=ib.WROpcode.RDMA_READ,
        send_flags=ib.SendFlags.SIGNALED, remote_addr=remote.addr,
        rkey=remote.rkey))
    wc = a.poll_one()
    assert wc.status == ib.WCStatus.SUCCESS, wc
    assert local.get_bytes(len(payload)) == payload

    local.close()
    remote.close()


def test_atomic_fetch_add(pair):
    a, b, pd = pair
    result = HostBuffer(pd, 64, fill=0)   # 8-byte fetched value lands here
    target = HostBuffer(pd, 64, fill=0)
    target.write_u64(100)

    a.qp.post_send(ib.SendWR(
        wr_id=40, sg_list=[result.sge(8)], opcode=ib.WROpcode.ATOMIC_FETCH_AND_ADD,
        send_flags=ib.SendFlags.SIGNALED, remote_addr=target.addr,
        rkey=target.rkey, compare_add=25))
    wc = a.poll_one()
    assert wc.status == ib.WCStatus.SUCCESS, wc
    assert result.read_u64() == 100          # original value returned
    assert target.read_u64() == 125          # 100 + 25 stored remotely

    result.close()
    target.close()


def test_atomic_compare_swap(pair):
    a, b, pd = pair
    result = HostBuffer(pd, 64, fill=0)
    target = HostBuffer(pd, 64, fill=0)
    target.write_u64(0x1122334455667788)

    a.qp.post_send(ib.SendWR(
        wr_id=50, sg_list=[result.sge(8)], opcode=ib.WROpcode.ATOMIC_CMP_AND_SWP,
        send_flags=ib.SendFlags.SIGNALED, remote_addr=target.addr,
        rkey=target.rkey, compare_add=0x1122334455667788, swap=0xAABBCCDD))
    wc = a.poll_one()
    assert wc.status == ib.WCStatus.SUCCESS, wc
    assert result.read_u64() == 0x1122334455667788   # original returned
    assert target.read_u64() == 0xAABBCCDD            # swapped

    result.close()
    target.close()


def test_unsignaled_send_then_signaled(pair):
    """Unsignaled WRs produce no completion; a later signaled one does."""
    a, b, pd = pair
    src = HostBuffer(pd, 64, fill=0)
    dst0 = HostBuffer(pd, 64, fill=0)
    dst1 = HostBuffer(pd, 64, fill=0)
    src.set_bytes(b"unsignaled-then-signaled!")

    a.qp.post_send(ib.SendWR(wr_id=60, sg_list=[src.sge(32)],
                             opcode=ib.WROpcode.RDMA_WRITE, send_flags=0,
                             remote_addr=dst0.addr, rkey=dst0.rkey))
    a.qp.post_send(ib.SendWR(wr_id=61, sg_list=[src.sge(32)],
                             opcode=ib.WROpcode.RDMA_WRITE,
                             send_flags=ib.SendFlags.SIGNALED,
                             remote_addr=dst1.addr, rkey=dst1.rkey))
    wc = a.poll_one()
    assert wc.wr_id == 61
    assert wc.status == ib.WCStatus.SUCCESS
    assert dst0.get_bytes(32) == src.get_bytes(32)
    assert dst1.get_bytes(32) == src.get_bytes(32)

    src.close()
    dst0.close()
    dst1.close()
