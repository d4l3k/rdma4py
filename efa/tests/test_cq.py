"""Completion queues: classic, extended, and completion channels."""

from __future__ import annotations

import select

import efa
import pytest
from _srd import Endpoint, HostBuffer, make_pair

pytestmark = pytest.mark.integration


def test_create_cq(ctx):
    cq = ctx.create_cq(64)
    try:
        assert cq.cqe >= 64
        assert cq.poll(4) == []
        try:
            attr = cq.query_efa()
        except RuntimeError as exc:
            pytest.skip(str(exc))
        assert attr.buffer_addr != 0
        assert attr.entry_size > 0
        assert attr.num_entries >= 64
        assert "CQAttr(" in repr(attr)
    finally:
        cq.close()
    with pytest.raises(efa.EfaError):
        cq.poll(1)


def test_create_cq_rejects_bad_args(ctx):
    with pytest.raises(ValueError):
        ctx.create_cq(0)
    with pytest.raises(ValueError):
        ctx.create_cq(64, comp_vector=10**9)


def test_create_cq_ex(ctx):
    cq = ctx.create_cq_ex(64)
    assert cq.cqe >= 64
    assert cq.poll(4) == []
    assert cq.sgid_enabled is False
    assert cq.unsolicited_enabled is False
    assert cq.query_efa().num_entries >= 64
    cq.close()


def test_create_cq_ex_with_sgid(ctx, efa_caps):
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.CQ_WITH_SGID)
    cq = ctx.create_cq_ex(64, sgid=True)
    assert cq.sgid_enabled is True
    cq.close()


def test_create_cq_ex_with_unsolicited(ctx, efa_caps):
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.UNSOLICITED_WRITE_RECV)
    cq = ctx.create_cq_ex(64, unsolicited=True)
    assert cq.unsolicited_enabled is True
    cq.close()


def test_cqex_reports_completions(ctx, pd):
    """The extended CQ must report the same completions as the classic one."""
    a = Endpoint(ctx, pd, cq_ex=True)
    b = Endpoint(ctx, pd, cq_ex=True)
    peer = a.peer_to(b)
    src = HostBuffer(pd, 1024, fill=7)
    dst = HostBuffer(pd, 1024)

    b.qp.post_recv(efa.RecvWR(wr_id=11, sg_list=[dst.sge()]))
    a.qp.post_send(
        efa.SendWR(
            wr_id=22,
            sg_list=[src.sge()],
            opcode=efa.WROpcode.SEND,
            send_flags=efa.SendFlags.SIGNALED,
            dest=peer,
        )
    )
    swc = a.poll_one()
    rwc = b.poll_one()
    assert swc.wr_id == 22
    assert swc.status == efa.WCStatus.SUCCESS
    assert swc.opcode == efa.WCOpcode.SEND
    assert rwc.wr_id == 11
    assert rwc.status == efa.WCStatus.SUCCESS
    assert rwc.opcode == efa.WCOpcode.RECV
    assert rwc.byte_len == 1024
    assert rwc.src_qp == a.qp.qp_num
    assert dst.get_bytes() == src.get_bytes()

    src.close()
    dst.close()
    peer.close()
    a.close()
    b.close()


def test_comp_channel_notification(ctx, pd):
    """req_notify + channel fd wakeup + get_cq_event + ack_events."""
    chan = ctx.create_comp_channel()
    assert chan.fd >= 0
    cq = ctx.create_cq(64, channel=chan)
    ep_cq_qp = pd.create_qp(efa.QPInitAttr(send_cq=cq, recv_cq=cq)).prepare(qkey=0x1234)
    peer = efa.local_endpoint_info(ep_cq_qp, qkey=0x1234).peer(pd)
    src = HostBuffer(pd, 64, fill=3)
    dst = HostBuffer(pd, 64)

    cq.req_notify()
    ep_cq_qp.post_recv(efa.RecvWR(wr_id=1, sg_list=[dst.sge()]))
    ep_cq_qp.post_send(
        efa.SendWR(
            wr_id=2,
            sg_list=[src.sge()],
            opcode=efa.WROpcode.SEND,
            send_flags=efa.SendFlags.SIGNALED,
            dest=peer,
        )
    )

    r, _, _ = select.select([chan.fd], [], [], 10.0)
    assert r, "no completion event within 10s"
    got = chan.get_cq_event()
    assert got is cq
    wcs = []
    for _ in range(2000000):
        wcs += cq.poll(4)
        if len(wcs) >= 2:
            break
    assert {w.wr_id for w in wcs} == {1, 2}
    cq.ack_events(1)

    src.close()
    dst.close()
    peer.close()
    ep_cq_qp.close()
    cq.close()
    chan.close()


def test_ack_more_events_than_received_raises(ctx):
    cq = ctx.create_cq(16)
    with pytest.raises(ValueError):
        cq.ack_events(1)
    cq.close()


def test_poll_batching(ctx, pd):
    """Post several sends and reap them in one poll call."""
    a, b, peer, _ = make_pair(ctx, pd)
    src = HostBuffer(pd, 256, fill=9)
    dst = HostBuffer(pd, 4096)

    n = 8
    for i in range(n):
        b.qp.post_recv(
            efa.RecvWR(wr_id=100 + i, sg_list=[dst.sge(256, offset=256 * i)])
        )
    a.qp.post_send(
        [
            efa.SendWR(
                wr_id=i,
                sg_list=[src.sge()],
                opcode=efa.WROpcode.SEND,
                send_flags=efa.SendFlags.SIGNALED,
                dest=peer,
            )
            for i in range(n)
        ]
    )
    swcs = a.poll_n(n)
    rwcs = b.poll_n(n)
    assert sorted(w.wr_id for w in swcs) == list(range(n))
    assert sorted(w.wr_id for w in rwcs) == [100 + i for i in range(n)]
    for w in swcs + rwcs:
        w.raise_for_status()

    src.close()
    dst.close()
    peer.close()
    a.close()
    b.close()
