"""Event-driven completions via a completion channel fd."""

from __future__ import annotations

import select

import ibverbs as ib
import pytest
from _rc import FULL_ACCESS, HostBuffer

pytestmark = pytest.mark.integration


def test_completion_channel_delivers_event(ctx, dev_name, first_active):
    from conftest import find_roce_gid

    _, port = first_active
    gid_index, gid = find_roce_gid(ctx, dev_name, port)
    pd = ctx.alloc_pd()
    pa = ctx.query_port(port)

    # Two endpoints, but the sender's CQ is bound to a completion channel.
    channel = ctx.create_comp_channel()
    scq = ctx.create_cq(16, channel=channel)
    rcq = ctx.create_cq(16)
    send_qp = pd.create_qp(
        ib.QPInitAttr(send_cq=scq, recv_cq=scq, qp_type=ib.QPType.RC)
    )
    recv_qp = pd.create_qp(
        ib.QPInitAttr(send_cq=rcq, recv_cq=rcq, qp_type=ib.QPType.RC)
    )

    info_s = ib.local_qp_info(send_qp, pa, gid, port=port)
    info_r = ib.local_qp_info(recv_qp, pa, gid, port=port)
    ib.connect_rc(send_qp, info_r, port=port, sgid_index=gid_index, access=FULL_ACCESS)
    ib.connect_rc(recv_qp, info_s, port=port, sgid_index=gid_index, access=FULL_ACCESS)

    src = HostBuffer(pd, 128)
    dst = HostBuffer(pd, 128)
    src.set_bytes(b"event-driven-completion")

    scq.req_notify()
    send_qp.post_send(
        ib.SendWR(
            wr_id=99,
            sg_list=[src.sge(64)],
            opcode=ib.WROpcode.RDMA_WRITE,
            send_flags=ib.SendFlags.SIGNALED,
            remote_addr=dst.addr,
            rkey=dst.rkey,
        )
    )

    # Block on the channel fd until the NIC signals a completion.
    r, _, _ = select.select([channel.fd], [], [], 10)
    assert r, "no completion event delivered on channel fd"

    event_cq = channel.get_cq_event()
    assert event_cq is scq
    event_cq.ack_events(1)
    event_cq.req_notify()

    wcs = scq.poll(4)
    assert wcs and wcs[0].status == ib.WCStatus.SUCCESS
    assert wcs[0].wr_id == 99
    assert dst.get_bytes(64) == src.get_bytes(64)

    src.close()
    dst.close()
    send_qp.close()
    recv_qp.close()
    scq.close()
    rcq.close()
    channel.close()
    pd.close()
