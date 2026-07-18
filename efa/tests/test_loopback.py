"""Single-device SRD loopback: exercise every send opcode against real hardware."""

from __future__ import annotations

import efa
import pytest
from _srd import Endpoint, HostBuffer, make_pair

pytestmark = pytest.mark.integration


@pytest.fixture()
def pair(ctx, pd):
    a, b, ab, ba = make_pair(ctx, pd)
    yield a, b, ab, ba, pd
    ab.close()
    ba.close()
    a.close()
    b.close()


def test_send_recv(pair):
    a, b, peer, _, pd = pair
    src = HostBuffer(pd, 256)
    dst = HostBuffer(pd, 256)
    payload = b"hello efa srd send/recv" + bytes(range(32))
    src.set_bytes(payload)

    b.qp.post_recv(efa.RecvWR(wr_id=1, sg_list=[dst.sge()]))
    a.qp.post_send(
        efa.SendWR(
            wr_id=2,
            sg_list=[src.sge(len(payload))],
            opcode=efa.WROpcode.SEND,
            send_flags=efa.SendFlags.SIGNALED,
            dest=peer,
        )
    )

    send_wc = a.poll_one()
    recv_wc = b.poll_one()
    assert send_wc.status == efa.WCStatus.SUCCESS, send_wc
    assert recv_wc.status == efa.WCStatus.SUCCESS, recv_wc
    assert recv_wc.byte_len == len(payload)
    assert recv_wc.src_qp == a.qp.qp_num
    assert dst.get_bytes(len(payload)) == payload

    src.close()
    dst.close()


def test_send_with_imm(pair):
    a, b, peer, _, pd = pair
    src = HostBuffer(pd, 64, fill=5)
    dst = HostBuffer(pd, 64)

    b.qp.post_recv(efa.RecvWR(wr_id=3, sg_list=[dst.sge()]))
    a.qp.post_send(
        efa.SendWR(
            wr_id=4,
            sg_list=[src.sge()],
            opcode=efa.WROpcode.SEND_WITH_IMM,
            send_flags=efa.SendFlags.SIGNALED,
            imm_data=0x12345678,
            dest=peer,
        )
    )
    a.poll_one().raise_for_status()
    rwc = b.poll_one()
    rwc.raise_for_status()
    assert rwc.imm_data == 0x12345678
    assert rwc.wc_flags & efa.WCFlags.WITH_IMM
    assert dst.get_bytes() == src.get_bytes()

    src.close()
    dst.close()


def test_rdma_write(pair, efa_caps):
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.RDMA_WRITE)
    a, b, peer, _, pd = pair
    src = HostBuffer(pd, 512)
    remote = HostBuffer(pd, 512)
    payload = bytes(range(200))
    src.set_bytes(payload)

    a.qp.post_send(
        efa.SendWR(
            wr_id=10,
            sg_list=[src.sge(len(payload))],
            opcode=efa.WROpcode.RDMA_WRITE,
            send_flags=efa.SendFlags.SIGNALED,
            remote_addr=remote.addr,
            rkey=remote.rkey,
            dest=peer,
        )
    )
    wc = a.poll_one()
    assert wc.status == efa.WCStatus.SUCCESS, wc
    assert remote.get_bytes(len(payload)) == payload

    src.close()
    remote.close()


def test_rdma_write_with_imm(pair, efa_caps):
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.RDMA_WRITE)
    a, b, peer, _, pd = pair
    src = HostBuffer(pd, 128)
    remote = HostBuffer(pd, 128)
    payload = b"immediate-data-write" + bytes(range(16))
    src.set_bytes(payload)

    # The receiver must have a recv posted to consume the immediate.
    b.qp.post_recv(efa.RecvWR(wr_id=20, sg_list=[remote.sge()]))
    a.qp.post_send(
        efa.SendWR(
            wr_id=21,
            sg_list=[src.sge(len(payload))],
            opcode=efa.WROpcode.RDMA_WRITE_WITH_IMM,
            send_flags=efa.SendFlags.SIGNALED,
            remote_addr=remote.addr,
            rkey=remote.rkey,
            imm_data=0xDEADBEEF,
            dest=peer,
        )
    )

    send_wc = a.poll_one()
    recv_wc = b.poll_one()
    assert send_wc.status == efa.WCStatus.SUCCESS, send_wc
    assert recv_wc.status == efa.WCStatus.SUCCESS, recv_wc
    assert recv_wc.imm_data == 0xDEADBEEF
    assert recv_wc.wc_flags & efa.WCFlags.WITH_IMM
    assert recv_wc.opcode == efa.WCOpcode.RECV_RDMA_WITH_IMM
    assert remote.get_bytes(len(payload)) == payload

    src.close()
    remote.close()


def test_unsolicited_completion_metadata(ctx, pd, efa_caps):
    from conftest import require_caps

    require_caps(
        efa_caps,
        efa.EfaDeviceCaps.RDMA_WRITE | efa.EfaDeviceCaps.UNSOLICITED_WRITE_RECV,
    )
    # Both peers must negotiate this QP feature. Mixing enabled and disabled
    # QPs completes with EFA's REMOTE_ERROR_FEATURE_MISMATCH status.
    a = Endpoint(ctx, pd, unsolicited=True)
    b = Endpoint(ctx, pd, unsolicited=True)
    peer = a.peer_to(b)
    reverse_peer = b.peer_to(a)
    src = HostBuffer(pd, 128, fill=0x5A)
    remote = HostBuffer(pd, 128)
    try:
        a.qp.post_send(
            efa.SendWR(
                wr_id=22,
                sg_list=[src.sge()],
                opcode=efa.WROpcode.RDMA_WRITE_WITH_IMM,
                send_flags=efa.SendFlags.SIGNALED,
                remote_addr=remote.addr,
                rkey=remote.rkey,
                imm_data=0x10203040,
                dest=peer,
            )
        )

        a.poll_one().raise_for_status()
        rwc = b.poll_one()
        rwc.raise_for_status()
        assert rwc.unsolicited is True
        assert rwc.imm_data == 0x10203040
        assert remote.get_bytes() == src.get_bytes()
    finally:
        a.close()
        b.close()
        src.close()
        remote.close()
        peer.close()
        reverse_peer.close()


def test_rdma_read(pair, efa_caps):
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.RDMA_READ)
    a, b, peer, _, pd = pair
    local = HostBuffer(pd, 512)
    remote = HostBuffer(pd, 512)
    payload = bytes((i * 7) & 0xFF for i in range(300))
    remote.set_bytes(payload)

    a.qp.post_send(
        efa.SendWR(
            wr_id=30,
            sg_list=[local.sge(len(payload))],
            opcode=efa.WROpcode.RDMA_READ,
            send_flags=efa.SendFlags.SIGNALED,
            remote_addr=remote.addr,
            rkey=remote.rkey,
            dest=peer,
        )
    )
    wc = a.poll_one()
    assert wc.status == efa.WCStatus.SUCCESS, wc
    assert local.get_bytes(len(payload)) == payload

    local.close()
    remote.close()


def test_inline_send(ctx, pd):
    caps = ctx.query_efa_device()
    if caps.inline_buf_size <= 0:
        pytest.skip("device does not support inline data")
    a = Endpoint(ctx, pd, max_inline_data=caps.inline_buf_size)
    b = Endpoint(ctx, pd)
    peer = a.peer_to(b)
    src = HostBuffer(pd, 64, fill=0xAB)
    dst = HostBuffer(pd, 64)

    n = caps.inline_buf_size
    b.qp.post_recv(efa.RecvWR(wr_id=40, sg_list=[dst.sge()]))
    a.qp.post_send(
        efa.SendWR(
            wr_id=41,
            sg_list=[src.sge(n)],
            opcode=efa.WROpcode.SEND,
            send_flags=efa.SendFlags.SIGNALED | efa.SendFlags.INLINE,
            dest=peer,
        )
    )
    a.poll_one().raise_for_status()
    rwc = b.poll_one()
    rwc.raise_for_status()
    assert rwc.byte_len == n
    assert dst.get_bytes(n) == src.get_bytes(n)

    src.close()
    dst.close()
    peer.close()
    a.close()
    b.close()


def test_scatter_gather_send(ctx, pd):
    """A 2-SGE send lands contiguously in the receive buffer."""
    dev = ctx.query_efa_device()
    if dev.max_sq_sge < 2 or dev.max_rq_sge < 2:
        pytest.skip("device supports fewer than 2 SGEs")
    a = Endpoint(ctx, pd, max_send_sge=2, max_recv_sge=2)
    b = Endpoint(ctx, pd, max_send_sge=2, max_recv_sge=2)
    peer = a.peer_to(b)
    src = HostBuffer(pd, 512)
    dst = HostBuffer(pd, 512)
    src.set_bytes(bytes(range(200)), offset=0)
    src.set_bytes(bytes(reversed(range(200))), offset=256)

    b.qp.post_recv(efa.RecvWR(wr_id=50, sg_list=[dst.sge()]))
    a.qp.post_send(
        efa.SendWR(
            wr_id=51,
            sg_list=[src.sge(200, offset=0), src.sge(200, offset=256)],
            opcode=efa.WROpcode.SEND,
            send_flags=efa.SendFlags.SIGNALED,
            dest=peer,
        )
    )
    a.poll_one().raise_for_status()
    rwc = b.poll_one()
    rwc.raise_for_status()
    assert rwc.byte_len == 400
    assert dst.get_bytes(200) == bytes(range(200))
    assert dst.get_bytes(200, offset=200) == bytes(reversed(range(200)))

    src.close()
    dst.close()
    peer.close()
    a.close()
    b.close()


def test_oversized_send_fails_with_completion(pair, ctx):
    """A SEND larger than max_msg_sz yields an error completion, not silence."""
    a, b, peer, _, pd = pair
    max_msg = ctx.query_port().max_msg_sz
    nbytes = max_msg + 4096
    src = HostBuffer(pd, nbytes)
    dst = HostBuffer(pd, nbytes)

    b.qp.post_recv(efa.RecvWR(wr_id=60, sg_list=[dst.sge()]))
    a.qp.post_send(
        efa.SendWR(
            wr_id=61,
            sg_list=[src.sge()],
            opcode=efa.WROpcode.SEND,
            send_flags=efa.SendFlags.SIGNALED,
            dest=peer,
        )
    )
    wc = a.poll_one()
    assert wc.status != efa.WCStatus.SUCCESS
    with pytest.raises(efa.CompletionError):
        wc.raise_for_status()

    # The failed SEND did not consume the posted receive. Destroying the
    # receiver QP cancels that WQE before its MR is deregistered.
    b.qp.close()
    src.close()
    dst.close()


def test_unsignaled_send_is_rejected(pair):
    """EFA requires SIGNALED on every send WR; the post must fail cleanly."""
    a, b, peer, _, pd = pair
    src = HostBuffer(pd, 64)
    with pytest.raises(efa.EfaError):
        a.qp.post_send(
            efa.SendWR(
                wr_id=70,
                sg_list=[src.sge()],
                opcode=efa.WROpcode.SEND,
                send_flags=0,
                dest=peer,
            )
        )
    src.close()


def test_chunked_write_helper(pair, efa_caps):
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.RDMA_WRITE)
    a, b, peer, _, pd = pair
    nbytes = 200_000
    src = HostBuffer(pd, nbytes)
    dst = HostBuffer(pd, nbytes)
    src.set_bytes(bytes(i % 251 for i in range(nbytes)))

    wrs = efa.write_wrs(src.mr, peer, dst.addr, dst.rkey, chunk=64 * 1024)
    assert len(wrs) == 4
    a.qp.post_send(wrs)
    for wc in a.poll_n(len(wrs)):
        wc.raise_for_status()
    assert dst.get_bytes() == src.get_bytes()

    src.close()
    dst.close()


def test_chunked_read_helper(pair, efa_caps):
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.RDMA_READ)
    a, b, peer, _, pd = pair
    nbytes = 150_000
    local = HostBuffer(pd, nbytes)
    remote = HostBuffer(pd, nbytes)
    remote.set_bytes(bytes((i * 3) % 256 for i in range(nbytes)))

    wrs = efa.read_wrs(local.mr, peer, remote.addr, remote.rkey, chunk=100_000)
    assert len(wrs) == 2
    a.qp.post_send(wrs)
    for wc in a.poll_n(len(wrs)):
        wc.raise_for_status()
    assert local.get_bytes() == remote.get_bytes()

    local.close()
    remote.close()


def test_sgid_reported_for_unknown_sender(ctx, pd, efa_caps):
    """With sgid=True and no local AH for the sender, WC.sgid is populated."""
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.CQ_WITH_SGID)
    a = Endpoint(ctx, pd)  # sender
    b = Endpoint(ctx, pd, sgid=True)  # receiver with sgid-enabled CQ
    peer = a.peer_to(b)
    src = HostBuffer(pd, 64, fill=1)
    dst = HostBuffer(pd, 64)

    b.qp.post_recv(efa.RecvWR(wr_id=80, sg_list=[dst.sge()]))
    a.qp.post_send(
        efa.SendWR(
            wr_id=81,
            sg_list=[src.sge()],
            opcode=efa.WROpcode.SEND,
            send_flags=efa.SendFlags.SIGNALED,
            dest=peer,
        )
    )
    a.poll_one().raise_for_status()
    rwc = b.poll_one()
    rwc.raise_for_status()
    # The sender's GID is only reported when it cannot be resolved to a local
    # AH; loopback traffic may resolve, so accept either form but require
    # a valid 16-byte GID when present.
    if rwc.sgid is not None:
        assert len(rwc.sgid) == 16
        assert rwc.sgid == ctx.query_gid().raw

    src.close()
    dst.close()
    peer.close()
    a.close()
    b.close()


def test_ud_qp_send_recv(ctx, pd):
    """UD QPs (non-SRD) also work through the same path."""
    a = Endpoint(ctx, pd, qkey=0x777, qp_type=efa.QPType.UD)
    b = Endpoint(ctx, pd, qkey=0x777, qp_type=efa.QPType.UD)
    peer = a.peer_to(b)
    src = HostBuffer(pd, 256, fill=0x42)
    # UD prepends a 40-byte GRH to received payloads.
    dst = HostBuffer(pd, 256 + 40)

    b.qp.post_recv(efa.RecvWR(wr_id=90, sg_list=[dst.sge()]))
    a.qp.post_send(
        efa.SendWR(
            wr_id=91,
            sg_list=[src.sge()],
            opcode=efa.WROpcode.SEND,
            send_flags=efa.SendFlags.SIGNALED,
            dest=peer,
        )
    )
    a.poll_one().raise_for_status()
    rwc = b.poll_one()
    rwc.raise_for_status()
    assert rwc.byte_len == 256 + 40
    # EFA includes the 40-byte GRH in UD receive buffers but does not set
    # IBV_WC_GRH on every provider/kernel combination.
    assert dst.get_bytes(256, offset=40) == src.get_bytes()

    src.close()
    dst.close()
    peer.close()
    a.close()
    b.close()
