"""Cross-NIC SRD transfers between two EFA devices on the same host."""

from __future__ import annotations

import efa
import numpy as np
import pytest
from _srd import Endpoint, FULL_ACCESS, HostBuffer

pytestmark = pytest.mark.integration


@pytest.fixture()
def two_devices(efa_devices):
    if len(efa_devices) < 2:
        pytest.skip("needs at least two EFA devices")
    ctx_a = efa_devices[0].open()
    ctx_b = efa_devices[1].open()
    pd_a = ctx_a.alloc_pd()
    pd_b = ctx_b.alloc_pd()
    yield ctx_a, pd_a, ctx_b, pd_b
    pd_a.close()
    pd_b.close()
    ctx_a.close()
    ctx_b.close()


def test_cross_nic_send_recv(two_devices):
    ctx_a, pd_a, ctx_b, pd_b = two_devices
    a = Endpoint(ctx_a, pd_a)
    b = Endpoint(ctx_b, pd_b)
    # Exchange endpoint info exactly as two processes would: over the wire
    # format. The AH must be created on the sender's PD.
    peer = efa.EndpointInfo.from_bytes(b.info().to_bytes()).peer(pd_a)

    src = HostBuffer(pd_a, 4096)
    dst = HostBuffer(pd_b, 4096)
    payload = bytes(i % 253 for i in range(4096))
    src.set_bytes(payload)

    b.qp.post_recv(efa.RecvWR(wr_id=1, sg_list=[dst.sge()]))
    a.qp.post_send(
        efa.SendWR(
            wr_id=2,
            sg_list=[src.sge()],
            opcode=efa.WROpcode.SEND,
            send_flags=efa.SendFlags.SIGNALED,
            dest=peer,
        )
    )
    a.poll_one().raise_for_status()
    rwc = b.poll_one()
    rwc.raise_for_status()
    assert rwc.byte_len == 4096
    assert dst.get_bytes() == payload

    src.close()
    dst.close()
    peer.close()
    a.close()
    b.close()


def test_cross_nic_rdma_write_read(two_devices, efa_caps):
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.RDMA_WRITE | efa.EfaDeviceCaps.RDMA_READ)
    ctx_a, pd_a, ctx_b, pd_b = two_devices
    a = Endpoint(ctx_a, pd_a)
    b = Endpoint(ctx_b, pd_b)
    peer = efa.EndpointInfo.from_bytes(b.info().to_bytes()).peer(pd_a)
    # EFA accepts one-sided operations only from a peer for which the
    # responder has an AH, even though no reverse traffic is posted here.
    reverse_peer = efa.EndpointInfo.from_bytes(a.info().to_bytes()).peer(pd_b)

    nbytes = 1 << 20
    local = np.frombuffer(
        bytearray(np.random.default_rng(7).bytes(nbytes)), dtype=np.uint8
    )
    remote = np.zeros(nbytes, dtype=np.uint8)
    lmr = efa.reg_tensor(pd_a, local, FULL_ACCESS)
    rmr = efa.reg_tensor(pd_b, remote, FULL_ACCESS)

    # Write local (dev A) -> remote (dev B)
    a.qp.post_send(efa.write_wrs(lmr, peer, rmr.addr, rmr.rkey))
    a.poll_one().raise_for_status()
    assert np.array_equal(local, remote)

    # Scramble local, then RDMA-read it back from B
    local[:] = 0
    a.qp.post_send(efa.read_wrs(lmr, peer, rmr.addr, rmr.rkey))
    a.poll_one().raise_for_status()
    assert np.array_equal(local, remote)

    lmr.close()
    rmr.close()
    peer.close()
    reverse_peer.close()
    a.close()
    b.close()


def test_all_devices_are_addressable(efa_devices):
    """Every EFA device reports an ACTIVE port and a non-zero GID."""
    for dev in efa_devices:
        with dev.open() as ctx:
            assert ctx.query_port().state == efa.PortState.ACTIVE
            assert ctx.query_gid().raw != b"\x00" * 16
