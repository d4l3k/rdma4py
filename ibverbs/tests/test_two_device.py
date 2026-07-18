"""RC over the real wire between two distinct NICs.

If the two NICs are not on a fabric that routes between them, the RDMA op will
report ``RETRY_EXC_ERR``; that reflects site fabric topology rather than a
binding bug, so the test skips in that case.
"""

from __future__ import annotations

import time

import ibverbs as ib
import pytest
from _rc import Endpoint, HostBuffer

pytestmark = pytest.mark.integration


def _poll_until(cq, timeout=10.0):
    """Poll ``cq`` for one completion up to ``timeout`` seconds, else None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        wcs = cq.poll(1)
        if wcs:
            return wcs[0]
    return None


def _distinct_active_ports(active_port_list):
    seen, out = set(), []
    for name, port in active_port_list:
        if name not in seen:
            seen.add(name)
            out.append((name, port))
    return out


def test_rdma_write_across_two_nics(active_port_list):
    from conftest import find_roce_gid

    ports = _distinct_active_ports(active_port_list)
    if len(ports) < 2:
        pytest.skip("need two distinct ACTIVE NICs")
    (name_a, port_a), (name_b, port_b) = ports[0], ports[1]

    ctx_a = None
    ctx_b = None
    pd_a = None
    pd_b = None
    ep_a = None
    ep_b = None
    src = None
    dst = None
    for dev in ib.get_device_list():
        if dev.name == name_a:
            ctx_a = dev.open()
        elif dev.name == name_b:
            ctx_b = dev.open()

    try:
        pd_a = ctx_a.alloc_pd()
        pd_b = ctx_b.alloc_pd()
        gi_a, gid_a = find_roce_gid(ctx_a, name_a, port_a)
        gi_b, gid_b = find_roce_gid(ctx_b, name_b, port_b)
        pa_a = ctx_a.query_port(port_a)
        pa_b = ctx_b.query_port(port_b)

        ep_a = Endpoint(ctx_a, pd_a, port_a)
        ep_b = Endpoint(ctx_b, pd_b, port_b)
        info_a = ep_a.info(pa_a, gid_a)
        info_b = ep_b.info(pa_b, gid_b)
        ep_a.connect(info_b, gi_a)
        ep_b.connect(info_a, gi_b)

        src = HostBuffer(pd_a, 512)
        dst = HostBuffer(pd_b, 512)
        payload = bytes((i ^ 0x5A) & 0xFF for i in range(256))
        src.set_bytes(payload)

        ep_a.qp.post_send(
            ib.SendWR(
                wr_id=1,
                sg_list=[src.sge(len(payload))],
                opcode=ib.WROpcode.RDMA_WRITE,
                send_flags=ib.SendFlags.SIGNALED,
                remote_addr=dst.addr,
                rkey=dst.rkey,
            )
        )
        wc = _poll_until(ep_a.cq, timeout=10.0)

        if wc is None or wc.status == ib.WCStatus.RETRY_EXC_ERR:
            pytest.skip(
                f"{name_a} and {name_b} are not on a mutually routable " "RoCE fabric"
            )
        assert wc.status == ib.WCStatus.SUCCESS, wc
        assert dst.get_bytes(len(payload)) == payload

    finally:
        if src is not None:
            src.close()
        if dst is not None:
            dst.close()
        if ep_a is not None:
            ep_a.close()
        if ep_b is not None:
            ep_b.close()
        if pd_a is not None:
            pd_a.close()
        if pd_b is not None:
            pd_b.close()
        if ctx_a is not None:
            ctx_a.close()
        if ctx_b is not None:
            ctx_b.close()
