"""Device enumeration, context, and query verbs."""

from __future__ import annotations

import ibverbs as ib
import pytest


def test_get_device_list_returns_devices():
    devs = ib.get_device_list()
    if not devs:
        pytest.skip("no RDMA devices present")
    assert all(isinstance(d, ib.Device) for d in devs)
    assert all(isinstance(d.name, str) and d.name for d in devs)
    assert all(isinstance(d.guid, int) for d in devs)


def test_open_and_query_device(ctx):
    da = ctx.query_device()
    assert da.max_qp > 0
    assert da.max_cqe > 0
    assert da.phys_port_cnt >= 1
    assert isinstance(da.fw_ver, str)


def test_query_port(ctx):
    pa = ctx.query_port(1)
    assert pa.gid_tbl_len > 0
    assert pa.state in {s.value for s in ib.PortState}
    assert pa.active_mtu in {m.value for m in ib.MTU}


def test_query_gid(ctx):
    gid = ctx.query_gid(1, 0)
    assert isinstance(gid, ib.Gid)
    assert len(gid.raw) == 16
    # subnet_prefix/interface_id are just views of the raw bytes.
    assert gid.subnet_prefix == int.from_bytes(gid.raw[:8], "big")
    assert gid.interface_id == int.from_bytes(gid.raw[8:], "big")


def test_num_comp_vectors(ctx):
    assert ctx.num_comp_vectors >= 1


def test_async_fd_nonblocking(ctx):
    import fcntl
    import os

    assert ctx.async_fd >= 0
    # Set non-blocking and confirm no event is pending right now.
    flags = fcntl.fcntl(ctx.async_fd, fcntl.F_GETFL)
    fcntl.fcntl(ctx.async_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    with pytest.raises(ib.VerbsError):
        ctx.get_async_event()


def test_open_missing_device_raises():
    with pytest.raises(ib.VerbsError):
        ib.Device("nonexistent_device_999", 0).open()


def test_context_is_context_manager(all_devices):
    with all_devices[0].open() as c:
        assert c.query_device().max_qp > 0


def test_context_rejects_close_with_open_child(ctx):
    pd = ctx.alloc_pd()
    with pytest.raises(ib.VerbsError) as exc_info:
        ctx.close()
    assert exc_info.value.errno != 0
    assert ctx.query_device().phys_port_cnt > 0
    pd.close()


def test_query_port_bad_port_raises(ctx):
    with pytest.raises(ib.VerbsError):
        ctx.query_port(99)
