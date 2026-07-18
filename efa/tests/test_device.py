"""Device discovery and attribute queries against real EFA hardware."""

from __future__ import annotations

import efa
import pytest

pytestmark = pytest.mark.integration


def test_get_device_list_contains_efa(efa_devices):
    all_devs = efa.get_device_list()
    assert {d.name for d in efa_devices} <= {d.name for d in all_devs}
    dev = efa_devices[0]
    assert dev.name
    assert repr(dev).startswith("Device(")


def test_open_and_is_efa(efa_devices):
    with efa_devices[0].open() as ctx:
        assert ctx.name == efa_devices[0].name
        assert ctx.is_efa() is True
        assert ctx.num_comp_vectors > 0
        assert ctx.async_fd >= 0


def test_query_device(ctx):
    attr = ctx.query_device()
    assert attr.vendor_id == 0x1D0F  # Amazon
    assert attr.max_qp > 0
    assert attr.max_cqe > 0
    assert attr.max_mr > 0
    assert attr.phys_port_cnt >= 1


def test_query_efa_device(ctx):
    attr = ctx.query_efa_device()
    assert attr.max_sq_wr > 0
    assert attr.max_rq_wr > 0
    assert attr.max_sq_sge >= 1
    assert attr.max_rq_sge >= 1
    assert attr.inline_buf_size >= 0
    caps = efa.EfaDeviceCaps(attr.device_caps)
    if caps & (efa.EfaDeviceCaps.RDMA_READ | efa.EfaDeviceCaps.RDMA_WRITE):
        assert attr.max_rdma_size > 0


def test_query_port(ctx):
    pa = ctx.query_port()
    assert pa.state == efa.PortState.ACTIVE
    assert pa.max_msg_sz > 0
    assert pa.gid_tbl_len >= 1


def test_query_gid(ctx):
    gid = ctx.query_gid()
    assert len(gid.raw) == 16
    assert gid.raw != b"\x00" * 16
    assert bytes(gid) == gid.raw
    assert gid == efa.Gid(gid.raw)
    assert "Gid(" in repr(gid)


def test_context_close_with_children_raises(ctx):
    pd = ctx.alloc_pd()
    with pytest.raises(efa.EfaError):
        ctx.close()
    pd.close()


def test_closed_context_raises(efa_devices):
    ctx = efa_devices[0].open()
    ctx.close()
    with pytest.raises(efa.EfaError):
        ctx.query_device()
