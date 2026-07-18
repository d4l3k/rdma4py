"""Protection domains and memory-region registration."""

from __future__ import annotations

import ibverbs as ib
import numpy as np
import pytest


@pytest.fixture()
def pd(ctx):
    p = ctx.alloc_pd()
    yield p
    p.close()


def test_alloc_pd(pd):
    assert pd is not None


def test_reg_mr_host_buffer(pd):
    buf = np.zeros(8192, dtype=np.uint8)
    access = (
        ib.AccessFlags.LOCAL_WRITE
        | ib.AccessFlags.REMOTE_WRITE
        | ib.AccessFlags.REMOTE_READ
    )
    mr = pd.reg_mr(buf.ctypes.data, buf.nbytes, access)
    assert mr.addr == buf.ctypes.data
    assert mr.length == buf.nbytes
    assert mr.lkey != 0
    assert mr.rkey != 0
    mr.close()


def test_reg_tensor_retains_owner_and_builds_bounded_sge(pd):
    buf = np.zeros(4096, dtype=np.uint8)
    mr = ib.reg_tensor(pd, buf, ib.AccessFlags.LOCAL_WRITE)
    assert mr.owner is buf
    sge = mr.sge(offset=64)
    assert sge.addr == buf.ctypes.data + 64
    assert sge.length == buf.nbytes - 64
    assert sge.owner is mr
    with pytest.raises(ValueError, match="exceeds"):
        mr.sge(length=4096, offset=1)
    mr.close()
    assert mr.owner is None


def test_closed_mr_properties_raise(pd):
    buf = np.zeros(4096, dtype=np.uint8)
    mr = pd.reg_mr(buf.ctypes.data, buf.nbytes, ib.AccessFlags.LOCAL_WRITE)
    mr.close()
    assert mr.closed
    with pytest.raises(ib.VerbsError) as exc_info:
        _ = mr.lkey
    assert exc_info.value.errno != 0


def test_sge_rejects_truncating_values():
    with pytest.raises(ValueError, match=r"2\*\*32"):
        ib.SGE(0x1000, 1 << 32, lkey=1)
    with pytest.raises(ValueError, match="non-negative"):
        ib.SGE(0x1000, 1, lkey=1, offset=-1)


def test_reg_mr_local_only_has_keys(pd):
    buf = np.zeros(4096, dtype=np.uint8)
    mr = pd.reg_mr(buf.ctypes.data, buf.nbytes, ib.AccessFlags.LOCAL_WRITE)
    assert mr.lkey != 0
    mr.close()


def test_reg_mr_context_manager(pd):
    buf = np.zeros(4096, dtype=np.uint8)
    with pd.reg_mr(buf.ctypes.data, buf.nbytes, ib.AccessFlags.LOCAL_WRITE) as mr:
        assert mr.handle >= 0


def test_reg_mr_bad_address_raises(pd):
    # Registering an unmapped address must fail, not crash.
    with pytest.raises(ib.VerbsError):
        pd.reg_mr(0xDEAD0000, 4096, ib.AccessFlags.LOCAL_WRITE)


def test_reg_dmabuf_bad_fd_raises(pd):
    # The dma-buf path exists; a bogus fd must raise cleanly.
    with pytest.raises(ib.VerbsError):
        pd.reg_dmabuf_mr(0, 4096, 0, -1, ib.AccessFlags.LOCAL_WRITE)


def test_double_close_is_safe(pd):
    buf = np.zeros(4096, dtype=np.uint8)
    mr = pd.reg_mr(buf.ctypes.data, buf.nbytes, ib.AccessFlags.LOCAL_WRITE)
    mr.close()
    mr.close()  # idempotent


def test_failed_parent_close_preserves_pd_and_mr(pd):
    buf = np.zeros(4096, dtype=np.uint8)
    mr = pd.reg_mr(buf.ctypes.data, buf.nbytes, ib.AccessFlags.LOCAL_WRITE)
    with pytest.raises(ib.VerbsError) as exc_info:
        pd.close()
    assert exc_info.value.errno != 0
    assert mr.lkey != 0
    mr.close()
