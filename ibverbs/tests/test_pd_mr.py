"""Protection domains and memory-region registration."""

from __future__ import annotations

import numpy as np
import pytest

import ibverbs as ib


@pytest.fixture()
def pd(ctx):
    p = ctx.alloc_pd()
    yield p
    p.close()


def test_alloc_pd(pd):
    assert pd is not None


def test_reg_mr_host_buffer(pd):
    buf = np.zeros(8192, dtype=np.uint8)
    access = (ib.AccessFlags.LOCAL_WRITE | ib.AccessFlags.REMOTE_WRITE
              | ib.AccessFlags.REMOTE_READ)
    mr = pd.reg_mr(buf.ctypes.data, buf.nbytes, access)
    assert mr.addr == buf.ctypes.data
    assert mr.length == buf.nbytes
    assert mr.lkey != 0
    assert mr.rkey != 0
    mr.close()


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
