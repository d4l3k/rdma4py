"""Protection domains and memory regions on real hardware."""

from __future__ import annotations

import efa
import numpy as np
import pytest
from _srd import FULL_ACCESS

pytestmark = pytest.mark.integration


def test_alloc_and_close_pd(ctx):
    pd = ctx.alloc_pd()
    pd.close()
    pd.close()  # idempotent


def test_reg_mr_basics(pd):
    buf = np.zeros(4096, dtype=np.uint8)
    mr = pd.reg_mr(buf.ctypes.data, buf.nbytes, FULL_ACCESS)
    assert mr.addr == buf.ctypes.data
    assert mr.length == 4096
    assert mr.lkey != 0
    assert mr.rkey != 0
    assert not mr.closed
    mr.close()
    assert mr.closed
    with pytest.raises(efa.EfaError):
        _ = mr.lkey


def test_reg_mr_rejects_bad_args(pd):
    with pytest.raises(ValueError):
        pd.reg_mr(-1, 4096, FULL_ACCESS)
    with pytest.raises(ValueError):
        pd.reg_mr(0x1000, 0, FULL_ACCESS)


def test_mr_sge_bounds(pd):
    buf = np.zeros(4096, dtype=np.uint8)
    with pd.reg_mr(buf.ctypes.data, buf.nbytes, FULL_ACCESS) as mr:
        sge = mr.sge()
        assert sge.addr == mr.addr
        assert sge.length == 4096
        sge = mr.sge(100, offset=200)
        assert sge.addr == mr.addr + 200
        assert sge.length == 100
        with pytest.raises(ValueError):
            mr.sge(4097)
        with pytest.raises(ValueError):
            mr.sge(offset=5000)


def test_sge_from_int_address():
    sge = efa.SGE(0xDEAD0000, 128, lkey=7)
    assert sge.addr == 0xDEAD0000
    assert sge.length == 128
    assert sge.lkey == 7
    with pytest.raises(ValueError):
        efa.SGE(-1, 8)
    with pytest.raises(ValueError):
        efa.SGE(0, 1 << 32)


def test_reg_tensor_numpy(pd):
    arr = np.arange(1024, dtype=np.uint8)
    mr = efa.reg_tensor(pd, arr, FULL_ACCESS)
    assert mr.addr == arr.ctypes.data
    assert mr.length == 1024
    assert mr.owner is arr  # retains the allocation
    mr.close()


def test_query_efa_mr(pd):
    buf = np.zeros(4096, dtype=np.uint8)
    with pd.reg_mr(buf.ctypes.data, buf.nbytes, FULL_ACCESS) as mr:
        try:
            attr = mr.query_efa()
        except RuntimeError as exc:
            pytest.skip(str(exc))
        assert attr.ic_id_validity >= 0
        assert "MRAttr(" in repr(attr)


def test_pd_close_order_is_safe(ctx):
    # Dropping all references must not crash even with a live child MR;
    # the MR keeps the PD alive until it is collected.
    pd = ctx.alloc_pd()
    buf = np.zeros(4096, dtype=np.uint8)
    mr = pd.reg_mr(buf.ctypes.data, buf.nbytes, FULL_ACCESS)
    del pd
    mr.close()
