"""Unit tests for CUDA helpers that do not require a CUDA device."""

from __future__ import annotations

import pytest

import ibverbs as ib
import ibverbs.cuda as ibcuda


class FakeTensor:
    is_cuda = True

    def __init__(self, *, ptr=0x10000, length=4096, contiguous=True):
        self._ptr = ptr
        self._length = length
        self._contiguous = contiguous

    def data_ptr(self):
        return self._ptr

    def numel(self):
        return self._length

    def element_size(self):
        return 1

    def is_contiguous(self):
        return self._contiguous


class FakeMR:
    lkey = 0x11
    rkey = 0x22

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakePD:
    def __init__(self):
        self.reg_mr_args = None

    def reg_mr(self, *args):
        self.reg_mr_args = args
        return FakeMR()


def test_cuda_register_retains_tensor_on_peermem_fallback(monkeypatch):
    tensor = FakeTensor()
    pd = FakePD()
    monkeypatch.setattr(
        ibcuda, "dmabuf_fd", lambda *args: (_ for _ in ()).throw(
            RuntimeError("no dma-buf")
        )
    )

    gmr = ibcuda.register_tensor(pd, tensor, ib.AccessFlags.LOCAL_WRITE)

    assert gmr.tensor is tensor
    assert pd.reg_mr_args[:2] == (tensor.data_ptr(), tensor.numel())
    sge = gmr.sge(offset=128)
    assert sge.addr == tensor.data_ptr() + 128
    assert sge.length == tensor.numel() - 128
    assert sge.owner is gmr
    gmr.close()
    assert gmr.tensor is None


def test_cuda_register_rejects_host_tensor():
    tensor = FakeTensor()
    tensor.is_cuda = False
    with pytest.raises(ValueError, match="ibverbs.reg_tensor"):
        ibcuda.register_tensor(FakePD(), tensor, ib.AccessFlags.LOCAL_WRITE)


def test_cuda_register_rejects_noncontiguous_and_empty_tensors():
    with pytest.raises(ValueError, match="contiguous"):
        ibcuda.register_tensor(
            FakePD(), FakeTensor(contiguous=False), ib.AccessFlags.LOCAL_WRITE
        )
    with pytest.raises(ValueError, match="empty"):
        ibcuda.register_tensor(
            FakePD(), FakeTensor(length=0), ib.AccessFlags.LOCAL_WRITE
        )


def test_gpu_sge_checks_bounds():
    gmr = ibcuda.GpuMR(FakeMR(), 0x1000, 1024)
    with pytest.raises(ValueError, match="exceeds"):
        gmr.sge(length=1024, offset=1)
    with pytest.raises(ValueError, match="outside"):
        gmr.sge(offset=-1)


def test_dmabuf_fd_validates_alignment_before_loading_cuda():
    with pytest.raises(ValueError, match="page-aligned"):
        ibcuda.dmabuf_fd(1, 4096)
    with pytest.raises(ValueError, match="positive and page-aligned"):
        ibcuda.dmabuf_fd(0x1000, 1)
