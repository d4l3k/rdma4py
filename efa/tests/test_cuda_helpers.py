"""Unit tests for efa.cuda that need no hardware (validation paths)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import efa.cuda as efacuda
import pytest

_PAGE = os.sysconf("SC_PAGESIZE")


def test_dmabuf_fd_validates_alignment():
    with pytest.raises(ValueError, match="page-aligned"):
        efacuda.dmabuf_fd(_PAGE + 1, _PAGE)
    with pytest.raises(ValueError, match="page-aligned"):
        efacuda.dmabuf_fd(_PAGE, _PAGE + 1)
    with pytest.raises(ValueError):
        efacuda.dmabuf_fd(-_PAGE, _PAGE)
    with pytest.raises(ValueError):
        efacuda.dmabuf_fd(_PAGE, 0)


def test_tensor_ptr_len_requires_data_ptr():
    with pytest.raises(TypeError, match="data_ptr"):
        efacuda.tensor_ptr_len(SimpleNamespace())


def test_tensor_ptr_len_torch_like():
    t = SimpleNamespace(
        data_ptr=lambda: 0x7000, numel=lambda: 10, element_size=lambda: 4
    )
    assert efacuda.tensor_ptr_len(t) == (0x7000, 40)


def test_tensor_ptr_len_nbytes_fallback():
    t = SimpleNamespace(data_ptr=lambda: 0x7000, nbytes=128)
    assert efacuda.tensor_ptr_len(t) == (0x7000, 128)


def test_register_tensor_rejects_host_tensor():
    t = SimpleNamespace(is_cuda=False)
    with pytest.raises(ValueError, match="requires a CUDA tensor"):
        efacuda.register_tensor(None, t, 1)


def test_register_tensor_rejects_empty():
    t = SimpleNamespace(
        is_cuda=True, data_ptr=lambda: 0x7000, numel=lambda: 0, element_size=lambda: 4
    )
    with pytest.raises(ValueError, match="empty"):
        efacuda.register_tensor(None, t, 1)


def test_gpu_mr_sge_bounds():
    class FakeMR:
        lkey = 1
        rkey = 2
        closed = False

        def close(self):
            self.closed = True

    gmr = efacuda.GpuMR(FakeMR(), 0x10000, 4096)
    sge = gmr.sge()
    assert sge.addr == 0x10000
    assert sge.length == 4096
    sge = gmr.sge(100, offset=200)
    assert sge.addr == 0x10000 + 200
    assert sge.length == 100
    with pytest.raises(ValueError):
        gmr.sge(4097)
    with pytest.raises(ValueError):
        gmr.sge(offset=5000)
    with pytest.raises(ValueError):
        gmr.sge(4000, offset=200)
    gmr.close()
    assert gmr.closed
    assert gmr.tensor is None
