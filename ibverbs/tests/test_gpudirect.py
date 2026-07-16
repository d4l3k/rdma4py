"""GPUDirect RDMA against real H100s.

Registers CUDA device memory with libibverbs and moves it over the NIC with
zero host-memory staging. Uses the dma-buf path (``ibv_reg_dmabuf_mr``); if
``nvidia_peermem`` is loaded the raw ``reg_mr`` path is also exercised.
"""

from __future__ import annotations

import pytest

import ibverbs as ib
from _rc import Endpoint, FULL_ACCESS

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("no CUDA GPU available", allow_module_level=True)

from _cuda import CudaUnavailable, CudaVMM  # noqa: E402


@pytest.fixture(scope="module")
def cuda():
    try:
        return CudaVMM(device=0)
    except CudaUnavailable as exc:
        pytest.skip(f"CUDA VMM unavailable: {exc}")


@pytest.fixture()
def loopback(ctx, dev_name, first_active):
    from conftest import find_roce_gid

    _, port = first_active
    gid_index, gid = find_roce_gid(ctx, dev_name, port)
    pd = ctx.alloc_pd()
    pa = ctx.query_port(port)
    a = Endpoint(ctx, pd, port)
    b = Endpoint(ctx, pd, port)
    ib.connect_rc(a.qp, b.info(pa, gid), port=port, sgid_index=gid_index,
                  access=FULL_ACCESS)
    ib.connect_rc(b.qp, a.info(pa, gid), port=port, sgid_index=gid_index,
                  access=FULL_ACCESS)
    yield a, b, pd
    a.close()
    b.close()
    pd.close()


def test_reg_dmabuf_gpu_memory(cuda, ctx):
    pd = ctx.alloc_pd()
    ptr, size, fd = cuda.alloc(1 << 16)
    mr = pd.reg_dmabuf_mr(0, size, ptr, fd, FULL_ACCESS)
    assert mr.lkey != 0
    assert mr.rkey != 0
    mr.close()
    pd.close()


def test_gpudirect_rdma_write_dmabuf(cuda, loopback):
    a, b, pd = loopback
    n = 4096
    sptr, ssz, sfd = cuda.alloc(n)
    dptr, dsz, dfd = cuda.alloc(n)
    smr = pd.reg_dmabuf_mr(0, ssz, sptr, sfd, FULL_ACCESS)
    dmr = pd.reg_dmabuf_mr(0, dsz, dptr, dfd, FULL_ACCESS)

    payload = bytes((i * 13) & 0xFF for i in range(n))
    cuda.memset_host_to_device(sptr, payload)
    cuda.memset_host_to_device(dptr, b"\x00" * n)

    a.qp.post_send(ib.SendWR(
        wr_id=1, sg_list=[ib.SGE(sptr, n, lkey=smr.lkey)],
        opcode=ib.WROpcode.RDMA_WRITE, send_flags=ib.SendFlags.SIGNALED,
        remote_addr=dptr, rkey=dmr.rkey))
    wc = a.poll_one()
    assert wc.status == ib.WCStatus.SUCCESS, wc

    # The write landed in GPU memory with no host staging on the data path.
    assert cuda.device_to_host(dptr, n) == payload

    smr.close()
    dmr.close()


def test_gpudirect_rdma_read_dmabuf(cuda, loopback):
    a, b, pd = loopback
    n = 2048
    lptr, lsz, lfd = cuda.alloc(n)
    rptr, rsz, rfd = cuda.alloc(n)
    lmr = pd.reg_dmabuf_mr(0, lsz, lptr, lfd, FULL_ACCESS)
    rmr = pd.reg_dmabuf_mr(0, rsz, rptr, rfd, FULL_ACCESS)

    payload = bytes((i * 7 + 1) & 0xFF for i in range(n))
    cuda.memset_host_to_device(rptr, payload)
    cuda.memset_host_to_device(lptr, b"\x00" * n)

    a.qp.post_send(ib.SendWR(
        wr_id=2, sg_list=[ib.SGE(lptr, n, lkey=lmr.lkey)],
        opcode=ib.WROpcode.RDMA_READ, send_flags=ib.SendFlags.SIGNALED,
        remote_addr=rptr, rkey=rmr.rkey))
    wc = a.poll_one()
    assert wc.status == ib.WCStatus.SUCCESS, wc
    assert cuda.device_to_host(lptr, n) == payload

    lmr.close()
    rmr.close()


def test_reg_mr_peermem_if_available(cuda, ctx):
    """If nvidia_peermem is loaded, a raw device pointer registers too."""
    import os

    if not os.path.exists("/sys/module/nvidia_peermem"):
        pytest.skip("nvidia_peermem not loaded; dma-buf path covers GPUDirect")
    pd = ctx.alloc_pd()
    t = torch.empty(4096, dtype=torch.uint8, device="cuda:0")
    mr = pd.reg_mr(t.data_ptr(), t.numel(), FULL_ACCESS)
    assert mr.rkey != 0
    mr.close()
    pd.close()
