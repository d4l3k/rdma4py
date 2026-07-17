"""GPUDirect RDMA against real GPUs, driven with torch tensors.

Allocation, fill, and verification all use torch; the only CUDA-specific step
(exporting a dma-buf fd) is handled by ``ibverbs.cuda.register_tensor``. The
library itself never imports torch.
"""

from __future__ import annotations

import pytest

import ibverbs as ib
import ibverbs.cuda as ibcuda
from _rc import Endpoint, FULL_ACCESS

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("no CUDA GPU available", allow_module_level=True)


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


def _register(pd, tensor):
    try:
        return ibcuda.register_tensor(pd, tensor, FULL_ACCESS)
    except RuntimeError as exc:
        pytest.skip(f"GPUDirect registration unavailable: {exc}")


def test_register_torch_cuda_tensor(ctx):
    pd = ctx.alloc_pd()
    t = torch.zeros(1 << 16, dtype=torch.uint8, device="cuda:0")
    gmr = _register(pd, t)
    assert gmr.lkey != 0
    assert gmr.rkey != 0
    assert gmr.addr == t.data_ptr()
    gmr.close()
    pd.close()


def test_gpudirect_rdma_write(loopback):
    a, b, pd = loopback
    src = (torch.arange(4096, dtype=torch.float32, device="cuda:0") * 1.5)
    dst = torch.zeros(4096, dtype=torch.float32, device="cuda:0")
    src_mr = _register(pd, src)
    dst_mr = _register(pd, dst)

    torch.cuda.synchronize(src.device)
    a.qp.post_send(ib.SendWR(
        wr_id=1, sg_list=[src_mr.sge()], opcode=ib.WROpcode.RDMA_WRITE,
        send_flags=ib.SendFlags.SIGNALED, remote_addr=dst_mr.addr,
        rkey=dst_mr.rkey))
    wc = a.poll_one()
    assert wc.status == ib.WCStatus.SUCCESS, wc

    with torch.cuda.device(dst.device):
        ibcuda.flush_gpudirect_writes()
    assert torch.equal(src, dst)   # data moved GPU->GPU with no host staging

    src_mr.close()
    dst_mr.close()


def test_gpudirect_rdma_read(loopback):
    a, b, pd = loopback
    remote = torch.randint(0, 255, (2048,), dtype=torch.uint8, device="cuda:0")
    local = torch.zeros(2048, dtype=torch.uint8, device="cuda:0")
    local_mr = _register(pd, local)
    remote_mr = _register(pd, remote)

    torch.cuda.synchronize(remote.device)
    a.qp.post_send(ib.SendWR(
        wr_id=2, sg_list=[local_mr.sge()], opcode=ib.WROpcode.RDMA_READ,
        send_flags=ib.SendFlags.SIGNALED, remote_addr=remote_mr.addr,
        rkey=remote_mr.rkey))
    wc = a.poll_one()
    assert wc.status == ib.WCStatus.SUCCESS, wc

    with torch.cuda.device(local.device):
        ibcuda.flush_gpudirect_writes()
    assert torch.equal(local, remote)

    local_mr.close()
    remote_mr.close()


def test_gpudirect_send_recv_between_gpus(loopback):
    """SEND from a tensor on GPU 0 into a recv tensor on another GPU."""
    a, b, pd = loopback
    ndev = torch.cuda.device_count()
    src = torch.arange(1024, dtype=torch.int32, device="cuda:0")
    dst = torch.zeros(1024, dtype=torch.int32, device=f"cuda:{1 if ndev > 1 else 0}")
    src_mr = _register(pd, src)
    dst_mr = _register(pd, dst)

    b.qp.post_recv(ib.RecvWR(wr_id=3, sg_list=[dst_mr.sge()]))
    torch.cuda.synchronize(src.device)
    a.qp.post_send(ib.SendWR(wr_id=4, sg_list=[src_mr.sge()],
                             opcode=ib.WROpcode.SEND,
                             send_flags=ib.SendFlags.SIGNALED))
    swc = a.poll_one()
    rwc = b.poll_one()
    assert swc.status == ib.WCStatus.SUCCESS, swc
    assert rwc.status == ib.WCStatus.SUCCESS, rwc

    with torch.cuda.device(dst.device):
        ibcuda.flush_gpudirect_writes()
    assert torch.equal(src.cpu(), dst.cpu())

    src_mr.close()
    dst_mr.close()


def test_reg_mr_peermem_if_available(ctx):
    """If nvidia_peermem is loaded, the raw reg_mr device-pointer path works too."""
    import os

    if not os.path.exists("/sys/module/nvidia_peermem"):
        pytest.skip("nvidia_peermem not loaded; dma-buf path covers GPUDirect")
    pd = ctx.alloc_pd()
    t = torch.empty(4096, dtype=torch.uint8, device="cuda:0")
    mr = pd.reg_mr(t.data_ptr(), t.numel(), FULL_ACCESS)
    assert mr.rkey != 0
    mr.close()
    pd.close()
