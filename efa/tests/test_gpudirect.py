"""GPUDirect RDMA over EFA against real GPUs, driven with torch tensors.

Allocation, fill, and verification all use torch; the only CUDA-specific step
(exporting a dma-buf fd) is handled by ``efa.cuda.register_tensor``. The
library itself never imports torch.
"""

from __future__ import annotations

import efa
import efa.cuda as efacuda
import pytest
from _srd import Endpoint, FULL_ACCESS

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("no CUDA GPU available", allow_module_level=True)


@pytest.fixture()
def loopback(ctx, pd):
    a = Endpoint(ctx, pd)
    b = Endpoint(ctx, pd)
    peer = a.peer_to(b)
    yield a, b, peer, pd
    peer.close()
    a.close()
    b.close()


def _register(pd, tensor):
    try:
        return efacuda.register_tensor(pd, tensor, FULL_ACCESS)
    except RuntimeError as exc:
        pytest.skip(f"GPUDirect registration unavailable: {exc}")


def test_register_torch_cuda_tensor(pd):
    t = torch.zeros(1 << 16, dtype=torch.uint8, device="cuda:0")
    gmr = _register(pd, t)
    assert gmr.lkey != 0
    assert gmr.rkey != 0
    assert gmr.addr == t.data_ptr()
    assert gmr.tensor is t
    gmr.close()
    assert gmr.closed


def test_gpudirect_rdma_write(loopback, efa_caps):
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.RDMA_WRITE)
    a, b, peer, pd = loopback
    src = torch.arange(1 << 20, dtype=torch.float32, device="cuda:0") * 1.5
    dst = torch.zeros(1 << 20, dtype=torch.float32, device="cuda:0")
    src_mr = _register(pd, src)
    dst_mr = _register(pd, dst)

    torch.cuda.synchronize(src.device)
    a.qp.post_send(
        efa.SendWR(
            wr_id=1,
            sg_list=[src_mr.sge()],
            opcode=efa.WROpcode.RDMA_WRITE,
            send_flags=efa.SendFlags.SIGNALED,
            remote_addr=dst_mr.addr,
            rkey=dst_mr.rkey,
            dest=peer,
        )
    )
    wc = a.poll_one()
    assert wc.status == efa.WCStatus.SUCCESS, wc

    with torch.cuda.device(dst.device):
        efacuda.flush_gpudirect_writes()
    assert torch.equal(src, dst)  # data moved GPU->GPU with no host staging

    src_mr.close()
    dst_mr.close()


def test_gpudirect_rdma_read(loopback, efa_caps):
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.RDMA_READ)
    a, b, peer, pd = loopback
    remote = torch.randint(0, 255, (1 << 18,), dtype=torch.uint8, device="cuda:0")
    local = torch.zeros(1 << 18, dtype=torch.uint8, device="cuda:0")
    local_mr = _register(pd, local)
    remote_mr = _register(pd, remote)

    torch.cuda.synchronize(remote.device)
    a.qp.post_send(
        efa.SendWR(
            wr_id=2,
            sg_list=[local_mr.sge()],
            opcode=efa.WROpcode.RDMA_READ,
            send_flags=efa.SendFlags.SIGNALED,
            remote_addr=remote_mr.addr,
            rkey=remote_mr.rkey,
            dest=peer,
        )
    )
    wc = a.poll_one()
    assert wc.status == efa.WCStatus.SUCCESS, wc

    with torch.cuda.device(local.device):
        efacuda.flush_gpudirect_writes()
    assert torch.equal(local, remote)

    local_mr.close()
    remote_mr.close()


def test_gpudirect_send_recv_between_gpus(loopback):
    """SEND from a tensor on GPU 0 into a recv tensor on another GPU."""
    a, b, peer, pd = loopback
    ndev = torch.cuda.device_count()
    n = 1024  # SEND payloads are limited to max_msg_sz (~8 KiB)
    src = torch.arange(n, dtype=torch.int32, device="cuda:0")
    dst = torch.zeros(n, dtype=torch.int32, device=f"cuda:{1 if ndev > 1 else 0}")
    src_mr = _register(pd, src)
    dst_mr = _register(pd, dst)

    b.qp.post_recv(efa.RecvWR(wr_id=3, sg_list=[dst_mr.sge()]))
    torch.cuda.synchronize(src.device)
    a.qp.post_send(
        efa.SendWR(
            wr_id=4,
            sg_list=[src_mr.sge()],
            opcode=efa.WROpcode.SEND,
            send_flags=efa.SendFlags.SIGNALED,
            dest=peer,
        )
    )
    swc = a.poll_one()
    rwc = b.poll_one()
    assert swc.status == efa.WCStatus.SUCCESS, swc
    assert rwc.status == efa.WCStatus.SUCCESS, rwc
    assert rwc.byte_len == n * 4

    with torch.cuda.device(dst.device):
        efacuda.flush_gpudirect_writes()
    assert torch.equal(src.cpu(), dst.cpu())

    src_mr.close()
    dst_mr.close()


def test_gpudirect_rdma_write_with_imm_notifies_receiver(loopback, efa_caps):
    """Write-with-imm gives the receiver a completion to key CUDA work off."""
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.RDMA_WRITE)
    a, b, peer, pd = loopback
    src = torch.full((4096,), 7.25, dtype=torch.float32, device="cuda:0")
    dst = torch.zeros(4096, dtype=torch.float32, device="cuda:0")
    src_mr = _register(pd, src)
    dst_mr = _register(pd, dst)

    b.qp.post_recv(efa.RecvWR(wr_id=5, sg_list=[dst_mr.sge()]))
    torch.cuda.synchronize(src.device)
    a.qp.post_send(
        efa.SendWR(
            wr_id=6,
            sg_list=[src_mr.sge()],
            opcode=efa.WROpcode.RDMA_WRITE_WITH_IMM,
            send_flags=efa.SendFlags.SIGNALED,
            remote_addr=dst_mr.addr,
            rkey=dst_mr.rkey,
            imm_data=42,
            dest=peer,
        )
    )
    a.poll_one().raise_for_status()
    rwc = b.poll_one()
    rwc.raise_for_status()
    assert rwc.imm_data == 42

    # The receive completion is the ordering point: flush, then consume.
    with torch.cuda.device(dst.device):
        efacuda.flush_gpudirect_writes()
    assert torch.equal(src, dst)

    src_mr.close()
    dst_mr.close()


def test_gpudirect_chunked_transfer(loopback, efa_caps):
    """A large tensor moves via the chunked-write helper (multiple WRs)."""
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.RDMA_WRITE)
    a, b, peer, pd = loopback
    n = 8 << 20  # 8 Mi floats = 32 MiB
    src = torch.rand(n, dtype=torch.float32, device="cuda:0")
    dst = torch.zeros(n, dtype=torch.float32, device="cuda:0")
    src_mr = _register(pd, src)
    dst_mr = _register(pd, dst)

    torch.cuda.synchronize(src.device)
    wrs = efa.write_wrs(src_mr, peer, dst_mr.addr, dst_mr.rkey, chunk=8 << 20)
    assert len(wrs) == 4
    a.qp.post_send(wrs)
    for wc in a.poll_n(len(wrs)):
        wc.raise_for_status()

    with torch.cuda.device(dst.device):
        efacuda.flush_gpudirect_writes()
    assert torch.equal(src, dst)

    src_mr.close()
    dst_mr.close()


def test_gpu_to_host_and_back(loopback, efa_caps):
    """RDMA-write GPU->host, then host->GPU, verifying both directions."""
    from conftest import require_caps

    require_caps(efa_caps, efa.EfaDeviceCaps.RDMA_WRITE)
    a, b, peer, pd = loopback
    n = 1 << 16
    gpu = torch.arange(n, dtype=torch.float32, device="cuda:0")
    host = torch.zeros(n, dtype=torch.float32)
    gpu_mr = _register(pd, gpu)
    host_mr = efa.reg_tensor(pd, host, FULL_ACCESS)

    # GPU -> host
    torch.cuda.synchronize(gpu.device)
    a.qp.post_send(
        efa.SendWR(
            wr_id=7,
            sg_list=[gpu_mr.sge()],
            opcode=efa.WROpcode.RDMA_WRITE,
            send_flags=efa.SendFlags.SIGNALED,
            remote_addr=host_mr.addr,
            rkey=host_mr.rkey,
            dest=peer,
        )
    )
    a.poll_one().raise_for_status()
    assert torch.equal(host, gpu.cpu())

    # host -> GPU (into a fresh zeroed tensor)
    gpu2 = torch.zeros(n, dtype=torch.float32, device="cuda:0")
    gpu2_mr = _register(pd, gpu2)
    host += 1
    a.qp.post_send(
        efa.SendWR(
            wr_id=8,
            sg_list=[host_mr.sge()],
            opcode=efa.WROpcode.RDMA_WRITE,
            send_flags=efa.SendFlags.SIGNALED,
            remote_addr=gpu2_mr.addr,
            rkey=gpu2_mr.rkey,
            dest=peer,
        )
    )
    a.poll_one().raise_for_status()
    with torch.cuda.device(gpu2.device):
        efacuda.flush_gpudirect_writes()
    assert torch.equal(gpu2.cpu(), host)

    gpu_mr.close()
    gpu2_mr.close()
    host_mr.close()


def test_reg_mr_peermem_if_available(pd):
    """If nvidia_peermem is loaded, the raw reg_mr device-pointer path works too."""
    import os

    if not os.path.exists("/sys/module/nvidia_peermem"):
        pytest.skip("nvidia_peermem not loaded; dma-buf path covers GPUDirect")
    t = torch.empty(4096, dtype=torch.uint8, device="cuda:0")
    mr = pd.reg_mr(t.data_ptr(), t.numel(), FULL_ACCESS)
    assert mr.rkey != 0
    mr.close()
