# efa

Low-level Python bindings for AWS Elastic Fabric Adapter (EFA), including
Scalable Reliable Datagram (SRD), one-sided RDMA, and GPUDirect transfers
to and from torch CUDA tensors.

The package wraps `libibverbs` and EFA's `libefa` direct-verbs API in Cython.
Its data path calls the provider's inline verbs directly, releases the GIL
around blocking and posting operations, and does not import torch or link
against CUDA.

- No Python runtime dependencies.
- SRD and UD queue pairs with SEND, RDMA read, and RDMA write operations.
- Host buffers, CUDA device pointers, and dma-buf memory registration.
- Classic and extended CQs, sender GID and unsolicited-write metadata, and
  direct EFA CQ, SQ, RQ, MR, and AH queries.
- One Linux `abi3` wheel for CPython 3.9 and newer.

## Requirements

- Linux on an AWS instance with one or more EFA devices attached.
- `libibverbs.so.1` and `libefa.so.1` at runtime. The AWS EFA installer and
  current `rdma-core` distributions provide both.
- The EFA device nodes available inside the process or container, normally
  `/dev/infiniband/uverbs*`.
- A C compiler, Cython, and the `rdma-core` development headers only when
  building from source.

For GPUDirect, the EFA device and instance type must support RDMA read/write,
and the NVIDIA driver must support dma-buf export or `nvidia_peermem`.

## Install

```bash
pip install efa
```

To build from this checkout:

```bash
pip install "Cython>=3.0" "setuptools>=77" wheel
pip install ./efa
```

## SRD Quickstart

SRD is reliable and connectionless. Each process creates a ready-to-send QP,
exchanges a 24-byte `EndpointInfo` out of band, and resolves the remote GID to
an address handle:

```python
import numpy as np
import efa

dev = efa.get_efa_device_list()[0]
ctx = dev.open()
pd = ctx.alloc_pd()
cq = ctx.create_cq(256)
qp = pd.create_qp(
    efa.QPInitAttr(send_cq=cq, recv_cq=cq)
).prepare(qkey=0x1234)

local_info = efa.local_endpoint_info(qp, qkey=0x1234)
# Exchange local_info.to_bytes() with the other process.
remote_info = efa.EndpointInfo.from_bytes(remote_bytes)
peer = remote_info.peer(pd)

buf = np.zeros(4096, dtype=np.uint8)
access = (
    efa.AccessFlags.LOCAL_WRITE
    | efa.AccessFlags.REMOTE_WRITE
    | efa.AccessFlags.REMOTE_READ
)
mr = efa.reg_tensor(pd, buf, access)

qp.post_send(efa.SendWR(
    wr_id=1,
    sg_list=[mr.sge()],
    opcode=efa.WROpcode.SEND,
    send_flags=efa.SendFlags.SIGNALED,
    dest=peer,
))
for wc in cq.poll(16):
    wc.raise_for_status()
```

Every handle is an idempotent context manager. A QP retains its PD and CQs,
an MR retains its PD and backing tensor, and an SGE retains the MR it addresses.

### One-sided RDMA

EFA requires the responder to have an address handle for the requester before
it accepts RDMA reads or writes. Both processes should therefore resolve the
other process's `EndpointInfo`, even if traffic is currently one-way. A
missing reverse AH completes with `REM_OP_ERR` and EFA vendor status `0x0e`
(`REMOTE_ERROR_UNKNOWN_PEER`).

Large buffers can be split at the device's `max_rdma_size`:

```python
wrs = efa.write_wrs(local_mr, peer, remote_addr, remote_rkey)
qp.post_send(wrs)
```

`read_wrs` provides the corresponding RDMA-read operation. Every generated WR
is signaled, as required by EFA.

## GPUDirect With Torch

The optional `efa.cuda` module is torch-free and duck-types objects exposing
`data_ptr()`, `numel()`, and `element_size()`. For torch's dma-buf path, enable
VMM-backed allocations before CUDA initializes:

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import efa
import efa.cuda

src = torch.arange(1 << 20, dtype=torch.float32, device="cuda:0")
dst = torch.zeros_like(src)

src_mr = efa.cuda.register_tensor(pd, src, access)
dst_mr = efa.cuda.register_tensor(pd, dst, access)

torch.cuda.synchronize(src.device)
qp.post_send(efa.SendWR(
    wr_id=2,
    sg_list=[src_mr.sge()],
    opcode=efa.WROpcode.RDMA_WRITE,
    send_flags=efa.SendFlags.SIGNALED,
    remote_addr=remote_dst_addr,
    rkey=remote_dst_rkey,
    dest=peer,
))
```

After the receiver observes a completion or protocol-level write
notification, it must order inbound NIC writes before CUDA consumes the
destination:

```python
with torch.cuda.device(dst.device):
    efa.cuda.flush_gpudirect_writes()
```

`register_tensor` first exports a dma-buf fd and calls `ibv_reg_dmabuf_mr`.
If that path is unavailable, it falls back to `ibv_reg_mr`, which requires
`nvidia_peermem`. The returned `GpuMR` retains the tensor allocation and keeps
the actual CUDA virtual address because `ibv_mr.addr` is not meaningful for a
dma-buf MR.

## Direct EFA API

| Area | API |
| --- | --- |
| Device capabilities | `Context.query_efa_device` |
| EFA CQ creation | `Context.create_cq_ex` |
| Sender and unsolicited metadata | `CQEx.poll`, `WC.sgid`, `WC.unsolicited` |
| CQ layout | `CQ.query_efa`, `CQEx.query_efa` |
| SQ/RQ layout | `QP.query_wqs` |
| MR interconnect IDs | `MR.query_efa` |
| Address-handle number | `AH.ahn` |
| SRD QP creation | `PD.create_qp` with `QPType.SRD` |

When using unsolicited RDMA write-with-immediate completions, create every
communicating QP with `QPInitAttr(..., unsolicited_write_recv=True)` and use an
extended CQ created with `unsolicited=True`. EFA requires peers to negotiate
the same QP feature set.

The direct layout queries expose process-local addresses for advanced
consumers. They do not transfer ownership of provider memory.

## Testing

```bash
pip install -e "./efa[test,gpu]"
cd efa
pytest -rs
```

Tests marked `integration` exercise real EFA hardware. Tests marked `gpu`
perform torch-verified GPU-to-GPU, GPU-to-host, and host-to-GPU transfers.
Unavailable hardware capabilities are skipped explicitly.

## License

BSD-3-Clause. See `LICENSE`.
