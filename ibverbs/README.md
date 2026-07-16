# ibverbs

Low-level, Pythonic bindings for **libibverbs** (RDMA), designed as a
foundation for building high-performance RDMA libraries in Python — including
**GPUDirect** transfers to/from GPU memory.

The bindings are a thin, faithful wrapper over the verbs API (device, PD, MR,
CQ, QP, SRQ, AH, work requests, completions, async events) plus a small,
optional set of RC connection helpers. They are written in Cython so the data
path (`post_send` / `post_recv` / `poll`) compiles to direct C calls and
releases the GIL, and so the `static inline` verbs fast-path functions are
called correctly (they can't be reached through `dlsym`).

- **No runtime dependencies.** Only libibverbs at load time.
- **No torch / CUDA linkage.** GPUDirect works by registering an integer
  device address or an exported dma-buf fd; CUDA stays entirely in your code.
- Targets CPython 3.9+ on Linux.

## Requirements

- Linux with an RDMA-capable NIC (tested on Mellanox/NVIDIA **mlx5**, RoCEv2).
- `libibverbs` at runtime and **`rdma-core` development headers** at build time
  (`libibverbs-dev` on Debian/Ubuntu, `rdma-core-devel` / `libibverbs-devel` on
  RHEL/Fedora).
- A C compiler and Cython (build time only).

## Install

From source (until published to PyPI):

```bash
# build deps
pip install "Cython>=3.0" "setuptools>=64" wheel
# the package
pip install ./ibverbs           # or: pip install -e ./ibverbs
```

The extension is compiled against your system `libibverbs` via `pkg-config`.

## Quickstart

```python
import ibverbs as ib

# 1. Open a device and set up resources.
dev = ib.get_device_list()[0]
ctx = dev.open()
pd = ctx.alloc_pd()
cq = ctx.create_cq(64)

# 2. Register memory (host or GPU address — any integer VA works).
import numpy as np
buf = np.zeros(4096, dtype=np.uint8)
access = ib.AccessFlags.LOCAL_WRITE | ib.AccessFlags.REMOTE_WRITE | ib.AccessFlags.REMOTE_READ
mr = pd.reg_mr(buf.ctypes.data, buf.nbytes, access)

# 3. Create a reliable-connected QP.
qp = pd.create_qp(ib.QPInitAttr(send_cq=cq, recv_cq=cq, qp_type=ib.QPType.RC))

# 4. Exchange connection info with the peer out-of-band, then connect.
port = 1
port_attr = ctx.query_port(port)
gid = ctx.query_gid(port, gid_index)              # pick a routable RoCEv2 GID
local = ib.local_qp_info(qp, port_attr, gid, port=port, psn=0)
# ... send local.to_bytes() to peer, receive remote_bytes ...
remote = ib.QPInfo.from_bytes(remote_bytes)
ib.connect_rc(qp, remote, port=port, sgid_index=gid_index, access=access)

# 5. Post an RDMA write and reap the completion.
qp.post_send(ib.SendWR(
    wr_id=1, sg_list=[ib.SGE(mr, 4096)], opcode=ib.WROpcode.RDMA_WRITE,
    send_flags=ib.SendFlags.SIGNALED, remote_addr=peer_addr, rkey=peer_rkey))
for wc in cq.poll(16):
    wc.raise_for_status()
```

Every resource is a context manager and frees its handle on `close()` /
garbage collection; children hold references to their parents, so destruction
order is always safe.

## GPUDirect with torch tensors

The library never imports torch or links CUDA. The optional `ibverbs.cuda`
helper (which only lazily `dlopen`s `libcuda`) registers a CUDA tensor for RDMA
in one call — handling the dma-buf export and page alignment for you:

```python
import os
# torch's CUDA memory must be VMM-backed to be dma-buf exportable. Set this
# BEFORE torch initializes CUDA:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import ibverbs as ib
import ibverbs.cuda

src = torch.arange(4096, dtype=torch.float32, device="cuda:0")
dst = torch.zeros(4096, dtype=torch.float32, device="cuda:0")

access = ib.AccessFlags.LOCAL_WRITE | ib.AccessFlags.REMOTE_WRITE
src_mr = ib.cuda.register_tensor(pd, src, access)   # -> GpuMR (dma-buf backed)
dst_mr = ib.cuda.register_tensor(pd, dst, access)

# RDMA-write one GPU buffer into another, with no host staging on the data path.
qp.post_send(ib.SendWR(
    wr_id=1, sg_list=[src_mr.sge()], opcode=ib.WROpcode.RDMA_WRITE,
    send_flags=ib.SendFlags.SIGNALED,
    remote_addr=dst_mr.addr, rkey=dst_mr.rkey))
for wc in qp.send_cq.poll(16):
    wc.raise_for_status()
```

`GpuMR` wraps the `MR` with the correct device address (`ibv_mr.addr` is not
meaningful for dma-buf MRs), and exposes `.sge()`, `.addr`, `.lkey`, `.rkey`.

Under the hood there are two registration paths, chosen automatically:

```python
# dma-buf fd (default; no kernel module needed):
mr = pd.reg_dmabuf_mr(offset, length, iova=device_va, fd=dmabuf_fd, access=access)
# raw device pointer (requires the nvidia_peermem kernel module):
mr = pd.reg_mr(tensor.data_ptr(), nbytes, access)
```

For a **host** (CPU) torch tensor or numpy array, `ib.reg_tensor(pd, tensor,
access)` registers it directly. `tests/test_gpudirect.py` performs real
GPU-to-GPU RDMA writes, reads, and sends verified with `torch.equal`.

## Feature coverage

| Area | Supported |
|------|-----------|
| Device / port / GID query | ✅ `get_device_list`, `Context.query_device/query_port/query_gid` |
| Protection domains | ✅ `alloc_pd` |
| Memory regions | ✅ `reg_mr`, `reg_dmabuf_mr` (GPUDirect) |
| Completion queues | ✅ `create_cq`, `poll`, comp channels + `req_notify`/`ack_events` |
| Queue pairs | ✅ RC / UC / UD; `modify`, `query`, `to_init`/`to_rtr`/`to_rts` |
| Work requests | ✅ SEND(/_IMM), RDMA_WRITE(/_IMM), RDMA_READ, ATOMIC_CMP_AND_SWP, ATOMIC_FETCH_AND_ADD, scatter/gather, inline/signaled/fenced/solicited flags |
| Shared receive queues | ✅ `create_srq`, `post_recv`, `modify`, `query` |
| Address handles | ✅ `create_ah` (UD) |
| Async events | ✅ `get_async_event` / `ack_async_event`, `async_fd` |
| Connection helpers | ✅ `QPInfo`, `local_qp_info`, `connect_rc` |

Out of scope for v1 (candidates for later): the extended `ibv_wr_*` / `qp_ex`
post API, device memory (`ibv_alloc_dm`), memory windows, and flow steering.

## Testing

The suite exercises real hardware and skips features the host lacks:

```bash
pip install -e "./ibverbs[test,gpu]"     # pytest, numpy, torch
cd ibverbs && pytest -rs                  # -rs shows skip reasons
pytest -m "not gpu"                       # skip GPUDirect tests
pytest -m integration                     # only real-hardware tests
```

Markers: `integration` (needs an RDMA NIC), `gpu` (needs CUDA + torch).

## License

BSD-3-Clause. See the repository `LICENSE`.
