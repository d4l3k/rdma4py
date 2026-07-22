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

- **No runtime dependencies.** Only libibverbs, which is `dlopen`ed at import.
- **No torch / CUDA linkage.** GPUDirect works by registering an integer
  device address or an exported dma-buf fd; CUDA stays entirely in your code.
- **One `abi3` wheel for all of CPython 3.9+** on Linux.

## Portability

The extension does **not** link `libibverbs`. It is compiled against the
rdma-core headers (for struct layouts and the `static inline` data-path verbs)
but resolves the exported verbs at import time with `dlopen`/`dlsym`. As a
result:

- The compiled module's only `NEEDED` library is `libc` — no external
  dependency for `auditwheel`, so a single **manylinux** wheel is portable
  across distros.
- It is built against the **CPython Limited API (abi3)**, so one wheel works on
  CPython **3.9 through 3.14+** — no per-version builds.
- A missing `libibverbs` yields a clean `ImportError`, not a loader crash.
- Newer verbs are optional: `ibv_reg_dmabuf_mr` (rdma-core ≥ 34) is loaded if
  present and only errors if you actually call `reg_dmabuf_mr`, so the wheel
  still imports on older systems.
- RDMA-CM is optional and lazily loads `librdmacm.so.1` only when `CMID` is
  used. `CMID.resolve()` returns the CM-owned device context and creates QPs
  whose state transitions and destruction remain owned by librdmacm.

Base verbs use only `libibverbs.so.1` at runtime (any `rdma-core` from the last
several years); `CMID` additionally needs `librdmacm.so.1`. The data path
(`post_send`/`poll`/…) stays compiled inline and dispatches through the
provider op table, so `dlopen` costs nothing on the hot path.

## Requirements

- Linux with an RDMA-capable NIC (tested on Mellanox/NVIDIA **mlx5**, RoCEv2).
- **Runtime:** `libibverbs.so.1` (`rdma-core` — `libibverbs1` on Debian/Ubuntu,
  `libibverbs` on RHEL/Fedora). No compiler or headers needed to *use* a wheel.
- **Build from source only:** a C compiler, Cython, and the `rdma-core`
  development headers (`libibverbs-dev` plus `librdmacm-dev` on Debian/Ubuntu,
  or `rdma-core-devel` on RPM-based distributions).

## Install

```bash
pip install ibverbs        # prebuilt abi3 manylinux wheel
```

Building from source (needs the rdma-core dev headers + a compiler):

```bash
pip install "Cython>=3.0" "setuptools>=77" wheel
pip install ./ibverbs       # or: pip install -e ./ibverbs
```

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
src_mr = ib.cuda.register_tensor(pd, src, access)   # retains src until close()
dst_mr = ib.cuda.register_tensor(pd, dst, access)

# RDMA-write one GPU buffer into another, with no host staging on the data path.
torch.cuda.synchronize(src.device)  # source-producing CUDA work must be done
qp.post_send(ib.SendWR(
    wr_id=1, sg_list=[src_mr.sge()], opcode=ib.WROpcode.RDMA_WRITE,
    send_flags=ib.SendFlags.SIGNALED,
    remote_addr=dst_mr.addr, rkey=dst_mr.rkey))
for wc in qp.send_cq.poll(16):
    wc.raise_for_status()

# On the receiver, after the peer has signaled that its write is complete,
# order the inbound NIC writes before launching CUDA work that consumes dst.
with torch.cuda.device(dst.device):
    ib.cuda.flush_gpudirect_writes()
```

`GpuMR` wraps the `MR` with the correct device address (`ibv_mr.addr` is not
meaningful for dma-buf MRs), retains the tensor allocation until `close()`, and
exposes `.sge()`, `.addr`, `.lkey`, `.rkey`.

CUDA work and NIC work are separate ordering domains. Synchronize the stream
that produced an outbound tensor before posting it. For inbound `SEND`, RDMA
read, or RDMA write, wait for the corresponding completion or protocol-level
notification, then call `flush_gpudirect_writes()` in the destination CUDA
context before consuming the tensor. A one-sided RDMA write does not create a
remote CQ entry by itself; use write-with-immediate or an out-of-band message
to notify the receiver.

Under the hood there are two registration paths, chosen automatically:

```python
# dma-buf fd (default; no kernel module needed):
mr = pd.reg_dmabuf_mr(offset, length, iova=device_va, fd=dmabuf_fd, access=access)
# raw device pointer (requires the nvidia_peermem kernel module):
mr = pd.reg_mr(tensor.data_ptr(), nbytes, access)
```

For a **host** (CPU) torch tensor or numpy array, `ib.reg_tensor(pd, tensor,
access)` registers it directly and retains the allocation. Both tensor helpers
require contiguous, non-empty tensors; split any single SGE larger than
`2**32 - 1` bytes into multiple entries. `tests/test_gpudirect.py` performs real
GPU-to-GPU RDMA writes, reads, and sends verified with `torch.equal`.

## GPU-initiated communication with GPUNetIO

`ibverbs.gpunetio` exports a connected mlx5 RC QP to NVIDIA DOCA GPUNetIO and
provides one device ABI for CUDA-derived kernels. Triton and CuTe DSL both link
the same architecture-specific LLVM bitcode, so WQE construction, doorbells,
and completion polling execute on the GPU without calling host libibverbs.

Install the framework adapter you use, plus the DOCA GPUNetIO runtime and
development headers from NVIDIA's DOCA repository:

```bash
pip install "ibverbs[gpunetio-triton]"
# or
pip install "ibverbs[gpunetio-cutedsl]"
```

Build the device library once for the target architecture. This requires
`clang++`, CUDA headers, and `doca-sdk-gpunetio-devel`:

```python
from ibverbs.gpunetio import build_bitcode

build_bitcode(arch="sm_90")  # cached under ~/.cache/rdma4py/gpunetio/
```

Connect and register memory before exporting the QP. The QP and both CQs must
be fresh, with no posted work or completions. Export permanently gives their
consumer and producer state to the external data path; do not call
`post_send`, `post_recv`, or `poll` on those objects afterward. Make the target
GPU's CUDA context current before export and keep it current while closing the
device handle. Synchronize all kernels before `DeviceQP.close()`.

```python
from ibverbs.gpunetio import DeviceQP

device_qp = DeviceQP.export(qp, gpu=0)  # requires a direct GPU doorbell
qp_ptr = device_qp.device_ptr
```

Triton functions accept integer device addresses and keys. Cast scalar kernel
arguments to the exact unsigned widths shown here:

```python
import triton
import triton.language as tl
from ibverbs.gpunetio import triton as gda

@triton.jit
def write_kernel(qp, remote, rkey, local, lkey, length, status):
    ticket = gda.put(
        tl.cast(qp, tl.uint64), tl.cast(remote, tl.uint64),
        tl.cast(rkey, tl.uint32), tl.cast(local, tl.uint64),
        tl.cast(lkey, tl.uint32), tl.cast(length, tl.uint64))
    tl.store(status, gda.wait_send(tl.cast(qp, tl.uint64), ticket))

write_kernel[(1,)](
    qp_ptr, peer_addr, peer_rkey, local_addr, local_lkey, nbytes, status,
    num_warps=1, extern_libs=gda.external_libraries())
```

CuTe DSL binds the same functions directly through its bitcode FFI:

```python
from cutlass import Uint32, Uint64, cute
from ibverbs.gpunetio.cutedsl import bind

gda = bind()

@cute.kernel
def write_kernel(qp: Uint64, remote: Uint64, rkey: Uint32,
                 local: Uint64, lkey: Uint32, length: Uint64):
    ticket = gda.put(qp, remote, rkey, local, lkey, length)
    gda.wait_send(qp, ticket)
```

The initial ABI includes RDMA Write/Read, Send/Receive, blocking completion
waits, and one-shot completion tests. `get_mcst` and `wait_recv_mcst` add the
DOCA dump-WQE memory-consistency sequence required on pre-Hopper GPUs; their
dump address must name at least one registered writable byte. Transfer lengths
must be positive, and a Receive is limited to `2**32 - 1` bytes.

The wrapper is deliberately hardware-specific:

| Component | Supported target |
|-----------|------------------|
| GPU | NVIDIA SM80 or newer data-center GPUs; SM90/Hopper tested |
| NIC | ConnectX-6 Dx or newer; BlueField-2/3 in NIC mode |
| Transport | mlx5 RC QPs, fixed 64-byte SQ/CQE and 16-byte RQ layout, no SRQ |
| Software | DOCA GPUNetIO 3.4 bridge ABI and a matching rdma-core/libmlx5 |

Ampere uses the `get_mcst` / `wait_recv_mcst` variants for inbound visibility;
Hopper and newer can use `get` / `wait_recv`. The bridge relies on mlx5 direct
verbs, CUDA host registration of provider queue memory, and NVIDIA's MMIO
doorbell mapping. It is not a portable implementation for non-mlx5 NICs or
AMD/Intel GPUs. GPUNetIO Verbs and its bridge API are experimental DOCA APIs,
so a new DOCA release may require an ABI update here.

`DeviceQP.export(..., nic_handler="auto")` permits DOCA's CPU-proxy fallback
and exposes `DeviceQP.progress()`, but the default `"gpu"` mode fails rather
than silently putting the CPU back on the critical path. CPU-proxy mode needs a
host thread to call `progress()` while a kernel can be posting work. The GPU and
NIC should also have a GPUDirect-friendly PCIe path for useful performance.

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
| RDMA connection manager | ✅ `CMID.resolve/create_qp/connect/disconnect` |
| GPU-initiated mlx5 data path | ✅ optional DOCA GPUNetIO bridge for Triton / CuTe DSL |

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
Set `RDMA4PY_SKIP_HARDWARE_TESTS=1` to force hardware-dependent tests to skip.

## License

BSD-3-Clause. See the repository `LICENSE`.
