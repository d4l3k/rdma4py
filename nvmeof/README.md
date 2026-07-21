# nvmeof

`nvmeof` is a userspace NVMe over Fabrics RDMA initiator for Python. It is a
separate package layered on
[`ibverbs`](https://github.com/d4l3k/rdma4py/tree/main/ibverbs) because NVMe controller and
namespace policy do not belong in low-level verbs bindings.

The data path uses NVMe/RDMA keyed SGL descriptors that point at an existing
ibverbs MR. The target transfers blocks directly to or from that MR, including
CUDA memory registered by `ibverbs.cuda`; there is no host staging buffer in
the NVMe-to-RDMA-to-GPU path.

## Why this does not bundle SPDK

SPDK is a strong choice for a managed native storage service, but it includes
an environment abstraction, hugepage/driver setup, native plugins, and a large
dependency graph. Embedding it would not produce a self-contained, portable
`pip`/manylinux wheel. This package instead implements the small NVMe/RDMA host
protocol directly over the existing portable `ibverbs` ABI3 extension.

`nvmeof` itself is a pure-Python `py3-none-any` wheel. Its `ibverbs` dependency
is an ABI3 manylinux wheel that `dlopen`s the host's `libibverbs.so.1` and,
only when NVMe/RDMA is used, `librdmacm.so.1`.

## Requirements

- Linux, an RDMA NIC, and a reachable NVMe/RDMA target.
- `libibverbs.so.1`, `librdmacm.so.1`, and the matching RDMA provider.
- For GPU I/O: CUDA, an HCA with GPUDirect RDMA, and dma-buf support or the
  `nvidia_peermem` module.
- The target must support keyed SGL data block descriptors. Separate namespace
  metadata and DH-HMAC-CHAP are not currently supported.

On Debian/Ubuntu the runtime packages are typically:

```bash
sudo apt-get install libibverbs1 librdmacm1 ibverbs-providers
python -m pip install nvmeof
```

## Controller options

`Controller.connect(host, subsystem_nqn, **options)` and the direct
`Controller(...)` constructor accept the same options:

| Option | Default | Meaning |
| --- | --- | --- |
| `port` | `4420` | Target NVMe/RDMA service port. |
| `host_id` | random UUID | Initiator host identifier. Supply a stable UUID when target policy depends on host identity. |
| `host_nqn` | derived from `host_id` | Initiator NQN sent in Fabrics Connect. |
| `queue_depth` | `128` | Requested I/O depth from 2 through 256; the controller may negotiate it downward. |
| `keep_alive_ms` | `0` | Keep-alive timeout. Non-zero values are currently unsupported because there is no background command worker. |
| `timeout` | `30.0` | Per-command timeout in seconds. |
| `source` | `None` | Local initiator IP used to bind RDMA-CM and select the HCA. |

## Host memory

```python
import nvmeof

with nvmeof.Controller.connect(
    "192.0.2.20",
    "nqn.2026-07.io.example:storage",
    source="192.0.2.21",
) as controller:
    namespace = controller.namespace(1)
    with controller.allocate(128 * 4096) as buffer:
        namespace.read(buffer, slba=0, blocks=128)
        payload = buffer.read()
```

`controller.register(array_or_cpu_tensor)` registers an existing contiguous
host allocation instead.

## Direct GPU memory

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import nvmeof

tensor = torch.empty(128 * 4096, dtype=torch.uint8, device="cuda:0")

with nvmeof.Controller.connect(
    "192.0.2.20", "nqn.2026-07.io.example:storage"
) as controller:
    namespace = controller.namespace(1)
    with controller.register_gpu(tensor) as gpu_mr:
        # Target NVMe -> target RDMA WRITE -> local GPU memory.
        # Namespace.read flushes completed GPUDirect writes before returning.
        with torch.cuda.device(tensor.device):
            namespace.read(gpu_mr, slba=0, blocks=128)

        # Local GPU memory -> target RDMA READ -> target NVMe.
        # Namespace.write synchronizes the current CUDA context first.
        with torch.cuda.device(tensor.device):
            namespace.write(gpu_mr, slba=1024, blocks=128)
            namespace.flush()
```

The CUDA context that owns the tensor must be current while issuing GPU I/O.
The registered MR must be closed before its controller.

`source` is optional. Set it to an address on the intended initiator HCA when
the host has multiple RDMA NICs. This makes an NVMe -> target NIC -> initiator
NIC -> GPU route explicit and prevents route selection from silently choosing
an HCA with poor GPU PCIe locality.

Reads and writes larger than the target's MDTS are split into multiple NVMe
commands while retaining direct placement in the same registered buffer. The
target used for the repository benchmarks advertises a 1 MiB MDTS; 4, 16, and
64 MiB GPU tensors were verified end to end.

## Scope

The first implementation provides dynamic controller connection, controller
enablement, Identify Controller/Namespace, one I/O queue, synchronous and
low-level asynchronous command submission, block read/write splitting, and
flush. It deliberately does not claim filesystem semantics, multipath,
reconnect, keep-alive, protection information, authentication, or target
functionality. KATO is therefore negotiated as zero.

Real-target tests use these environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NVME4PY_TARGET` | unset | Target hostname or IP; required to enable integration tests. |
| `NVME4PY_SUBSYSTEM_NQN` | unset | Target subsystem NQN; required to enable integration tests. |
| `NVME4PY_SOURCE` | unset | Optional local initiator IP/HCA binding. |
| `NVME4PY_NSID` | `1` | Namespace ID used by integration tests. |
| `NVME4PY_DESTRUCTIVE` | unset | Must equal `1` to enable the write/read GPU round trip. Use only with a disposable namespace. |
| `NVME4PY_TEST_SLBA` | `0` | Starting LBA overwritten by the destructive test. |

Measured multi-QP bandwidth, latency, QP scaling, and PCIe locality results
are in [`BENCHMARKS.md`](BENCHMARKS.md).
