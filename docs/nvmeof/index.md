# NVMe over Fabrics RDMA

`nvmeof` is a userspace NVMe/RDMA initiator layered on `ibverbs`. NVMe keyed
SGL descriptors address the caller's registered MR directly, so the same API
works with host memory and `ibverbs.cuda.GpuMR` without staging block data
through the CPU.

```python
import nvmeof

with nvmeof.Controller.connect(
    target_ip, subsystem_nqn, source=initiator_ip
) as controller:
    namespace = controller.namespace(1)
    with controller.allocate(4096) as block:
        namespace.read(block, slba=0, blocks=1)
```

## Controller options

`Controller.connect` and `Controller` accept `port` (default `4420`),
`host_id` (random UUID), `host_nqn` (derived from the host ID), `queue_depth`
(`128`, negotiated downward when necessary), `keep_alive_ms` (only `0` is
currently supported), `timeout` (`30.0` seconds), and an optional local
`source` IP/HCA binding. See the [API reference](api) for parameter and
lifecycle details.

GPU memory is registered with `controller.register_gpu(tensor)`. Make the
tensor's CUDA context current around I/O; writes synchronize CUDA before the
target reads the MR, and reads flush inbound GPUDirect writes before returning.
On multi-HCA hosts, `source` binds RDMA-CM to the initiator address selected for
the GPU's PCIe domain. Transfers larger than MDTS are split across commands
without staging the registered buffer through host memory.

The package is separate from `ibverbs`: verbs and RDMA-CM are general transport
mechanisms, while controller setup, NVMe commands, namespace geometry, and
storage error handling are a higher-level protocol.

## Asynchronous commands

`Controller.io` is the connected low-level `RDMAQueue`. Its `submit` method
assigns a command ID and returns a `Request`; call `poll` until the request is
done. A request completes only after both the command-capsule SEND and matching
NVMe response complete. Pass the registered buffer as `data_owner` so it stays
alive while its keyed SGL is in flight.

```python
from nvmeof import protocol

command = protocol.rw_command(
    protocol.OPC_READ,
    namespace.nsid,
    slba=0,
    blocks=1,
    buffer=buffer,
    lba_size=namespace.lba_size,
)
request = controller.io.submit(command, data_owner=buffer)
while not request.done:
    controller.io.poll()
request.response.raise_for_status()
```

The high-level `Namespace.read` and `Namespace.write` methods remain the
preferred interface: they validate ranges, split transfers at MDTS, and apply
the required CUDA ordering around GPU I/O.

See the [package README](https://github.com/d4l3k/rdma4py/tree/main/nvmeof)
for installation, target requirements, full examples, and current scope. The
[API reference](api) covers the controller, transport, and wire-layout helpers.
