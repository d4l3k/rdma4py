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

GPU memory is registered with `controller.register_gpu(tensor)`. Make the
tensor's CUDA context current around I/O; writes synchronize CUDA before the
target reads the MR, and reads flush inbound GPUDirect writes before returning.
On multi-HCA hosts, `source` binds RDMA-CM to the initiator address selected for
the GPU's PCIe domain. Transfers larger than MDTS are split across commands
without staging the registered buffer through host memory.

The package is separate from `ibverbs`: verbs and RDMA-CM are general transport
mechanisms, while controller setup, NVMe commands, namespace geometry, and
storage error handling are a higher-level protocol.

See the [package README](https://github.com/d4l3k/rdma4py/tree/main/nvmeof)
for installation, target requirements, full examples, and current scope.
