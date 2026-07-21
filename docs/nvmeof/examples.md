# NVMe-oF examples and best practices

`nvmeof` is a userspace NVMe/RDMA initiator. NVMe keyed SGL descriptors carry
the registered address and `rkey`, allowing the target to place data directly
in host or CUDA memory:

- An NVMe read makes the target RDMA-write storage data into the initiator MR.
- An NVMe write makes the target RDMA-read data from the initiator MR.

The target must advertise keyed SGL support. These examples assume a reachable
NVMe/RDMA target and a namespace that the initiator is authorized to access.

## Read directly into a CUDA tensor

```python
import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import torch
import nvmeof

target = os.environ["NVME4PY_TARGET"]
subsystem_nqn = os.environ["NVME4PY_SUBSYSTEM_NQN"]
source = os.getenv("NVME4PY_SOURCE")
nsid = int(os.getenv("NVME4PY_NSID", "1"))

torch.cuda.set_device(0)
gpu = torch.device("cuda", 0)

with nvmeof.Controller.connect(
    target,
    subsystem_nqn,
    source=source,
) as controller:
    namespace = controller.namespace(nsid)
    tensor = torch.empty(namespace.lba_size, dtype=torch.uint8, device=gpu)

    with controller.register_gpu(tensor) as mr:
        with torch.cuda.device(gpu):
            namespace.read(mr, slba=0, blocks=1)

    print(tensor[:16].cpu())
```

`Namespace.read` waits for the NVMe response and flushes completed GPUDirect
writes before returning. Keep the destination CUDA context current around the
call.

## Write from CUDA memory

:::{warning}
An NVMe write changes namespace contents. Use only a disposable namespace and
an LBA range explicitly reserved for the application.
:::

```python
slba = 1024
tensor = torch.arange(
    namespace.lba_size,
    dtype=torch.int64,
    device=gpu,
).remainder(251).to(torch.uint8)

with controller.register_gpu(tensor) as mr:
    with torch.cuda.device(gpu):
        namespace.write(mr, slba=slba, blocks=1)
        namespace.flush()
```

The high-level write synchronizes CUDA before the target reads the source MR.
The command response means the write command completed, not necessarily that
the media is persistent. Issue `Namespace.flush()` when persistence matters.

## Asynchronous command submission

The high-level namespace methods are synchronous. `Controller.io` exposes the
underlying I/O queue when an application needs multiple outstanding commands:

```python
from nvmeof import protocol

command = protocol.rw_command(
    protocol.OPC_READ,
    namespace.nsid,
    slba=0,
    blocks=1,
    buffer=mr,
    lba_size=namespace.lba_size,
)
request = controller.io.submit(command, data_owner=mr)

while not request.done:
    controller.io.poll()

request.response.raise_for_status()
```

Passing `data_owner=mr` retains the keyed-SGL memory while the request is in
flight. A request becomes done only after both the command-capsule SEND and the
matching NVMe response have completed.

## Multiple I/O QPs

The current high-level `Controller` creates one admin QP and one I/O QP. The
benchmark's multi-QP mode creates independent controller lanes, each with its
own admin connection, I/O QP, PD, MR, and GPU tensor. It then assigns disjoint
LBA ranges to the I/O QPs and polls them round-robin.

The following condensed example submits one direct GPU read per lane:

```python
import ibverbs.cuda as ibcuda
from nvmeof import protocol

num_qps = 4
depth_per_qp = 8
blocks_per_lane = 256
lanes = []

try:
    for lane_index in range(num_qps):
        # RDMAQueue makes depth - 1 command IDs available.
        controller = nvmeof.Controller.connect(
            target,
            subsystem_nqn,
            source=source,
            queue_depth=depth_per_qp + 1,
        )
        namespace = controller.namespace(nsid)
        blocks = min(
            blocks_per_lane,
            controller.max_transfer_bytes // namespace.lba_size,
        )
        length = blocks * namespace.lba_size
        tensor = torch.empty(length, dtype=torch.uint8, device=gpu)
        mr = controller.register_gpu(tensor)
        command = protocol.rw_command(
            protocol.OPC_READ,
            namespace.nsid,
            slba=lane_index * blocks_per_lane,
            blocks=blocks,
            buffer=mr,
            lba_size=namespace.lba_size,
        )
        request = controller.io.submit(command, data_owner=mr)
        lanes.append((controller, mr, tensor, request))

    while any(not request.done for _, _, _, request in lanes):
        for controller, _, _, request in lanes:
            if not request.done:
                controller.io.poll()

    for _, _, _, request in lanes:
        request.response.raise_for_status()

    with torch.cuda.device(gpu):
        ibcuda.flush_gpudirect_writes()
finally:
    for controller, mr, _, _ in reversed(lanes):
        mr.close()
        controller.close()
```

This is lane-level orchestration, not multiple `qid` values under one
controller. It is useful for throughput experiments but has higher setup and
registration cost. Reuse lanes for steady-state traffic instead of reconnecting
for every transfer.

## Splitting and signalling

- A keyed SGL contains the target-visible address, byte count, and `rkey`.
- The target performs the one-sided RDMA data movement and returns an NVMe
  completion capsule.
- `RDMAQueue.submit` signals its command-capsule SEND. `RDMAQueue.poll` waits
  for both that local completion and the target's NVMe response.
- A successful write response does not replace an NVMe Flush command.
- `Namespace.read` and `Namespace.write` split logical transfers at the
  controller's MDTS and the keyed-SGL length limit. The current high-level path
  submits those split commands serially on its one I/O queue.
- Multi-QP applications must divide LBA and buffer ranges themselves and wait
  for every lane's response.

## Checklist

- Bind `source` to an initiator address on the HCA nearest the selected GPU.
  RDMA-CM route selection otherwise may choose a cross-socket path.
- Keep the CUDA context owning the tensor current while registering and issuing
  GPU I/O.
- Register long-lived tensors once per controller PD and close every MR before
  closing its controller.
- Size queue depth for the intended concurrency. At most `depth - 1` commands
  are outstanding on an `RDMAQueue`.
- Align I/O sizes and offsets to `Namespace.lba_size`.
- Expect transfers larger than MDTS to become multiple NVMe commands.
- Poll every active lane; submitting work without progressing its CQ will
  eventually exhaust command IDs and receive credits.
- Treat storage writes as destructive and use a dedicated test namespace for
  benchmarks.
- Use `Namespace.flush()` when durability, rather than command completion, is
  the required boundary.
