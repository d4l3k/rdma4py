# ibverbs examples and best practices

These examples use two `torchrun` ranks and CUDA tensors. They show the verbs
operations that a transfer layer normally manages: QP bootstrap, memory-key
exchange, CUDA ordering, completion handling, and transfer striping. The
snippets focus on the data path; production code should also add timeouts and
propagate failures through its control plane.

## Two-rank RC bootstrap

Use Gloo only as the control plane. The payload moves through the RC QP, not
through `torch.distributed`.

```python
import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import torch
import torch.distributed as dist
import ibverbs as ib
import ibverbs.cuda as ibcuda

dist.init_process_group("gloo")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
if world_size != 2:
    raise RuntimeError("this example requires two ranks")

torch.cuda.set_device(local_rank)
gpu = torch.device("cuda", local_rank)
peer_rank = 1 - rank
port = int(os.getenv("RDMA_PORT", "1"))
gid_index = int(os.getenv("RDMA_GID_INDEX", "0"))

device = ib.get_device_list()[0]
context = device.open()
pd = context.alloc_pd()
cq = context.create_cq(32)
qp = pd.create_qp(
    ib.QPInitAttr(
        send_cq=cq,
        recv_cq=cq,
        qp_type=ib.QPType.RC,
        max_send_wr=16,
        max_recv_wr=16,
    )
)

port_attr = context.query_port(port)
gid = context.query_gid(port, gid_index)
local_info = ib.local_qp_info(qp, port_attr, gid, port=port)
local_wire = torch.tensor(list(local_info.to_bytes()), dtype=torch.uint8)
wires = [torch.empty_like(local_wire) for _ in range(world_size)]
dist.all_gather(wires, local_wire)
remote_info = ib.QPInfo.from_bytes(bytes(wires[peer_rank].tolist()))

# Enable only the incoming operations this rank is prepared to serve.
incoming_access = (
    ib.AccessFlags.REMOTE_WRITE | ib.AccessFlags.REMOTE_READ
    if rank == 1
    else 0
)
ib.connect_rc(
    qp,
    remote_info,
    port=port,
    sgid_index=gid_index,
    access=incoming_access,
)
```

Keep `context`, `pd`, `cq`, and `qp` alive until all work completes. Close
resources in reverse dependency order: MR, QP, CQ, PD, context.

## One-sided write with receiver notification

`RDMA_WRITE_WITH_IMM` places the payload through the destination address and
`rkey`. Its immediate value creates a receiver CQ entry, but that notification
still consumes one receive-queue entry. The posted receive has no payload SGE.

```python
WRITE_COMPLETE = 0xC0DE


def wait(cq):
    while True:
        completions = cq.poll(1)
        if completions:
            completions[0].raise_for_status()
            return completions[0]


tensor = (
    torch.arange(1024, dtype=torch.int32, device=gpu)
    if rank == 0
    else torch.zeros(1024, dtype=torch.int32, device=gpu)
)

mr_access = ib.AccessFlags.LOCAL_WRITE
if rank == 1:
    mr_access |= ib.AccessFlags.REMOTE_WRITE

with ibcuda.register_tensor(pd, tensor, mr_access) as mr:
    target = torch.zeros(2, dtype=torch.int64)
    if rank == 1:
        target[0] = mr.addr
        target[1] = mr.rkey
        qp.post_recv(ib.RecvWR(wr_id=1, sg_list=[]))

    # Rank 1 enters this collective only after its notification receive exists.
    dist.broadcast(target, src=1)

    if rank == 0:
        torch.cuda.current_stream(gpu).synchronize()
        qp.post_send(
            ib.SendWR(
                wr_id=2,
                sg_list=[mr.sge()],
                opcode=ib.WROpcode.RDMA_WRITE_WITH_IMM,
                send_flags=ib.SendFlags.SIGNALED,
                remote_addr=int(target[0]),
                rkey=int(target[1]),
                imm_data=WRITE_COMPLETE,
            )
        )
        wait(cq)
    else:
        notification = wait(cq)
        assert notification.opcode == ib.WCOpcode.RECV_RDMA_WITH_IMM
        assert notification.wc_flags & ib.WCFlags.WITH_IMM
        assert notification.imm_data == WRITE_COMPLETE
        with torch.cuda.device(gpu):
            ibcuda.flush_gpudirect_writes()
        assert torch.equal(
            tensor,
            torch.arange(1024, dtype=torch.int32, device=gpu),
        )
```

`SendFlags.SIGNALED` requests a completion only on rank 0. The immediate value
is what notifies rank 1. A plain `RDMA_WRITE` produces no receiver CQ entry.

## One-sided read

An RDMA read is initiated by the destination. Here rank 0 pulls rank 1's
tensor. Rank 1 synchronizes the stream that produced its source before
publishing the memory key. Rank 0 flushes inbound NIC writes before CUDA
consumes the local destination.

```python
tensor = (
    torch.zeros(1024, dtype=torch.int32, device=gpu)
    if rank == 0
    else torch.arange(1024, dtype=torch.int32, device=gpu)
)
mr_access = ib.AccessFlags.LOCAL_WRITE
if rank == 1:
    mr_access |= ib.AccessFlags.REMOTE_READ

with ibcuda.register_tensor(pd, tensor, mr_access) as mr:
    source = torch.zeros(2, dtype=torch.int64)
    if rank == 1:
        torch.cuda.current_stream(gpu).synchronize()
        source[0] = mr.addr
        source[1] = mr.rkey

    dist.broadcast(source, src=1)

    if rank == 0:
        qp.post_send(
            ib.SendWR(
                wr_id=3,
                sg_list=[mr.sge()],
                opcode=ib.WROpcode.RDMA_READ,
                send_flags=ib.SendFlags.SIGNALED,
                remote_addr=int(source[0]),
                rkey=int(source[1]),
            )
        )
        wait(cq)
        with torch.cuda.device(gpu):
            ibcuda.flush_gpudirect_writes()
```

The remote rank receives no verbs completion for an RDMA read. Use the control
plane if it needs notification after the requester finishes.

## Multiple QPs

Use multiple QPs to increase outstanding work or to reduce contention for
small transfers. A single large-transfer RC QP can already reach line rate on
many devices, so measure before adding QPs.

One CQ and one MR can be shared by QPs in the same protection domain. Exchange
one `QPInfo` per QP, connect matching indices, then assign disjoint buffer
ranges:

```python
from contextlib import ExitStack

num_qps = 4
chunk_size = tensor.numel() // num_qps

with ExitStack() as stack:
    cq = stack.enter_context(context.create_cq(num_qps * 4))
    qp_init = ib.QPInitAttr(
        send_cq=cq,
        recv_cq=cq,
        qp_type=ib.QPType.RC,
        max_send_wr=4,
        max_recv_wr=4,
    )
    qps = [
        stack.enter_context(pd.create_qp(qp_init))
        for _ in range(num_qps)
    ]

    # Exchange every QPInfo and connect local qps[i] to remote qps[i].
    ...

    if rank == 1:
        for index, lane_qp in enumerate(qps):
            lane_qp.post_recv(ib.RecvWR(wr_id=index, sg_list=[]))

    ...  # Exchange the destination MR address and rkey.

    if rank == 0:
        torch.cuda.current_stream(gpu).synchronize()
        for index, lane_qp in enumerate(qps):
            offset = index * chunk_size
            lane_qp.post_send(
                ib.SendWR(
                    wr_id=index,
                    sg_list=[mr.sge(chunk_size, offset=offset)],
                    opcode=ib.WROpcode.RDMA_WRITE_WITH_IMM,
                    send_flags=ib.SendFlags.SIGNALED,
                    remote_addr=remote_addr + offset,
                    rkey=remote_rkey,
                    imm_data=index,
                )
            )

    # Poll num_qps successful completions before reusing or consuming memory.
    ...
```

Ordering is guaranteed within one RC QP, not across QPs. Treat the transfer as
complete only after every range has completed.

## Completion and CUDA ordering

| Mechanism | Where it completes | Receiver requirement |
| --- | --- | --- |
| `SendFlags.SIGNALED` | Sender CQ | None |
| `RDMA_WRITE` | Sender CQ only | Out-of-band notification if needed |
| `RDMA_WRITE_WITH_IMM` | Sender and receiver CQs | One posted receive per notification |
| `RDMA_READ` | Requester CQ only | Publish readable address and `rkey` |
| `SEND` / `SEND_WITH_IMM` | Sender and receiver CQs | Posted receive containing payload SGEs |

## Checklist

- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing or
  initializing torch when using the dma-buf registration path.
- Register long-lived buffers once and reuse their MRs. Registration is a
  control-plane operation, not part of the steady-state data path.
- Synchronize only the CUDA stream that produced outbound data. Do not rely on
  MR registration or work-request posting to order CUDA and NIC operations.
- After an inbound GPU completion, call `flush_gpudirect_writes()` in the
  destination device context before launching CUDA work that reads the data.
- Keep tensors alive until their MRs close, and keep MRs alive until all work
  requests using their keys have completed.
- Use the minimum remote access flags needed by each rank.
- Check every completion with `raise_for_status()` and bound polling with a
  timeout in production code.
- Keep the GPU, HCA, and polling CPU on the same PCIe/NUMA domain when possible.
- Queue unsignaled ibverbs operations only when CQ accounting is explicit;
  periodically post a signaled operation so completed work can be reclaimed.

Launch a two-rank example with:

```bash
RDMA_GID_INDEX=3 torchrun --standalone --nproc-per-node=2 example.py
```

For two hosts, use the same `--master-addr`, `--master-port`, `--nnodes=2`, and
distinct `--node-rank` values on both hosts.
