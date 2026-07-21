# EFA examples and best practices

EFA uses reliable, connectionless SRD queue pairs. Applications exchange a
small endpoint description and create an address handle instead of driving an
RC QP through a connection handshake. Every work request names its destination
with `dest=peer`.

## SRD bootstrap with torchrun

The following fragment creates one QP per rank and exchanges endpoint metadata
through a Gloo control plane:

```python
import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import torch
import torch.distributed as dist
import efa
import efa.cuda as efa_cuda

dist.init_process_group("gloo")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
gpu = torch.device("cuda", local_rank)
peer_rank = 1 - rank

qkey = 0x1234
port = int(os.getenv("RDMA_PORT", "1"))
gid_index = int(os.getenv("RDMA_GID_INDEX", "0"))

device = efa.get_efa_device_list()[0]
context = device.open()
pd = context.alloc_pd()
cq = context.create_cq(32)
qp = pd.create_qp(
    efa.QPInitAttr(
        send_cq=cq,
        recv_cq=cq,
        max_send_wr=16,
        max_recv_wr=16,
    )
).prepare(qkey=qkey)

local_info = efa.local_endpoint_info(
    qp,
    qkey=qkey,
    gid_index=gid_index,
    port=port,
)
local_wire = torch.tensor(list(local_info.to_bytes()), dtype=torch.uint8)
wires = [torch.empty_like(local_wire) for _ in range(world_size)]
dist.all_gather(wires, local_wire)
remote_info = efa.EndpointInfo.from_bytes(bytes(wires[peer_rank].tolist()))
peer = remote_info.peer(pd, sgid_index=gid_index, port=port)
```

Resolve `peer` on **both** ranks, even for one-way data. EFA responders require
a reverse address handle before accepting RDMA reads or writes from that peer.
Close `peer` before its protection domain.

## One-sided GPU write with notification

This is the one-QP data path. Rank 0 writes a CUDA tensor into rank 1's
registered memory and attaches an immediate value for receiver notification.

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
mr_access = efa.AccessFlags.LOCAL_WRITE
if rank == 1:
    mr_access |= efa.AccessFlags.REMOTE_WRITE

with efa_cuda.register_tensor(pd, tensor, mr_access) as mr:
    target = torch.zeros(2, dtype=torch.int64)
    if rank == 1:
        target[0] = mr.addr
        target[1] = mr.rkey
        qp.post_recv(efa.RecvWR(wr_id=1, sg_list=[mr.sge()]))

    # Rank 1 publishes its address/rkey after posting the notification receive.
    dist.broadcast(target, src=1)

    if rank == 0:
        torch.cuda.current_stream(gpu).synchronize()
        qp.post_send(
            efa.SendWR(
                wr_id=2,
                sg_list=[mr.sge()],
                opcode=efa.WROpcode.RDMA_WRITE_WITH_IMM,
                send_flags=efa.SendFlags.SIGNALED,
                remote_addr=int(target[0]),
                rkey=int(target[1]),
                imm_data=WRITE_COMPLETE,
                dest=peer,
            )
        )
        wait(cq)
    else:
        notification = wait(cq)
        assert notification.opcode == efa.WCOpcode.RECV_RDMA_WITH_IMM
        assert notification.wc_flags & efa.WCFlags.WITH_IMM
        assert notification.imm_data == WRITE_COMPLETE
        with torch.cuda.device(gpu):
            efa_cuda.flush_gpudirect_writes()
```

The receive SGE is not the write payload path; the remote address and `rkey`
place the data. The receive work request supplies the notification credit.

## One-sided GPU read

Rank 0 can instead pull a tensor exposed by rank 1. No receive work request is
needed, and rank 1 receives no verbs completion for the read.

```python
tensor = (
    torch.zeros(1024, dtype=torch.int32, device=gpu)
    if rank == 0
    else torch.arange(1024, dtype=torch.int32, device=gpu)
)
mr_access = efa.AccessFlags.LOCAL_WRITE
if rank == 1:
    mr_access |= efa.AccessFlags.REMOTE_READ

with efa_cuda.register_tensor(pd, tensor, mr_access) as mr:
    source = torch.zeros(2, dtype=torch.int64)
    if rank == 1:
        torch.cuda.current_stream(gpu).synchronize()
        source[0] = mr.addr
        source[1] = mr.rkey

    dist.broadcast(source, src=1)

    if rank == 0:
        qp.post_send(
            efa.SendWR(
                wr_id=3,
                sg_list=[mr.sge()],
                opcode=efa.WROpcode.RDMA_READ,
                send_flags=efa.SendFlags.SIGNALED,
                remote_addr=int(source[0]),
                rkey=int(source[1]),
                dest=peer,
            )
        )
        wait(cq)
        with torch.cuda.device(gpu):
            efa_cuda.flush_gpudirect_writes()
```

Use a control-plane message if rank 1 needs to know that rank 0 has finished
reading its memory.

## QPs and physical lanes

EFA has two independent scaling dimensions:

| Dimension | Resources | Purpose |
| --- | --- | --- |
| More QPs on one EFA | Shared context, PD, CQ, and MR | More outstanding work, especially for small transfers |
| More physical EFA lanes | One context, PD, CQ, peer, and MR registration per EFA | Aggregate bandwidth across physical links |

Within one lane, divide the tensor into disjoint regions and post work to every
QP before polling completions:

```python
for qp_index, lane_qp in enumerate(qps):
    offset = qp_index * chunk_size
    lane_qp.post_send(
        efa.SendWR(
            wr_id=qp_index,
            sg_list=[lane_mr.sge(chunk_size, offset=offset)],
            opcode=efa.WROpcode.RDMA_WRITE,
            send_flags=efa.SendFlags.SIGNALED,
            remote_addr=remote_addr + offset,
            rkey=remote_rkey,
            dest=peers[qp_index],
        )
    )

# Drain one successful completion for each posted request.
...
```

For multiple physical lanes, repeat that resource set for each selected EFA.
Register the shared torch allocation in every lane's PD, publish the address
and lane-specific `rkey`, post all lanes, and only then begin polling. Pair
each GPU with PCIe-local EFAs instead of selecting devices by enumeration
order.

One QP is generally enough to saturate a lane for large messages. Multiple QPs
primarily improve the number of small operations in flight; multiple physical
lanes are what increase the large-transfer bandwidth ceiling.

## Signalling choices

- Every EFA RDMA read/write work request should be signaled. Drain successful
  sender completions before reusing local ranges.
- A plain `RDMA_WRITE` produces no receiver completion.
- `RDMA_WRITE_WITH_IMM` notifies the receiver after consuming a posted receive.
- Unsolicited write-with-immediate removes that posted-receive requirement, but
  every peer must enable `unsolicited_write_recv` and use an extended CQ with
  unsolicited metadata enabled.
- An RDMA read produces a completion only on the requester.

## Checklist

- Query `Context.query_efa_device()` and require the matching
  `EfaDeviceCaps.RDMA_WRITE` or `RDMA_READ` bit before selecting one-sided I/O.
- Split a transfer at `EfaDeviceAttr.max_rdma_size`; the `write_wrs` and
  `read_wrs` helpers perform this chunking.
- Resolve endpoint metadata in both directions before one-sided traffic.
- Register long-lived buffers once per PD and keep the owning tensor alive.
- Synchronize the CUDA stream that produced an outbound buffer before the NIC
  reads it.
- Flush inbound GPUDirect writes in the destination CUDA context after the
  matching completion or notification.
- Check every completion with `raise_for_status()` and retain vendor status in
  error logs.
- Keep each GPU, EFA, and polling CPU within the same PCIe/NUMA domain when
  possible.

Launch a local two-rank smoke test with:

```bash
torchrun --standalone --nproc-per-node=2 example.py
```

Production EFA traffic is normally cross-host; use one rank per host and the
standard multi-node `torchrun` rendezvous options.
