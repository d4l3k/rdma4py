# EFA

`efa` is a low-level wrapper for AWS Elastic Fabric Adapter, libibverbs, and
the EFA direct-verbs API. It supports reliable connectionless SRD traffic,
one-sided RDMA, and direct transfers involving CUDA tensors.

## Setup

The process needs `libibverbs.so.1`, `libefa.so.1`, and access to the host's
`/dev/infiniband/uverbs*` devices.

```python
import efa

device = efa.get_efa_device_list()[0]
context = device.open()
pd = context.alloc_pd()
cq = context.create_cq(256)
qp = pd.create_qp(
    efa.QPInitAttr(send_cq=cq, recv_cq=cq)
).prepare(qkey=0x1234)
```

Exchange `efa.local_endpoint_info(qp, qkey=0x1234).to_bytes()` out of band.
Resolve the bytes received from the other process with
`efa.EndpointInfo.from_bytes(...).peer(pd)`.

## EFA peer rule

Each SEND names a destination address handle, QP number, and qkey. For RDMA
read or write, the responder must also have created an address handle for the
requester. Applications should resolve endpoint information in both
directions even when data currently flows only one way.

## Torch and GPUDirect

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch initializes
CUDA, then register a tensor without introducing a torch dependency into the
package:

```python
import efa.cuda

gpu_mr = efa.cuda.register_tensor(pd, tensor, access)
```

Synchronize CUDA work that produced an outbound tensor before posting the
network operation. After an inbound completion or write notification, call
`efa.cuda.flush_gpudirect_writes()` in the destination CUDA context before
launching work that consumes the tensor.

The full installation, SRD, one-sided RDMA, and GPUDirect guide is maintained
in the [EFA package README](https://github.com/d4l3k/rdma4py/tree/main/efa).

For unsolicited RDMA write-with-immediate notifications, enable
`unsolicited_write_recv` on every communicating QP and `unsolicited` on the
extended receive CQ. EFA rejects peers that negotiate different QP feature
sets.
