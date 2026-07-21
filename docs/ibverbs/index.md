# ibverbs

`ibverbs` is a general low-level wrapper for Linux RDMA verbs. It exposes
devices, protection domains, memory regions, completion queues, queue pairs,
shared receive queues, address handles, work requests, completions, and async
events.

## Setup

```python
import ibverbs as ib

device = ib.get_device_list()[0]
context = device.open()
pd = context.alloc_pd()
cq = context.create_cq(256)
qp = pd.create_qp(
    ib.QPInitAttr(send_cq=cq, recv_cq=cq, qp_type=ib.QPType.RC)
)
```

Use `ib.local_qp_info` and `ib.connect_rc` to exchange and apply the remote RC
path information. The raw `QP.modify`, `QP.to_init`, `QP.to_rtr`, and
`QP.to_rts` methods remain available when an application needs direct control.

## Configuration and feature flags

`QPInitAttr` controls transport type, CQs, optional SRQ, queue capacities, SGE
and inline limits, and send signaling. The QP transition helpers document path
MTU, access, retry, timeout, atomic-depth, and route options. `CMID.connect`
separately exposes RDMA-CM private data, initiator/responder depth, and retry
controls.

The [API reference](api) documents every option and each member of
`AccessFlags`, `QPAttrMask`, `SendFlags`, and `WCFlags`. Optional access modes
and transports remain subject to provider and device support.

## Memory and GPU registration

`ib.reg_tensor` registers contiguous NumPy arrays and CPU torch tensors while
retaining their allocations. `ibverbs.cuda.register_tensor` exports CUDA
memory as dma-buf when available and falls back to `nvidia_peermem`.

The full transport, API coverage, and GPUDirect guide is maintained in the
[ibverbs package README](https://github.com/d4l3k/rdma4py/tree/main/ibverbs).
