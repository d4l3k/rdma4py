# rdma4py

Low-level Python bindings for high-performance RDMA and direct GPU networking.
The repository contains two independently installable packages.

## EFA

[`efa`](efa/index) provides AWS Elastic Fabric Adapter bindings for SRD,
one-sided RDMA, EFA direct-verbs metadata, and torch-friendly GPUDirect
registration.

```bash
pip install efa
```

## ibverbs

[`ibverbs`](ibverbs/index) provides general libibverbs bindings for RC, UC,
and UD transports, including queue pairs, completion queues, shared receive
queues, atomics, and GPUDirect registration.

```bash
pip install ibverbs
```

Both packages use Cython for inline provider dispatch, release the GIL on data
path operations, load RDMA libraries at runtime, and publish one Linux `abi3`
wheel for CPython 3.9 and newer.

```{toctree}
:maxdepth: 2
:hidden:

efa/index
efa/api
ibverbs/index
ibverbs/api
```
