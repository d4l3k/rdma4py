# rdma4py

[![efa on PyPI](https://img.shields.io/pypi/v/efa?label=efa)](https://pypi.org/project/efa/)
[![ibverbs on PyPI](https://img.shields.io/pypi/v/ibverbs?label=ibverbs)](https://pypi.org/project/ibverbs/)
[![nvmeof on PyPI](https://img.shields.io/pypi/v/nvmeof?label=nvmeof)](https://pypi.org/project/nvmeof/)

High-performance RDMA for Python.

Read the [PyTorch developer blog introduction to rdma4py](https://docs.pytorch.org/devlogs/distributed/2026-07-21-rdma4py/).

Documentation for all packages is published at
[d4l3k.github.io/rdma4py](https://d4l3k.github.io/rdma4py/).

## Benchmarks

See the [benchmark overview](BENCHMARKS.md) and the detailed backend reports:
[EFA multi-lane](efa/BENCHMARKS.md),
[ibverbs multi-QP](ibverbs/BENCHMARKS.md), and
[NVMe-oF/RDMA to GPU](nvmeof/BENCHMARKS.md).

![Measured bandwidth by message size](benchmarks/bandwidth.svg)

![Measured latency by message size](benchmarks/latency.svg)

## [`efa/`](efa/) - AWS EFA and SRD bindings

Low-level bindings for AWS Elastic Fabric Adapter, including SRD SEND and
one-sided RDMA, EFA direct-verbs queries, and torch-friendly GPUDirect memory
registration. No Python runtime dependencies and no torch/CUDA linkage.

Install the [EFA package from PyPI](https://pypi.org/project/efa/):

```bash
pip install efa
```

See [`efa/README.md`](efa/README.md) for system setup, API examples, EFA peer
rules, GPUDirect usage, source installation, and hardware testing.

## [`ibverbs/`](ibverbs/) — low-level libibverbs bindings

Pythonic, Cython-based bindings to `libibverbs` intended as the foundation for
building high-performance RDMA libraries in Python, with first-class
**GPUDirect** support (register a GPU device pointer or an exported dma-buf fd).
No runtime dependencies, no torch/CUDA linkage.

Install the [ibverbs package from PyPI](https://pypi.org/project/ibverbs/):

```bash
pip install ibverbs
```

See [`ibverbs/README.md`](ibverbs/README.md) for system setup, the API,
quickstart, feature coverage, source installation, and testing instructions.

## [`nvmeof/`](nvmeof/) - NVMe over Fabrics RDMA initiator

Userspace NVMe/RDMA controller and namespace access layered on `ibverbs`, with
keyed SGL data placement directly into registered host or CUDA GPU memory. It
is shipped separately because NVMe protocol policy is above the verbs layer.

```bash
pip install nvmeof
```

See [`nvmeof/README.md`](nvmeof/README.md) for target requirements, host and
GPU examples, ordering rules, and current protocol scope.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
