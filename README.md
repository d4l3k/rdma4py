# rdma4py

[![efa on PyPI](https://img.shields.io/pypi/v/efa?label=efa)](https://pypi.org/project/efa/)
[![ibverbs on PyPI](https://img.shields.io/pypi/v/ibverbs?label=ibverbs)](https://pypi.org/project/ibverbs/)

High-performance RDMA for Python.

Documentation for both packages is published at
[d4l3k.github.io/rdma4py](https://d4l3k.github.io/rdma4py/).

## [`efa/`](efa/) - AWS EFA and SRD bindings

Low-level bindings for AWS Elastic Fabric Adapter, including SRD SEND and
one-sided RDMA, EFA direct-verbs queries, and torch-friendly GPUDirect memory
registration. No Python runtime dependencies and no torch/CUDA linkage.

Install the [EFA package from PyPI](https://pypi.org/project/efa/):

```bash
python -m pip install efa
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
python -m pip install ibverbs
```

See [`ibverbs/README.md`](ibverbs/README.md) for system setup, the API,
quickstart, feature coverage, source installation, and testing instructions.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
