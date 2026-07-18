# rdma4py

High-performance RDMA for Python.

Documentation for both packages is published at
[d4l3k.github.io/rdma4py](https://d4l3k.github.io/rdma4py/).

## [`efa/`](efa/) - AWS EFA and SRD bindings

Low-level bindings for AWS Elastic Fabric Adapter, including SRD SEND and
one-sided RDMA, EFA direct-verbs queries, and torch-friendly GPUDirect memory
registration. No Python runtime dependencies and no torch/CUDA linkage.

See [`efa/README.md`](efa/README.md) for setup, API examples, EFA peer rules,
and hardware testing.

```bash
pip install ./efa
```

## [`ibverbs/`](ibverbs/) — low-level libibverbs bindings

Pythonic, Cython-based bindings to `libibverbs` intended as the foundation for
building high-performance RDMA libraries in Python, with first-class
**GPUDirect** support (register a GPU device pointer or an exported dma-buf fd).
No runtime dependencies, no torch/CUDA linkage.

See [`ibverbs/README.md`](ibverbs/README.md) for the API, quickstart, feature
coverage, and testing instructions.

```bash
pip install "Cython>=3.0" "setuptools>=64" wheel
pip install ./ibverbs
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
