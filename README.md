# rdma4py

High-performance RDMA for Python.

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
