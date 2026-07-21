# GPUDirect RDMA benchmarks

Measured on 2026-07-21 with the `ibverbs` GPUDirect dma-buf path and torch
CUDA tensors. These are intra-host transfers through two physical ConnectX-7
ports on different NUMA domains, not CUDA copies over NVLink.

## Topology

Each GPU is paired with its nearest NIC. `nvidia-smi topo -m` reports `PIX`
between each GPU/NIC pair:

```text
NUMA 0                                              NUMA 1
GPU 0 (0000:1b:00.0)                               GPU 4 (0000:a3:00.0)
        | PIX, PCIe 5.0 x16                               | PIX, PCIe 5.0 x16
mlx5_0 (0000:18:00.0) ===== 400 Gb/s RoCE ===== mlx5_6 (0000:9a:00.0)
```

The host has two Intel Xeon Platinum 8480C sockets. GPU 0 and `mlx5_0` are on
NUMA node 0; GPU 4 and `mlx5_6` are on NUMA node 1. Both PCIe links negotiated
32.0 GT/s x16. The benchmark process was pinned to CPU 0 in the source HCA's
NUMA node. Although the two GPUs also have an NV18 path, it is not used by
these RDMA writes.

| Component | Version / configuration |
|---|---|
| GPUs | 2x NVIDIA H100 80GB HBM3 |
| GPU driver | 580.82.07 |
| HCAs | 2x ConnectX-7, firmware 28.38.1002 |
| HCA ports | Ethernet, RoCE v2, 400 Gb/s, MTU 4096 |
| HCA PCIe | 32.0 GT/s x16 on both endpoints |
| Kernel | Linux 6.16.1 |
| libibverbs | 61.0 |
| Python | 3.14.5 |
| torch / CUDA | torch 2.13.0+cu130 / CUDA 13.0 |
| Source revision | `823a56452630ffc0a683e05ce654985430318652` |

## Method

- Source and destination are separate contiguous `torch.uint8` CUDA tensors.
  Each tensor is registered against its local HCA's protection domain with
  `ibverbs.cuda.register_tensor` and the dma-buf path.
- The data operation is a one-way RC `RDMA_WRITE` from GPU 0 to GPU 4. Source
  CUDA work is synchronized before timing. GPUDirect writes are flushed and
  the destination pattern is checked after timing.
- Latency is the median and p99 host time from posting one signaled write to
  polling its successful sender completion. Writes are serialized on one QP.
  It includes Python/Cython posting and polling overhead, but not a CUDA flush
  or an application-level ping-pong.
- Bandwidth uses 16 RC QPs with queue depth 64. QPs write non-overlapping
  tensor regions; only the last request in each 64-write batch is signaled.
  The result is useful payload bytes divided by host wall time, reported as
  decimal GB/s and Gb/s. Each value is the median of five runs.
- Each size is warmed up first. Latency uses 10,000 iterations for small
  messages and at least 100 for large messages. Bandwidth targets 8 GiB per
  run, caps small-message runs at 1,048,576 writes, and uses at least four
  full batches per QP.

## Results

| Message size | p50 latency (us) | p99 latency (us) | Bandwidth (GB/s) | Bandwidth (Gb/s) |
|---:|---:|---:|---:|---:|
| 8 B | 5.022 | 6.705 | 0.114 | 0.91 |
| 64 B | 5.035 | 5.753 | 0.912 | 7.30 |
| 1 KiB | 5.135 | 5.897 | 14.026 | 112.21 |
| 4 KiB | 5.364 | 5.995 | 42.347 | 338.77 |
| 16 KiB | 5.842 | 6.464 | 46.910 | 375.28 |
| 64 KiB | 6.844 | 7.667 | 48.126 | 385.01 |
| 256 KiB | 13.046 | 14.368 | 48.470 | 387.76 |
| 1 MiB | 29.260 | 29.583 | **48.549** | **388.39** |
| 4 MiB | 94.247 | 95.107 | 48.525 | 388.20 |
| 16 MiB | 353.884 | 354.902 | 48.459 | 387.67 |
| 64 MiB | 1390.150 | 1594.362 | 48.433 | 387.47 |

Peak useful bandwidth is 48.55 GB/s, or 388.39 Gb/s. That is 97.1% of the
nominal 400 Gb/s port rate. Bandwidth is effectively saturated from 64 KiB
through 64 MiB.

### QP scaling

The full size sweep uses 16 QPs because it produced the highest median at
1 MiB. A single QP was already close to line rate, so the additional QPs
improved this workload by only 0.9%.

| QPs | Queue depth per QP | 1 MiB bandwidth (GB/s) | Bandwidth (Gb/s) |
|---:|---:|---:|---:|
| 1 | 64 | 48.130 | 385.04 |
| 2 | 64 | 48.369 | 386.95 |
| 4 | 64 | 48.478 | 387.82 |
| 8 | 64 | 48.529 | 388.23 |
| 16 | 64 | **48.548** | **388.38** |

## Reproduce

Run from the repository root after installing the package and torch into a
virtual environment. Torch must use its VMM-backed allocator so the tensors
can be exported as dma-buf file descriptors.

```bash
taskset -c 0 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python ibverbs/benchmarks/gpudirect.py \
  --src-gpu 0 --src-hca mlx5_0 \
  --dst-gpu 4 --dst-hca mlx5_6 \
  --qps 16
```

The script prints JSON containing the topology metadata, iteration counts,
and unrounded results. Change the GPU/HCA arguments only as pairs after
checking locality with `nvidia-smi topo -m`.
