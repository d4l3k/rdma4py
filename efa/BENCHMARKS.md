# EFA GPUDirect Benchmarks

These point measurements exercise this package's one-sided EFA path directly
between torch CUDA tensors on two GPUs in one host.

## Hardware and topology

Measurements were collected on 2026-07-21.

| Component | Configuration |
| --- | --- |
| Instance | AWS `p5.48xlarge` |
| CPU | 2-socket AMD EPYC 7R13, 192 logical CPUs, 2 NUMA nodes |
| GPUs | 4 assigned NVIDIA H100 80GB HBM3 GPUs |
| Source | GPU0, PCI `0000:53:00.0`, NUMA 0 |
| Destination | GPU1, PCI `0000:64:00.0`, NUMA 0 |
| GPU topology | `NV18`; CUDA peer access enabled |
| Source EFA | `rdmap79s0`, PCI `0000:4f:00.0`, NUMA 0 |
| Destination EFA | `rdmap96s0`, PCI `0000:60:00.0`, NUMA 0 |
| CPU affinity | CPU 0, NUMA 0 |
| Software | Ubuntu 24.04, Linux 6.12.83, torch 2.13.0+cu130, CUDA 13.0, NVIDIA driver 610.43.02, rdma-core 61.0 |

The benchmark automatically selected distinct EFA pairs by NUMA node and PCI
hierarchy. Every selected EFA has a PCI distance of 4 from its GPU in the
benchmark's sysfs-tree metric.

| Lane | Source EFA | Source PCI | Destination EFA | Destination PCI |
| ---: | --- | --- | --- | --- |
| 0 | `rdmap79s0` | `0000:4f:00.0` | `rdmap96s0` | `0000:60:00.0` |
| 1 | `rdmap80s0` | `0000:50:00.0` | `rdmap97s0` | `0000:61:00.0` |
| 2 | `rdmap81s0` | `0000:51:00.0` | `rdmap98s0` | `0000:62:00.0` |
| 3 | `rdmap82s0` | `0000:52:00.0` | `rdmap99s0` | `0000:63:00.0` |

Both tested GPUs are in the same `NV18` NVSwitch domain and NUMA node, so this
run does not compare cross-NVLink-domain or cross-NUMA behavior.

## Method

Run from the `efa` directory:

```bash
PYTHONPATH=src .venv-gpu/bin/python benchmarks/efa_gpu_to_gpu.py \
  --source-gpu 0 --destination-gpu 1 --cpu 0
```

The benchmark allocates VMM-backed `torch.uint8` CUDA tensors and registers
them with `ibv_reg_dmabuf_mr`. EFA writes move GPU0 to GPU1; EFA reads are
issued by the GPU1-side endpoint and pull GPU0 into GPU1. It flushes completed
GPUDirect writes before torch verifies every destination byte.

Latency uses one QP at queue depth 1 and reports host-observed post-to-completion
time, including Python and polling overhead. Bandwidth uses queue depth 16 per
QP and 1, 2, 4, or 8 QPs. Each QP uses a distinct tensor region and every work
request is signaled, as required by EFA. Rates use decimal GB/s; tensor sizes
use binary units. Each point follows four warmup batches. The target measurement
windows are 0.75 seconds for latency, 1.5 seconds for EFA bandwidth, and 0.5
seconds for the torch baseline.

The multi-lane test stripes each batch across 1, 2, or 4 EFA pairs, posts work
to every lane before polling, and reports aggregate payload bandwidth. It uses
four QPs per lane at queue depth 16. Each lane and QP gets a disjoint tensor
region, and each EFA device has its own context, protection domain, completion
queue, QPs, and registration of the shared torch allocation.

The torch baseline uses `Tensor.copy_` between the same GPU pair and CUDA event
timing. It follows the NVLink path and is not an EFA protocol measurement.

## Completion latency

| Tensor size | EFA write (us) | EFA read (us) | torch P2P copy (us) |
| ---: | ---: | ---: | ---: |
| 4 KiB | 23.07 | 33.58 | 11.24 |
| 16 KiB | 25.43 | 35.26 | 10.84 |
| 64 KiB | 32.97 | 42.00 | 10.54 |
| 256 KiB | 47.97 | 55.86 | 10.66 |
| 1 MiB | 112.96 | 124.38 | 10.80 |
| 4 MiB | 371.20 | 395.89 | 26.65 |
| 16 MiB | 1403.95 | 1467.28 | 56.38 |
| 64 MiB | 5524.06 | 5604.49 | 185.89 |

## Aggregate bandwidth

### EFA RDMA write

| Tensor size | 1 QP (GB/s) | 2 QPs (GB/s) | 4 QPs (GB/s) | 8 QPs (GB/s) |
| ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 1.716 | 2.816 | 4.287 | 5.651 |
| 16 KiB | 5.017 | 7.192 | 9.021 | 10.341 |
| 64 KiB | 8.849 | 10.268 | 11.143 | 11.632 |
| 256 KiB | 11.136 | 11.633 | 11.910 | 12.068 |
| 1 MiB | 11.915 | 12.066 | 12.140 | 12.179 |
| 4 MiB | 12.136 | 12.180 | 12.200 | 12.210 |
| 16 MiB | 12.200 | 12.210 | 12.216 | 12.219 |
| 64 MiB | 12.216 | 12.219 | 12.220 | 12.221 |

### EFA RDMA read

| Tensor size | 1 QP (GB/s) | 2 QPs (GB/s) | 4 QPs (GB/s) | 8 QPs (GB/s) |
| ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 1.273 | 2.201 | 3.490 | 5.014 |
| 16 KiB | 4.119 | 6.164 | 8.081 | 9.606 |
| 64 KiB | 8.204 | 9.744 | 10.809 | 11.383 |
| 256 KiB | 10.894 | 11.369 | 11.363 | 11.805 |
| 1 MiB | 11.729 | 11.728 | 11.823 | 12.029 |
| 4 MiB | 11.963 | 12.017 | 12.036 | 12.161 |
| 16 MiB | 12.177 | 12.187 | 12.193 | 12.206 |
| 64 MiB | 12.219 | 12.221 | 12.222 | 12.226 |

## Multi-lane aggregate bandwidth

Each lane uses four QPs at queue depth 16.

### EFA RDMA write

| Tensor size | 1 lane (GB/s) | 2 lanes (GB/s) | 4 lanes (GB/s) |
| ---: | ---: | ---: | ---: |
| 4 KiB | 4.330 | 5.503 | 8.665 |
| 16 KiB | 9.017 | 15.017 | 25.363 |
| 64 KiB | 11.127 | 20.807 | 39.079 |
| 256 KiB | 11.916 | 23.368 | 45.875 |
| 1 MiB | 12.138 | 24.152 | 48.071 |
| 4 MiB | 12.199 | 24.366 | 48.672 |
| 16 MiB | 12.216 | 24.424 | 48.831 |
| 64 MiB | 12.220 | 24.438 | 48.871 |

### EFA RDMA read

| Tensor size | 1 lane (GB/s) | 2 lanes (GB/s) | 4 lanes (GB/s) |
| ---: | ---: | ---: | ---: |
| 4 KiB | 3.501 | 4.546 | 7.539 |
| 16 KiB | 8.093 | 13.365 | 22.817 |
| 64 KiB | 10.831 | 20.075 | 37.419 |
| 256 KiB | 11.349 | 22.314 | 42.609 |
| 1 MiB | 11.810 | 23.558 | 46.314 |
| 4 MiB | 12.017 | 23.973 | 47.865 |
| 16 MiB | 12.191 | 24.373 | 48.718 |
| 64 MiB | 12.224 | 24.440 | 48.877 |

## torch NVLink baseline

| Tensor size | Bandwidth (GB/s) |
| ---: | ---: |
| 4 KiB | 0.365 |
| 16 KiB | 1.511 |
| 64 KiB | 6.220 |
| 256 KiB | 24.585 |
| 1 MiB | 97.088 |
| 4 MiB | 157.369 |
| 16 MiB | 297.552 |
| 64 MiB | 361.007 |

The single-lane EFA path reaches 12.226 GB/s (97.81 Gbit/s). Multiple QPs
matter most for small tensors: at 4 KiB, eight QPs improve write bandwidth by
3.29x and read bandwidth by 3.94x over one QP. At 1 MiB and above, one QP is
already close to the approximately 100 Gbit/s EFA lane limit, so additional
QPs on that lane provide little gain.

Physical lanes continue scaling: four lanes reach 48.871 GB/s for 64 MiB
writes and 48.877 GB/s for reads, almost exactly 4x one-lane bandwidth. At 4
KiB, four lanes improve aggregate bandwidth by only 2.00x for writes and 2.15x
for reads because sequential Python submission and completion polling dominate.

These are single-run results without error bars. EFA latency includes Python
submission and completion polling, and all tested GPUs share one NVLink and
NUMA domain.
