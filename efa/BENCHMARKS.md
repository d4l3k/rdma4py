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

The benchmark automatically selected a distinct EFA for each GPU by NUMA node
and PCI hierarchy. Each selected EFA has a PCI distance of 4 from its GPU in
the benchmark's sysfs-tree metric. All assigned GPUs are in the same `NV18`
NVSwitch domain and NUMA node, so this host cannot compare cross-NVLink-domain
or cross-NUMA behavior.

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

The torch baseline uses `Tensor.copy_` between the same GPU pair and CUDA event
timing. It follows the NVLink path and is not an EFA protocol measurement.

## Completion latency

| Tensor size | EFA write (us) | EFA read (us) | torch P2P copy (us) |
| ---: | ---: | ---: | ---: |
| 4 KiB | 23.09 | 33.45 | 11.17 |
| 16 KiB | 25.57 | 35.38 | 11.12 |
| 64 KiB | 32.97 | 41.98 | 11.18 |
| 256 KiB | 47.98 | 55.84 | 11.16 |
| 1 MiB | 112.88 | 124.21 | 11.30 |
| 4 MiB | 371.16 | 395.65 | 27.17 |
| 16 MiB | 1403.30 | 1467.89 | 56.81 |
| 64 MiB | 5522.62 | 5612.54 | 185.97 |

## Aggregate bandwidth

### EFA RDMA write

| Tensor size | 1 QP (GB/s) | 2 QPs (GB/s) | 4 QPs (GB/s) | 8 QPs (GB/s) |
| ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 1.715 | 2.849 | 4.330 | 5.665 |
| 16 KiB | 4.963 | 7.167 | 9.005 | 10.329 |
| 64 KiB | 8.830 | 10.277 | 11.138 | 11.637 |
| 256 KiB | 11.135 | 11.633 | 11.916 | 12.066 |
| 1 MiB | 11.915 | 12.064 | 12.141 | 12.181 |
| 4 MiB | 12.139 | 12.181 | 12.200 | 12.211 |
| 16 MiB | 12.200 | 12.211 | 12.216 | 12.219 |
| 64 MiB | 12.216 | 12.219 | 12.220 | 12.221 |

### EFA RDMA read

| Tensor size | 1 QP (GB/s) | 2 QPs (GB/s) | 4 QPs (GB/s) | 8 QPs (GB/s) |
| ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 1.265 | 2.194 | 3.498 | 5.071 |
| 16 KiB | 4.111 | 6.158 | 8.088 | 9.668 |
| 64 KiB | 8.216 | 9.741 | 10.828 | 11.361 |
| 256 KiB | 10.898 | 11.373 | 11.363 | 11.816 |
| 1 MiB | 11.725 | 11.732 | 11.808 | 12.040 |
| 4 MiB | 11.973 | 12.014 | 12.027 | 12.163 |
| 16 MiB | 12.180 | 12.193 | 12.193 | 12.204 |
| 64 MiB | 12.218 | 12.221 | 12.225 | 12.226 |

### torch NVLink baseline

| Tensor size | Bandwidth (GB/s) |
| ---: | ---: |
| 4 KiB | 0.367 |
| 16 KiB | 1.473 |
| 64 KiB | 5.861 |
| 256 KiB | 23.489 |
| 1 MiB | 92.793 |
| 4 MiB | 154.378 |
| 16 MiB | 295.340 |
| 64 MiB | 360.853 |

The EFA path reaches 12.226 GB/s (97.81 Gbit/s). Multiple QPs matter most for
small tensors: at 4 KiB, eight QPs improve write bandwidth by 3.30x and read
bandwidth by 4.01x over one QP. At 1 MiB and above, one QP is already close to
the approximately 100 Gbit/s EFA lane limit, so additional QPs provide little
gain.

These are single-run results without error bars. EFA latency includes Python
submission and completion polling, and all tested GPUs share one NVLink and
NUMA domain.
