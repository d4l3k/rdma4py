# NVMe/RDMA GPUDirect benchmarks

Measured on 2026-07-21 with the `nvmeof` userspace initiator, the Linux
`nvmet_rdma` target, and CUDA tensors registered through the `ibverbs`
dma-buf path. Every result uses two different physical HCAs:

```text
NVMe file -> target HCA -> RDMA fabric -> initiator HCA -> GPU
```

The payload is placed directly in GPU memory on the initiator. There is no
initiator-side host staging buffer.

## Safety and target

The namespace was an 8 GiB NOCOW file created solely for this benchmark on
the host's existing Btrfs filesystem. The file backend used direct I/O
(`buffered_io=0`). Cleanup removed the file after the run. No raw NVMe device,
partition, filesystem metadata, or existing file was exposed through `nvmet`.

Btrfs is backed by `/dev/md0`, an eight-drive RAID0 with a 512 KiB chunk.
Every drive contains live RAID data, so exporting an individual drive would
have been destructive. These results are for the aggregate RAID. Four members
are attached to each NUMA domain:

| NUMA | NVMe controllers | PCI addresses |
|---:|---|---|
| 0 | `nvme5n1` through `nvme8n1` | `0000:1a:00.0`, `3c:00.0`, `4d:00.0`, `5e:00.0` |
| 1 | `nvme1n1` through `nvme4n1` | `0000:9c:00.0`, `bc:00.0`, `cc:00.0`, `dc:00.0` |

Large RAID I/O stripes across both NVMe groups. Reversing the target listener
changes the target NIC/NVMe PCIe domain, but cannot isolate one group without
putting the live array at risk.

| HCA | HCA PCI / NUMA | PIX-local GPU | Local RAID members |
|---|---|---:|---|
| `mlx5_0` | `0000:18:00.0` / 0 | GPU 0 (`0000:1b:00.0`) | `nvme5n1` - `nvme8n1` |
| `mlx5_6` | `0000:9a:00.0` / 1 | GPU 4 (`0000:a3:00.0`) | `nvme1n1` - `nvme4n1` |

Both HCAs negotiated PCIe 5.0 x16 and expose 400 Gb/s RoCE ports. This is a
same-host test through the RDMA fabric, not HCA loopback and not a cross-host
measurement. `rdma resource show qp` confirmed the two endpoints during a
live I/O connection:

```text
mlx5_0 lqpn 1879 rqpn 863 type RC state RTS comm [rdma_cm]
mlx5_6 lqpn  863 rqpn 1879 type RC state RTS pid  python
```

| Component | Version / configuration |
|---|---|
| CPU | 2x Intel Xeon Platinum 8480C |
| GPUs | 8x NVIDIA H100 80GB HBM3 |
| GPU driver | 580.82.07 |
| HCAs | ConnectX-7, firmware 28.38.1002 |
| NVMe | 8x Samsung MZOL63T8HDLT-00AFB, firmware LDA67F2Q |
| Storage | 8-drive RAID0, 512 KiB chunk, Btrfs NOCOW file |
| Kernel | 6.16.1-0_fbk4_rc4_0_g0171af6f7fa9 |
| rdma-core / libibverbs | 57.0 / 61.0 |
| torch / CUDA | torch 2.13.0+cu130 / CUDA 13.0 |
| Target LBA / MDTS | 4 KiB / 1 MiB |
| Source base revision | `064e89613e3d84de3d0426a826c6f2b1d1b76e7a` |

## Method

- `--source` binds the initiator to one HCA while `--target` addresses the
  other. `--expected-hca` aborts if RDMA-CM resolves a different device.
- One NVMe controller and I/O QP are created per lane. Small transfers use 64
  QPs at queue depth 64. Large transfers use 32 QPs at queue depth 4.
- Latency is measured on one QP with serialized logical transfers. It is host
  time from command submission through the NVMe response CQE, including the
  Python/Cython submission and polling path.
- Small-transfer bandwidth values are medians of three runs targeting 2 GiB
  per run. Each transfer through 1 MiB is one NVMe command.
- Transfers above the 1 MiB MDTS are split into 1 MiB commands. Large-transfer
  bandwidth is the logical tensor payload divided by the completion time for
  every command. The latency value covers the complete logical transfer.
- Write completion does not include an NVMe Flush command. A namespace Flush
  is issued after timed work and before integrity verification.
- Every run finishes with a GPU -> NVMe -> GPU round trip and compares the
  destination tensor. All reported runs passed.

## `mlx5_0` target to `mlx5_6` initiator

Path: RAID -> `mlx5_0` -> fabric -> `mlx5_6` -> GPU 4. The process is pinned
to CPU 56 on NUMA node 1 with the initiator HCA and GPU.

| I/O size | Write p50 (us) | Write p99 (us) | Write GB/s | Read p50 (us) | Read p99 (us) | Read GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 4 KiB | 50.853 | 79.285 | 0.392 | 95.744 | 134.507 | 0.547 |
| 16 KiB | 57.625 | 76.409 | 1.533 | 107.948 | 172.264 | 2.200 |
| 64 KiB | 74.714 | 94.098 | 5.995 | 119.004 | 179.956 | 9.365 |
| 256 KiB | 140.469 | 171.528 | 18.686 | 189.207 | 1345.822 | 22.230 |
| 1 MiB | 268.995 | 292.899 | 25.269 | 311.241 | 353.167 | 27.818 |

## `mlx5_6` target to `mlx5_0` initiator

Path: RAID -> `mlx5_6` -> fabric -> `mlx5_0` -> GPU 0. The process is pinned
to CPU 0 on NUMA node 0 with the initiator HCA and GPU.

| I/O size | Write p50 (us) | Write p99 (us) | Write GB/s | Read p50 (us) | Read p99 (us) | Read GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 4 KiB | 48.769 | 76.941 | 0.395 | 96.147 | 144.712 | 0.546 |
| 16 KiB | 57.064 | 80.947 | 1.478 | 108.863 | 193.156 | 2.133 |
| 64 KiB | 73.756 | 100.225 | 6.118 | 121.524 | 183.654 | 9.327 |
| 256 KiB | 140.620 | 197.254 | 18.709 | 188.639 | 224.366 | 22.358 |
| 1 MiB | 269.178 | 288.287 | **26.074** | 311.084 | 371.889 | **27.850** |

## Transfers larger than 1 MiB

These results use locality-optimized GPUs, 32 QPs, and queue depth 4. A 4,
16, or 64 MiB tensor becomes 4, 16, or 64 NVMe commands respectively.

| Direction | Tensor | Commands | Write p50 (us) | Write GB/s | Read p50 (us) | Read GB/s |
|---|---:|---:|---:|---:|---:|---:|
| `mlx5_0` -> `mlx5_6` -> GPU 4 | 4 MiB | 4 | 355.596 | 20.737 | 628.526 | 18.323 |
| `mlx5_0` -> `mlx5_6` -> GPU 4 | 16 MiB | 16 | 1423.047 | 20.113 | 2551.750 | 20.728 |
| `mlx5_0` -> `mlx5_6` -> GPU 4 | 64 MiB | 64 | 5724.594 | 20.339 | 8870.129 | 21.290 |
| `mlx5_6` -> `mlx5_0` -> GPU 0 | 4 MiB | 4 | 359.132 | 20.963 | 635.913 | 18.489 |
| `mlx5_6` -> `mlx5_0` -> GPU 0 | 16 MiB | 16 | 1700.682 | 21.185 | 2543.264 | 20.810 |
| `mlx5_6` -> `mlx5_0` -> GPU 0 | 64 MiB | 64 | 5747.600 | 20.628 | 8897.215 | 21.507 |

Logical-transfer latency grows approximately with tensor size because the
target advertises a 1 MiB MDTS. Bandwidth remains around 20-21.5 GB/s through
64 MiB, so larger tensors are supported without a host bounce buffer.

## QP scaling

This sweep uses RAID -> `mlx5_0` -> fabric -> `mlx5_6` -> GPU 4, 1 MiB I/O,
and queue depth 64 per QP.

| I/O QPs | Write GB/s | Read GB/s |
|---:|---:|---:|
| 1 | 16.561 | 7.798 |
| 2 | 20.698 | 13.432 |
| 4 | 23.211 | 20.579 |
| 8 | 25.103 | 22.710 |
| 16 | 24.704 | 25.763 |
| 32 | **25.693** | 27.611 |
| 64 | 25.321 | **27.830** |

Multiple QPs are particularly important for reads. At 1 MiB the curve is
effectively saturated by 32-64 QPs. The large-transfer sweep uses fewer QPs
because each tensor is already split into many concurrent commands.

## PCIe locality

The crossed cases keep the CPU local to the initiator HCA and register a GPU
from the other NUMA node. Both still use distinct target and initiator HCAs.

| Target -> initiator HCA / GPU | GPU locality | CPU | 1 MiB write GB/s | 1 MiB read GB/s |
|---|---|---:|---:|---:|
| `mlx5_6` -> `mlx5_0` / GPU 0 | PIX, NUMA 0 | 0 | **26.074** | **27.850** |
| `mlx5_6` -> `mlx5_0` / GPU 4 | cross-socket | 0 | 13.528 | 7.106 |
| `mlx5_0` -> `mlx5_6` / GPU 4 | PIX, NUMA 1 | 56 | 25.269 | 27.818 |
| `mlx5_0` -> `mlx5_6` / GPU 0 | cross-socket | 56 | 13.135 | 8.389 |

Cross-socket GPU/HCA placement cuts 1 MiB write bandwidth roughly in half and
read bandwidth by about 70-75%. Keeping the GPU on the initiator HCA's PCIe
domain matters more than adding QPs after the local path is saturated.

## Reproduce

The benchmark requires an already configured disposable NVMe/RDMA namespace.
Never point it at a namespace containing data that must be preserved. The
target and source addresses must resolve to different HCAs.

```bash
taskset -c 56 env \
  PYTHONPATH="$PWD/ibverbs/src:$PWD/nvmeof/src" \
  python nvmeof/benchmarks/gpudirect.py \
    --target 198.18.1.1 --source 198.18.2.1 --port 4420 \
    --subsystem-nqn nqn.2026-07.io.rdma4py:benchmark-file \
    --gpu 4 --expected-hca mlx5_6 \
    --qps 64 --queue-depth 64
```

The command prints JSON with unrounded results, topology metadata, iteration
counts, commands per logical transfer, and the integrity status. Check NIC,
GPU, and CPU placement with `nvidia-smi topo -m`, `ibdev2netdev`, and
`/sys/class/infiniband/*/device/numa_node` before selecting the source address.
