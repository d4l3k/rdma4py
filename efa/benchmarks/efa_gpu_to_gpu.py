#!/usr/bin/env python3
"""Benchmark single-host EFA GPUDirect transfers between torch GPU tensors."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

# CUDA dma-buf export requires VMM-backed torch allocations. This must be set
# before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import efa
import efa.cuda as efa_cuda
import torch

QKEY = 0x11111111
FULL_ACCESS = (
    efa.AccessFlags.LOCAL_WRITE
    | efa.AccessFlags.REMOTE_WRITE
    | efa.AccessFlags.REMOTE_READ
)


@dataclass(frozen=True)
class DeviceLocation:
    name: str
    pci: str
    numa: int
    path: Path


@dataclass
class Endpoint:
    qp: object

    def info(self) -> efa.EndpointInfo:
        return efa.local_endpoint_info(self.qp, qkey=QKEY)


@dataclass(frozen=True)
class EfaResult:
    operation: str
    size: int
    qp_count: int
    queue_depth: int
    operations: int
    seconds: float

    @property
    def latency_us(self) -> float:
        return self.seconds * 1e6 / self.operations

    @property
    def gb_per_second(self) -> float:
        return self.operations * self.size / self.seconds / 1e9


@dataclass(frozen=True)
class TorchResult:
    size: int
    iterations: int
    seconds: float

    @property
    def latency_us(self) -> float:
        return self.seconds * 1e6 / self.iterations

    @property
    def gb_per_second(self) -> float:
        return self.iterations * self.size / self.seconds / 1e9


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([kmgt]?i?b?)?\s*", value.lower())
    if match is None:
        raise argparse.ArgumentTypeError("size must look like 4096, 4KiB, or 1MiB")
    number = int(match.group(1))
    suffix = match.group(2)
    powers = {
        "": 0,
        "b": 0,
        "k": 1,
        "kb": 1,
        "ki": 1,
        "kib": 1,
        "m": 2,
        "mb": 2,
        "mi": 2,
        "mib": 2,
        "g": 3,
        "gb": 3,
        "gi": 3,
        "gib": 3,
        "t": 4,
        "tb": 4,
        "ti": 4,
        "tib": 4,
    }
    if suffix not in powers:
        raise argparse.ArgumentTypeError("size must look like 4096, 4KiB, or 1MiB")
    result = number * (1024 ** powers[suffix])
    if result <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return result


def format_size(size: int) -> str:
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if size >= scale and size % scale == 0:
            return f"{size // scale} {unit}"
    return f"{size} B"


def _pci_location(name: str, sysfs_path: Path) -> DeviceLocation:
    path = sysfs_path.resolve()
    pci = path.name
    numa_path = path / "numa_node"
    numa = int(numa_path.read_text().strip()) if numa_path.exists() else -1
    return DeviceLocation(name=name, pci=pci, numa=numa, path=path)


def gpu_location(index: int) -> DeviceLocation:
    props = torch.cuda.get_device_properties(index)
    pci = (
        f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:"
        f"{props.pci_device_id:02x}.0"
    )
    return _pci_location(f"cuda:{index}", Path("/sys/bus/pci/devices") / pci)


def efa_location(name: str) -> DeviceLocation:
    return _pci_location(name, Path("/sys/class/infiniband") / name / "device")


def pci_distance(left: DeviceLocation, right: DeviceLocation) -> int:
    left_parts = left.path.parts
    right_parts = right.path.parts
    common = 0
    for left_part, right_part in zip(left_parts, right_parts):
        if left_part != right_part:
            break
        common += 1
    distance = len(left_parts) + len(right_parts) - 2 * common
    if left.numa >= 0 and right.numa >= 0 and left.numa != right.numa:
        distance += 1000
    return distance


def select_efa_devices(
    devices: Sequence[object],
    source_gpu: DeviceLocation,
    destination_gpu: DeviceLocation,
    source_name: Optional[str],
    destination_name: Optional[str],
) -> tuple[object, object, DeviceLocation, DeviceLocation]:
    by_name = {device.name: device for device in devices}
    locations = {name: efa_location(name) for name in by_name}

    def choose(gpu: DeviceLocation, requested: Optional[str], excluded: set[str]):
        if requested is not None:
            if requested not in by_name:
                raise RuntimeError(f"EFA device {requested!r} was not found")
            if requested in excluded:
                raise RuntimeError("source and destination EFA devices must differ")
            return by_name[requested], locations[requested]
        candidates = [
            (pci_distance(gpu, location), name, by_name[name], location)
            for name, location in locations.items()
            if name not in excluded
        ]
        if not candidates:
            raise RuntimeError("at least two accessible EFA devices are required")
        _, _, device, location = min(candidates)
        return device, location

    source_device, source_location = choose(source_gpu, source_name, set())
    destination_device, destination_location = choose(
        destination_gpu, destination_name, {source_device.name}
    )
    return (
        source_device,
        destination_device,
        source_location,
        destination_location,
    )


def nvlink_relation(source_gpu: int, destination_gpu: int) -> str:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    lines = [line.split() for line in output.splitlines() if line.strip()]
    header_index = next(
        (index for index, line in enumerate(lines) if "GPU0" in line), None
    )
    if header_index is None:
        return "unknown"
    header = lines[header_index]
    row = next(
        (line for line in lines[header_index + 1 :] if line[0] == f"GPU{source_gpu}"),
        None,
    )
    if row is None:
        return "unknown"
    try:
        return row[header.index(f"GPU{destination_gpu}") + 1]
    except (ValueError, IndexError):
        return "unknown"


def format_cpu_set(cpus: set[int]) -> str:
    ranges = []
    first = previous = None
    for cpu in sorted(cpus):
        if first is None:
            first = previous = cpu
        elif cpu == previous + 1:
            previous = cpu
        else:
            ranges.append(str(first) if first == previous else f"{first}-{previous}")
            first = previous = cpu
    if first is not None:
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


def create_qps(pd, cq, count: int, queue_depth: int) -> list[Endpoint]:
    endpoints = []
    for _ in range(count):
        qp = pd.create_qp(
            efa.QPInitAttr(
                send_cq=cq,
                recv_cq=cq,
                max_send_wr=max(32, queue_depth),
                max_recv_wr=8,
            )
        ).prepare(qkey=QKEY)
        endpoints.append(Endpoint(qp=qp))
    return endpoints


def poll_completions(cq, count: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    completed = 0
    while completed < count:
        completions = cq.poll(min(256, count - completed))
        for completion in completions:
            completion.raise_for_status()
        completed += len(completions)
        if not completions and time.monotonic() >= deadline:
            raise TimeoutError(f"received {completed}/{count} EFA completions")


def make_wrs(
    operation: str,
    size: int,
    queue_depth: int,
    qp_count: int,
    region_size: int,
    local_mr,
    remote_mr,
    peers: Sequence[efa.Peer],
) -> list[list[efa.SendWR]]:
    opcode = efa.WROpcode.RDMA_WRITE if operation == "write" else efa.WROpcode.RDMA_READ
    result = []
    for qp_index in range(qp_count):
        offset = qp_index * region_size
        result.append(
            [
                efa.SendWR(
                    wr_id=(qp_index << 32) | wr_index,
                    sg_list=[local_mr.sge(size, offset=offset)],
                    opcode=opcode,
                    send_flags=efa.SendFlags.SIGNALED,
                    remote_addr=remote_mr.addr + offset,
                    rkey=remote_mr.rkey,
                    dest=peers[qp_index],
                )
                for wr_index in range(queue_depth)
            ]
        )
    return result


def run_efa(
    operation: str,
    size: int,
    qp_count: int,
    queue_depth: int,
    duration: float,
    warmup_batches: int,
    timeout: float,
    source_endpoints: Sequence[Endpoint],
    destination_endpoints: Sequence[Endpoint],
    source_cq,
    destination_cq,
    source_mr,
    destination_mr,
    source_peers: Sequence[efa.Peer],
    destination_peers: Sequence[efa.Peer],
    region_size: int,
) -> EfaResult:
    if operation == "write":
        endpoints = source_endpoints
        cq = source_cq
        local_mr = source_mr
        remote_mr = destination_mr
        peers = source_peers
    else:
        endpoints = destination_endpoints
        cq = destination_cq
        local_mr = destination_mr
        remote_mr = source_mr
        peers = destination_peers

    wrs = make_wrs(
        operation,
        size,
        queue_depth,
        qp_count,
        region_size,
        local_mr,
        remote_mr,
        peers,
    )
    completions_per_batch = qp_count * queue_depth

    def run_batch() -> None:
        for endpoint, qp_wrs in zip(endpoints, wrs):
            endpoint.qp.post_send(qp_wrs)
        poll_completions(cq, completions_per_batch, timeout)

    for _ in range(warmup_batches):
        run_batch()

    operations = 0
    start = time.perf_counter()
    deadline = start + duration
    while True:
        run_batch()
        operations += completions_per_batch
        if time.perf_counter() >= deadline:
            break
    elapsed = time.perf_counter() - start
    return EfaResult(
        operation=operation,
        size=size,
        qp_count=qp_count,
        queue_depth=queue_depth,
        operations=operations,
        seconds=elapsed,
    )


def run_torch_copy(
    source: torch.Tensor,
    destination: torch.Tensor,
    size: int,
    target_seconds: float,
) -> TorchResult:
    source_view = source[:size]
    destination_view = destination[:size]
    with torch.cuda.device(destination.device):
        for _ in range(16):
            destination_view.copy_(source_view, non_blocking=True)
        torch.cuda.synchronize(destination.device)

        pilot_start = torch.cuda.Event(enable_timing=True)
        pilot_end = torch.cuda.Event(enable_timing=True)
        pilot_start.record()
        for _ in range(32):
            destination_view.copy_(source_view, non_blocking=True)
        pilot_end.record()
        pilot_end.synchronize()
        pilot_seconds = max(pilot_start.elapsed_time(pilot_end) / 1000, 1e-9)
        iterations = max(16, min(100_000, int(target_seconds * 32 / pilot_seconds)))

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            destination_view.copy_(source_view, non_blocking=True)
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) / 1000
    return TorchResult(size=size, iterations=iterations, seconds=seconds)


def print_environment(
    source_gpu_index: int,
    destination_gpu_index: int,
    source_gpu: DeviceLocation,
    destination_gpu: DeviceLocation,
    source_efa: DeviceLocation,
    destination_efa: DeviceLocation,
) -> None:
    print("# Environment")
    print()
    print(f"- torch: `{torch.__version__}` (CUDA `{torch.version.cuda}`)")
    print(
        f"- source GPU: `cuda:{source_gpu_index}` "
        f"({torch.cuda.get_device_name(source_gpu_index)}), PCI `{source_gpu.pci}`, "
        f"NUMA `{source_gpu.numa}`"
    )
    print(
        f"- destination GPU: `cuda:{destination_gpu_index}` "
        f"({torch.cuda.get_device_name(destination_gpu_index)}), "
        f"PCI `{destination_gpu.pci}`, NUMA `{destination_gpu.numa}`"
    )
    print(
        f"- GPU peer access: "
        f"`{torch.cuda.can_device_access_peer(source_gpu_index, destination_gpu_index)}`"
    )
    print(
        f"- NVIDIA topology relation: "
        f"`{nvlink_relation(source_gpu_index, destination_gpu_index)}`"
    )
    print(
        f"- source EFA: `{source_efa.name}`, PCI `{source_efa.pci}`, "
        f"NUMA `{source_efa.numa}`, PCI distance `{pci_distance(source_gpu, source_efa)}`"
    )
    print(
        f"- destination EFA: `{destination_efa.name}`, "
        f"PCI `{destination_efa.pci}`, NUMA `{destination_efa.numa}`, "
        f"PCI distance `{pci_distance(destination_gpu, destination_efa)}`"
    )
    print(f"- process CPU affinity: `{format_cpu_set(os.sched_getaffinity(0))}`")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--destination-gpu", type=int, default=1)
    parser.add_argument("--source-device")
    parser.add_argument("--destination-device")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=parse_size,
        default=[
            parse_size(size)
            for size in (
                "4KiB",
                "16KiB",
                "64KiB",
                "256KiB",
                "1MiB",
                "4MiB",
                "16MiB",
                "64MiB",
            )
        ],
    )
    parser.add_argument(
        "--operations", nargs="+", choices=("write", "read"), default=("write", "read")
    )
    parser.add_argument("--qp-counts", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--queue-depth", type=int, default=16)
    parser.add_argument("--latency-seconds", type=float, default=0.75)
    parser.add_argument("--bandwidth-seconds", type=float, default=1.5)
    parser.add_argument("--torch-seconds", type=float, default=0.5)
    parser.add_argument("--warmup-batches", type=int, default=4)
    parser.add_argument("--completion-timeout", type=float, default=30.0)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--skip-torch-baseline", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if min(args.source_gpu, args.destination_gpu) < 0:
        raise ValueError("GPU indices must be non-negative")
    if args.source_gpu == args.destination_gpu:
        raise ValueError("source and destination GPUs must differ")
    if min(args.sizes) <= 0:
        raise ValueError("tensor sizes must be positive")
    if min(args.qp_counts) <= 0:
        raise ValueError("QP counts must be positive")
    if args.queue_depth <= 0:
        raise ValueError("queue depth must be positive")
    if args.warmup_batches < 0:
        raise ValueError("warmup batches must be non-negative")
    if args.completion_timeout <= 0:
        raise ValueError("completion timeout must be positive")
    if min(args.latency_seconds, args.bandwidth_seconds, args.torch_seconds) <= 0:
        raise ValueError("benchmark durations must be positive")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() <= max(args.source_gpu, args.destination_gpu):
        raise RuntimeError("requested GPU index is unavailable")

    devices = efa.get_efa_device_list()
    if len(devices) < 2:
        raise RuntimeError("at least two accessible EFA devices are required")

    source_gpu_location = gpu_location(args.source_gpu)
    destination_gpu_location = gpu_location(args.destination_gpu)
    (
        source_device,
        destination_device,
        source_efa_location,
        destination_efa_location,
    ) = select_efa_devices(
        devices,
        source_gpu_location,
        destination_gpu_location,
        args.source_device,
        args.destination_device,
    )
    print_environment(
        args.source_gpu,
        args.destination_gpu,
        source_gpu_location,
        destination_gpu_location,
        source_efa_location,
        destination_efa_location,
    )

    max_qp_count = max(args.qp_counts)
    max_size = max(args.sizes)
    allocation_size = max_qp_count * max_size
    max_cqe = max(1024, max_qp_count * args.queue_depth * 4)

    source_ctx = source_device.open()
    destination_ctx = destination_device.open()
    source_pd = source_ctx.alloc_pd()
    destination_pd = destination_ctx.alloc_pd()
    source_cq = source_ctx.create_cq(max_cqe)
    destination_cq = destination_ctx.create_cq(max_cqe)
    source_endpoints = create_qps(source_pd, source_cq, max_qp_count, args.queue_depth)
    destination_endpoints = create_qps(
        destination_pd,
        destination_cq,
        max_qp_count,
        args.queue_depth,
    )
    source_peers = [
        efa.EndpointInfo.from_bytes(endpoint.info().to_bytes()).peer(source_pd)
        for endpoint in destination_endpoints
    ]
    destination_peers = [
        efa.EndpointInfo.from_bytes(endpoint.info().to_bytes()).peer(destination_pd)
        for endpoint in source_endpoints
    ]

    source_tensor = torch.full(
        (allocation_size,),
        0xA5,
        dtype=torch.uint8,
        device=f"cuda:{args.source_gpu}",
    )
    destination_tensor = torch.zeros(
        allocation_size,
        dtype=torch.uint8,
        device=f"cuda:{args.destination_gpu}",
    )
    torch.cuda.synchronize(args.source_gpu)
    torch.cuda.synchronize(args.destination_gpu)
    source_mr = efa_cuda.register_tensor(source_pd, source_tensor, FULL_ACCESS)
    destination_mr = efa_cuda.register_tensor(
        destination_pd, destination_tensor, FULL_ACCESS
    )

    try:
        latency_results = []
        bandwidth_results = []
        for operation in args.operations:
            destination_tensor.zero_()
            torch.cuda.synchronize(args.destination_gpu)
            for size in args.sizes:
                latency_results.append(
                    run_efa(
                        operation,
                        size,
                        1,
                        1,
                        args.latency_seconds,
                        args.warmup_batches,
                        args.completion_timeout,
                        source_endpoints,
                        destination_endpoints,
                        source_cq,
                        destination_cq,
                        source_mr,
                        destination_mr,
                        source_peers,
                        destination_peers,
                        max_size,
                    )
                )
                for qp_count in args.qp_counts:
                    bandwidth_results.append(
                        run_efa(
                            operation,
                            size,
                            qp_count,
                            args.queue_depth,
                            args.bandwidth_seconds,
                            args.warmup_batches,
                            args.completion_timeout,
                            source_endpoints,
                            destination_endpoints,
                            source_cq,
                            destination_cq,
                            source_mr,
                            destination_mr,
                            source_peers,
                            destination_peers,
                            max_size,
                        )
                    )

            with torch.cuda.device(args.destination_gpu):
                efa_cuda.flush_gpudirect_writes()
            torch.cuda.synchronize(args.destination_gpu)
            if not bool(torch.all(destination_tensor == 0xA5).item()):
                raise RuntimeError(f"{operation} correctness check failed")

        print("# EFA completion latency")
        print()
        print("| Operation | Tensor size | QPs | QD | Latency (us) | Samples |")
        print("|---|---:|---:|---:|---:|---:|")
        for result in latency_results:
            print(
                f"| {result.operation} | {format_size(result.size)} | "
                f"{result.qp_count} | {result.queue_depth} | "
                f"{result.latency_us:.2f} | {result.operations} |"
            )
        print()

        print("# EFA aggregate bandwidth")
        print()
        print(
            "| Operation | Tensor size | QPs | QD/QP | "
            "Outstanding | GB/s | Gbit/s | Samples |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for result in bandwidth_results:
            print(
                f"| {result.operation} | {format_size(result.size)} | "
                f"{result.qp_count} | {result.queue_depth} | "
                f"{result.qp_count * result.queue_depth} | "
                f"{result.gb_per_second:.3f} | "
                f"{result.gb_per_second * 8:.2f} | {result.operations} |"
            )
        print()

        if not args.skip_torch_baseline:
            torch_results = [
                run_torch_copy(
                    source_tensor,
                    destination_tensor,
                    size,
                    args.torch_seconds,
                )
                for size in args.sizes
            ]
            print("# torch P2P copy baseline")
            print()
            print("| Tensor size | Latency (us) | GB/s | Samples |")
            print("|---:|---:|---:|---:|")
            for result in torch_results:
                print(
                    f"| {format_size(result.size)} | {result.latency_us:.2f} | "
                    f"{result.gb_per_second:.3f} | {result.iterations} |"
                )
    finally:
        source_mr.close()
        destination_mr.close()
        for peer in source_peers:
            peer.close()
        for peer in destination_peers:
            peer.close()
        for endpoint in source_endpoints:
            endpoint.qp.close()
        for endpoint in destination_endpoints:
            endpoint.qp.close()
        source_cq.close()
        destination_cq.close()
        source_pd.close()
        destination_pd.close()
        source_ctx.close()
        destination_ctx.close()


if __name__ == "__main__":
    main()
