#!/usr/bin/env python3
"""Benchmark RC RDMA writes between torch CUDA tensors."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

# dma-buf export requires torch's VMM-backed allocator. This must be set before
# torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import ibverbs as ib
import ibverbs.cuda as ibcuda
import torch

ACCESS = (
    ib.AccessFlags.LOCAL_WRITE
    | ib.AccessFlags.REMOTE_WRITE
    | ib.AccessFlags.REMOTE_READ
)
DEFAULT_SIZES = "8,64,1024,4096,16384,65536,262144,1048576,4194304,16777216,67108864"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _is_link_local(raw: bytes) -> bool:
    return raw[0] == 0xFE and (raw[1] & 0xC0) == 0x80


def _find_roce_gid(ctx, device: str, port: int):
    """Choose a non-link-local RoCE v2 GID when one is available."""
    port_attr = ctx.query_port(port)
    best = None
    for index in range(port_attr.gid_tbl_len):
        gid = ctx.query_gid(port, index)
        if gid.raw == b"\x00" * 16:
            continue
        gid_type = _read_text(
            Path("/sys/class/infiniband")
            / device
            / "ports"
            / str(port)
            / "gid_attrs/types"
            / str(index)
        )
        score = 4 if gid_type and "v2" in gid_type.lower() else 0
        score += 2 if not _is_link_local(gid.raw) else 0
        if best is None or score > best[0]:
            best = (score, index, gid)
    if best is None:
        raise RuntimeError(f"no usable GID on {device} port {port}")
    return best[1], best[2]


class Endpoint:
    def __init__(self, ctx, pd, port: int, queue_depth: int):
        self.ctx = ctx
        self.pd = pd
        self.port = port
        self.cq = ctx.create_cq(queue_depth + 8)
        self.qp = pd.create_qp(
            ib.QPInitAttr(
                send_cq=self.cq,
                recv_cq=self.cq,
                qp_type=ib.QPType.RC,
                max_send_wr=queue_depth + 8,
                max_recv_wr=8,
            )
        )

    def info(self, gid):
        return ib.local_qp_info(
            self.qp,
            self.ctx.query_port(self.port),
            gid,
            port=self.port,
        )

    def connect(self, remote, gid_index: int) -> None:
        ib.connect_rc(
            self.qp,
            remote,
            port=self.port,
            sgid_index=gid_index,
            access=ACCESS,
        )

    def poll_one(self, timeout: float = 30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            completions = self.cq.poll(1)
            if completions:
                completions[0].raise_for_status()
                return completions[0]
        raise TimeoutError("timed out waiting for a send completion")

    def close(self) -> None:
        self.qp.close()
        self.cq.close()


def _percentile(values: list[int], percentile: float) -> float:
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return float(ordered[max(0, index)])


def _make_writes(source_mr, destination_mr, size: int, count: int):
    return _make_writes_at_offset(source_mr, destination_mr, size, count, 0)


def _make_writes_at_offset(
    source_mr, destination_mr, size: int, count: int, offset: int
):
    sge = source_mr.sge(size, offset=offset)
    writes = []
    for index in range(count):
        flags = ib.SendFlags.SIGNALED if index == count - 1 else 0
        writes.append(
            ib.SendWR(
                wr_id=index,
                sg_list=[sge],
                opcode=ib.WROpcode.RDMA_WRITE,
                send_flags=flags,
                remote_addr=destination_mr.addr + offset,
                rkey=destination_mr.rkey,
            )
        )
    return writes


def _warm_up(endpoint: Endpoint, write, iterations: int) -> None:
    for _ in range(iterations):
        endpoint.qp.post_send(write)
        endpoint.poll_one()


def _measure_latency(endpoint: Endpoint, write, iterations: int):
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        endpoint.qp.post_send(write)
        endpoint.poll_one()
        samples.append(time.perf_counter_ns() - start)
    return {
        "iterations": iterations,
        "p50_us": statistics.median(samples) / 1_000.0,
        "p99_us": _percentile(samples, 0.99) / 1_000.0,
    }


def _measure_bandwidth(
    endpoints: list[Endpoint],
    source_mr,
    destination_mr,
    size: int,
    slot_stride: int,
    queue_depth: int,
    total_bytes: int,
    repeats: int,
):
    work_per_round = queue_depth * len(endpoints)
    iterations = max(work_per_round * 4, math.ceil(total_bytes / size))
    iterations = min(iterations, 1_048_576)
    iterations = math.ceil(iterations / work_per_round) * work_per_round
    writes = [
        _make_writes_at_offset(
            source_mr,
            destination_mr,
            size,
            queue_depth,
            index * slot_stride,
        )
        for index in range(len(endpoints))
    ]
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations // work_per_round):
            for endpoint, queue_writes in zip(endpoints, writes):
                endpoint.qp.post_send(queue_writes)
            for endpoint in endpoints:
                endpoint.poll_one()
        elapsed = (time.perf_counter_ns() - start) / 1e9
        samples.append(iterations * size / elapsed)
    bytes_per_second = statistics.median(samples)
    return {
        "iterations": iterations,
        "repeats": repeats,
        "GBps": bytes_per_second / 1e9,
        "Gbps": bytes_per_second * 8 / 1e9,
    }


def _device_metadata(device: str):
    base = Path("/sys/class/infiniband") / device / "device"
    try:
        pci_address = base.resolve().name
    except OSError:
        pci_address = None
    return {
        "name": device,
        "pci_address": pci_address,
        "numa_node": _read_text(base / "numa_node"),
        "current_link_speed": _read_text(base / "current_link_speed"),
        "current_link_width": _read_text(base / "current_link_width"),
    }


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("torch cannot access CUDA")
    if args.src_gpu == args.dst_gpu:
        raise ValueError("source and destination GPUs must be different")

    devices = {device.name: device for device in ib.get_device_list()}
    try:
        src_device = devices[args.src_hca]
        dst_device = devices[args.dst_hca]
    except KeyError as exc:
        raise RuntimeError(f"RDMA device not found: {exc.args[0]}") from exc

    src_ctx = src_device.open()
    dst_ctx = dst_device.open()
    src_pd = src_ctx.alloc_pd()
    dst_pd = dst_ctx.alloc_pd()
    src_endpoints = [
        Endpoint(src_ctx, src_pd, args.port, args.queue_depth) for _ in range(args.qps)
    ]
    dst_endpoints = [
        Endpoint(dst_ctx, dst_pd, args.port, args.queue_depth) for _ in range(args.qps)
    ]
    src_mr = None
    dst_mr = None
    try:
        src_gid_index, src_gid = _find_roce_gid(src_ctx, args.src_hca, args.port)
        dst_gid_index, dst_gid = _find_roce_gid(dst_ctx, args.dst_hca, args.port)
        for src_endpoint, dst_endpoint in zip(src_endpoints, dst_endpoints):
            src_endpoint.connect(dst_endpoint.info(dst_gid), src_gid_index)
            dst_endpoint.connect(src_endpoint.info(src_gid), dst_gid_index)

        max_size = max(args.sizes)
        allocation_size = max_size * args.qps
        source = torch.full(
            (allocation_size,), 0xA5, dtype=torch.uint8, device=args.src_gpu
        )
        destination = torch.zeros(
            allocation_size, dtype=torch.uint8, device=args.dst_gpu
        )
        src_mr = ibcuda.register_tensor(src_pd, source, ACCESS)
        dst_mr = ibcuda.register_tensor(dst_pd, destination, ACCESS)
        torch.cuda.synchronize(source.device)

        results = []
        for size in args.sizes:
            latency_write = _make_writes(src_mr, dst_mr, size, 1)[0]
            warmup = min(args.warmup, max(4, (64 * 1024 * 1024) // size))
            _warm_up(src_endpoints[0], latency_write, warmup)
            latency_iterations = max(
                100,
                min(args.latency_iterations, (512 * 1024 * 1024) // size),
            )
            latency = _measure_latency(
                src_endpoints[0], latency_write, latency_iterations
            )
            bandwidth = _measure_bandwidth(
                src_endpoints,
                src_mr,
                dst_mr,
                size,
                max_size,
                args.queue_depth,
                args.bandwidth_total_bytes,
                args.bandwidth_repeats,
            )
            results.append(
                {"size_bytes": size, "latency": latency, "bandwidth": bandwidth}
            )

        with torch.cuda.device(destination.device):
            ibcuda.flush_gpudirect_writes()
        if not bool(torch.all(destination == 0xA5).item()):
            raise RuntimeError("destination tensor does not match source tensor")

        return {
            "config": {
                "src_gpu": args.src_gpu,
                "dst_gpu": args.dst_gpu,
                "src_hca": _device_metadata(args.src_hca),
                "dst_hca": _device_metadata(args.dst_hca),
                "port": args.port,
                "qps": args.qps,
                "queue_depth": args.queue_depth,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(args.src_gpu),
            },
            "results": results,
        }
    finally:
        if src_mr is not None:
            src_mr.close()
        if dst_mr is not None:
            dst_mr.close()
        for endpoint in src_endpoints:
            endpoint.close()
        for endpoint in dst_endpoints:
            endpoint.close()
        src_pd.close()
        dst_pd.close()
        src_ctx.close()
        dst_ctx.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-gpu", type=int, required=True)
    parser.add_argument("--dst-gpu", type=int, required=True)
    parser.add_argument("--src-hca", required=True)
    parser.add_argument("--dst-hca", required=True)
    parser.add_argument("--port", type=int, default=1)
    parser.add_argument("--qps", type=int, default=1)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--latency-iterations", type=int, default=10_000)
    parser.add_argument("--bandwidth-total-bytes", type=int, default=8 << 30)
    parser.add_argument("--bandwidth-repeats", type=int, default=5)
    args = parser.parse_args()
    args.sizes = [int(size) for size in args.sizes.split(",")]
    if any(size <= 0 or size > 0xFFFFFFFF for size in args.sizes):
        parser.error("sizes must be between 1 and 2**32 - 1 bytes")
    if args.queue_depth <= 0:
        parser.error("queue depth must be positive")
    if args.qps <= 0:
        parser.error("QP count must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
