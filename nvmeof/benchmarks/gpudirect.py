#!/usr/bin/env python3
"""Benchmark NVMe/RDMA reads and writes directly to CUDA memory."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import ibverbs.cuda as ibcuda
import nvmeof
import torch
from nvmeof import protocol as p

DEFAULT_SIZES = "4096,16384,65536,262144,1048576"
PATTERN = 0xA5


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _percentile(values: list[int], percentile: float) -> float:
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return float(ordered[max(0, index)])


def _hca_metadata(name: str) -> dict:
    base = Path("/sys/class/infiniband") / name / "device"
    try:
        pci_address = base.resolve().name
    except OSError:
        pci_address = None
    return {
        "name": name,
        "pci_address": pci_address,
        "numa_node": _read_text(base / "numa_node"),
        "current_link_speed": _read_text(base / "current_link_speed"),
        "current_link_width": _read_text(base / "current_link_width"),
    }


@dataclass
class Lane:
    controller: nvmeof.Controller
    namespace: nvmeof.Namespace
    tensor: object
    mr: object

    @property
    def queue(self):
        return self.controller.io

    def close(self) -> None:
        self.mr.close()
        self.controller.close()


def _make_lanes(args, allocation_size: int) -> list[Lane]:
    lanes = []
    try:
        for _ in range(args.qps):
            controller = nvmeof.Controller.connect(
                args.target,
                args.subsystem_nqn,
                port=args.port,
                source=args.source,
                queue_depth=args.queue_depth + 1,
                timeout=args.timeout,
            )
            try:
                if (
                    args.expected_hca
                    and controller.io.context.name != args.expected_hca
                ):
                    raise RuntimeError(
                        f"target resolved to {controller.io.context.name}, "
                        f"expected {args.expected_hca}"
                    )
                namespace = controller.namespace(args.nsid)
                with torch.cuda.device(args.gpu):
                    tensor = torch.full(
                        (allocation_size,),
                        PATTERN,
                        dtype=torch.uint8,
                        device=f"cuda:{args.gpu}",
                    )
                mr = controller.register_gpu(tensor)
                lanes.append(Lane(controller, namespace, tensor, mr))
            except Exception:
                controller.close()
                raise
        return lanes
    except Exception:
        for lane in reversed(lanes):
            lane.close()
        raise


def _make_commands(
    lanes: list[Lane],
    opcode: int,
    size: int,
    queue_depth: int,
    slot_stride: int,
    start_lba: int,
    command_limit: int,
) -> list[list[bytes]]:
    lba_size = lanes[0].namespace.lba_size
    commands = []
    for lane_index, lane in enumerate(lanes):
        lane_commands = []
        for slot in range(queue_depth):
            byte_offset = (lane_index * queue_depth + slot) * slot_stride
            transfer_offset = 0
            while transfer_offset < size:
                command_bytes = min(command_limit, size - transfer_offset)
                lane_commands.append(
                    p.rw_command(
                        opcode,
                        lane.namespace.nsid,
                        start_lba + (byte_offset + transfer_offset) // lba_size,
                        command_bytes // lba_size,
                        lane.mr,
                        lba_size=lba_size,
                        buffer_offset=slot * slot_stride + transfer_offset,
                    )
                )
                transfer_offset += command_bytes
        commands.append(lane_commands)
    return commands


def _run_batch(lanes: list[Lane], commands: list[list[bytes]]) -> None:
    requests = []
    for lane, lane_commands in zip(lanes, commands):
        requests.append(
            [lane.queue.submit(command, lane.mr) for command in lane_commands]
        )
    while True:
        pending = False
        for lane, lane_requests in zip(lanes, requests):
            if any(not request.done for request in lane_requests):
                pending = True
                lane.queue.poll()
        if not pending:
            return


def _run_command_set(
    lanes: list[Lane], commands: list[list[bytes]], queue_depth: int
) -> None:
    command_count = len(commands[0])
    for offset in range(0, command_count, queue_depth):
        _run_batch(
            lanes,
            [
                lane_commands[offset : offset + queue_depth]
                for lane_commands in commands
            ],
        )


def _measure_latency(
    lane: Lane,
    commands: list[bytes],
    queue_depth: int,
    warmup: int,
    iterations: int,
):
    for _ in range(warmup):
        _run_command_set([lane], [commands], queue_depth)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        _run_command_set([lane], [commands], queue_depth)
        samples.append(time.perf_counter_ns() - start)
    return {
        "iterations": iterations,
        "p50_us": statistics.median(samples) / 1_000.0,
        "p99_us": _percentile(samples, 0.99) / 1_000.0,
    }


def _measure_bandwidth(
    lanes: list[Lane],
    commands: list[list[bytes]],
    size: int,
    queue_depth: int,
    total_bytes: int,
    repeats: int,
):
    work_per_round = queue_depth * len(lanes)
    iterations = max(work_per_round * 4, math.ceil(total_bytes / size))
    iterations = min(iterations, 1_048_576)
    rounds = math.ceil(iterations / work_per_round)
    iterations = rounds * work_per_round
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(rounds):
            _run_command_set(lanes, commands, queue_depth)
        elapsed = (time.perf_counter_ns() - start) / 1e9
        samples.append(iterations * size / elapsed)
    bytes_per_second = statistics.median(samples)
    return {
        "iterations": iterations,
        "repeats": repeats,
        "GBps": bytes_per_second / 1e9,
        "Gbps": bytes_per_second * 8 / 1e9,
    }


def _verify_round_trip(lane: Lane, size: int, start_lba: int) -> None:
    namespace = lane.namespace
    blocks = size // namespace.lba_size
    with torch.cuda.device(lane.tensor.device):
        lane.tensor[:size].fill_(PATTERN)
        destination = torch.zeros(size, dtype=torch.uint8, device=lane.tensor.device)
        ibcuda.synchronize()
        destination_mr = lane.controller.register_gpu(destination)
        try:
            namespace.write(lane.mr, start_lba, blocks)
            namespace.flush()
            namespace.read(destination_mr, start_lba, blocks)
            if not bool(torch.all(destination == PATTERN).item()):
                raise RuntimeError("NVMe/RDMA GPU round-trip verification failed")
        finally:
            destination_mr.close()


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("torch cannot access CUDA")
    max_size = max(args.sizes)
    allocation_size = max_size * args.queue_depth
    lanes = _make_lanes(args, allocation_size)
    try:
        namespace = lanes[0].namespace
        lba_size = namespace.lba_size
        if any(size % lba_size for size in args.sizes):
            raise ValueError(
                f"every size must be a multiple of the {lba_size}-byte LBA"
            )
        transfer_limit = min(lane.controller.max_transfer_bytes for lane in lanes)
        required_bytes = args.qps * args.queue_depth * max_size
        namespace_bytes = namespace.info.capacity_lbas * lba_size
        start_byte = args.start_lba * lba_size
        if start_byte + required_bytes > namespace_bytes:
            raise ValueError("benchmark slots exceed the namespace capacity")

        results = []
        for size in args.sizes:
            row = {"size_bytes": size}
            for operation in args.operations:
                opcode = p.OPC_READ if operation == "read" else p.OPC_WRITE
                commands = _make_commands(
                    lanes,
                    opcode,
                    size,
                    args.queue_depth,
                    max_size,
                    args.start_lba,
                    transfer_limit,
                )
                if operation == "write":
                    with torch.cuda.device(args.gpu):
                        for lane in lanes:
                            lane.tensor.fill_(PATTERN)
                        ibcuda.synchronize()
                else:
                    precondition = _make_commands(
                        lanes,
                        p.OPC_WRITE,
                        size,
                        args.queue_depth,
                        max_size,
                        args.start_lba,
                        transfer_limit,
                    )
                    _run_command_set(lanes, precondition, args.queue_depth)
                    namespace.flush()
                    with torch.cuda.device(args.gpu):
                        for lane in lanes:
                            lane.tensor.zero_()
                        ibcuda.synchronize()

                latency_iterations = max(
                    100,
                    min(args.latency_iterations, (256 * 1024 * 1024) // size),
                )
                commands_per_transfer = math.ceil(size / transfer_limit)
                latency_commands = commands[0][:commands_per_transfer]
                latency = _measure_latency(
                    lanes[0],
                    latency_commands,
                    args.queue_depth,
                    args.warmup,
                    latency_iterations,
                )
                bandwidth = _measure_bandwidth(
                    lanes,
                    commands,
                    size,
                    args.queue_depth,
                    args.bandwidth_total_bytes,
                    args.bandwidth_repeats,
                )
                if operation == "read":
                    ibcuda.flush_gpudirect_writes()
                row[operation] = {
                    "commands_per_transfer": commands_per_transfer,
                    "latency": latency,
                    "bandwidth": bandwidth,
                }
            results.append(row)

        lanes[0].namespace.flush()
        _verify_round_trip(lanes[0], max_size, args.start_lba)
        hca = lanes[0].controller.io.context.name
        return {
            "config": {
                "target": args.target,
                "source": args.source,
                "port": args.port,
                "subsystem_nqn": args.subsystem_nqn,
                "nsid": args.nsid,
                "gpu": args.gpu,
                "gpu_name": torch.cuda.get_device_name(args.gpu),
                "hca": _hca_metadata(hca),
                "qps": args.qps,
                "queue_depth_per_qp": args.queue_depth,
                "lba_size": lba_size,
                "target_mdts": transfer_limit,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "results": results,
            "verified": True,
        }
    finally:
        for lane in reversed(lanes):
            lane.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--port", type=int, default=4420)
    parser.add_argument("--subsystem-nqn", required=True)
    parser.add_argument("--nsid", type=int, default=1)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--expected-hca")
    parser.add_argument("--qps", type=int, default=1)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--start-lba", type=int, default=0)
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--operations", default="write,read")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--latency-iterations", type=int, default=10_000)
    parser.add_argument("--bandwidth-total-bytes", type=int, default=2 << 30)
    parser.add_argument("--bandwidth-repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    args.sizes = [int(size) for size in args.sizes.split(",")]
    args.operations = [operation.strip() for operation in args.operations.split(",")]
    if not args.sizes or any(size <= 0 for size in args.sizes):
        parser.error("sizes must be positive")
    if not args.operations or any(
        op not in ("read", "write") for op in args.operations
    ):
        parser.error("operations must contain only read and write")
    if args.qps <= 0:
        parser.error("QP count must be positive")
    if args.queue_depth <= 0 or args.queue_depth >= p.MAX_QUEUE_DEPTH:
        parser.error("queue depth must be between 1 and 255")
    if args.start_lba < 0:
        parser.error("start LBA must be non-negative")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
