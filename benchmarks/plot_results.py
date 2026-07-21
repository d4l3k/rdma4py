#!/usr/bin/env python3
"""Generate the benchmark overview charts from the backend reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["svg.hashsalt"] = "rdma4py-benchmarks"
plt.rcParams["svg.fonttype"] = "none"

KIB = 1 << 10
MIB = 1 << 20
SIZES = [4 * KIB, 16 * KIB, 64 * KIB, 256 * KIB, MIB, 4 * MIB, 16 * MIB, 64 * MIB]
SIZE_LABELS = [
    "4 KiB",
    "16 KiB",
    "64 KiB",
    "256 KiB",
    "1 MiB",
    "4 MiB",
    "16 MiB",
    "64 MiB",
]

RESULTS = {
    "EFA: 4 lanes x 4 QPs": {
        "write_bandwidth": [
            8.665,
            25.363,
            39.079,
            45.875,
            48.071,
            48.672,
            48.831,
            48.871,
        ],
        "read_bandwidth": [
            7.539,
            22.817,
            37.419,
            42.609,
            46.314,
            47.865,
            48.718,
            48.877,
        ],
        "write_latency": [23.07, 25.43, 32.97, 47.97, 112.96, 371.20, 1403.95, 5524.06],
        "read_latency": [33.58, 35.26, 42.00, 55.86, 124.38, 395.89, 1467.28, 5604.49],
    },
    "ibverbs: 16 QPs": {
        "write_bandwidth": [
            42.347,
            46.910,
            48.126,
            48.470,
            48.549,
            48.525,
            48.459,
            48.433,
        ],
        "read_bandwidth": None,
        "write_latency": [
            5.364,
            5.842,
            6.844,
            13.046,
            29.260,
            94.247,
            353.884,
            1390.150,
        ],
        "read_latency": None,
    },
    "NVMe-oF: 64/32 QPs": {
        "write_bandwidth": [
            0.395,
            1.478,
            6.118,
            18.709,
            26.074,
            20.963,
            21.185,
            20.628,
        ],
        "read_bandwidth": [0.546, 2.133, 9.327, 22.358, 27.850, 18.489, 20.810, 21.507],
        "write_latency": [
            48.769,
            57.064,
            73.756,
            140.620,
            269.178,
            359.132,
            1700.682,
            5747.600,
        ],
        "read_latency": [
            96.147,
            108.863,
            121.524,
            188.639,
            311.084,
            635.913,
            2543.264,
            8897.215,
        ],
    },
}

COLORS = {"write": "#0f766e", "read": "#c2410c"}
LATENCY_TITLES = {
    "EFA: 4 lanes x 4 QPs": "EFA: 1 QP, serialized",
    "ibverbs: 16 QPs": "ibverbs: 1 QP, serialized",
    "NVMe-oF: 64/32 QPs": "NVMe-oF: 1 I/O QP, serialized",
}


def _style_axis(axis, title: str) -> None:
    axis.set_title(title, fontsize=12, fontweight="bold", loc="left")
    axis.set_xscale("log", base=2)
    axis.set_xlim(SIZES[0] / 1.3, SIZES[-1] * 1.3)
    axis.set_xticks(SIZES, SIZE_LABELS, rotation=30, ha="right")
    axis.grid(True, which="major", color="#d1d5db", linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot(metric: str, output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 10), sharex=True)
    for axis, (title, values) in zip(axes, RESULTS.items()):
        display_title = LATENCY_TITLES[title] if metric == "latency" else title
        _style_axis(axis, display_title)
        for operation in ("write", "read"):
            samples = values[f"{operation}_{metric}"]
            if samples is None:
                continue
            axis.plot(
                SIZES,
                samples,
                color=COLORS[operation],
                linewidth=2.2,
                marker="o",
                markersize=4.5,
                label=operation.capitalize(),
            )
        if title.startswith("NVMe-oF"):
            axis.axvline(MIB, color="#6b7280", linestyle="--", linewidth=1)
            axis.text(
                1.12 * MIB,
                0.08 if metric == "bandwidth" else 0.89,
                "32-QP MDTS-split region",
                color="#4b5563",
                fontsize=8.5,
                transform=axis.get_xaxis_transform(),
            )
        axis.legend(frameon=False, loc="upper left")

    if metric == "bandwidth":
        figure.suptitle(
            "Measured GPUDirect bandwidth by message size",
            fontsize=16,
            fontweight="bold",
        )
        for axis in axes:
            axis.set_ylim(0, 54)
            axis.set_ylabel("GB/s")
        description = "Bandwidth charts for EFA four-lane, ibverbs 16-QP, and NVMe-oF 64/32-QP GPUDirect benchmarks."
    else:
        figure.suptitle(
            "Measured GPUDirect completion latency by message size",
            fontsize=16,
            fontweight="bold",
        )
        for axis in axes:
            axis.set_yscale("log")
            axis.set_ylim(4, 14_000)
            axis.set_ylabel("p50 latency (us)")
        description = "Latency charts for serialized EFA, ibverbs, and NVMe-oF GPUDirect transfers."

    axes[-1].set_xlabel("Message or tensor size")
    figure.text(
        0.5,
        0.01,
        "Measured on different systems; use each backend report for topology and method details.",
        ha="center",
        color="#4b5563",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.04, 0.04, 0.98, 0.96))
    figure.savefig(
        output,
        format="svg",
        metadata={
            "Title": figure._suptitle.get_text(),
            "Description": description,
            "Creator": "rdma4py benchmarks/plot_results.py",
            "Date": "2026-07-21",
        },
    )
    plt.close(figure)
    lines = output.read_text().splitlines()
    output.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    _plot("bandwidth", output_dir / "bandwidth.svg")
    _plot("latency", output_dir / "latency.svg")


if __name__ == "__main__":
    main()
