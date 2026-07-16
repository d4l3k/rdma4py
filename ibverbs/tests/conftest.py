"""Shared fixtures for the ibverbs test suite.

The integration tests run against whatever RDMA hardware is present on the
host. Tests that need a feature the hardware lacks are skipped, never failed.
"""

from __future__ import annotations

import os

# torch's CUDA memory must be VMM-backed to be dma-buf exportable for
# GPUDirect. Set this before torch is ever imported (conftest loads first).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pytest

import ibverbs as ib


def _read_sysfs(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _is_link_local(raw: bytes) -> bool:
    # IPv6 link-local (fe80::/10) GIDs do not loop back / route to themselves.
    return raw[0] == 0xFE and (raw[1] & 0xC0) == 0x80


def find_roce_gid(ctx, dev_name: str, port: int):
    """Return ``(gid_index, Gid)`` for the best usable RoCE GID on ``port``.

    Ranks candidates so a routable **RoCE v2, non-link-local** GID (which is
    the one that actually loops back and routes on this fabric) wins over a
    link-local or RoCE v1 entry.
    """
    pa = ctx.query_port(port)
    best = None  # (score, index, gid)
    for idx in range(pa.gid_tbl_len):
        gid = ctx.query_gid(port, idx)
        if gid.raw == b"\x00" * 16:
            continue
        gtype = _read_sysfs(
            f"/sys/class/infiniband/{dev_name}/ports/{port}/gid_attrs/types/{idx}"
        )
        score = 0
        if gtype and "v2" in gtype.lower():
            score += 4
        if not _is_link_local(gid.raw):
            score += 2
        if best is None or score > best[0]:
            best = (score, idx, gid)
    if best is None:
        pytest.skip(f"no usable GID on {dev_name} port {port}")
    return best[1], best[2]


def active_ports():
    """Yield ``(device, port)`` for every ACTIVE port on the host."""
    out = []
    for dev in ib.get_device_list():
        ctx = dev.open()
        try:
            da = ctx.query_device()
            for port in range(1, da.phys_port_cnt + 1):
                pa = ctx.query_port(port)
                if pa.state == ib.PortState.ACTIVE:
                    out.append((dev.name, port))
        finally:
            ctx.close()
    return out


@pytest.fixture(scope="session")
def all_devices():
    devs = ib.get_device_list()
    if not devs:
        pytest.skip("no RDMA devices present")
    return devs


@pytest.fixture(scope="session")
def active_port_list():
    ports = active_ports()
    if not ports:
        pytest.skip("no ACTIVE RDMA ports present")
    return ports


@pytest.fixture()
def ctx(all_devices):
    context = all_devices[0].open()
    yield context
    context.close()


@pytest.fixture()
def dev_name(all_devices):
    return all_devices[0].name


@pytest.fixture()
def first_active(active_port_list):
    """``(device_name, port)`` of the first ACTIVE port."""
    return active_port_list[0]
