"""Shared fixtures for the efa test suite.

The integration tests run against whatever EFA hardware is present on the
host. Tests that need a feature the hardware lacks are skipped, never failed.
"""

from __future__ import annotations

import os

# torch's CUDA memory must be VMM-backed to be dma-buf exportable for
# GPUDirect. Set this before torch is ever imported (conftest loads first).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import efa
import pytest


@pytest.fixture(scope="session")
def efa_devices():
    if os.environ.get("RDMA4PY_SKIP_HARDWARE_TESTS") == "1":
        pytest.skip("hardware tests disabled")
    devs = efa.get_efa_device_list()
    if not devs:
        pytest.skip("no EFA devices present")
    return devs


@pytest.fixture()
def ctx(efa_devices):
    context = efa_devices[0].open()
    yield context
    context.close()


@pytest.fixture()
def pd(ctx):
    pd = ctx.alloc_pd()
    yield pd
    pd.close()


@pytest.fixture()
def efa_caps(ctx):
    return efa.EfaDeviceCaps(ctx.query_efa_device().device_caps)


def require_caps(caps, needed):
    if not (caps & needed):
        pytest.skip(f"device lacks {needed!r}")
