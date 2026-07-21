"""Opt-in, read-only tests against a real NVMe/RDMA target."""

from __future__ import annotations

import os

import nvmeof
import pytest

pytestmark = pytest.mark.integration

TARGET = os.environ.get("NVME4PY_TARGET")
SOURCE = os.environ.get("NVME4PY_SOURCE")
SUBSYSTEM_NQN = os.environ.get("NVME4PY_SUBSYSTEM_NQN")
NSID = int(os.environ.get("NVME4PY_NSID", "1"))

if not TARGET or not SUBSYSTEM_NQN:
    pytest.skip(
        "set NVME4PY_TARGET and NVME4PY_SUBSYSTEM_NQN for target tests",
        allow_module_level=True,
    )


@pytest.fixture()
def namespace():
    controller = nvmeof.Controller.connect(TARGET, SUBSYSTEM_NQN, source=SOURCE)
    yield controller.namespace(NSID)
    controller.close()


def test_read_one_lba_to_host(namespace):
    with namespace.controller.allocate(namespace.lba_size) as buffer:
        namespace.read(buffer, 0, 1)
        assert len(buffer.read()) == namespace.lba_size


@pytest.mark.gpu
def test_read_one_lba_directly_to_gpu(namespace):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tensor = torch.empty(namespace.lba_size, dtype=torch.uint8, device="cuda:0")
    with namespace.controller.register_gpu(tensor) as gpu_mr:
        with torch.cuda.device(tensor.device):
            namespace.read(gpu_mr, 0, 1)
    assert tensor.cpu().numel() == namespace.lba_size


@pytest.mark.gpu
def test_disposable_target_gpu_round_trip(namespace):
    if os.environ.get("NVME4PY_DESTRUCTIVE") != "1":
        pytest.skip("set NVME4PY_DESTRUCTIVE=1 only for a disposable namespace")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    slba = int(os.environ.get("NVME4PY_TEST_SLBA", "0"))
    source = torch.arange(namespace.lba_size, dtype=torch.int64, device="cuda:0")
    source = source.remainder(251).to(torch.uint8)
    destination = torch.zeros_like(source)
    with namespace.controller.register_gpu(source) as source_mr:
        with namespace.controller.register_gpu(destination) as destination_mr:
            with torch.cuda.device(source.device):
                namespace.write(source_mr, slba, 1)
                namespace.flush()
                namespace.read(destination_mr, slba, 1)
    assert torch.equal(source, destination)
