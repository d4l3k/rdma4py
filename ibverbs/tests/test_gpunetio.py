"""Unit and hardware-smoke tests for the optional GPUNetIO bridge."""

from __future__ import annotations

import ctypes

import ibverbs as ib
import ibverbs.gpunetio as gpunetio
import pytest
from ibverbs.gpunetio import _runtime


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeLibrary:
    def __init__(self, *, proxy=False):
        self.calls = []
        self.proxy = proxy
        self.doca_gpu_create = FakeFunction(self._create)
        self.doca_gpu_destroy = FakeFunction(self._destroy)
        self.doca_gpu_verbs_bridge_export_qp = FakeFunction(self._export)
        self.doca_gpu_verbs_get_qp_dev = FakeFunction(self._get_device)
        self.doca_gpu_verbs_unexport_qp = FakeFunction(self._unexport)
        self.doca_gpu_verbs_cpu_proxy_enabled = FakeFunction(self._proxy)
        self.doca_gpu_verbs_cpu_proxy_progress = FakeFunction(self._progress)

    def _create(self, bus, out):
        self.calls.append(("create", bus))
        out._obj.value = 0x1000
        return 0

    def _destroy(self, gpu):
        self.calls.append(("destroy", gpu.value))
        return 0

    def _export(self, *args):
        self.calls.append(("export", args))
        args[-1]._obj.value = 0x2000
        return 0

    def _get_device(self, exported, out):
        self.calls.append(("get_device", exported.value))
        out._obj.value = 0x3000
        return 0

    def _unexport(self, gpu, exported):
        self.calls.append(("unexport", gpu.value, exported.value))
        return 0

    def _proxy(self, exported, out):
        out._obj.value = int(self.proxy)
        return 0

    def _progress(self, exported):
        self.calls.append(("progress", exported.value))
        return 0


class FakeQP:
    def __init__(self):
        self.transferred = False
        self.active = 0

    def _mlx5dv_bridge_info(self):
        self.transferred = True
        return {
            "sq_qpn": 1,
            "sq_wqe_addr": 0x10,
            "sq_wqe_num": 16,
            "sq_wqe_stride": 64,
            "sq_dbrec": 0x20,
            "sq_db": 0x30,
            "uar_size": 4096,
            "sq_cqn": 2,
            "sq_cqe_addr": 0x40,
            "sq_cqe_num": 32,
            "sq_cqe_size": 64,
            "sq_cq_dbrec": 0x50,
            "rq_qpn": 1,
            "rq_wqe_addr": 0x60,
            "rq_wqe_num": 16,
            "rq_dbrec": 0x70,
            "rcv_wqe_size": 16,
            "rq_cqn": 2,
            "rq_cqe_addr": 0x40,
            "rq_cqe_num": 32,
            "rq_cqe_size": 64,
            "rq_cq_dbrec": 0x50,
        }

    def _acquire_external_datapath(self):
        self.active += 1

    def _release_external_datapath(self):
        self.active -= 1


class FakeCuda:
    def __init__(self):
        self.unregistered = []

    def cuMemHostUnregister(self, address):
        self.unregistered.append(address.value)
        return 0


def test_device_qp_lifecycle(monkeypatch):
    lib = FakeLibrary()
    qp = FakeQP()
    monkeypatch.setattr(_runtime, "_load_library", lambda: lib)
    monkeypatch.setattr(_runtime, "_cuda_bus_id", lambda index=None: "1b:00.0")
    monkeypatch.setattr(_runtime, "_initialize_cqs", lambda info: None)
    mappings = type("Mappings", (), {"close": lambda self: None})()
    monkeypatch.setattr(
        _runtime,
        "_register_queue_memory",
        lambda info: (info, mappings),
    )

    device_qp = gpunetio.DeviceQP.export(qp, gpu="00000000:1B:00.0", nic_handler="gpu")

    assert qp.transferred
    assert qp.active == 1
    assert device_qp.device_ptr == 0x3000
    assert int(device_qp) == 0x3000
    assert device_qp.gpu_bus_id == "0000:1b:00.0"
    assert not device_qp.cpu_proxy
    with pytest.raises(RuntimeError, match="direct GPU"):
        device_qp.progress()
    device_qp.close()
    device_qp.close()
    assert device_qp.closed
    assert qp.active == 0
    assert [call[0] for call in lib.calls].count("unexport") == 1


def test_device_qp_explicit_cpu_proxy(monkeypatch):
    lib = FakeLibrary(proxy=True)
    monkeypatch.setattr(_runtime, "_load_library", lambda: lib)
    monkeypatch.setattr(_runtime, "_cuda_bus_id", lambda index=None: "1b:00.0")
    monkeypatch.setattr(_runtime, "_initialize_cqs", lambda info: None)
    mappings = type("Mappings", (), {"close": lambda self: None})()
    monkeypatch.setattr(
        _runtime,
        "_register_queue_memory",
        lambda info: (info, mappings),
    )
    device_qp = gpunetio.DeviceQP.export(FakeQP(), gpu="1b:00.0", nic_handler="auto")
    assert device_qp.cpu_proxy
    device_qp.progress()
    assert ("progress", 0x2000) in lib.calls
    device_qp.close()


def test_device_qp_rejects_bad_handler_before_transfer():
    qp = FakeQP()
    with pytest.raises(ValueError, match="nic_handler"):
        gpunetio.DeviceQP.export(qp, gpu="1b:00.0", nic_handler="other")
    assert not qp.transferred


def test_device_qp_requires_selected_gpu_context(monkeypatch):
    qp = FakeQP()
    monkeypatch.setattr(_runtime, "_load_library", FakeLibrary)
    monkeypatch.setattr(
        _runtime,
        "_cuda_bus_id",
        lambda index=None: "2b:00.0" if index is None else "1b:00.0",
    )
    with pytest.raises(RuntimeError, match="current CUDA context"):
        gpunetio.DeviceQP.export(qp, gpu=0)
    assert not qp.transferred


def test_initialize_cqs_sets_empty_owner_phase():
    cq = (ctypes.c_uint8 * 128)()
    cq[63] = 0xF0
    cq[127] = 0xF0
    dbrec = ctypes.c_uint32(123)
    info = FakeQP()._mlx5dv_bridge_info()
    info.update(
        sq_cqe_addr=ctypes.addressof(cq),
        sq_cqe_num=2,
        sq_cq_dbrec=ctypes.addressof(dbrec),
        rq_cqe_addr=ctypes.addressof(cq),
        rq_cqe_num=2,
        rq_cq_dbrec=ctypes.addressof(dbrec),
    )

    _runtime._validate_queue_layout(info)
    _runtime._initialize_cqs(info)

    assert cq[63] == 0xF1
    assert cq[127] == 0xF1
    assert dbrec.value == 0


def test_initialize_cqs_rejects_used_cq():
    cq = (ctypes.c_uint8 * 64)()
    cq[63] = 0
    dbrec = ctypes.c_uint32()
    info = FakeQP()._mlx5dv_bridge_info()
    info.update(
        sq_cqe_addr=ctypes.addressof(cq),
        sq_cqe_num=1,
        sq_cq_dbrec=ctypes.addressof(dbrec),
        rq_cqe_addr=ctypes.addressof(cq),
        rq_cqe_num=1,
        rq_cq_dbrec=ctypes.addressof(dbrec),
    )

    with pytest.raises(ValueError, match="fresh CQs"):
        _runtime._initialize_cqs(info)


def test_cuda_host_mapping_reference_counts_shared_pages(monkeypatch):
    page = 0x12345000
    cuda = FakeCuda()
    monkeypatch.setitem(_runtime._host_memory_refs, page, 2)
    first = _runtime._CudaHostMappings(cuda, [page])
    second = _runtime._CudaHostMappings(cuda, [page])

    first.close()
    assert _runtime._host_memory_refs[page] == 1
    assert not cuda.unregistered

    second.close()
    assert page not in _runtime._host_memory_refs
    assert cuda.unregistered == [page]


def test_bitcode_path_and_arch_validation(tmp_path):
    bitcode = tmp_path / "device.bc"
    bitcode.write_bytes(b"BC")
    assert gpunetio.bitcode_path(bitcode) == bitcode.resolve()
    with pytest.raises(ValueError, match="sm_80"):
        gpunetio.cache_bitcode_path("sm_75")
    with pytest.raises(FileNotFoundError, match="build_bitcode"):
        gpunetio.bitcode_path(tmp_path / "missing.bc")


@pytest.mark.integration
def test_mlx5_bridge_exposes_queues_and_transfers_ownership(ctx, dev_name):
    if not dev_name.startswith("mlx5"):
        pytest.skip("GPUNetIO bridge requires mlx5")
    pd = ctx.alloc_pd()
    cq = ctx.create_cq(32)
    qp = pd.create_qp(ib.QPInitAttr(send_cq=cq, recv_cq=cq, qp_type=ib.QPType.RC))
    try:
        info = qp._mlx5dv_bridge_info()
        assert info["sq_qpn"] == qp.qp_num
        assert info["sq_wqe_addr"] != 0
        assert info["rq_wqe_addr"] != 0
        assert info["sq_db"] != 0
        assert info["sq_cqe_num"] >= 32
        with pytest.raises(ib.VerbsError, match="external GPU"):
            cq.poll(1)
        with pytest.raises(ib.VerbsError, match="external GPU"):
            qp.post_recv(ib.RecvWR(wr_id=1, sg_list=[]))
    finally:
        qp.close()
        cq.close()
        pd.close()
