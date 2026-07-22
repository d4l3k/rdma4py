"""Lazy host-side bindings for exporting an mlx5 QP to GPUNetIO."""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import re
import threading

_HANDLERS = {
    "auto": 0,
    "cpu": 1,
    "gpu": 2,
}
_CUDA_HOST_REGISTER_PORTABLE = 0x01
_CUDA_HOST_REGISTER_DEVICE_MAP = 0x02
_MLX5_CQE_INVALID = 0xF
_host_memory_lock = threading.Lock()
_host_memory_refs = {}
_lib = None


class _CudaHostMappings:
    def __init__(self, cuda, pages):
        self._cuda = cuda
        self._pages = pages

    def close(self):
        error = None
        remaining = []
        with _host_memory_lock:
            for base in self._pages:
                references = _host_memory_refs[base]
                if references > 1:
                    _host_memory_refs[base] = references - 1
                    continue
                status = self._cuda.cuMemHostUnregister(ctypes.c_void_p(base))
                if status == 0:
                    del _host_memory_refs[base]
                else:
                    remaining.append(base)
                    error = error or status
        self._pages = remaining
        if error is not None:
            raise RuntimeError(f"cuMemHostUnregister failed with CUDA status {error}")


def _load_cuda_driver():
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise RuntimeError("libcuda.so.1 is required by GPUNetIO") from exc
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda.cuCtxGetDevice.restype = ctypes.c_int
    cuda.cuDeviceGetPCIBusId.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    cuda.cuDeviceGetPCIBusId.restype = ctypes.c_int
    cuda.cuMemHostRegister_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    cuda.cuMemHostRegister_v2.restype = ctypes.c_int
    cuda.cuMemHostGetDevicePointer_v2.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    cuda.cuMemHostGetDevicePointer_v2.restype = ctypes.c_int
    cuda.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
    cuda.cuMemHostUnregister.restype = ctypes.c_int
    if cuda.cuInit(0) != 0:
        raise RuntimeError("CUDA driver initialization failed")
    return cuda


def _validate_queue_layout(info):
    if info["sq_wqe_stride"] != 64:
        raise ValueError("GPUNetIO requires 64-byte mlx5 SQ WQEs")
    if info["rcv_wqe_size"] != 16:
        raise ValueError("GPUNetIO requires 16-byte mlx5 receive WQEs")
    if info["sq_cqe_size"] != 64 or info["rq_cqe_size"] != 64:
        raise ValueError("GPUNetIO requires 64-byte mlx5 CQEs")


def _register_queue_memory(info):
    ranges = [
        ("sq_wqe_addr", info["sq_wqe_num"] * info["sq_wqe_stride"]),
        ("rq_wqe_addr", info["rq_wqe_num"] * info["rcv_wqe_size"]),
        ("sq_dbrec", 4),
        ("rq_dbrec", 4),
        ("sq_cqe_addr", info["sq_cqe_num"] * info["sq_cqe_size"]),
        ("sq_cq_dbrec", 4),
        ("rq_cqe_addr", info["rq_cqe_num"] * info["rq_cqe_size"]),
        ("rq_cq_dbrec", 4),
    ]
    page_size = mmap.PAGESIZE
    pages = set()
    for key, size in ranges:
        address = info[key]
        base = address & ~(page_size - 1)
        end = (address + size + page_size - 1) & ~(page_size - 1)
        pages.update(range(base, end, page_size))
    pages = sorted(pages)

    cuda = _load_cuda_driver()
    acquired = []
    try:
        with _host_memory_lock:
            for base in pages:
                references = _host_memory_refs.get(base, 0)
                if references == 0:
                    status = cuda.cuMemHostRegister_v2(
                        ctypes.c_void_p(base),
                        page_size,
                        _CUDA_HOST_REGISTER_PORTABLE | _CUDA_HOST_REGISTER_DEVICE_MAP,
                    )
                    if status != 0:
                        raise RuntimeError(
                            "cuMemHostRegister failed for mlx5 queue memory "
                            f"with CUDA status {status}"
                        )
                _host_memory_refs[base] = references + 1
                acquired.append(base)
    except Exception:
        if acquired:
            _CudaHostMappings(cuda, acquired).close()
        raise

    mappings = _CudaHostMappings(cuda, acquired)
    try:
        mapped = dict(info)
        for key, _ in ranges:
            device = ctypes.c_uint64()
            status = cuda.cuMemHostGetDevicePointer_v2(
                ctypes.byref(device), ctypes.c_void_p(info[key]), 0
            )
            if status != 0:
                raise RuntimeError(
                    "cuMemHostGetDevicePointer failed for mlx5 queue memory "
                    f"with CUDA status {status}"
                )
            mapped[key] = device.value
        return mapped, mappings
    except Exception:
        mappings.close()
        raise


def _initialize_cqs(info):
    cqs = []
    seen = set()
    for prefix in ("sq", "rq"):
        address = info[f"{prefix}_cqe_addr"]
        count = info[f"{prefix}_cqe_num"]
        size = info[f"{prefix}_cqe_size"]
        identity = (address, count, size)
        if identity in seen:
            continue
        seen.add(identity)
        cqs.append((prefix, address, count, size))
        for index in range(count):
            op_own = ctypes.c_uint8.from_address(address + index * size + size - 1)
            if op_own.value >> 4 != _MLX5_CQE_INVALID:
                raise ValueError(
                    "GPUNetIO export requires fresh CQs with no completions"
                )

    for prefix, address, count, size in cqs:
        for index in range(count):
            op_own = ctypes.c_uint8.from_address(address + index * size + size - 1)
            op_own.value |= 1
        ctypes.c_uint32.from_address(info[f"{prefix}_cq_dbrec"]).value = 0


class GPUNetIOError(RuntimeError):
    """An error returned by a DOCA GPUNetIO control-path function."""

    def __init__(self, function: str, status: int):
        self.function = function
        self.status = int(status)
        super().__init__(f"{function} failed with DOCA status {status}")


def _configure_library(lib):
    void_p = ctypes.c_void_p
    lib.doca_gpu_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(void_p)]
    lib.doca_gpu_create.restype = ctypes.c_int
    lib.doca_gpu_destroy.argtypes = [void_p]
    lib.doca_gpu_destroy.restype = ctypes.c_int
    lib.doca_gpu_verbs_bridge_export_qp.argtypes = [
        void_p,
        ctypes.c_uint32,
        void_p,
        ctypes.c_uint16,
        void_p,
        void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        void_p,
        ctypes.c_uint32,
        void_p,
        ctypes.c_uint32,
        void_p,
        ctypes.c_uint16,
        void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        void_p,
        ctypes.c_uint32,
        void_p,
        ctypes.c_int,
        ctypes.POINTER(void_p),
    ]
    lib.doca_gpu_verbs_bridge_export_qp.restype = ctypes.c_int
    lib.doca_gpu_verbs_get_qp_dev.argtypes = [void_p, ctypes.POINTER(void_p)]
    lib.doca_gpu_verbs_get_qp_dev.restype = ctypes.c_int
    lib.doca_gpu_verbs_unexport_qp.argtypes = [void_p, void_p]
    lib.doca_gpu_verbs_unexport_qp.restype = ctypes.c_int
    lib.doca_gpu_verbs_cpu_proxy_enabled.argtypes = [
        void_p,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.doca_gpu_verbs_cpu_proxy_enabled.restype = ctypes.c_int
    lib.doca_gpu_verbs_cpu_proxy_progress.argtypes = [void_p]
    lib.doca_gpu_verbs_cpu_proxy_progress.restype = ctypes.c_int
    return lib


def _load_library():
    global _lib
    if _lib is not None:
        return _lib
    names = [
        ctypes.util.find_library("doca_gpunetio"),
        "libdoca_gpunetio.so.2",
        "libdoca_gpunetio.so",
    ]
    error = None
    for name in names:
        if not name:
            continue
        try:
            _lib = _configure_library(
                ctypes.CDLL(name, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            )
            return _lib
        except OSError as exc:
            error = exc
    raise RuntimeError(
        "DOCA GPUNetIO runtime was not found; install doca-sdk-gpunetio"
    ) from error


def _cuda_bus_id(index=None) -> str:
    cuda = _load_cuda_driver()
    if index is None:
        device = ctypes.c_int()
        if cuda.cuCtxGetDevice(ctypes.byref(device)) != 0:
            raise RuntimeError(
                "no CUDA context is current; initialize the target GPU first"
            )
        index = device.value
    buf = ctypes.create_string_buffer(32)
    rc = cuda.cuDeviceGetPCIBusId(buf, len(buf), int(index))
    if rc != 0:
        raise RuntimeError(f"cuDeviceGetPCIBusId failed with CUDA status {rc}")
    return buf.value.decode()


def _normalize_bus_id(gpu) -> str:
    value = (
        _cuda_bus_id(gpu if gpu is not None else None)
        if not isinstance(gpu, str)
        else gpu
    )
    match = re.fullmatch(
        r"(?:(?P<domain>[0-9a-fA-F]{4,8}):)?(?P<bus>[0-9a-fA-F]{2}):"
        r"(?P<device>[0-9a-fA-F]{2})\.(?P<function>[0-7])",
        value,
    )
    if match is None:
        raise ValueError(f"invalid GPU PCI address: {value!r}")
    domain = (match.group("domain") or "0000")[-4:]
    return "%s:%s:%s.%s" % (
        domain.lower(),
        match.group("bus").lower(),
        match.group("device").lower(),
        match.group("function"),
    )


def _check(function: str, status: int):
    if status != 0:
        raise GPUNetIOError(function, status)


class DeviceQP:
    """A GPUNetIO device handle exported from an mlx5 RC queue pair.

    Export permanently transfers the QP and both CQs to the external GPU data
    path. Keep this object alive while kernels use :attr:`device_ptr`, and
    close it before closing the underlying ibverbs resources.
    """

    def __init__(self):
        raise TypeError("use DeviceQP.export(qp, ...)")

    @classmethod
    def export(cls, qp, *, gpu=None, nic_handler: str = "gpu") -> "DeviceQP":
        """Export ``qp`` for the current CUDA context.

        ``gpu`` may be a CUDA device index or PCI address. The default uses the
        current CUDA context. ``nic_handler='gpu'`` requires a direct GPU
        doorbell; use ``'auto'`` only when CPU-proxy fallback is acceptable.
        The RC QP and both CQs must be fresh and use no SRQ. Export permanently
        transfers their data-path ownership, even after this handle is closed.
        """
        if nic_handler not in _HANDLERS:
            raise ValueError("nic_handler must be one of %s" % ", ".join(_HANDLERS))
        if not hasattr(qp, "_mlx5dv_bridge_info"):
            raise TypeError("qp must be an ibverbs.QP")

        lib = _load_library()
        gpu_bus_id = _normalize_bus_id(gpu)
        current_bus_id = _normalize_bus_id(_cuda_bus_id())
        if gpu_bus_id != current_bus_id:
            raise RuntimeError(
                f"GPU {gpu_bus_id} is not the current CUDA context "
                f"({current_bus_id})"
            )
        info = qp._mlx5dv_bridge_info()
        _validate_queue_layout(info)
        _initialize_cqs(info)
        info, mappings = _register_queue_memory(info)
        gpu_handle = ctypes.c_void_p()
        exported = ctypes.c_void_p()
        device = ctypes.c_void_p()
        try:
            _check(
                "doca_gpu_create",
                lib.doca_gpu_create(
                    gpu_bus_id.encode("ascii"), ctypes.byref(gpu_handle)
                ),
            )
            args = (
                gpu_handle,
                info["sq_qpn"],
                ctypes.c_void_p(info["sq_wqe_addr"]),
                info["sq_wqe_num"],
                ctypes.c_void_p(info["sq_dbrec"]),
                ctypes.c_void_p(info["sq_db"]),
                info["uar_size"],
                info["sq_cqn"],
                ctypes.c_void_p(info["sq_cqe_addr"]),
                info["sq_cqe_num"],
                ctypes.c_void_p(info["sq_cq_dbrec"]),
                info["rq_qpn"],
                ctypes.c_void_p(info["rq_wqe_addr"]),
                info["rq_wqe_num"],
                ctypes.c_void_p(info["rq_dbrec"]),
                info["rcv_wqe_size"],
                info["rq_cqn"],
                ctypes.c_void_p(info["rq_cqe_addr"]),
                info["rq_cqe_num"],
                ctypes.c_void_p(info["rq_cq_dbrec"]),
                _HANDLERS[nic_handler],
                ctypes.byref(exported),
            )
            _check(
                "doca_gpu_verbs_bridge_export_qp",
                lib.doca_gpu_verbs_bridge_export_qp(*args),
            )
            _check(
                "doca_gpu_verbs_get_qp_dev",
                lib.doca_gpu_verbs_get_qp_dev(exported, ctypes.byref(device)),
            )
            proxy = ctypes.c_uint8()
            _check(
                "doca_gpu_verbs_cpu_proxy_enabled",
                lib.doca_gpu_verbs_cpu_proxy_enabled(exported, ctypes.byref(proxy)),
            )
            if nic_handler == "gpu" and proxy.value:
                raise RuntimeError(
                    "DOCA did not provide the requested direct GPU doorbell"
                )
            qp._acquire_external_datapath()
        except Exception:
            if exported.value:
                lib.doca_gpu_verbs_unexport_qp(gpu_handle, exported)
            if gpu_handle.value:
                lib.doca_gpu_destroy(gpu_handle)
            try:
                mappings.close()
            except Exception:
                pass
            raise

        self = cls.__new__(cls)
        self._lib = lib
        self._gpu = gpu_handle
        self._exported = exported
        self._device = device
        self._qp = qp
        self._mappings = mappings
        self.gpu_bus_id = gpu_bus_id
        self.nic_handler = nic_handler
        self.cpu_proxy = bool(proxy.value)
        return self

    @property
    def device_ptr(self) -> int:
        """GPU address of ``struct doca_gpu_dev_verbs_qp`` for kernels."""
        if not self._device.value:
            raise RuntimeError("DeviceQP is closed")
        return int(self._device.value)

    @property
    def closed(self) -> bool:
        """Whether the GPUNetIO handle has been unexported."""
        return not bool(self._device.value)

    def progress(self) -> None:
        """Ring pending doorbells once when CPU-proxy mode is active."""
        if self.closed:
            raise RuntimeError("DeviceQP is closed")
        if not self.cpu_proxy:
            raise RuntimeError("this QP uses direct GPU doorbells")
        _check(
            "doca_gpu_verbs_cpu_proxy_progress",
            self._lib.doca_gpu_verbs_cpu_proxy_progress(self._exported),
        )

    def close(self) -> None:
        """Unexport after GPU work ends; the ibverbs QP stays externally owned."""
        if self.closed:
            return
        self._mappings.close()
        rc = self._lib.doca_gpu_verbs_unexport_qp(self._gpu, self._exported)
        if rc != 0:
            raise GPUNetIOError("doca_gpu_verbs_unexport_qp", rc)
        self._qp._release_external_datapath()
        rc = self._lib.doca_gpu_destroy(self._gpu)
        self._device = ctypes.c_void_p()
        self._exported = ctypes.c_void_p()
        self._gpu = ctypes.c_void_p()
        self._qp = None
        self._mappings = None
        if rc != 0:
            raise GPUNetIOError("doca_gpu_destroy", rc)

    def __int__(self):
        return self.device_ptr

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


__all__ = ["DeviceQP", "GPUNetIOError"]
