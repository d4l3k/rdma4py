"""High-level NVMe-oF controller and namespace APIs."""

from __future__ import annotations

import time
import uuid

import ibverbs as ib

from . import protocol as p
from .rdma import HostBuffer, RDMAQueue

_DATA_ACCESS = (
    ib.AccessFlags.LOCAL_WRITE
    | ib.AccessFlags.REMOTE_WRITE
    | ib.AccessFlags.REMOTE_READ
)


def _is_gpu_memory(buffer) -> bool:
    tensor = getattr(buffer, "tensor", None)
    return tensor is not None and bool(getattr(tensor, "is_cuda", True))


class Controller:
    """A userspace NVMe-oF/RDMA controller with one direct I/O queue."""

    def __init__(
        self,
        host: str,
        subsystem_nqn: str,
        *,
        port=p.NVME_RDMA_PORT,
        host_id=None,
        host_nqn=None,
        queue_depth=128,
        keep_alive_ms=0,
        timeout=30.0,
        source=None,
    ):
        self.host = host
        self.port = int(port)
        self.source = source
        self.subsystem_nqn = subsystem_nqn
        self.host_id = uuid.UUID(str(host_id)) if host_id else uuid.uuid4()
        self.host_nqn = host_nqn or (
            "nqn.2014-08.org.nvmexpress:uuid:%s" % self.host_id
        )
        self.keep_alive_ms = int(keep_alive_ms)
        if self.keep_alive_ms != 0:
            raise NotImplementedError(
                "non-zero keep-alive requires a background command worker"
            )
        self.timeout = float(timeout)
        self.admin = None
        self.io = None
        self.controller_id = None
        self.capabilities = None
        self.info = None
        self.max_transfer_bytes = None
        try:
            self._connect(int(queue_depth))
        except Exception:
            try:
                self.close()
            except OSError:
                pass
            raise

    @classmethod
    def connect(cls, host: str, subsystem_nqn: str, **kwargs) -> "Controller":
        return cls(host, subsystem_nqn, **kwargs)

    def _connect(self, requested_depth: int) -> None:
        if requested_depth < 2 or requested_depth > p.MAX_QUEUE_DEPTH:
            raise ValueError("queue_depth must be between 2 and 256")
        self.admin = RDMAQueue(
            self.host,
            self.port,
            qid=0,
            depth=p.ADMIN_QUEUE_DEPTH,
            source=self.source,
        )
        with HostBuffer(self.admin.pd, 1024) as data:
            data.write(
                p.connect_data(self.host_id, self.subsystem_nqn, self.host_nqn, 0xFFFF)
            )
            completion = self.admin.execute(
                p.fabrics_connect(0, p.ADMIN_QUEUE_DEPTH, data, self.keep_alive_ms),
                data_owner=data,
                timeout=self.timeout,
            )
        self.controller_id = completion.result & 0xFFFF
        auth_required = completion.result & ((1 << 17) | (1 << 18))
        if auth_required:
            raise NotImplementedError("NVMe DH-HMAC-CHAP authentication is required")

        self.capabilities = self._property_get(p.REG_CAP, 8)
        mpsmin = (self.capabilities >> 48) & 0xF
        self.controller_page_size = 1 << (12 + mpsmin)
        css = (self.capabilities >> 37) & 0xFF
        cc = p.CC_ENABLE | p.CC_IOSQES | p.CC_IOCQES | (mpsmin << p.CC_MPS_SHIFT)
        if css & (1 << 6):
            cc |= p.CC_CSS_CSI
        self._property_set(p.REG_CC, cc)
        self._wait_ready()

        with HostBuffer(self.admin.pd, 4096) as identify:
            self.admin.execute(
                p.identify(identify, controller=True),
                data_owner=identify,
                timeout=self.timeout,
            )
            self.info = p.ControllerInfo.from_bytes(identify.read())
        if self.info.controller_id != self.controller_id:
            raise RuntimeError("Connect and Identify returned different controller IDs")
        if not (self.info.sgls & (1 << 2)):
            raise RuntimeError("NVMe/RDMA target does not advertise keyed SGL support")
        if self.info.icdoff:
            raise RuntimeError("non-zero NVMe in-capsule data offsets are unsupported")

        queue_limit = (self.capabilities & 0xFFFF) + 1
        if self.info.max_commands:
            queue_limit = min(queue_limit, self.info.max_commands)
        depth = min(requested_depth, queue_limit, p.MAX_QUEUE_DEPTH)
        queue_count = (0 << 16) | 0  # Request one submission and one completion queue.
        completion = self.admin.execute(
            p.set_features(p.FEAT_NUMBER_OF_QUEUES, queue_count),
            timeout=self.timeout,
        )
        if min(completion.result & 0xFFFF, completion.result >> 16) + 1 < 1:
            raise RuntimeError("NVMe target did not allocate an I/O queue")

        self.io = RDMAQueue(
            self.host,
            self.port,
            qid=1,
            depth=depth,
            controller_id=self.controller_id,
            source=self.source,
        )
        with HostBuffer(self.io.pd, 1024) as data:
            data.write(
                p.connect_data(
                    self.host_id,
                    self.subsystem_nqn,
                    self.host_nqn,
                    self.controller_id,
                )
            )
            self.io.execute(
                p.fabrics_connect(1, depth, data),
                data_owner=data,
                timeout=self.timeout,
            )
        mdts = p.MAX_KEYED_SGL_LENGTH
        if self.info.mdts:
            mdts = min(mdts, self.controller_page_size << self.info.mdts)
        self.max_transfer_bytes = mdts

    def _property_get(self, offset: int, size: int) -> int:
        return self.admin.execute(
            p.property_get(offset, size), timeout=self.timeout
        ).result

    def _property_set(self, offset: int, value: int, size: int = 4) -> None:
        self.admin.execute(p.property_set(offset, value, size), timeout=self.timeout)

    def _wait_ready(self) -> None:
        timeout = ((self.capabilities >> 24) & 0xFF) + 1
        deadline = time.monotonic() + max(0.5, timeout * 0.5)
        while True:
            csts = self._property_get(p.REG_CSTS, 4)
            if csts & p.CSTS_FATAL:
                raise RuntimeError("NVMe controller entered fatal status")
            if csts & p.CSTS_READY:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("NVMe controller did not become ready")
            time.sleep(0.01)

    def identify_namespace(self, nsid: int) -> "Namespace":
        nsid = int(nsid)
        if nsid <= 0 or nsid > self.info.namespace_count:
            raise ValueError("namespace ID is outside the controller's range")
        with HostBuffer(self.admin.pd, 4096) as identify:
            self.admin.execute(
                p.identify(identify, nsid=nsid),
                data_owner=identify,
                timeout=self.timeout,
            )
            info = p.NamespaceInfo.from_bytes(nsid, identify.read())
        if info.metadata_size:
            raise NotImplementedError(
                "namespaces with separate metadata are unsupported"
            )
        return Namespace(self, info)

    namespace = identify_namespace

    def allocate(self, length: int) -> HostBuffer:
        """Allocate host memory registered for direct NVMe/RDMA I/O."""
        return HostBuffer(self.io.pd, length, _DATA_ACCESS)

    def register(self, tensor):
        """Register contiguous host tensor/array memory for NVMe/RDMA I/O."""
        return ib.reg_tensor(self.io.pd, tensor, _DATA_ACCESS)

    def register_gpu(self, tensor):
        """Register CUDA tensor memory for direct NVMe/RDMA I/O."""
        import ibverbs.cuda as ibcuda

        return ibcuda.register_tensor(self.io.pd, tensor, _DATA_ACCESS)

    def close(self) -> None:
        error = None
        if self.io is not None:
            try:
                self.io.close()
                self.io = None
            except OSError as exc:
                error = exc
        if self.admin is not None:
            try:
                self.admin.close()
                self.admin = None
            except OSError as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Namespace:
    """An NVM namespace accessed through the controller's direct I/O queue."""

    def __init__(self, controller: Controller, info: p.NamespaceInfo):
        self.controller = controller
        self.info = info
        self.nsid = info.nsid
        self.lba_size = info.lba_size

    def _transfer(
        self, opcode: int, buffer, slba: int, blocks=None, buffer_offset: int = 0
    ) -> None:
        slba = int(slba)
        buffer_offset = int(buffer_offset)
        available = int(buffer.length) - buffer_offset
        if blocks is None:
            if available % self.lba_size:
                raise ValueError("buffer length is not a whole number of LBAs")
            blocks = available // self.lba_size
        blocks = int(blocks)
        if blocks <= 0 or slba < 0 or slba + blocks > self.info.size_lbas:
            raise ValueError("I/O range is outside the namespace")
        total = blocks * self.lba_size
        if buffer_offset < 0 or total > available:
            raise ValueError("I/O range is outside the registered buffer")

        gpu = _is_gpu_memory(buffer)
        if gpu and opcode == p.OPC_WRITE:
            import ibverbs.cuda as ibcuda

            ibcuda.synchronize()

        max_blocks = min(65536, self.controller.max_transfer_bytes // self.lba_size)
        if max_blocks == 0:
            raise RuntimeError(
                "namespace LBA size exceeds the controller transfer limit"
            )
        remaining = blocks
        offset = buffer_offset
        current_lba = slba
        while remaining:
            count = min(remaining, max_blocks)
            command = p.rw_command(
                opcode,
                self.nsid,
                current_lba,
                count,
                buffer,
                lba_size=self.lba_size,
                buffer_offset=offset,
            )
            self.controller.io.execute(
                command, data_owner=buffer, timeout=self.controller.timeout
            )
            current_lba += count
            offset += count * self.lba_size
            remaining -= count

        if gpu and opcode == p.OPC_READ:
            import ibverbs.cuda as ibcuda

            ibcuda.flush_gpudirect_writes()

    def read(self, buffer, slba: int, blocks=None, *, buffer_offset: int = 0) -> None:
        """Read namespace LBAs directly into a registered host/GPU buffer."""
        self._transfer(p.OPC_READ, buffer, slba, blocks, buffer_offset)

    def write(self, buffer, slba: int, blocks=None, *, buffer_offset: int = 0) -> None:
        """Write namespace LBAs directly from a registered host/GPU buffer."""
        self._transfer(p.OPC_WRITE, buffer, slba, blocks, buffer_offset)

    def flush(self) -> None:
        command = p._command(p.OPC_FLUSH, self.nsid)
        self.controller.io.execute(bytes(command), timeout=self.controller.timeout)
