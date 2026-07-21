"""NVMe/RDMA queue transport built directly on :mod:`ibverbs`."""

from __future__ import annotations

import ctypes
import struct
import time
from collections import deque
from dataclasses import dataclass

import ibverbs as ib

from .protocol import Completion, parse_rdma_cm_response, rdma_cm_request

_SEND_TAG = 1 << 63
_RECV_TAG = 1 << 62
_INDEX_MASK = (1 << 32) - 1


class QueueFullError(RuntimeError):
    """Raised when every command ID on an NVMe/RDMA queue is in use."""


class HostBuffer:
    """Pinned host allocation registered with an NVMe/RDMA queue's PD.

    Args:
        pd: Protection domain used to register the allocation.
        length: Allocation size in bytes.
        access: Optional verbs access mask. By default the buffer permits the
            local and remote operations required by NVMe/RDMA.

    The buffer is a context manager. Close it before closing the protection
    domain or controller that owns ``pd``.
    """

    def __init__(self, pd, length: int, access=None):
        length = int(length)
        if length <= 0:
            raise ValueError("buffer length must be positive")
        if access is None:
            access = (
                ib.AccessFlags.LOCAL_WRITE
                | ib.AccessFlags.REMOTE_WRITE
                | ib.AccessFlags.REMOTE_READ
            )
        self._allocation = (ctypes.c_ubyte * length)()
        self.addr = ctypes.addressof(self._allocation)
        self.length = length
        self.mr = pd.reg_mr(self.addr, length, access)

    @property
    def rkey(self) -> int:
        """Return the remote key encoded in NVMe keyed SGL descriptors."""
        return self.mr.rkey

    @property
    def lkey(self) -> int:
        """Return the local key used for initiator work requests."""
        return self.mr.lkey

    @property
    def closed(self) -> bool:
        """Whether the underlying memory region has been deregistered."""
        return self.mr.closed

    def write(self, data: bytes, offset: int = 0) -> None:
        """Copy ``data`` into this allocation at byte ``offset``."""
        data = bytes(data)
        offset = int(offset)
        if offset < 0 or len(data) > self.length - offset:
            raise ValueError("write exceeds the host buffer")
        ctypes.memmove(self.addr + offset, data, len(data))

    def read(self, length=None, offset: int = 0) -> bytes:
        """Copy bytes from this allocation into a new :class:`bytes` object.

        ``length`` defaults to all bytes from ``offset`` through the end of
        the allocation.
        """
        offset = int(offset)
        if length is None:
            length = self.length - offset
        length = int(length)
        if offset < 0 or length < 0 or length > self.length - offset:
            raise ValueError("read exceeds the host buffer")
        return ctypes.string_at(self.addr + offset, length)

    def close(self) -> None:
        """Deregister the memory region."""
        self.mr.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@dataclass
class Request:
    """State retained for an asynchronously submitted NVMe command.

    A request becomes :attr:`done` only after both the capsule SEND and the
    matching NVMe response complete. ``data_owner`` keeps any keyed-SGL
    allocation alive while the command is outstanding. Callers should treat
    the state fields as read-only.
    """

    command_id: int
    data_owner: object = None
    send_complete: bool = False
    response: Completion | None = None
    done: bool = False


class RDMAQueue:
    """One connected NVMe/RDMA submission/completion queue pair.

    This is the low-level command transport used by :class:`Controller`.
    Queue construction resolves the route, creates an RC QP, connects with
    NVMe/RDMA private data, and posts the response receives.

    Args:
        host: Target hostname or IP address.
        port: Target NVMe/RDMA service port.
        qid: NVMe queue identifier. Zero creates an admin queue.
        depth: Queue depth from 2 through 256. At most ``depth - 1`` commands
            may be outstanding.
        controller_id: Connected controller ID for an I/O queue.
        source: Optional initiator IP address used to select the local HCA.
    """

    def __init__(
        self,
        host: str,
        port: int,
        qid: int,
        depth: int,
        controller_id: int = 0,
        source=None,
    ):
        self.host = host
        self.port = int(port)
        self.source = source
        self.qid = int(qid)
        self.depth = int(depth)
        rdma_cm_request(self.qid, self.depth, controller_id)
        self.cm = None
        self.context = None
        self.pd = None
        self.cq = None
        self.qp = None
        self.commands = None
        self.responses = None
        self._pending = {}
        self._completed = deque()
        self._free = deque(range(self.depth - 1))
        self._closed = False
        try:
            self.cm = ib.CMID.resolve(  # pyre-ignore[16]: Cython export.
                host, port, source=source
            )
            self.context = self.cm.context
            self.pd = self.context.alloc_pd()
            self.cq = self.context.create_cq(2 * self.depth + 1)
            self.qp = self.cm.create_qp(
                self.pd,
                ib.QPInitAttr(  # pyre-ignore[16]: Cython export.
                    send_cq=self.cq,
                    recv_cq=self.cq,
                    qp_type=ib.QPType.RC,
                    max_send_wr=self.depth,
                    max_recv_wr=self.depth,
                    max_send_sge=1,
                    max_recv_sge=1,
                ),
            )
            device = self.context.query_device()
            peer = self.cm.connect(
                rdma_cm_request(self.qid, self.depth, controller_id),
                responder_resources=min(255, device.max_qp_rd_atom),
            )
            self.controller_receive_depth = parse_rdma_cm_response(peer)
            if self.controller_receive_depth < self.depth - 1:
                raise RuntimeError(
                    "target receive depth %d is smaller than host send depth %d"
                    % (self.controller_receive_depth, self.depth - 1)
                )
            self.commands = HostBuffer(self.pd, 64 * (self.depth - 1))
            self.responses = HostBuffer(self.pd, 16 * self.depth)
            self.qp.post_recv([self._recv_wr(slot) for slot in range(self.depth)])
        except Exception:
            try:
                self.close()
            except OSError:
                pass
            raise

    @property
    def closed(self) -> bool:
        """Whether queue shutdown has begun."""
        return self._closed

    @property
    def outstanding(self) -> int:
        """Return the number of commands awaiting transport completion."""
        return len(self._pending)

    def _recv_wr(self, slot: int):
        return ib.RecvWR(  # pyre-ignore[16]: Cython export.
            wr_id=_RECV_TAG | slot,
            sg_list=[self.responses.mr.sge(16, slot * 16)],
        )

    def submit(self, command: bytes, data_owner=None) -> Request:
        """Submit one 64-byte command capsule without waiting.

        The queue assigns and writes the command ID. ``data_owner`` should own
        any buffer addressed by the command's keyed SGL so it remains alive
        until the returned request completes. Call :meth:`poll` to advance
        and retrieve completed requests.

        Raises:
            QueueFullError: If ``depth - 1`` commands are already outstanding.
        """
        if self._closed:
            raise RuntimeError("NVMe/RDMA queue is closed")
        if len(command) != 64:
            raise ValueError("an NVMe command capsule must be exactly 64 bytes")
        if not self._free:
            raise QueueFullError("NVMe submission queue is full")
        cid = self._free.popleft()
        capsule = bytearray(command)
        struct.pack_into("<H", capsule, 2, cid)
        self.commands.write(capsule, cid * 64)
        request = Request(cid, data_owner=data_owner)
        self._pending[cid] = request
        try:
            self.qp.post_send(
                ib.SendWR(  # pyre-ignore[16]: Cython export.
                    wr_id=_SEND_TAG | cid,
                    sg_list=[self.commands.mr.sge(64, cid * 64)],
                    opcode=ib.WROpcode.SEND,
                    send_flags=ib.SendFlags.SIGNALED,
                )
            )
        except Exception:
            del self._pending[cid]
            self._free.appendleft(cid)
            raise
        return request

    def _finish_if_ready(self, request: Request) -> None:
        if request.send_complete and request.response is not None:
            request.done = True
            del self._pending[request.command_id]
            self._free.append(request.command_id)
            self._completed.append(request)

    def poll(self, max_entries=None):
        """Poll transport completions and return newly completed requests.

        A request is returned only after its capsule SEND completion and
        matching NVMe response have both arrived. This method does not block;
        ``max_entries`` defaults to enough entries to drain the CQ.
        """
        if self._closed:
            return []
        if max_entries is None:
            max_entries = 2 * self.depth + 1
        for wc in self.cq.poll(int(max_entries)):
            wc.raise_for_status()
            if wc.wr_id & _SEND_TAG:
                cid = wc.wr_id & _INDEX_MASK
                request = self._pending.get(cid)
                if request is None:
                    raise RuntimeError("send completion has an unknown command ID")
                request.send_complete = True
                self._finish_if_ready(request)
                continue
            if wc.wr_id & _RECV_TAG:
                slot = wc.wr_id & _INDEX_MASK
                if wc.byte_len < 16 or slot >= self.depth:
                    raise RuntimeError("invalid NVMe/RDMA response completion")
                completion = Completion.from_bytes(self.responses.read(16, slot * 16))
                self.qp.post_recv(self._recv_wr(slot))
                request = self._pending.get(completion.command_id)
                if request is None:
                    raise RuntimeError("response has an unknown command ID")
                request.response = completion
                self._finish_if_ready(request)
                continue
            raise RuntimeError("unexpected work completion on NVMe/RDMA CQ")
        completed = list(self._completed)
        self._completed.clear()
        return completed

    def execute(self, command: bytes, data_owner=None, timeout: float = 30.0):
        """Submit a command and wait for its successful NVMe completion.

        Args:
            command: Exactly one 64-byte NVMe command capsule.
            data_owner: Optional owner of the command's keyed-SGL buffer.
            timeout: Maximum wait in seconds.

        Returns:
            The matching :class:`~nvmeof.protocol.Completion`.

        Raises:
            TimeoutError: If the request does not finish before ``timeout``.
            nvmeof.protocol.NVMeStatusError: If the target returns a non-zero
                NVMe status.
        """
        request = self.submit(command, data_owner=data_owner)
        deadline = time.monotonic() + float(timeout)
        while not request.done:
            self.poll()
            if time.monotonic() >= deadline:
                raise TimeoutError("NVMe command timed out")
        response = request.response
        if response is None:
            raise RuntimeError("NVMe request completed without a response")
        response.raise_for_status()
        return response

    def close(self) -> None:
        """Disconnect and close every RDMA resource owned by this queue."""
        if self._closed and all(
            resource is None
            for resource in (
                self.commands,
                self.responses,
                self.qp,
                self.cq,
                self.pd,
                self.cm,
            )
        ):
            return
        self._closed = True
        errors = []
        if self.cm is not None and self.cm.connected:
            try:
                self.cm.disconnect()
            except OSError:
                # The peer may already have disconnected; local resource
                # destruction below is still required and authoritative.
                pass
        for name in ("commands", "responses", "qp", "cq", "pd"):
            resource = getattr(self, name)
            if resource is not None:
                try:
                    resource.close()
                    setattr(self, name, None)
                except OSError as exc:
                    errors.append(exc)
        if self.cm is not None:
            try:
                self.cm.close()
                self.cm = None
            except OSError as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
