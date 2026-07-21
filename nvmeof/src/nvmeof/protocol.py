"""NVMe, NVMe-oF, and NVMe/RDMA wire layouts used by the initiator."""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass

#: IANA service port for NVMe over Fabrics/RDMA.
NVME_RDMA_PORT = 4420
#: Queue depth used for the controller's admin queue.
ADMIN_QUEUE_DEPTH = 32
#: Largest queue supported by this initiator's command-ID layout.
MAX_QUEUE_DEPTH = 256
#: Largest byte count representable by one keyed SGL descriptor.
MAX_KEYED_SGL_LENGTH = (1 << 24) - 1

#: NVM Flush opcode.
OPC_FLUSH = 0x00
#: NVM Write opcode.
OPC_WRITE = 0x01
#: NVM Read opcode.
OPC_READ = 0x02
#: Admin Identify opcode.
OPC_IDENTIFY = 0x06
#: Admin Set Features opcode.
OPC_SET_FEATURES = 0x09
#: Fabrics command opcode.
OPC_FABRICS = 0x7F

#: Fabrics Property Set command type.
FCTYPE_PROPERTY_SET = 0x00
#: Fabrics Connect command type.
FCTYPE_CONNECT = 0x01
#: Fabrics Property Get command type.
FCTYPE_PROPERTY_GET = 0x04

#: Controller Capabilities property offset.
REG_CAP = 0x00
#: Controller Version property offset.
REG_VS = 0x08
#: Controller Configuration property offset.
REG_CC = 0x14
#: Controller Status property offset.
REG_CSTS = 0x1C

#: Controller Configuration enable bit.
CC_ENABLE = 1
#: Controller Configuration command-set selection for CSI.
CC_CSS_CSI = 6 << 4
#: Controller Configuration memory-page-size shift.
CC_MPS_SHIFT = 7
#: Controller Configuration I/O submission queue entry size.
CC_IOSQES = 6 << 16
#: Controller Configuration I/O completion queue entry size.
CC_IOCQES = 4 << 20
#: Controller Status ready bit.
CSTS_READY = 1
#: Controller Status fatal-status bit.
CSTS_FATAL = 1 << 1

#: Set Features identifier for Number of Queues.
FEAT_NUMBER_OF_QUEUES = 0x07
#: Command data-pointer flag selecting an SGL.
PSDT_SGL = 1 << 6
#: NVMe/RDMA keyed data block SGL descriptor type.
KEYED_DATA_BLOCK = 0x40


class NVMeStatusError(OSError):
    """An NVMe command completed with a non-zero status."""

    def __init__(self, completion: "Completion"):
        self.completion = completion
        message = "NVMe command %d failed: SCT=%d SC=0x%02x%s" % (
            completion.command_id,
            completion.status_type,
            completion.status_code,
            " DNR" if completion.do_not_retry else "",
        )
        super().__init__(5, message)


@dataclass(frozen=True)
class Completion:
    """Decoded 16-byte NVMe completion queue entry.

    ``status_field`` retains the phase bit exactly as received. The derived
    status properties remove that bit and expose the NVMe status code, status
    code type, and do-not-retry flag.
    """

    result: int
    sq_head: int
    sq_id: int
    command_id: int
    status_field: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "Completion":
        """Decode the first 16 bytes of an NVMe completion queue entry."""
        if len(data) < 16:
            raise ValueError("an NVMe completion must be at least 16 bytes")
        return cls(*struct.unpack_from("<QHHHH", data))

    @property
    def status(self) -> int:
        """Return the status field with its phase bit removed."""
        return self.status_field >> 1

    @property
    def status_code(self) -> int:
        """Return the eight-bit NVMe status code (SC)."""
        return self.status & 0xFF

    @property
    def status_type(self) -> int:
        """Return the three-bit status code type (SCT)."""
        return (self.status >> 8) & 0x7

    @property
    def do_not_retry(self) -> bool:
        """Whether the target marked this command as not retryable."""
        return bool(self.status & (1 << 14))

    @property
    def succeeded(self) -> bool:
        """Whether both the status code and status code type are zero."""
        return (self.status & 0x7FF) == 0

    def raise_for_status(self) -> None:
        """Raise :class:`NVMeStatusError` unless the command succeeded."""
        if not self.succeeded:
            raise NVMeStatusError(self)


@dataclass(frozen=True)
class ControllerInfo:
    """Controller fields consumed from a 4096-byte Identify response.

    The record includes display identity, MDTS, controller and namespace
    limits, keyed-SGL capability bits, and NVMe-oF capsule sizing fields.
    """

    serial: str
    model: str
    firmware: str
    mdts: int
    controller_id: int
    version: int
    max_commands: int
    namespace_count: int
    sgls: int
    ioccsz: int
    iorcsz: int
    icdoff: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "ControllerInfo":
        """Parse an Identify Controller data structure."""
        if len(data) < 4096:
            raise ValueError("Identify Controller data must be 4096 bytes")

        def text(offset: int, length: int) -> str:
            return data[offset : offset + length].decode("ascii", "replace").strip()

        return cls(
            serial=text(4, 20),
            model=text(24, 40),
            firmware=text(64, 8),
            mdts=data[77],
            controller_id=struct.unpack_from("<H", data, 78)[0],
            version=struct.unpack_from("<I", data, 80)[0],
            max_commands=struct.unpack_from("<H", data, 514)[0],
            namespace_count=struct.unpack_from("<I", data, 516)[0],
            sgls=struct.unpack_from("<I", data, 536)[0],
            ioccsz=struct.unpack_from("<I", data, 1792)[0],
            iorcsz=struct.unpack_from("<I", data, 1796)[0],
            icdoff=struct.unpack_from("<H", data, 1800)[0],
        )


@dataclass(frozen=True)
class NamespaceInfo:
    """Namespace capacity and active LBA-format information."""

    nsid: int
    size_lbas: int
    capacity_lbas: int
    used_lbas: int
    lba_size: int
    metadata_size: int

    @classmethod
    def from_bytes(cls, nsid: int, data: bytes) -> "NamespaceInfo":
        """Parse an Identify Namespace data structure for ``nsid``."""
        if len(data) < 4096:
            raise ValueError("Identify Namespace data must be 4096 bytes")
        fmt = data[26] & 0x0F
        if fmt > data[25]:
            raise ValueError("Identify Namespace selected an invalid LBA format")
        lbaf = 128 + fmt * 4
        metadata_size, data_shift = struct.unpack_from("<HB", data, lbaf)
        if data_shift < 9 or data_shift > 31:
            raise ValueError("unsupported namespace LBA data size")
        return cls(
            nsid=int(nsid),
            size_lbas=struct.unpack_from("<Q", data, 0)[0],
            capacity_lbas=struct.unpack_from("<Q", data, 8)[0],
            used_lbas=struct.unpack_from("<Q", data, 16)[0],
            lba_size=1 << data_shift,
            metadata_size=metadata_size,
        )


def _command(opcode: int, nsid: int = 0) -> bytearray:
    command = bytearray(64)
    struct.pack_into("<BBHI", command, 0, opcode, PSDT_SGL, 0, nsid)
    command[39] = KEYED_DATA_BLOCK
    return command


def set_keyed_sgl(command: bytearray, buffer, length=None, offset=0) -> None:
    """Point a command at a registered host MR or ``ibverbs.cuda.GpuMR``.

    ``buffer`` must expose ``addr``, ``length``, and ``rkey``. The selected
    range must fit one 24-bit keyed data block descriptor.
    """
    offset = int(offset)
    total = int(buffer.length)
    if offset < 0 or offset > total:
        raise ValueError("data offset is outside the registered memory region")
    if length is None:
        length = total - offset
    length = int(length)
    if length <= 0 or length > total - offset:
        raise ValueError("data range is outside the registered memory region")
    if length > MAX_KEYED_SGL_LENGTH:
        raise ValueError("one NVMe/RDMA keyed SGL cannot exceed 2**24 - 1 bytes")
    addr = int(buffer.addr) + offset
    rkey = int(buffer.rkey)
    if addr < 0 or addr > (1 << 64) - 1:
        raise ValueError("data address is outside the uint64 range")
    if rkey < 0 or rkey > (1 << 32) - 1:
        raise ValueError("rkey is outside the uint32 range")
    command[1] |= PSDT_SGL
    struct.pack_into("<Q", command, 24, addr)
    command[32:35] = length.to_bytes(3, "little")
    struct.pack_into("<I", command, 35, rkey)
    command[39] = KEYED_DATA_BLOCK


def rdma_cm_request(qid: int, depth: int, controller_id: int = 0) -> bytes:
    """Build the 32-byte NVMe/RDMA connection-management request record.

    The admin queue (``qid=0``) always encodes controller ID zero. I/O queues
    encode the ID returned by the admin Fabrics Connect command.
    """
    qid = int(qid)
    depth = int(depth)
    if qid < 0 or qid > 0xFFFF:
        raise ValueError("qid must fit in uint16")
    if depth < 2 or depth > MAX_QUEUE_DEPTH:
        raise ValueError("queue depth must be between 2 and 256")
    cntlid = int(controller_id) if qid else 0
    if cntlid < 0 or cntlid > 0xFFFF:
        raise ValueError("controller_id must fit in uint16")
    return struct.pack("<HHHHH22x", 0, qid, depth, depth - 1, cntlid)


def parse_rdma_cm_response(data: bytes) -> int:
    """Validate NVMe/RDMA CM response data and return its receive queue size."""
    if len(data) < 4:
        raise ValueError("NVMe/RDMA CM response is shorter than 4 bytes")
    recfmt, crqsize = struct.unpack_from("<HH", data)
    if recfmt != 0:
        raise ValueError("unsupported NVMe/RDMA CM record format %d" % recfmt)
    if crqsize == 0:
        raise ValueError("NVMe/RDMA target returned a zero receive queue size")
    return crqsize


def connect_data(
    host_id, subsystem_nqn: str, host_nqn: str, controller_id: int
) -> bytes:
    """Build the 1024-byte Fabrics Connect data structure.

    ``host_id`` accepts a :class:`uuid.UUID` or UUID string. Both NQNs must
    contain 11 through 223 non-NUL bytes when UTF-8 encoded.
    """
    host_uuid = host_id if isinstance(host_id, uuid.UUID) else uuid.UUID(str(host_id))
    data = bytearray(1024)
    data[:16] = host_uuid.bytes
    struct.pack_into("<H", data, 16, int(controller_id))
    for offset, value, name in (
        (256, subsystem_nqn, "subsystem NQN"),
        (512, host_nqn, "host NQN"),
    ):
        encoded = value.encode("utf-8")
        if len(encoded) < 11 or len(encoded) > 223 or b"\x00" in encoded:
            raise ValueError("%s must contain 11 to 223 non-NUL UTF-8 bytes" % name)
        data[offset : offset + len(encoded)] = encoded
    return bytes(data)


def fabrics_connect(qid: int, depth: int, data_buffer, kato_ms: int = 0) -> bytes:
    """Build a Fabrics Connect command using ``data_buffer`` as its payload."""
    if kato_ms < 0 or kato_ms > 0xFFFFFFFF:
        raise ValueError("kato_ms must fit in uint32")
    command = _command(OPC_FABRICS)
    command[4] = FCTYPE_CONNECT
    set_keyed_sgl(command, data_buffer, 1024)
    struct.pack_into("<HHHBBI", command, 40, 0, qid, depth - 1, 0, 0, kato_ms)
    return bytes(command)


def property_get(offset: int, size: int) -> bytes:
    """Build a Fabrics Property Get command for a 4- or 8-byte register."""
    if size not in (4, 8):
        raise ValueError("property size must be 4 or 8 bytes")
    command = _command(OPC_FABRICS)
    command[4] = FCTYPE_PROPERTY_GET
    command[40] = 1 if size == 8 else 0
    struct.pack_into("<I", command, 44, int(offset))
    return bytes(command)


def property_set(offset: int, value: int, size: int = 4) -> bytes:
    """Build a Fabrics Property Set command for a 4- or 8-byte register."""
    if size not in (4, 8):
        raise ValueError("property size must be 4 or 8 bytes")
    command = _command(OPC_FABRICS)
    command[4] = FCTYPE_PROPERTY_SET
    command[40] = 1 if size == 8 else 0
    struct.pack_into("<IQ", command, 44, int(offset), int(value))
    return bytes(command)


def identify(data_buffer, *, nsid: int = 0, controller: bool = False) -> bytes:
    """Build an Identify Controller or Identify Namespace command.

    ``data_buffer`` must provide at least 4096 registered bytes. Pass
    ``controller=True`` for controller data; otherwise ``nsid`` selects the
    namespace.
    """
    command = _command(OPC_IDENTIFY, nsid)
    set_keyed_sgl(command, data_buffer, 4096)
    command[40] = 1 if controller else 0
    return bytes(command)


def set_features(feature_id: int, value: int) -> bytes:
    """Build a Set Features command with command-dword 11 ``value``."""
    command = _command(OPC_SET_FEATURES)
    struct.pack_into("<II", command, 40, int(feature_id), int(value))
    return bytes(command)


def rw_command(
    opcode: int,
    nsid: int,
    slba: int,
    blocks: int,
    buffer,
    *,
    lba_size: int,
    buffer_offset: int = 0,
) -> bytes:
    """Build one NVM Read or Write command with a keyed SGL.

    A command may transfer 1 through 65536 logical blocks, subject to the
    24-bit keyed-SGL length limit. Higher-level namespace I/O splits larger
    transfers before calling this builder.
    """
    if opcode not in (OPC_READ, OPC_WRITE):
        raise ValueError("opcode must be READ or WRITE")
    if slba < 0 or slba > (1 << 64) - 1:
        raise ValueError("starting LBA is outside the uint64 range")
    if blocks <= 0 or blocks > 65536:
        raise ValueError("one NVMe command must contain 1 to 65536 blocks")
    lba_size = int(lba_size)
    if lba_size <= 0:
        raise ValueError("lba_size must be positive")
    length = int(blocks) * lba_size
    command = _command(opcode, nsid)
    set_keyed_sgl(command, buffer, length, buffer_offset)
    struct.pack_into("<QH", command, 40, int(slba), int(blocks) - 1)
    return bytes(command)
