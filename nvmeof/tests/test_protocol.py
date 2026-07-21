from __future__ import annotations

import struct
import uuid

import pytest
from nvmeof import protocol as p


class Buffer:
    addr = 0x1234_5000
    length = 1 << 20
    rkey = 0x89ABCDEF


def test_rdma_cm_private_data_layout():
    request = p.rdma_cm_request(qid=3, depth=128, controller_id=7)
    assert len(request) == 32
    assert struct.unpack_from("<HHHHH", request) == (0, 3, 128, 127, 7)
    assert p.parse_rdma_cm_response(struct.pack("<HH28x", 0, 128)) == 128


def test_connect_data_layout():
    host_id = uuid.UUID("00112233-4455-6677-8899-aabbccddeeff")
    data = p.connect_data(
        host_id,
        "nqn.2026-07.io.example:storage",
        "nqn.2014-08.org.nvmexpress:uuid:%s" % host_id,
        0x1234,
    )
    assert len(data) == 1024
    assert data[:16] == host_id.bytes
    assert struct.unpack_from("<H", data, 16)[0] == 0x1234
    assert data[256:].startswith(b"nqn.2026-07.io.example:storage\x00")


def test_keyed_sgl_and_rw_layout():
    command = p.rw_command(
        p.OPC_READ,
        nsid=9,
        slba=0x1122334455667788,
        blocks=8,
        buffer=Buffer(),
        lba_size=4096,
        buffer_offset=4096,
    )
    assert len(command) == 64
    assert command[0] == p.OPC_READ
    assert command[1] == p.PSDT_SGL
    assert struct.unpack_from("<I", command, 4)[0] == 9
    assert struct.unpack_from("<Q", command, 24)[0] == Buffer.addr + 4096
    assert int.from_bytes(command[32:35], "little") == 8 * 4096
    assert struct.unpack_from("<I", command, 35)[0] == Buffer.rkey
    assert command[39] == p.KEYED_DATA_BLOCK
    assert struct.unpack_from("<QH", command, 40) == (
        0x1122334455667788,
        7,
    )


def test_keyed_sgl_rejects_24_bit_overflow():
    buffer = Buffer()
    buffer.length = 1 << 25
    with pytest.raises(ValueError, match=r"2\*\*24"):
        p.set_keyed_sgl(bytearray(64), buffer, p.MAX_KEYED_SGL_LENGTH + 1)


def test_completion_decodes_phase_and_status():
    success = p.Completion.from_bytes(struct.pack("<QHHHH", 7, 1, 2, 3, 1))
    assert success.succeeded
    assert success.result == 7
    success.raise_for_status()

    failed = p.Completion.from_bytes(
        struct.pack("<QHHHH", 0, 0, 1, 4, (2 << 1) | 1 | (1 << 15))
    )
    assert failed.status_code == 2
    assert failed.do_not_retry
    with pytest.raises(p.NVMeStatusError):
        failed.raise_for_status()


def test_identify_controller_offsets():
    data = bytearray(4096)
    data[4:24] = b"serial".ljust(20)
    data[24:64] = b"model".ljust(40)
    data[64:72] = b"fw1".ljust(8)
    data[77] = 8
    struct.pack_into("<HI", data, 78, 17, 0x020100)
    struct.pack_into("<HI", data, 514, 128, 4)
    struct.pack_into("<I", data, 536, 1 << 2)
    struct.pack_into("<IIH", data, 1792, 4, 1, 0)
    info = p.ControllerInfo.from_bytes(data)
    assert (info.serial, info.model, info.firmware) == ("serial", "model", "fw1")
    assert info.controller_id == 17
    assert info.namespace_count == 4
    assert info.ioccsz == 4


def test_identify_namespace_lba_format():
    data = bytearray(4096)
    struct.pack_into("<QQQ", data, 0, 1000, 900, 100)
    data[25] = 0
    data[26] = 0
    struct.pack_into("<HBB", data, 128, 0, 12, 0)
    info = p.NamespaceInfo.from_bytes(2, data)
    assert info.nsid == 2
    assert info.lba_size == 4096
    assert info.size_lbas == 1000
