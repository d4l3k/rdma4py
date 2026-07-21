from __future__ import annotations

import struct
from types import SimpleNamespace

from nvmeof import protocol as p, rdma


class FakeMR:
    def __init__(self, addr, length):
        self.addr = addr
        self.length = length
        self.lkey = 11
        self.rkey = 22
        self.closed = False

    def sge(self, length, offset):
        return SimpleNamespace(addr=self.addr + offset, length=length, lkey=self.lkey)

    def close(self):
        self.closed = True


class FakePD:
    def reg_mr(self, addr, length, access):
        return FakeMR(addr, length)

    def close(self):
        pass


class FakeCQ:
    def __init__(self):
        self.completions = []

    def poll(self, count):
        result, self.completions = self.completions[:count], self.completions[count:]
        return result

    def close(self):
        pass


class FakeQP:
    def __init__(self):
        self.recvs = []
        self.sends = []

    def post_recv(self, wr):
        self.recvs.extend(wr if isinstance(wr, list) else [wr])

    def post_send(self, wr):
        self.sends.append(wr)

    def close(self):
        pass


class FakeContext:
    def __init__(self):
        self.pd = FakePD()
        self.cq = FakeCQ()
        self.qp = FakeQP()

    def query_device(self):
        return SimpleNamespace(max_qp_rd_atom=16)

    def alloc_pd(self):
        return self.pd

    def create_cq(self, count):
        return self.cq


class FakeCM:
    def __init__(self):
        self.context = FakeContext()
        self.connected = False

    @classmethod
    def resolve(cls, host, port, source=None):
        endpoint = cls()
        endpoint.source = source
        return endpoint

    def create_qp(self, pd, attr):
        return self.context.qp

    def connect(self, private_data, **kwargs):
        self.connected = True
        return struct.pack("<HH28x", 0, 8)

    def disconnect(self):
        self.connected = False

    def close(self):
        pass


class FakeWC:
    def __init__(self, wr_id, byte_len=0):
        self.wr_id = wr_id
        self.byte_len = byte_len

    def raise_for_status(self):
        pass


def test_queue_waits_for_send_and_response_completions(monkeypatch):
    monkeypatch.setattr(rdma.ib, "CMID", FakeCM)
    queue = rdma.RDMAQueue(
        "target", 4420, qid=1, depth=8, controller_id=3, source="source"
    )
    assert queue.cm.source == "source"
    request = queue.submit(bytes(p._command(p.OPC_FLUSH, 1)))

    response = struct.pack("<QHHHH", 99, 0, 1, request.command_id, 1)
    queue.responses.write(response, 0)
    queue.cq.completions.append(FakeWC(rdma._RECV_TAG, 16))
    queue.poll()
    assert request.response.result == 99
    assert not request.done

    queue.cq.completions.append(FakeWC(rdma._SEND_TAG | request.command_id))
    completed = queue.poll()
    assert completed == [request]
    assert request.done
    assert queue.outstanding == 0
    assert len(queue.qp.recvs) == queue.depth + 1
    queue.close()
