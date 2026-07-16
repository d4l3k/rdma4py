"""Test-side helpers to stand up connected RC queue pairs and buffers.

Kept out of the shipped library on purpose: the library provides the raw verbs
and thin state-machine helpers; wiring a full connection (including the
out-of-band QPInfo exchange) is the consumer's job. These helpers exercise that
path for the integration tests.
"""

from __future__ import annotations

import numpy as np

import ibverbs as ib

FULL_ACCESS = (
    ib.AccessFlags.LOCAL_WRITE
    | ib.AccessFlags.REMOTE_WRITE
    | ib.AccessFlags.REMOTE_READ
    | ib.AccessFlags.REMOTE_ATOMIC
)


class HostBuffer:
    """A registered host buffer backed by a numpy array."""

    def __init__(self, pd, nbytes, access=FULL_ACCESS, fill=0):
        self.array = np.full(nbytes, fill, dtype=np.uint8)
        self.pd = pd
        self.mr = pd.reg_mr(self.array.ctypes.data, nbytes, access)

    @property
    def addr(self) -> int:
        return int(self.array.ctypes.data)

    @property
    def rkey(self) -> int:
        return self.mr.rkey

    @property
    def lkey(self) -> int:
        return self.mr.lkey

    def sge(self, length=None, offset=0):
        return ib.SGE(self.mr, length if length is not None else self.array.nbytes,
                      offset=offset)

    def set_bytes(self, data: bytes, offset=0):
        self.array[offset:offset + len(data)] = np.frombuffer(data, dtype=np.uint8)

    def get_bytes(self, length=None, offset=0) -> bytes:
        end = self.array.nbytes if length is None else offset + length
        return self.array[offset:end].tobytes()

    def read_u64(self, offset=0) -> int:
        return int(self.array[offset:offset + 8].view(np.uint64)[0])

    def write_u64(self, value, offset=0):
        self.array[offset:offset + 8].view(np.uint64)[0] = np.uint64(value)

    def close(self):
        self.mr.close()


class Endpoint:
    """A single RC queue pair with its own completion queue."""

    def __init__(self, ctx, pd, port, cqe=64):
        self.ctx = ctx
        self.pd = pd
        self.port = port
        self.cq = ctx.create_cq(cqe)
        self.qp = pd.create_qp(
            ib.QPInitAttr(send_cq=self.cq, recv_cq=self.cq, qp_type=ib.QPType.RC,
                          max_send_wr=cqe, max_recv_wr=cqe)
        )

    def info(self, port_attr, gid, psn=0):
        return ib.local_qp_info(self.qp, port_attr, gid, port=self.port, psn=psn)

    def connect(self, remote, gid_index, access=FULL_ACCESS):
        ib.connect_rc(self.qp, remote, port=self.port, sgid_index=gid_index,
                      access=access)

    def poll_one(self, spins=200000):
        for _ in range(spins):
            wcs = self.cq.poll(1)
            if wcs:
                return wcs[0]
        raise TimeoutError("no completion")

    def close(self):
        self.qp.close()
        self.cq.close()


def make_connected_pair(ctx, pd, dev_name, port, gid_index, gid):
    """Return two connected loopback RC endpoints (a, b) on the same port."""
    pa = ctx.query_port(port)
    a = Endpoint(ctx, pd, port)
    b = Endpoint(ctx, pd, port)
    info_a = a.info(pa, gid)
    info_b = b.info(pa, gid)
    a.connect(info_b, gid_index)
    b.connect(info_a, gid_index)
    return a, b
