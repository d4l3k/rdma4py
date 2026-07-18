"""Test-side helpers to stand up SRD endpoints and buffers.

Kept out of the shipped library on purpose: the library provides the raw
verbs, the endpoint-info exchange and thin state-machine helpers; wiring a
full application (including the out-of-band EndpointInfo exchange) is the
consumer's job. These helpers exercise that path for the integration tests.
"""

from __future__ import annotations

import efa
import numpy as np

FULL_ACCESS = (
    efa.AccessFlags.LOCAL_WRITE
    | efa.AccessFlags.REMOTE_WRITE
    | efa.AccessFlags.REMOTE_READ
)

QKEY = 0x11111111


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
        return efa.SGE(
            self.mr, length if length is not None else self.array.nbytes, offset=offset
        )

    def set_bytes(self, data: bytes, offset=0):
        self.array[offset : offset + len(data)] = np.frombuffer(data, dtype=np.uint8)

    def get_bytes(self, length=None, offset=0) -> bytes:
        end = self.array.nbytes if length is None else offset + length
        return self.array[offset:end].tobytes()

    def close(self):
        self.mr.close()


class Endpoint:
    """A single SRD queue pair with its own completion queue."""

    def __init__(
        self,
        ctx,
        pd,
        qkey=QKEY,
        cqe=256,
        cq_ex=False,
        sgid=False,
        unsolicited=False,
        **qp_kwargs,
    ):
        self.ctx = ctx
        self.pd = pd
        self.qkey = qkey
        if cq_ex or sgid or unsolicited:
            self.cq = ctx.create_cq_ex(cqe, sgid=sgid, unsolicited=unsolicited)
        else:
            self.cq = ctx.create_cq(cqe)
        if unsolicited:
            qp_kwargs.setdefault("unsolicited_write_recv", True)
        self.qp = pd.create_qp(
            efa.QPInitAttr(
                send_cq=self.cq,
                recv_cq=self.cq,
                max_send_wr=128,
                max_recv_wr=128,
                **qp_kwargs,
            )
        ).prepare(qkey=qkey)

    def info(self) -> efa.EndpointInfo:
        return efa.local_endpoint_info(self.qp, qkey=self.qkey)

    def peer_to(self, other: "Endpoint") -> efa.Peer:
        """Resolve a Peer for ``other`` via the wire-format round trip."""
        return efa.EndpointInfo.from_bytes(other.info().to_bytes()).peer(self.pd)

    def poll_one(self, spins=2000000):
        for _ in range(spins):
            wcs = self.cq.poll(1)
            if wcs:
                return wcs[0]
        raise TimeoutError("no completion")

    def poll_n(self, n, spins=2000000):
        got = []
        for _ in range(spins):
            got += self.cq.poll(16)
            if len(got) >= n:
                return got
        raise TimeoutError(f"got {len(got)}/{n} completions")

    def close(self):
        self.qp.close()
        self.cq.close()


def make_pair(ctx, pd, **kwargs):
    """Return two SRD endpoints on the same device plus a->b and b->a peers."""
    a = Endpoint(ctx, pd, **kwargs)
    b = Endpoint(ctx, pd, **kwargs)
    return a, b, a.peer_to(b), b.peer_to(a)
