"""Triton extern functions for the GPUNetIO device bitcode ABI."""

from __future__ import annotations

from triton.language import core

from ._build import bitcode_path

_U64 = core.dtype("uint64")
_U32 = core.dtype("uint32")
_I32 = core.dtype("int32")


def external_libraries(path=None, *, arch="sm_90") -> dict[str, str]:
    """Return the ``extern_libs=`` mapping for a Triton kernel launch."""
    return {"rdma4py_gpunetio": str(bitcode_path(path, arch=arch))}


def _call(symbol, args, types, result, semantic):
    return core.extern_elementwise(
        "rdma4py_gpunetio",
        "",
        args,
        {tuple(types): (symbol, result)},
        is_pure=False,
        _semantic=semantic,
    )


@core.extern
def put(qp, remote_addr, rkey, local_addr, lkey, length, _semantic=None):
    """Post an RDMA Write and return its send-CQ ticket."""
    return _call(
        "rdma4py_gpunetio_put",
        [qp, remote_addr, rkey, local_addr, lkey, length],
        [_U64, _U64, _U32, _U64, _U32, _U64],
        _U64,
        _semantic,
    )


@core.extern
def get(qp, remote_addr, rkey, local_addr, lkey, length, _semantic=None):
    """Post an RDMA Read and return its send-CQ ticket (Hopper or newer)."""
    return _call(
        "rdma4py_gpunetio_get",
        [qp, remote_addr, rkey, local_addr, lkey, length],
        [_U64, _U64, _U32, _U64, _U32, _U64],
        _U64,
        _semantic,
    )


@core.extern
def get_mcst(
    qp,
    remote_addr,
    rkey,
    local_addr,
    lkey,
    length,
    dump_addr,
    dump_lkey,
    _semantic=None,
):
    """Post a pre-Hopper RDMA Read with a memory-consistency dump WQE."""
    return _call(
        "rdma4py_gpunetio_get_mcst",
        [
            qp,
            remote_addr,
            rkey,
            local_addr,
            lkey,
            length,
            dump_addr,
            dump_lkey,
        ],
        [_U64, _U64, _U32, _U64, _U32, _U64, _U64, _U32],
        _U64,
        _semantic,
    )


@core.extern
def send(qp, local_addr, lkey, length, _semantic=None):
    """Post an RDMA Send and return its send-CQ ticket."""
    return _call(
        "rdma4py_gpunetio_send",
        [qp, local_addr, lkey, length],
        [_U64, _U64, _U32, _U64],
        _U64,
        _semantic,
    )


@core.extern
def recv(qp, local_addr, lkey, length, _semantic=None):
    """Post an RDMA Receive and return its receive-CQ ticket."""
    return _call(
        "rdma4py_gpunetio_recv",
        [qp, local_addr, lkey, length],
        [_U64, _U64, _U32, _U64],
        _U64,
        _semantic,
    )


def _completion(symbol, qp, ticket, semantic):
    return _call(symbol, [qp, ticket], [_U64, _U64], _I32, semantic)


@core.extern
def wait_send(qp, ticket, _semantic=None):
    """Wait for a send completion and return zero or a negative error."""
    return _completion("rdma4py_gpunetio_wait_send", qp, ticket, _semantic)


@core.extern
def test_send(qp, ticket, _semantic=None):
    """Poll a send ticket once; return ``EBUSY`` while incomplete."""
    return _completion("rdma4py_gpunetio_test_send", qp, ticket, _semantic)


@core.extern
def wait_recv(qp, ticket, _semantic=None):
    """Wait for a receive completion and return zero or a negative error."""
    return _completion("rdma4py_gpunetio_wait_recv", qp, ticket, _semantic)


@core.extern
def test_recv(qp, ticket, _semantic=None):
    """Poll a receive ticket once; return ``EBUSY`` while incomplete."""
    return _completion("rdma4py_gpunetio_test_recv", qp, ticket, _semantic)


@core.extern
def wait_recv_mcst(qp, ticket, dump_addr, dump_lkey, _semantic=None):
    """Wait for a pre-Hopper receive and apply GPUNetIO memory consistency."""
    return _call(
        "rdma4py_gpunetio_wait_recv_mcst",
        [qp, ticket, dump_addr, dump_lkey],
        [_U64, _U64, _U64, _U32],
        _I32,
        _semantic,
    )


__all__ = [
    "external_libraries",
    "get",
    "get_mcst",
    "put",
    "recv",
    "send",
    "test_recv",
    "test_send",
    "wait_recv",
    "wait_recv_mcst",
    "wait_send",
]
