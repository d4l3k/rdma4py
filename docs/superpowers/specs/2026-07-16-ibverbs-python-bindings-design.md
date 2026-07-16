# ibverbs — low-level Python bindings for libibverbs

**Date:** 2026-07-16
**Status:** Approved

## Goal

Provide low-level, Pythonic bindings to `libibverbs` that expose all key RDMA
verbs features, support GPUDirect memory registration, and are fast enough to
build a high-performance RDMA library on top of (cf.
`gloo/transport/ibverbs`). The bindings live in the `ibverbs/` subdirectory as
a standalone project intended for eventual PyPI publication. The library does
**not** depend on torch; torch may be used in tests only. Dependencies are kept
minimal (no runtime deps).

## Key decisions

- **Binding technology: Cython.** Compiles against `<infiniband/verbs.h>`,
  correctly emits the `static inline` fast-path dispatch functions
  (`ibv_post_send`, `ibv_post_recv`, `ibv_poll_cq`, `ibv_req_notify_cq`) that
  cannot be reliably reached via `dlsym`, releases the GIL on hot paths, and
  needs zero runtime dependencies. This mirrors rdma-core's own `pyverbs`.
- **Dev/test env:** torch + CUDA tensors must be available for GPUDirect
  tests. Python 3.12 was the fallback, but the configured index turned out to
  ship torch `2.13.0+cu130` wheels for the repo's existing Python **3.14** venv
  (CUDA available, 8× H100 visible), so that venv is retained. The library
  itself targets CPython 3.9+.
- **API scope: raw verbs + thin RC helpers.** 1:1 Pythonic wrappers of the key
  verbs plus a small optional layer for RC QP state transitions and
  QP-info exchange.
- **GPUDirect: both `reg_mr` and `reg_dmabuf_mr`.** `reg_mr` accepts any integer
  address (incl. a CUDA device pointer, via `nvidia_peermem`); `reg_dmabuf_mr`
  supports the dma-buf fd path. No CUDA linkage in the library.

## Package layout

```
ibverbs/
  pyproject.toml             # build-system: setuptools + Cython; deps; extras
  setup.py                   # Extension built via pkg-config libibverbs
  README.md
  CLAUDE.md
  src/ibverbs/
    __init__.py              # public API re-exports + __version__
    _ibverbs.pyx             # Cython extension: all cdef classes
    _libverbs.pxd            # `cdef extern` declarations of infiniband/verbs.h
    enums.py                 # IntEnum/IntFlag: access, qp_state, opcodes, wc_status…
    helpers.py               # QPInfo + RC state-machine helpers (thin layer)
  tests/
    conftest.py              # device/port fixtures, skip logic, markers
    test_enums.py test_device.py test_pd_mr.py test_cq.py test_qp.py
    test_loopback.py         # single-NIC RC: SEND/RECV, WRITE, WRITE_IMM, READ, atomics
    test_two_device.py       # RC across two mlx5 NICs (real wire)
    test_comp_channel.py     # event-driven completions via channel fd
    test_gpudirect.py        # torch CUDA tensors: reg_mr + reg_dmabuf RDMA
```

Import name is `ibverbs`. Hot-path C classes live in the compiled `_ibverbs`;
pure-Python enums and helpers are layered on top.

## Object model (Pythonic, RAII)

Each `cdef class` owns its C handle and frees it in `__dealloc__`. Children hold
Python references to parents (MR→PD, QP→PD+CQ, CQ→Context/CompChannel) so the
garbage collector cannot free a parent before its children. All resource
objects are context managers with an explicit `close()`.

- `get_device_list() -> list[Device]`; `Device(name, guid, node_type).open() -> Context`
- `Context`: `query_device()`, `query_port(port)`, `query_gid(port, index)`,
  `alloc_pd()`, `create_cq(cqe, channel=None, comp_vector=0)`,
  `create_comp_channel()`, `get_async_event()/ack_async_event()`,
  `num_comp_vectors`
- `PD`: `reg_mr(addr:int, length:int, access) -> MR`,
  `reg_dmabuf_mr(offset, length, iova, fd, access) -> MR`,
  `create_qp(init_attr)`, `create_ah(attr)`, `create_srq(attr)`
- `MR`: `.addr .length .lkey .rkey`
- `CQ`: `poll(n) -> list[WC]`, `req_notify(solicited_only=False)`, `ack_events(n)`
- `CompChannel`: `.fd` (for epoll/select), `get_cq_event()`
- `QP`: `.qp_num .state`, `modify(attr, mask)`, `query()`, `post_send(wrs)`,
  `post_recv(wrs)`
- `SRQ`, `AH` (for UD)

## Work requests & completions

- `SGE(addr|mr, offset, length)`
- `SendWR(wr_id, sg, opcode, flags, *, remote_addr, rkey, imm_data,
  compare_add, swap, ah, remote_qpn, remote_qkey)`
- `RecvWR(wr_id, sg)`
- Opcodes: `SEND`, `SEND_WITH_IMM`, `RDMA_WRITE`, `RDMA_WRITE_WITH_IMM`,
  `RDMA_READ`, `ATOMIC_FETCH_AND_ADD`, `ATOMIC_CMP_AND_SWP`
- Send flags: `SIGNALED`, `FENCE`, `SOLICITED`, `INLINE`
- `WC(wr_id, status, opcode, byte_len, imm_data, qp_num, src_qp, wc_flags)` plus
  `wc.raise_for_status()`
- Cython builds the C `ibv_send_wr`/`ibv_sge` chains on the stack and calls
  `ibv_post_send`/`ibv_poll_cq` **`nogil`** for throughput.

## GPUDirect

The library links no CUDA. `reg_mr(tensor.data_ptr(), nbytes, access)`
registers a device pointer (nvidia_peermem path). `reg_dmabuf_mr(offset,
length, iova, fd, access)` supports the dma-buf path; tests obtain the fd via
`cuMemGetHandleForAddressRange` (ctypes → libcuda) and skip gracefully when
unsupported.

## Thin RC helpers (`helpers.py`)

`QPInfo` dataclass (qp_num, psn, lid, gid bytes, port, mtu) with
`to_bytes()/from_bytes()` for out-of-band exchange, plus `qp.to_init(port,
access)`, `qp.to_rtr(remote, sgid_index, …)`, `qp.to_rts(psn, …)` wrapping
`ibv_modify_qp` masks for RC. The raw `modify()` remains available.

## Error handling

`VerbsError(OSError)` carries `errno` and the failing operation name; raised on
NULL / negative returns via `PyErr_SetFromErrno`. A bad WC status does **not**
auto-raise (it is returned in the `WC`); `raise_for_status()` is opt-in.

## Build, dependencies, testing

- **Build deps:** Cython, setuptools, wheel. Requires `rdma-core-devel` /
  `libibverbs-dev` present at build time (documented). **Runtime deps: none.**
- **Extras:** `[test]` = pytest, numpy; `[gpu]` = torch.
- **Tests:** unit (no wire) + integration on the local mlx5 fabric (single-NIC
  loopback covering every opcode; two-NIC real-wire; event-driven comp channel)
  + GPUDirect (H100 tensors, both registration paths). pytest markers
  `integration` / `gpu`; auto-skip when hardware or torch is absent.

## Out of scope for v1 (documented as future work)

Extended `ibv_wr_*` / `qp_ex` post API, device memory (`ibv_alloc_dm`), memory
windows, and flow steering.
