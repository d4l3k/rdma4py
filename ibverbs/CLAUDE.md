# CLAUDE.md — ibverbs

Guidance for working on the `ibverbs` package (low-level libibverbs bindings).

## What this is

A Cython wrapper over `libibverbs`. The goal is a **faithful, low-level** verbs
API in Python that a higher-level RDMA library can be built on — not a
transport itself. Keep it thin: one Python object per verbs object, minimal
policy, no hidden threads, no runtime dependencies, and **no torch/CUDA
linkage** (GPUDirect works via raw addresses / dma-buf fds).

## Layout

```
ibverbs/
  pyproject.toml          # build (setuptools+Cython), metadata, pytest markers
  setup.py                # Extension built via `pkg-config libibverbs`
  src/ibverbs/
    __init__.py           # public re-exports + __version__
    _libverbs.pxd         # `cdef extern` declarations of <infiniband/verbs.h>
    _ibverbs.pyx          # all cdef classes + posting logic (the real binding)
    enums.py              # IntEnum/IntFlag mirrors of the rdma-core ABI
    helpers.py            # QPInfo, RC connect helpers, reg_tensor (pure Python)
    cuda.py               # optional GPUDirect: register_tensor/GpuMR (lazy libcuda)
  tests/
    conftest.py           # device/port fixtures, RoCE GID selection, markers
    _rc.py                # HostBuffer + Endpoint + connected-pair helpers
    test_*.py             # unit + integration (see README feature table)
```

## Build & test

```bash
# Dev venv lives at repo-root .venv (Python 3.14, has torch+cython+pytest+numpy).
uv pip install --python ../.venv -e . --no-build-isolation   # rebuild after .pyx/.pxd edits
../.venv/bin/python -m pytest -rs -q                          # run the suite
```

- Editing `.py` (enums/helpers/tests): no rebuild needed.
- Editing `.pyx` / `.pxd`: **must rebuild** (`uv pip install -e . --no-build-isolation`).
- The Cython-generated `_ibverbs.c` and the `.so` are git-ignored.

## How the C API is declared (`_libverbs.pxd`)

- Structs are declared **partially** — only the fields we touch. Cython trusts
  the real header for layout, so this is safe even for stack-allocated structs.
- Enum-typed fields/params are declared as plain `int` (C converts implicitly);
  this keeps the file small and robust across rdma-core versions.
- `ibv_reg_mr` and `ibv_query_port` are **macros** in the header. We bind the
  underlying real symbols instead: `ibv_reg_mr_iova2` (with `iova == addr`) and
  `___ibv_query_port`.
- The fast-path functions (`ibv_post_send/recv`, `ibv_poll_cq`,
  `ibv_req_notify_cq`) are `static inline`; declaring them as extern functions
  lets the C compiler inline them at the call site — this is *why* we use a
  compiled extension rather than ctypes.
- `ibv_send_wr.wr` is a C **union** (`rdma` / `atomic` / `ud`); declared with
  named nested `cdef struct`s that are never emitted (extern), only used for
  field-access type-checking.

## Conventions in `_ibverbs.pyx`

- **RAII:** each `cdef class` frees its handle in `__dealloc__` and in an
  idempotent `close()`; all are context managers.
- **Lifetime:** children hold a Python reference to their parent (MR→PD,
  QP→PD+CQs+SRQ, CQ→Context/CompChannel) so the GC can't free a parent first.
- **GIL:** build C structs while holding the GIL, then wrap only the actual
  `ibv_*` call in `with nogil:`. Coercing a Python object inside `nogil` is a
  compile error — convert to a C local first.
- **Errors:** verbs failures raise `VerbsError(OSError)` carrying `errno`; a bad
  *work completion* status does **not** raise (it's returned in the `WC`);
  `WC.raise_for_status()` is opt-in.
- **Byte order:** `imm_data` crosses the wire in network order — `htonl` on
  send, `ntohl` on completion, so Python always sees host order.
- **CQ ↔ event mapping:** `create_cq` passes `<void*>cq` as the CQ context, so
  `CompChannel.get_cq_event()` can hand back the exact Python `CQ`.

## Adding a wrapped verb

1. Declare the C signature/struct fields in `_libverbs.pxd` (partial is fine).
2. Add the method/class in `_ibverbs.pyx`; follow the RAII + GIL + error rules
   above; add any new enum values to `enums.py`.
3. Re-export new public names in `__init__.py` (and its `__all__`).
4. Write a test in `tests/` — a unit test if it needs no wire, otherwise add to
   the loopback/integration set. Rebuild, then `pytest`.

## RDMA gotchas learned here (don't regress these)

- **RoCE GID selection matters.** Link-local (`fe80::`) GIDs do **not** loop
  back or route to themselves on this fabric; use a global **RoCE v2** GID.
  `tests/conftest.find_roce_gid` scores candidates accordingly.
- **Union aliasing.** In `post_send`, only populate the `wr` union member for
  the actual opcode — writing `atomic.*` over `rdma.*` clobbers the rkey.
- **`char*` comparison.** Comparing device names must use `strcmp`, not `==`
  (which compares pointers in Cython).
- **GPUDirect here.** `nvidia_peermem` may fail to load (kernel/driver skew);
  the dma-buf path (`reg_dmabuf_mr` + `cuMemGetHandleForAddressRange`) is the
  reliable route and needs no kernel module. `ibverbs.cuda.register_tensor`
  wraps it — it stays torch-free (duck-typed on `data_ptr()`) and imports no
  CUDA at build. Requirements learned the hard way: torch memory must be
  VMM-backed (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`); the dma-buf
  export needs the **base page-aligned down and length page-aligned up**
  (`register_tensor` handles this via `offset`/`iova`); and `ibv_mr.addr` is
  **not** meaningful for dma-buf MRs, so `GpuMR` carries the real device VA.
- Loopback (two QPs on one port) is a real NIC round-trip and is the primary
  integration signal; two-NIC tests skip when the NICs aren't mutually routable.

## Hardware this was validated on

12× Mellanox mlx5 NICs (RoCEv2, Ethernet link layer), 8× NVIDIA H100, CUDA
driver 580.x, rdma-core 61, kernel 6.16.
