# EFA Package Guide

`efa` is a thin Cython wrapper over `libibverbs` and EFA's direct-verbs API.
Keep the package low-level: one Python object per verbs resource, explicit
state transitions, no hidden threads, no torch import, and no CUDA linkage.

## Layout

```text
efa/
  setup.py
  pyproject.toml
  src/efa/
    _libefa.pxd
    _efa.pyx
    enums.py
    helpers.py
    cuda.py
  tests/
```

## Build And Test

```bash
uv pip install --python .venv/bin/python -e . --no-build-isolation
.venv/bin/python -m pytest -rs
```

Rebuild after changing `_efa.pyx` or `_libefa.pxd`. Generated C, extension,
cache, wheel, and sdist files are ignored.

## Binding Conventions

- Resolve exported `ibv_*` and `efadv_*` symbols with `dlopen`/`dlsym`.
- Call header-only provider dispatch functions directly from Cython.
- Hold Python parent references for every live C resource.
- Make `close()` idempotent and release resources in `__dealloc__`.
- Convert Python values before entering `nogil`.
- Preserve positive return-code errors; do not assume every failure uses
  `errno`.
- Convert immediate data to network byte order when posting and back to host
  order on completion.
- Treat optional EFA 1.3/1.4 entry points as optional symbols with clear
  runtime errors.

## EFA Details

SRD uses `IBV_QPT_DRIVER` plus `EFADV_QP_DRIVER_TYPE_SRD`. It follows the UD
RESET to INIT to RTR to RTS transitions, and each send supplies an AH, remote
QPN, and qkey. One-sided operations also require the responder to have an AH
for the requester; otherwise EFA reports vendor status `0x0e`.

All EFA sends are signaled unless `sq_sig_all` was enabled. RDMA operations
use the extended `ibv_wr_*` API. SEND size is bounded by
`PortAttr.max_msg_sz`; RDMA size is bounded by
`EfaDeviceAttr.max_rdma_size`.

GPU registration prefers dma-buf and falls back to `nvidia_peermem`.
Outbound CUDA work must complete before posting. Inbound data needs
`cuFlushGPUDirectRDMAWrites` after the network completion or notification and
before CUDA consumption.
