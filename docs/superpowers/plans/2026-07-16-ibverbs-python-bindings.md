# ibverbs Python Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build low-level, Pythonic Cython bindings to libibverbs (`ibverbs/`), covering all key verbs features + GPUDirect, with a full integration test suite exercised on the local mlx5 + H100 hardware.

**Architecture:** A single Cython extension `_ibverbs` wraps `<infiniband/verbs.h>` as RAII `cdef class`es (Context/PD/MR/CQ/CompChannel/QP/SRQ/AH). Pure-Python `enums.py` and `helpers.py` layer Pythonic enums and RC state-machine helpers on top. Hot paths (`post_send`/`post_recv`/`poll`/`get_cq_event`) run `nogil`. GPUDirect is address-based (no CUDA linkage).

**Tech Stack:** Cython 3.x, setuptools, libibverbs (rdma-core 61 / verbs.h 1.15), pytest, numpy, torch (tests only).

## Global Constraints

- Import name: `ibverbs`. Standalone project rooted at `ibverbs/` with its own `pyproject.toml` (PyPI target).
- **No runtime dependencies.** Build deps: Cython, setuptools, wheel. Test extras: `[test]`=pytest,numpy; `[gpu]`=torch.
- **No torch dependency in the library**; torch used only in tests.
- Target CPython 3.9+; dev/test on the repo's Python 3.14 venv (`.venv`, torch 2.13.0+cu130, 8× H100).
- Build against installed `rdma-core-devel` (pkg-config `libibverbs`).
- License header/style: BSD-3 (matches repo LICENSE, © 2026 Tristan Rice).
- Errors raise `VerbsError(OSError)` with `.errno` + operation name.

---

### Task 1: Project scaffolding + buildable empty extension

**Files:**
- Create: `ibverbs/pyproject.toml`, `ibverbs/setup.py`, `ibverbs/src/ibverbs/__init__.py`, `ibverbs/src/ibverbs/_ibverbs.pyx`, `ibverbs/src/ibverbs/_libverbs.pxd`, `ibverbs/.gitignore`
- Test: `ibverbs/tests/test_import.py`

**Interfaces:**
- Produces: importable `ibverbs` package; `ibverbs.__version__: str`.

- [ ] `pyproject.toml`: `[build-system] requires=["setuptools>=64","Cython>=3.0","wheel"]`; project name `ibverbs`, version `0.1.0`, py>=3.9, optional-deps `test`/`gpu`.
- [ ] `setup.py`: build `Extension("ibverbs._ibverbs", ["src/ibverbs/_ibverbs.pyx"])` using `pkg-config --cflags --libs libibverbs` (fallback `-libverbs`), `cythonize(language_level=3)`.
- [ ] `_libverbs.pxd`: minimal `cdef extern from "infiniband/verbs.h"` stub (expand in Task 2).
- [ ] `_ibverbs.pyx`: `__all__`, a trivial `def _linked() -> bool` calling `ibv_fork_init`-free check.
- [ ] `__init__.py`: `__version__="0.1.0"`, re-export from `_ibverbs`.
- [ ] Test `test_import.py`: `import ibverbs; assert ibverbs.__version__`.
- [ ] Build (`uv pip install -e ./ibverbs`) and run test → PASS. Commit.

### Task 2: C API declarations (`_libverbs.pxd`)

**Files:** Modify `ibverbs/src/ibverbs/_libverbs.pxd`

**Interfaces:**
- Produces: `cdef extern` decls for all structs/enums/functions used by later tasks: device list, `ibv_context`, `ibv_device_attr`, `ibv_port_attr`, `ibv_gid`, `ibv_pd`, `ibv_mr`, `ibv_reg_mr`, `ibv_reg_dmabuf_mr`, `ibv_cq`, `ibv_comp_channel`, `ibv_wc`, `ibv_qp`, `ibv_qp_init_attr`, `ibv_qp_attr`, `ibv_modify_qp`, `ibv_send_wr`, `ibv_recv_wr`, `ibv_sge`, `ibv_ah`, `ibv_ah_attr`, `ibv_srq`, `ibv_async_event`, all relevant enums, and the `static inline` fast-path fns (`ibv_post_send/recv`, `ibv_poll_cq`, `ibv_req_notify_cq`).

- [ ] Declare structs with only the fields we touch (Cython allows partial structs). Include unions (`imm_data`, `wr.rdma/atomic/ud`, `ibv_gid.raw/global`).
- [ ] Build a throwaway compile to verify decls match headers. Commit.

### Task 3: Enums (`enums.py`)

**Files:** Create `ibverbs/src/ibverbs/enums.py`; Test `ibverbs/tests/test_enums.py`

**Interfaces:**
- Produces: `AccessFlags(IntFlag)`, `QPType(IntEnum)`, `QPState(IntEnum)`, `WROpcode(IntEnum)`, `SendFlags(IntFlag)`, `WCStatus(IntEnum)`, `WCOpcode(IntEnum)`, `WCFlags(IntFlag)`, `PortState(IntEnum)`, `MTU(IntEnum)`, `NodeType`, `MigState`, `qp attr masks` (`QPAttrMask(IntFlag)`).

- [ ] TDD: test known numeric values (`AccessFlags.LOCAL_WRITE==1`, `REMOTE_WRITE==2`, `REMOTE_READ==4`, `REMOTE_ATOMIC==8`; `WROpcode.RDMA_WRITE==0`, `SEND==2`, `RDMA_READ==4`, `ATOMIC_CMP_AND_SWP==5`, `ATOMIC_FETCH_AND_ADD==6`; `WCStatus.SUCCESS==0`).
- [ ] Implement enums from verbs.h values. Commit.

### Task 4: VerbsError, Device enumeration, Context

**Files:** Modify `_ibverbs.pyx`, `_libverbs.pxd`, `__init__.py`; Test `ibverbs/tests/test_device.py`

**Interfaces:**
- Produces: `class VerbsError(OSError)`; `get_device_list()->list[Device]`; `Device.name:str`, `.guid:int`, `.node_type`, `.open()->Context`; `Context.query_device()->DeviceAttr`, `.query_port(port:int)->PortAttr`, `.query_gid(port:int,index:int)->Gid`, `.query_gid_table()`, `.num_comp_vectors:int`, `.name`, context-manager + `.close()`. `Gid.raw:bytes`, `.subnet_prefix`, `.interface_id`. `DeviceAttr`/`PortAttr` as attrs objects/dataclasses.

- [ ] TDD: `get_device_list()` returns ≥1 `Device` with name startswith `mlx5`; open, query_device has `.max_qp>0`; query_port(1) `.state`, `.gid_tbl_len`; query_gid returns 16-byte gid. Error path: `Device` open of bogus raises `VerbsError`.
- [ ] Implement. Release GIL on `ibv_open_device`? no (fast) — but on nothing blocking here. Commit.

### Task 5: PD + MR (host + GPUDirect registration)

**Files:** Modify `_ibverbs.pyx`; Test `ibverbs/tests/test_pd_mr.py`

**Interfaces:**
- Produces: `Context.alloc_pd()->PD`; `PD.reg_mr(addr:int,length:int,access:int)->MR`, `PD.reg_dmabuf_mr(offset:int,length:int,iova:int,fd:int,access:int)->MR`, `PD.close()`; `MR.addr:int`, `.length:int`, `.lkey:int`, `.rkey:int`, `.close()`. MR holds ref to PD; PD holds ref to Context.

- [ ] TDD: reg_mr over a `numpy`/`bytearray` buffer address with LOCAL_WRITE|REMOTE_WRITE|REMOTE_READ → nonzero lkey/rkey; dereg; reg_mr with bad length/addr raises VerbsError. `reg_dmabuf_mr` with bad fd raises VerbsError (path exists).
- [ ] Implement (release GIL around `ibv_reg_mr`). Commit.

### Task 6: CQ + CompChannel

**Files:** Modify `_ibverbs.pyx`; Test `ibverbs/tests/test_cq.py`

**Interfaces:**
- Produces: `Context.create_comp_channel()->CompChannel` (`.fd:int`, `.get_cq_event()->CQ`, `.close()`); `Context.create_cq(cqe:int, channel:CompChannel|None=None, comp_vector:int=0)->CQ`; `CQ.poll(num:int)->list[WC]`, `.req_notify(solicited_only=False)`, `.ack_events(n:int)`, `.cqe:int`, `.close()`. `WC` dataclass-like: `wr_id,status,opcode,byte_len,imm_data,qp_num,src_qp,wc_flags` + `raise_for_status()`.

- [ ] TDD: create_cq(16) → `.cqe>=16`; poll on empty → `[]`; comp channel `.fd>=0`; destroy order safe.
- [ ] Implement `poll` `nogil` into a stack/heap `ibv_wc` buffer; `get_cq_event` `nogil` (blocking). Commit.

### Task 7: QP + work requests + completions

**Files:** Modify `_ibverbs.pyx`; Test `ibverbs/tests/test_qp.py`

**Interfaces:**
- Produces: `SGE(addr:int|MR, length:int, lkey:int=0)` (if MR passed, derive addr+lkey; `offset` via addr arithmetic by caller); `SendWR(wr_id:int, sg:list[SGE], opcode:WROpcode, send_flags:int=0, remote_addr:int=0, rkey:int=0, imm_data:int=0, compare_add:int=0, swap:int=0, ah:AH=None, remote_qpn:int=0, remote_qkey:int=0)`; `RecvWR(wr_id:int, sg:list[SGE])`; `QPInitAttr(send_cq,recv_cq,qp_type,max_send_wr,max_recv_wr,max_send_sge,max_recv_sge,max_inline_data,srq=None,sq_sig_all=False)`; `PD.create_qp(init_attr)->QP`; `QP.qp_num:int`, `.state`, `.modify(attr:dict,mask:int)`, `.query()->(QPAttr,QPInitAttr)`, `.post_send(wrs:list[SendWR])`, `.post_recv(wrs:list[RecvWR])`, `.close()`. QP holds refs to PD, send_cq, recv_cq, srq.

- [ ] TDD: create RC QP → `.qp_num>0`, state RESET; `modify` to INIT sets state INIT; query returns INIT; post_recv on INIT QP OK; destroy.
- [ ] Implement WR chain building `nogil`; `modify` builds `ibv_qp_attr` from dict keyed by attr name. Commit.

### Task 8: AH + UD support + SRQ

**Files:** Modify `_ibverbs.pyx`; Test `ibverbs/tests/test_qp.py` (extend)

**Interfaces:**
- Produces: `AHAttr(dgid:bytes, sgid_index:int, dlid:int=0, is_global:bool=True, port_num:int=1, hop_limit:int=1, traffic_class:int=0, sl:int=0)`; `PD.create_ah(attr)->AH`; `Context`/`PD` `create_srq(max_wr,max_sge)->SRQ`; `SRQ.post_recv(wrs)`, `.modify()`, `.close()`.

- [ ] TDD: create SRQ → post_recv OK; create UD QP with SRQ; create_ah from a queried gid → object valid.
- [ ] Implement. Commit.

### Task 9: helpers.py — QPInfo + RC state machine

**Files:** Create `ibverbs/src/ibverbs/helpers.py`; Test `ibverbs/tests/test_helpers.py`

**Interfaces:**
- Produces: `@dataclass QPInfo(qp_num:int, psn:int, lid:int, gid:bytes, port:int, mtu:int)` with `to_bytes()->bytes`/`from_bytes(b)->QPInfo` (fixed struct.pack layout); `to_init(qp, port, access)`, `to_rtr(qp, remote:QPInfo, *, sgid_index:int, mtu:int|None=None, hop_limit:int=1, min_rnr_timer:int=12, max_dest_rd_atomic:int=1, sl:int=0)`, `to_rts(qp, *, psn:int, timeout:int=14, retry_cnt:int=7, rnr_retry:int=7, max_rd_atomic:int=1)`. Exposed as `QP.to_init/to_rtr/to_rts` bound methods too.

- [ ] TDD: `QPInfo` round-trips through `to_bytes/from_bytes`; helper builds correct attr+mask (test to_init transitions a real QP to INIT).
- [ ] Implement using `QPAttrMask`. Commit.

### Task 10: async events

**Files:** Modify `_ibverbs.pyx`; Test in `test_device.py`

**Interfaces:**
- Produces: `Context.get_async_event()->AsyncEvent(event_type,element)`, `Context.ack_async_event(ev)`. Context sets `O_NONBLOCK` optionally via `.async_fd`.

- [ ] TDD: `.async_fd>=0`; nonblocking get returns None/raises EAGAIN wrapped. Commit.

### Task 11: Integration — single-NIC RC loopback (all opcodes)

**Files:** Create `ibverbs/tests/conftest.py`, `ibverbs/tests/_rc.py` (test helper), `ibverbs/tests/test_loopback.py`

**Interfaces:**
- conftest fixtures: `first_device`, `active_port(dev)`, `roce_gid_index`. Markers `integration`, `gpu` registered. `_rc.connect(pd, cq, port, gid_index)` builds two connected RC QPs (loopback: each connects to the other on same device).

- [ ] TDD: SEND/RECV moves bytes; RDMA_WRITE lands in remote buffer; RDMA_WRITE_WITH_IMM delivers imm on recv WC; RDMA_READ pulls remote buffer; ATOMIC_FETCH_AND_ADD and CMP_AND_SWP update 8-byte target. Each verifies buffer contents + WC status SUCCESS.
- [ ] Commit.

### Task 12: Integration — two-NIC real wire

**Files:** Create `ibverbs/tests/test_two_device.py`

- [ ] TDD: pick two active RoCE ports (skip if <2); connect RC QP on each; RDMA_WRITE from dev A buffer to dev B buffer; verify contents. Commit.

### Task 13: Integration — event-driven completions

**Files:** Create `ibverbs/tests/test_comp_channel.py`

- [ ] TDD: attach CQ to comp channel; `req_notify`; post signaled RDMA_WRITE; `select` on `channel.fd`; `get_cq_event()`; `ack_events(1)`; poll finds the WC. Commit.

### Task 14: Integration — GPUDirect

**Files:** Create `ibverbs/tests/test_gpudirect.py`, `ibverbs/tests/_cuda.py` (ctypes dmabuf fd helper)

**Interfaces:** `_cuda.dmabuf_fd(ptr, length)->int|None` via `libcuda.cuMemGetHandleForAddressRange`.

- [ ] TDD (skip if no torch/cuda): reg_mr on `torch.empty(n, device='cuda').data_ptr()`; RDMA_WRITE GPU→GPU (two cuda buffers) over one NIC loopback; verify with `torch.equal` after `.cpu()`. Then reg_dmabuf_mr path (skip if fd unsupported). Commit.

### Task 15: Docs — README + CLAUDE.md

**Files:** Create `ibverbs/README.md`, `ibverbs/CLAUDE.md`

- [ ] README: what it is, install/build (needs rdma-core-devel), quickstart (device→pd→mr→qp→connect→post→poll), GPUDirect note, feature matrix, test instructions, license.
- [ ] CLAUDE.md: architecture, file map, build/test commands, verbs concepts cheat-sheet, gotchas (destruction order, GIL, RoCE gid index), how to add a wrapped verb. Commit.

### Task 16: Finalize — full suite, top-level README, commit & push

- [ ] Run full pytest suite on hardware; ensure all pass (or skip w/ reason).
- [ ] Update top-level `rdma4py/README.md` to point at `ibverbs/`.
- [ ] Commit and `git push`.

## Self-Review notes
- Spec coverage: device/port/gid ✓(T4), PD/MR/dmabuf ✓(T5), CQ/comp-channel ✓(T6), QP/WR/WC/opcodes ✓(T7,T11), AH/UD/SRQ ✓(T8), helpers ✓(T9), async ✓(T10), GPUDirect ✓(T5,T14), tests ✓(T11-14), docs ✓(T15). Out-of-scope (qp_ex, dm, MW, flow) intentionally omitted.
- Type consistency: `access:int` accepts `AccessFlags`; `SendWR.opcode:WROpcode`; `WC.status:WCStatus`.
