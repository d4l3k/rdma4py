# cython: language_level=3, embedsignature=True, binding=True
"""Low-level Cython bindings for AWS EFA (libibverbs + libefa).

Every RDMA resource is a RAII ``cdef class`` that owns its C handle and frees
it in ``__dealloc__``. Children keep a Python reference to their parent so the
garbage collector cannot free a parent before its children. Hot paths
(``post_send``/``post_recv``/``poll``/``get_cq_event``) release the GIL.

EFA specifics vs. classic verbs:

- The native transport is **SRD** (Scalable Reliable Datagram): reliable,
  connectionless, out-of-order. QPs are created through the EFA provider
  (``efadv_create_qp_ex``) and use the UD-style state machine; no remote
  info is needed to reach RTS; each *send* names its destination via an
  address handle + remote QP number + qkey.
- RDMA read/write on SRD are only reachable through the **extended**
  work-request API (``ibv_wr_*``); ``post_send`` here is implemented on top
  of it, so every opcode (SEND(_WITH_IMM), RDMA_READ, RDMA_WRITE(_WITH_IMM))
  is posted uniformly.
- SEND messages are limited to the port MTU-ish ``PortAttr.max_msg_sz``
  (~8 KiB); RDMA read/write go up to ``EfaDeviceAttr.max_rdma_size`` (1 GiB
  on current hardware). Oversized posts fail with an error *completion*.
"""

import os

from libc.errno cimport EBADF, EBUSY, EIO, ENOENT, errno
from libc.stddef cimport size_t
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, uintptr_t
from libc.stdlib cimport calloc, free
from libc.string cimport memset, memcpy, strcmp

cimport efa._libefa as c


# --------------------------------------------------------------------------- #
# Runtime binding to libibverbs + libefa (dlopen/dlsym)
# --------------------------------------------------------------------------- #
# The exported (non-inline) verbs are resolved at import time from
# libibverbs.so.1 / libefa.so.1 via dlopen instead of being hard-linked. This
# means the extension has no external NEEDED dependency (trivially
# manylinux-compliant), gives a clean ImportError when the libraries are
# absent, and tolerates older rdma-core by treating newer symbols as optional.
# The data-path fast paths (ibv_wr_*/poll_cq/start_poll/...) remain compiled
# inline and dispatch through provider op tables, so they need no symbol
# resolution.

cdef extern from "dlfcn.h" nogil:
    void *dlopen(const char *filename, int flag)
    void *dlsym(void *handle, const char *symbol)
    char *dlerror()
    int RTLD_NOW
    int RTLD_GLOBAL

ctypedef c.ibv_device **(*fp_get_device_list)(int *) noexcept nogil
ctypedef void (*fp_free_device_list)(c.ibv_device **) noexcept nogil
ctypedef const char *(*fp_get_device_name)(c.ibv_device *) noexcept nogil
ctypedef uint64_t (*fp_get_device_guid)(c.ibv_device *) noexcept nogil
ctypedef c.ibv_context *(*fp_open_device)(c.ibv_device *) noexcept nogil
ctypedef int (*fp_close_device)(c.ibv_context *) noexcept nogil
ctypedef int (*fp_query_device)(c.ibv_context *, c.ibv_device_attr *) noexcept nogil
ctypedef int (*fp_query_port)(c.ibv_context *, uint8_t, c.ibv_port_attr *) noexcept nogil
ctypedef int (*fp_query_gid)(c.ibv_context *, uint8_t, int, c.ibv_gid *) noexcept nogil
ctypedef c.ibv_pd *(*fp_alloc_pd)(c.ibv_context *) noexcept nogil
ctypedef int (*fp_dealloc_pd)(c.ibv_pd *) noexcept nogil
ctypedef c.ibv_mr *(*fp_reg_mr_iova2)(c.ibv_pd *, void *, size_t, uint64_t, unsigned int) noexcept nogil
ctypedef c.ibv_mr *(*fp_reg_dmabuf_mr)(c.ibv_pd *, uint64_t, size_t, uint64_t, int, int) noexcept nogil
ctypedef int (*fp_dereg_mr)(c.ibv_mr *) noexcept nogil
ctypedef c.ibv_comp_channel *(*fp_create_comp_channel)(c.ibv_context *) noexcept nogil
ctypedef int (*fp_destroy_comp_channel)(c.ibv_comp_channel *) noexcept nogil
ctypedef c.ibv_cq *(*fp_create_cq)(c.ibv_context *, int, void *, c.ibv_comp_channel *, int) noexcept nogil
ctypedef int (*fp_destroy_cq)(c.ibv_cq *) noexcept nogil
ctypedef int (*fp_get_cq_event)(c.ibv_comp_channel *, c.ibv_cq **, void **) noexcept nogil
ctypedef void (*fp_ack_cq_events)(c.ibv_cq *, unsigned int) noexcept nogil
ctypedef int (*fp_destroy_qp)(c.ibv_qp *) noexcept nogil
ctypedef int (*fp_modify_qp)(c.ibv_qp *, c.ibv_qp_attr *, int) noexcept nogil
ctypedef int (*fp_query_qp)(c.ibv_qp *, c.ibv_qp_attr *, int, c.ibv_qp_init_attr *) noexcept nogil
ctypedef c.ibv_qp_ex *(*fp_qp_to_qp_ex)(c.ibv_qp *) noexcept nogil
ctypedef c.ibv_ah *(*fp_create_ah)(c.ibv_pd *, c.ibv_ah_attr *) noexcept nogil
ctypedef int (*fp_destroy_ah)(c.ibv_ah *) noexcept nogil
ctypedef int (*fp_get_async_event)(c.ibv_context *, c.ibv_async_event *) noexcept nogil
ctypedef void (*fp_ack_async_event)(c.ibv_async_event *) noexcept nogil
ctypedef const char *(*fp_str_from_int)(int) noexcept nogil

# libefa (efadv_*) entry points.
ctypedef int (*fp_efadv_query_device)(c.ibv_context *, c.rdma4py_efadv_device_attr *, uint32_t) noexcept nogil
ctypedef int (*fp_efadv_query_ah)(c.ibv_ah *, c.rdma4py_efadv_ah_attr *, uint32_t) noexcept nogil
ctypedef c.ibv_qp *(*fp_efadv_create_qp_ex)(c.ibv_context *, c.ibv_qp_init_attr_ex *, c.rdma4py_efadv_qp_init_attr *, uint32_t) noexcept nogil
ctypedef c.ibv_cq_ex *(*fp_efadv_create_cq)(c.ibv_context *, c.ibv_cq_init_attr_ex *, c.rdma4py_efadv_cq_init_attr *, uint32_t) noexcept nogil
ctypedef int (*fp_efadv_query_cq)(c.ibv_cq *, c.rdma4py_efadv_cq_attr *, uint32_t) noexcept nogil
ctypedef c.rdma4py_efadv_cq *(*fp_efadv_cq_from_ibv_cq_ex)(c.ibv_cq_ex *) noexcept nogil
ctypedef int (*fp_efadv_query_mr)(c.ibv_mr *, c.rdma4py_efadv_mr_attr *, uint32_t) noexcept nogil
ctypedef int (*fp_efadv_query_qp_wqs)(c.ibv_qp *, c.rdma4py_efadv_wq_attr *, c.rdma4py_efadv_wq_attr *, uint32_t) noexcept nogil

cdef void *_libverbs_handle = NULL
cdef void *_libefa_handle = NULL
cdef fp_get_device_list _v_get_device_list = NULL
cdef fp_free_device_list _v_free_device_list = NULL
cdef fp_get_device_name _v_get_device_name = NULL
cdef fp_get_device_guid _v_get_device_guid = NULL
cdef fp_open_device _v_open_device = NULL
cdef fp_close_device _v_close_device = NULL
cdef fp_query_device _v_query_device = NULL
cdef fp_query_port _v_query_port = NULL
cdef fp_query_gid _v_query_gid = NULL
cdef fp_alloc_pd _v_alloc_pd = NULL
cdef fp_dealloc_pd _v_dealloc_pd = NULL
cdef fp_reg_mr_iova2 _v_reg_mr_iova2 = NULL
cdef fp_reg_dmabuf_mr _v_reg_dmabuf_mr = NULL
cdef fp_dereg_mr _v_dereg_mr = NULL
cdef fp_create_comp_channel _v_create_comp_channel = NULL
cdef fp_destroy_comp_channel _v_destroy_comp_channel = NULL
cdef fp_create_cq _v_create_cq = NULL
cdef fp_destroy_cq _v_destroy_cq = NULL
cdef fp_get_cq_event _v_get_cq_event = NULL
cdef fp_ack_cq_events _v_ack_cq_events = NULL
cdef fp_destroy_qp _v_destroy_qp = NULL
cdef fp_modify_qp _v_modify_qp = NULL
cdef fp_query_qp _v_query_qp = NULL
cdef fp_qp_to_qp_ex _v_qp_to_qp_ex = NULL
cdef fp_create_ah _v_create_ah = NULL
cdef fp_destroy_ah _v_destroy_ah = NULL
cdef fp_get_async_event _v_get_async_event = NULL
cdef fp_ack_async_event _v_ack_async_event = NULL
cdef fp_str_from_int _v_event_type_str = NULL
cdef fp_str_from_int _v_wc_status_str = NULL

cdef fp_efadv_query_device _e_query_device = NULL
cdef fp_efadv_query_ah _e_query_ah = NULL
cdef fp_efadv_create_qp_ex _e_create_qp_ex = NULL
cdef fp_efadv_create_cq _e_create_cq = NULL
cdef fp_efadv_query_cq _e_query_cq = NULL
cdef fp_efadv_cq_from_ibv_cq_ex _e_cq_from_cq_ex = NULL
cdef fp_efadv_query_mr _e_query_mr = NULL
cdef fp_efadv_query_qp_wqs _e_query_qp_wqs = NULL


cdef void *_req_sym(void *h, const char *lib, const char *name) except NULL:
    cdef void *sym = dlsym(h, name)
    if sym is NULL:
        raise ImportError("%s is missing required symbol %s" % (
            (<bytes>lib).decode(), (<bytes>name).decode()))
    return sym


cdef void *_dl(const char *soname) except NULL:
    cdef void *h = dlopen(soname, RTLD_NOW | RTLD_GLOBAL)
    cdef char *e
    if h is NULL:
        e = dlerror()
        raise ImportError(
            "could not load %s - install rdma-core (libibverbs + the EFA "
            "provider libefa) (%s)" % ((<bytes>soname).decode(),
            (<bytes>e).decode() if e is not NULL else "not found"))
    return h


cdef int _load_libraries() except -1:
    global _libverbs_handle, _libefa_handle
    global _v_get_device_list, _v_free_device_list, _v_get_device_name
    global _v_get_device_guid, _v_open_device, _v_close_device, _v_query_device
    global _v_query_port, _v_query_gid, _v_alloc_pd, _v_dealloc_pd
    global _v_reg_mr_iova2, _v_reg_dmabuf_mr, _v_dereg_mr
    global _v_create_comp_channel, _v_destroy_comp_channel, _v_create_cq
    global _v_destroy_cq, _v_get_cq_event, _v_ack_cq_events
    global _v_destroy_qp, _v_modify_qp, _v_query_qp, _v_qp_to_qp_ex
    global _v_create_ah, _v_destroy_ah
    global _v_get_async_event, _v_ack_async_event, _v_event_type_str
    global _v_wc_status_str
    global _e_query_device, _e_query_ah, _e_create_qp_ex, _e_create_cq
    global _e_query_cq, _e_cq_from_cq_ex, _e_query_mr, _e_query_qp_wqs
    cdef void *hv = _dl(b"libibverbs.so.1")
    _libverbs_handle = hv
    _v_get_device_list = <fp_get_device_list>_req_sym(hv, b"libibverbs", b"ibv_get_device_list")
    _v_free_device_list = <fp_free_device_list>_req_sym(hv, b"libibverbs", b"ibv_free_device_list")
    _v_get_device_name = <fp_get_device_name>_req_sym(hv, b"libibverbs", b"ibv_get_device_name")
    _v_get_device_guid = <fp_get_device_guid>_req_sym(hv, b"libibverbs", b"ibv_get_device_guid")
    _v_open_device = <fp_open_device>_req_sym(hv, b"libibverbs", b"ibv_open_device")
    _v_close_device = <fp_close_device>_req_sym(hv, b"libibverbs", b"ibv_close_device")
    _v_query_device = <fp_query_device>_req_sym(hv, b"libibverbs", b"ibv_query_device")
    _v_query_port = <fp_query_port>_req_sym(hv, b"libibverbs", b"ibv_query_port")
    _v_query_gid = <fp_query_gid>_req_sym(hv, b"libibverbs", b"ibv_query_gid")
    _v_alloc_pd = <fp_alloc_pd>_req_sym(hv, b"libibverbs", b"ibv_alloc_pd")
    _v_dealloc_pd = <fp_dealloc_pd>_req_sym(hv, b"libibverbs", b"ibv_dealloc_pd")
    _v_reg_mr_iova2 = <fp_reg_mr_iova2>_req_sym(hv, b"libibverbs", b"ibv_reg_mr_iova2")
    _v_dereg_mr = <fp_dereg_mr>_req_sym(hv, b"libibverbs", b"ibv_dereg_mr")
    _v_create_comp_channel = <fp_create_comp_channel>_req_sym(hv, b"libibverbs", b"ibv_create_comp_channel")
    _v_destroy_comp_channel = <fp_destroy_comp_channel>_req_sym(hv, b"libibverbs", b"ibv_destroy_comp_channel")
    _v_create_cq = <fp_create_cq>_req_sym(hv, b"libibverbs", b"ibv_create_cq")
    _v_destroy_cq = <fp_destroy_cq>_req_sym(hv, b"libibverbs", b"ibv_destroy_cq")
    _v_get_cq_event = <fp_get_cq_event>_req_sym(hv, b"libibverbs", b"ibv_get_cq_event")
    _v_ack_cq_events = <fp_ack_cq_events>_req_sym(hv, b"libibverbs", b"ibv_ack_cq_events")
    _v_destroy_qp = <fp_destroy_qp>_req_sym(hv, b"libibverbs", b"ibv_destroy_qp")
    _v_modify_qp = <fp_modify_qp>_req_sym(hv, b"libibverbs", b"ibv_modify_qp")
    _v_query_qp = <fp_query_qp>_req_sym(hv, b"libibverbs", b"ibv_query_qp")
    _v_qp_to_qp_ex = <fp_qp_to_qp_ex>_req_sym(hv, b"libibverbs", b"ibv_qp_to_qp_ex")
    _v_create_ah = <fp_create_ah>_req_sym(hv, b"libibverbs", b"ibv_create_ah")
    _v_destroy_ah = <fp_destroy_ah>_req_sym(hv, b"libibverbs", b"ibv_destroy_ah")
    _v_get_async_event = <fp_get_async_event>_req_sym(hv, b"libibverbs", b"ibv_get_async_event")
    _v_ack_async_event = <fp_ack_async_event>_req_sym(hv, b"libibverbs", b"ibv_ack_async_event")
    _v_event_type_str = <fp_str_from_int>_req_sym(hv, b"libibverbs", b"ibv_event_type_str")
    _v_wc_status_str = <fp_str_from_int>_req_sym(hv, b"libibverbs", b"ibv_wc_status_str")
    # Optional: added in rdma-core 34 (IBVERBS_1.12). Absent on older systems.
    _v_reg_dmabuf_mr = <fp_reg_dmabuf_mr>dlsym(hv, b"ibv_reg_dmabuf_mr")

    # libefa: the EFA provider's direct-verbs entry points. libibverbs loads
    # the provider itself for device operation; we dlopen the same shared
    # object to reach the efadv_* API.
    cdef void *he = _dl(b"libefa.so.1")
    _libefa_handle = he
    _e_query_device = <fp_efadv_query_device>_req_sym(he, b"libefa", b"efadv_query_device")
    _e_query_ah = <fp_efadv_query_ah>_req_sym(he, b"libefa", b"efadv_query_ah")
    _e_create_qp_ex = <fp_efadv_create_qp_ex>_req_sym(he, b"libefa", b"efadv_create_qp_ex")
    _e_create_cq = <fp_efadv_create_cq>_req_sym(he, b"libefa", b"efadv_create_cq")
    _e_cq_from_cq_ex = <fp_efadv_cq_from_ibv_cq_ex>_req_sym(he, b"libefa", b"efadv_cq_from_ibv_cq_ex")
    # Optional: EFA_1.3 / EFA_1.4 additions.
    _e_query_mr = <fp_efadv_query_mr>dlsym(he, b"efadv_query_mr")
    _e_query_cq = <fp_efadv_query_cq>dlsym(he, b"efadv_query_cq")
    _e_query_qp_wqs = <fp_efadv_query_qp_wqs>dlsym(he, b"efadv_query_qp_wqs")
    return 0


_load_libraries()


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class EfaError(OSError):
    """Raised when a libibverbs/libefa call fails.

    Subclasses :class:`OSError`, so ``.errno`` and ``.strerror`` are populated.
    ``.operation`` names the call that failed.
    """

    def __init__(self, str operation, int err, detail=None):
        self.operation = operation
        msg = ("%s failed: %s" % (operation, os.strerror(err))
               if detail is None else str(detail))
        super().__init__(err, msg)


cdef int _error_from_rc(int rc):
    if rc > 0:
        return rc
    if errno != 0:
        return errno
    if rc < -1:
        return -rc
    return EIO


cdef int _fail(str op) except -1:
    raise EfaError(op, errno if errno != 0 else EIO)


cdef int _fail_rc(str op, int rc) except -1:
    raise EfaError(op, _error_from_rc(rc))


# --------------------------------------------------------------------------- #
# Linkage smoke test (kept for the import test)
# --------------------------------------------------------------------------- #
def _linked() -> bool:
    """Return True if libibverbs and libefa were loaded at import."""
    return _libverbs_handle is not NULL and _libefa_handle is not NULL


def _has_dmabuf() -> bool:
    """Return True if this libibverbs provides ``ibv_reg_dmabuf_mr`` (rdma-core >= 34)."""
    return _v_reg_dmabuf_mr is not NULL


# --------------------------------------------------------------------------- #
# Result value objects
# --------------------------------------------------------------------------- #
cdef class Gid:
    """A 16-byte GID (the EFA device address)."""

    cdef readonly bytes raw

    def __init__(self, bytes raw):
        if len(raw) != 16:
            raise ValueError("gid must be exactly 16 bytes")
        self.raw = raw

    @property
    def subnet_prefix(self) -> int:
        return int.from_bytes(self.raw[:8], "big")

    @property
    def interface_id(self) -> int:
        return int.from_bytes(self.raw[8:], "big")

    def __bytes__(self):
        return self.raw

    def __eq__(self, other):
        if isinstance(other, Gid):
            return self.raw == (<Gid>other).raw
        if isinstance(other, (bytes, bytearray)):
            return self.raw == bytes(other)
        return NotImplemented

    def __repr__(self):
        cdef int i
        parts = ["%02x%02x" % (self.raw[i], self.raw[i + 1]) for i in range(0, 16, 2)]
        return "Gid('%s')" % ":".join(parts)


cdef class DeviceAttr:
    """Generic verbs device capabilities from :meth:`Context.query_device`."""

    cdef readonly str fw_ver
    cdef readonly uint64_t node_guid
    cdef readonly uint64_t sys_image_guid
    cdef readonly uint64_t max_mr_size
    cdef readonly uint32_t vendor_id
    cdef readonly uint32_t vendor_part_id
    cdef readonly uint32_t hw_ver
    cdef readonly int max_qp
    cdef readonly int max_qp_wr
    cdef readonly int max_sge
    cdef readonly int max_cq
    cdef readonly int max_cqe
    cdef readonly int max_mr
    cdef readonly int max_pd
    cdef readonly int max_ah
    cdef readonly int max_pkeys
    cdef readonly int phys_port_cnt

    def __repr__(self):
        return "DeviceAttr(fw_ver=%r, max_qp=%d, max_cqe=%d, phys_port_cnt=%d)" % (
            self.fw_ver, self.max_qp, self.max_cqe, self.phys_port_cnt)


cdef class EfaDeviceAttr:
    """EFA-specific capabilities from :meth:`Context.query_efa_device`."""

    cdef readonly uint32_t max_sq_wr
    cdef readonly uint32_t max_rq_wr
    cdef readonly int max_sq_sge
    cdef readonly int max_rq_sge
    cdef readonly int inline_buf_size
    cdef readonly uint32_t device_caps
    cdef readonly uint32_t max_rdma_size

    def __repr__(self):
        return ("EfaDeviceAttr(max_sq_wr=%d, max_rq_wr=%d, inline_buf_size=%d, "
                "device_caps=0x%x, max_rdma_size=%d)") % (
            self.max_sq_wr, self.max_rq_wr, self.inline_buf_size,
            self.device_caps, self.max_rdma_size)


cdef class PortAttr:
    """Port attributes from :meth:`Context.query_port`.

    ``max_msg_sz`` is the SEND size limit; RDMA read/write are limited by
    :attr:`EfaDeviceAttr.max_rdma_size` instead.
    """

    cdef readonly int state
    cdef readonly int max_mtu
    cdef readonly int active_mtu
    cdef readonly int gid_tbl_len
    cdef readonly uint32_t port_cap_flags
    cdef readonly uint32_t max_msg_sz
    cdef readonly int pkey_tbl_len
    cdef readonly int lid
    cdef readonly int active_width
    cdef readonly int active_speed
    cdef readonly int link_layer

    def __repr__(self):
        return "PortAttr(state=%d, active_mtu=%d, max_msg_sz=%d)" % (
            self.state, self.active_mtu, self.max_msg_sz)


cdef class WC:
    """A work completion returned by :meth:`CQ.poll` / :meth:`CQEx.poll`.

    For completions from a :class:`CQEx` created with ``sgid=True``,
    :attr:`sgid` carries the sender's GID on receive completions from peers
    not covered by a local AH (else ``None``).
    """

    cdef readonly uint64_t wr_id
    cdef readonly int status
    cdef readonly int opcode
    cdef readonly uint32_t vendor_err
    cdef readonly uint32_t byte_len
    cdef readonly uint32_t imm_data
    cdef readonly uint32_t qp_num
    cdef readonly uint32_t src_qp
    cdef readonly unsigned int wc_flags
    cdef readonly object sgid
    cdef readonly bint unsolicited

    @property
    def is_success(self) -> bool:
        return self.status == 0

    @property
    def status_str(self) -> str:
        return _v_wc_status_str(self.status).decode()

    def raise_for_status(self):
        """Raise :class:`CompletionError` if this completion did not succeed."""
        if self.status != 0:
            raise CompletionError(self)

    def __repr__(self):
        return ("WC(wr_id=%d, status=%d (%s), opcode=%d, byte_len=%d, "
                "qp_num=%d)") % (self.wr_id, self.status, self.status_str,
                                 self.opcode, self.byte_len, self.qp_num)


class CompletionError(EfaError):
    """Raised by :meth:`WC.raise_for_status` for a failed completion."""

    def __init__(self, WC wc):
        self.wc = wc
        self.status = wc.status
        self.vendor_err = wc.vendor_err
        super().__init__(
            "work completion",
            EIO,
            "work completion failed: %s (status=%d, vendor_err=0x%x)" % (
                wc.status_str, wc.status, wc.vendor_err),
        )


cdef class CQAttr:
    """An EFA completion queue's direct layout from ``efadv_query_cq``."""

    cdef readonly uint64_t buffer_addr
    cdef readonly uint32_t entry_size
    cdef readonly uint32_t num_entries
    cdef readonly uint64_t doorbell_addr

    def __repr__(self):
        return ("CQAttr(buffer_addr=0x%x, entry_size=%d, num_entries=%d, "
                "doorbell_addr=0x%x)") % (
            self.buffer_addr, self.entry_size, self.num_entries,
            self.doorbell_addr)


# --------------------------------------------------------------------------- #
# Work request inputs
# --------------------------------------------------------------------------- #
cdef class SGE:
    """A scatter/gather entry.

    ``SGE(target, length, lkey=0, offset=0)`` where ``target`` is either an
    :class:`MR` (``lkey``/``addr`` derived from it) or an integer address.
    """

    cdef readonly uint64_t addr
    cdef readonly uint32_t length
    cdef readonly uint32_t lkey
    cdef readonly object owner

    def __init__(self, target, length, lkey=0, offset=0):
        cdef MR mr
        cdef object py_length = int(length)
        cdef object py_offset = int(offset)
        cdef object py_lkey = int(lkey)
        cdef object py_addr
        if py_length < 0 or py_length > 0xFFFFFFFF:
            raise ValueError("SGE length must be between 0 and 2**32 - 1")
        if py_offset < 0:
            raise ValueError("SGE offset must be non-negative")
        if py_lkey < 0 or py_lkey > 0xFFFFFFFF:
            raise ValueError("SGE lkey must be between 0 and 2**32 - 1")
        if isinstance(target, MR):
            mr = <MR>target
            mr._ensure()
            if py_offset > mr._mr.length or py_length > mr._mr.length - py_offset:
                raise ValueError("SGE range exceeds the memory region")
            self.addr = ((<uint64_t><uintptr_t>mr._mr.addr)
                         + <uint64_t>py_offset)
            self.lkey = mr._mr.lkey if py_lkey == 0 else <uint32_t>py_lkey
            self.owner = mr
        else:
            py_addr = int(target)
            if py_addr < 0 or py_addr > 0xFFFFFFFFFFFFFFFF:
                raise ValueError("SGE address must be between 0 and 2**64 - 1")
            if py_addr + py_offset > 0xFFFFFFFFFFFFFFFF:
                raise ValueError("SGE address plus offset exceeds 2**64 - 1")
            self.addr = <uint64_t>(py_addr + py_offset)
            self.lkey = <uint32_t>py_lkey
            self.owner = None
        self.length = <uint32_t>py_length

    def _keepalive(self, owner):
        """Retain ``owner`` for as long as this SGE is retained."""
        self.owner = owner
        return self

    def __repr__(self):
        return "SGE(addr=0x%x, length=%d, lkey=0x%x)" % (
            self.addr, self.length, self.lkey)


cdef class SendWR:
    """A send work request.

    SRD/UD is connectionless: every send names its destination. Set either
    ``dest=`` (anything with ``.ah``, ``.qp_num`` and ``.qkey`` attributes,
    e.g. :class:`efa.helpers.Peer`) or the explicit ``ah``/``remote_qpn``/
    ``remote_qkey`` trio.

    Opcodes: SEND, SEND_WITH_IMM, RDMA_READ, RDMA_WRITE, RDMA_WRITE_WITH_IMM
    (RDMA opcodes additionally need ``remote_addr``/``rkey``).
    """

    cdef public uint64_t wr_id
    cdef public list sg_list
    cdef public int opcode
    cdef public unsigned int send_flags
    cdef public uint64_t remote_addr
    cdef public uint32_t rkey
    cdef public uint32_t imm_data
    cdef public object ah
    cdef public uint32_t remote_qpn
    cdef public uint32_t remote_qkey

    def __init__(self, wr_id=0, sg_list=None, opcode=2, send_flags=0,
                 remote_addr=0, rkey=0, imm_data=0, ah=None, remote_qpn=0,
                 remote_qkey=0, dest=None):
        self.wr_id = wr_id
        self.sg_list = list(sg_list) if sg_list is not None else []
        self.opcode = opcode
        self.send_flags = send_flags
        self.remote_addr = remote_addr
        self.rkey = rkey
        self.imm_data = imm_data
        if dest is not None:
            self.ah = dest.ah
            self.remote_qpn = dest.qp_num
            self.remote_qkey = dest.qkey
        else:
            self.ah = ah
            self.remote_qpn = remote_qpn
            self.remote_qkey = remote_qkey


cdef class RecvWR:
    """A receive work request."""

    cdef public uint64_t wr_id
    cdef public list sg_list

    def __init__(self, wr_id=0, sg_list=None):
        self.wr_id = wr_id
        self.sg_list = list(sg_list) if sg_list is not None else []


cdef class QPCap:
    """Queue-pair capacity limits."""

    cdef readonly uint32_t max_send_wr
    cdef readonly uint32_t max_recv_wr
    cdef readonly uint32_t max_send_sge
    cdef readonly uint32_t max_recv_sge
    cdef readonly uint32_t max_inline_data

    def __repr__(self):
        return ("QPCap(max_send_wr=%d, max_recv_wr=%d, max_send_sge=%d, "
                "max_recv_sge=%d, max_inline_data=%d)") % (
            self.max_send_wr, self.max_recv_wr, self.max_send_sge,
            self.max_recv_sge, self.max_inline_data)


class QPInitAttr:
    """Parameters for :meth:`PD.create_qp`.

    ``qp_type`` is :attr:`~efa.enums.QPType.SRD` (default) or ``UD``.
    ``send_ops_flags`` defaults to every operation the QP type supports
    (SEND/SEND_WITH_IMM, plus the RDMA ops for SRD); pass an explicit
    :class:`~efa.enums.QPExSendOpsFlags` mask to restrict it; the provider
    rejects masks including ops the device cannot do, so pass e.g.
    ``SEND | SEND_WITH_IMM`` on RDMA-less instance types.
    ``unsolicited_write_recv`` opts in to RDMA-write-with-imm completions
    that consume no posted recv (check
    :attr:`~efa.enums.EfaDeviceCaps.UNSOLICITED_WRITE_RECV`).
    """

    def __init__(self, send_cq, recv_cq, qp_type=None, max_send_wr=128,
                 max_recv_wr=128, max_send_sge=1, max_recv_sge=1,
                 max_inline_data=0, sq_sig_all=False, send_ops_flags=None,
                 sl=0, unsolicited_write_recv=False):
        self.send_cq = send_cq
        self.recv_cq = recv_cq
        # enums.QPType.SRD; kept numeric here so _efa has no enums import.
        self.qp_type = int(qp_type) if qp_type is not None else 0xFF
        self.max_send_wr = max_send_wr
        self.max_recv_wr = max_recv_wr
        self.max_send_sge = max_send_sge
        self.max_recv_sge = max_recv_sge
        self.max_inline_data = max_inline_data
        self.sq_sig_all = bool(sq_sig_all)
        self.send_ops_flags = (int(send_ops_flags)
                               if send_ops_flags is not None else None)
        self.sl = int(sl)
        self.unsolicited_write_recv = bool(unsolicited_write_recv)


cdef class AHAttr:
    """Address-handle attributes.

    EFA addresses are GIDs: only ``dgid`` (the remote device's GID),
    ``sgid_index`` and ``port_num`` matter; the rest exists for parity with
    generic verbs.
    """

    cdef public bytes dgid
    cdef public uint32_t flow_label
    cdef public uint8_t sgid_index
    cdef public uint8_t hop_limit
    cdef public uint8_t traffic_class
    cdef public uint8_t is_global
    cdef public uint8_t port_num

    def __init__(self, dgid=None, sgid_index=0, port_num=1, hop_limit=1,
                 traffic_class=0, flow_label=0, is_global=1):
        self.dgid = bytes(dgid) if dgid is not None else b"\x00" * 16
        if len(self.dgid) != 16:
            raise ValueError("dgid must be 16 bytes")
        self.sgid_index = sgid_index
        self.port_num = port_num
        self.hop_limit = hop_limit
        self.traffic_class = traffic_class
        self.flow_label = flow_label
        self.is_global = is_global


cdef void _fill_ah_attr(c.ibv_ah_attr *dst, AHAttr src):
    memset(dst, 0, sizeof(c.ibv_ah_attr))
    dst.is_global = src.is_global
    dst.port_num = src.port_num
    memcpy(&dst.grh.dgid.raw[0], <char*>src.dgid, 16)
    dst.grh.flow_label = src.flow_label
    dst.grh.sgid_index = src.sgid_index
    dst.grh.hop_limit = src.hop_limit
    dst.grh.traffic_class = src.traffic_class


# --------------------------------------------------------------------------- #
# Device / Context
# --------------------------------------------------------------------------- #
cdef class Device:
    """An RDMA device discovered by :func:`get_device_list`."""

    cdef readonly str name
    cdef readonly uint64_t guid

    def __init__(self, str name, uint64_t guid):
        self.name = name
        self.guid = guid

    def open(self) -> "Context":
        """Open this device, returning a :class:`Context`."""
        cdef int num = 0
        cdef c.ibv_device **lst = _v_get_device_list(&num)
        cdef c.ibv_context *ctx = NULL
        cdef bytes want = self.name.encode()
        cdef int i
        if lst is NULL:
            _fail("ibv_get_device_list")
        try:
            for i in range(num):
                if strcmp(_v_get_device_name(lst[i]), <char*>want) == 0:
                    ctx = _v_open_device(lst[i])
                    break
        finally:
            _v_free_device_list(lst)
        if ctx is NULL:
            raise EfaError("ibv_open_device(%s)" % self.name, errno)
        return Context._wrap(ctx, self.name)

    def __repr__(self):
        return "Device(name=%r, guid=0x%x)" % (self.name, self.guid)


def get_device_list() -> list:
    """Return every RDMA :class:`Device` on this host (EFA or not).

    Use :func:`get_efa_device_list` to keep only EFA devices.
    """
    cdef int num = 0
    cdef c.ibv_device **lst = _v_get_device_list(&num)
    cdef int i
    if lst is NULL:
        _fail("ibv_get_device_list")
    out = []
    try:
        for i in range(num):
            name = _v_get_device_name(lst[i]).decode()
            guid = _v_get_device_guid(lst[i])
            out.append(Device(name, guid))
    finally:
        _v_free_device_list(lst)
    return out


def get_efa_device_list() -> list:
    """Return the :class:`Device` list filtered to EFA devices.

    Each device is briefly opened and probed with ``efadv_query_device``.
    """
    out = []
    for dev in get_device_list():
        try:
            ctx = dev.open()
        except EfaError:
            continue
        try:
            if ctx.is_efa():
                out.append(dev)
        finally:
            ctx.close()
    return out


cdef class Context:
    """An open device context (``ibv_context``)."""

    cdef c.ibv_context *_ctx
    cdef readonly str name
    cdef unsigned int _children

    def __cinit__(self):
        self._ctx = NULL
        self._children = 0

    @staticmethod
    cdef Context _wrap(c.ibv_context *ctx, str name):
        cdef Context self = Context.__new__(Context)
        self._ctx = ctx
        self.name = name
        return self

    cdef int _ensure(self) except -1:
        if self._ctx is NULL:
            raise EfaError("context is closed", EBADF)
        return 0

    cdef void _add_child(self):
        self._children += 1

    cdef void _release_child(self):
        if self._children > 0:
            self._children -= 1

    @property
    def num_comp_vectors(self) -> int:
        self._ensure()
        return self._ctx.num_comp_vectors

    @property
    def async_fd(self) -> int:
        self._ensure()
        return self._ctx.async_fd

    def is_efa(self) -> bool:
        """Return True if this device is driven by the EFA provider."""
        self._ensure()
        cdef c.rdma4py_efadv_device_attr a
        memset(&a, 0, sizeof(a))
        return _e_query_device(self._ctx, &a, sizeof(a)) == 0

    def query_device(self) -> DeviceAttr:
        self._ensure()
        cdef c.ibv_device_attr a
        cdef int rc = _v_query_device(self._ctx, &a)
        if rc != 0:
            _fail_rc("ibv_query_device", rc)
        cdef DeviceAttr r = DeviceAttr.__new__(DeviceAttr)
        r.fw_ver = a.fw_ver.decode() if a.fw_ver[0] != 0 else ""
        r.node_guid = a.node_guid
        r.sys_image_guid = a.sys_image_guid
        r.max_mr_size = a.max_mr_size
        r.vendor_id = a.vendor_id
        r.vendor_part_id = a.vendor_part_id
        r.hw_ver = a.hw_ver
        r.max_qp = a.max_qp
        r.max_qp_wr = a.max_qp_wr
        r.max_sge = a.max_sge
        r.max_cq = a.max_cq
        r.max_cqe = a.max_cqe
        r.max_mr = a.max_mr
        r.max_pd = a.max_pd
        r.max_ah = a.max_ah
        r.max_pkeys = a.max_pkeys
        r.phys_port_cnt = a.phys_port_cnt
        return r

    def query_efa_device(self) -> EfaDeviceAttr:
        """EFA capabilities (``efadv_query_device``).

        Raises :class:`EfaError` if the device is not an EFA device.
        """
        self._ensure()
        cdef c.rdma4py_efadv_device_attr a
        memset(&a, 0, sizeof(a))
        cdef int rc = _e_query_device(self._ctx, &a, sizeof(a))
        if rc != 0:
            _fail_rc("efadv_query_device", rc)
        cdef EfaDeviceAttr r = EfaDeviceAttr.__new__(EfaDeviceAttr)
        r.max_sq_wr = a.max_sq_wr
        r.max_rq_wr = a.max_rq_wr
        r.max_sq_sge = a.max_sq_sge
        r.max_rq_sge = a.max_rq_sge
        r.inline_buf_size = a.inline_buf_size
        r.device_caps = a.device_caps
        r.max_rdma_size = a.max_rdma_size
        return r

    def query_port(self, int port_num=1) -> PortAttr:
        self._ensure()
        cdef c.ibv_port_attr a
        memset(&a, 0, sizeof(a))
        cdef int rc = _v_query_port(self._ctx, <uint8_t>port_num, &a)
        if rc != 0:
            _fail_rc("ibv_query_port", rc)
        cdef PortAttr r = PortAttr.__new__(PortAttr)
        r.state = a.state
        r.max_mtu = a.max_mtu
        r.active_mtu = a.active_mtu
        r.gid_tbl_len = a.gid_tbl_len
        r.port_cap_flags = a.port_cap_flags
        r.max_msg_sz = a.max_msg_sz
        r.pkey_tbl_len = a.pkey_tbl_len
        r.lid = a.lid
        r.active_width = a.active_width
        r.active_speed = a.active_speed
        r.link_layer = a.link_layer
        return r

    def query_gid(self, int port_num=1, int index=0) -> Gid:
        """The device GID, which is the EFA device's network address."""
        self._ensure()
        cdef c.ibv_gid g
        if _v_query_gid(self._ctx, <uint8_t>port_num, index, &g) != 0:
            _fail("ibv_query_gid")
        return Gid(bytes(bytearray(g.raw[:16])))

    def alloc_pd(self) -> "PD":
        self._ensure()
        cdef c.ibv_pd *pd = _v_alloc_pd(self._ctx)
        if pd is NULL:
            _fail("ibv_alloc_pd")
        return PD._wrap(pd, self)

    def create_comp_channel(self) -> "CompChannel":
        self._ensure()
        cdef c.ibv_comp_channel *ch = _v_create_comp_channel(self._ctx)
        if ch is NULL:
            _fail("ibv_create_comp_channel")
        return CompChannel._wrap(ch, self)

    def create_cq(self, int cqe, channel=None, int comp_vector=0) -> "CQ":
        """Create a classic completion queue (``ibv_create_cq``)."""
        self._ensure()
        if cqe <= 0:
            raise ValueError("cqe must be positive")
        if comp_vector < 0 or comp_vector >= self._ctx.num_comp_vectors:
            raise ValueError("comp_vector is outside the context's range")
        cdef c.ibv_comp_channel *chp = NULL
        cdef CompChannel ch = None
        if channel is not None:
            ch = <CompChannel>channel
            ch._ensure()
            if ch.context is not self:
                raise ValueError("completion channel must belong to this context")
            chp = ch._chan
        cdef CQ cq = CQ.__new__(CQ)
        cq.context = self
        cq.channel = ch
        cq._cq = _v_create_cq(self._ctx, cqe, <void*>cq, chp, comp_vector)
        if cq._cq is NULL:
            _fail("ibv_create_cq")
        self._add_child()
        return cq

    def create_cq_ex(self, int cqe, channel=None, int comp_vector=0,
                     wc_flags=None, bint sgid=False,
                     bint unsolicited=False) -> "CQEx":
        """Create an extended EFA completion queue (``efadv_create_cq``).

        ``sgid=True`` requests sender-GID reporting on receive completions
        (needs :attr:`~efa.enums.EfaDeviceCaps.CQ_WITH_SGID`); the GID shows
        up as :attr:`WC.sgid` for peers not matched by a local AH.

        ``unsolicited=True`` requests an indicator for unsolicited
        write-with-immediate receive completions (needs
        :attr:`~efa.enums.EfaDeviceCaps.UNSOLICITED_WRITE_RECV`).
        """
        self._ensure()
        if cqe <= 0:
            raise ValueError("cqe must be positive")
        if comp_vector < 0 or comp_vector >= self._ctx.num_comp_vectors:
            raise ValueError("comp_vector is outside the context's range")
        cdef c.ibv_comp_channel *chp = NULL
        cdef CompChannel ch = None
        if channel is not None:
            ch = <CompChannel>channel
            ch._ensure()
            if ch.context is not self:
                raise ValueError("completion channel must belong to this context")
            chp = ch._chan
        cdef c.ibv_cq_init_attr_ex a
        memset(&a, 0, sizeof(a))
        a.cqe = cqe
        a.channel = chp
        a.comp_vector = comp_vector
        # IBV_WC_STANDARD_FLAGS: byte_len, imm, qp_num, src_qp, slid, sl,
        # dlid_path_bits.
        a.wc_flags = int(wc_flags) if wc_flags is not None else 0x7F
        cdef c.rdma4py_efadv_cq_init_attr ea
        memset(&ea, 0, sizeof(ea))
        if sgid:
            ea.wc_flags = 1  # EFADV_WC_EX_WITH_SGID
        if unsolicited:
            ea.wc_flags |= 2  # EFADV_WC_EX_WITH_IS_UNSOLICITED
        cdef CQEx cq = CQEx.__new__(CQEx)
        cq.context = self
        cq.channel = ch
        cq._cqx = _e_create_cq(self._ctx, &a, &ea, sizeof(ea))
        if cq._cqx is NULL:
            _fail("efadv_create_cq")
        cq._cqx.cq_context = <void*>cq
        cq._ecq = _e_cq_from_cq_ex(cq._cqx) if ea.wc_flags else NULL
        cq._sgid = sgid
        cq._unsolicited = unsolicited
        self._add_child()
        return cq

    def get_async_event(self) -> "AsyncEvent":
        self._ensure()
        cdef AsyncEvent ev = AsyncEvent.__new__(AsyncEvent)
        cdef int rc
        with nogil:
            rc = _v_get_async_event(self._ctx, &ev._ev)
        if rc != 0:
            _fail("ibv_get_async_event")
        ev._acked = False
        ev.event_type = ev._ev.event_type
        return ev

    def ack_async_event(self, AsyncEvent ev):
        if not ev._acked:
            _v_ack_async_event(&ev._ev)
            ev._acked = True

    def close(self):
        if self._ctx is not NULL:
            if self._children != 0:
                raise EfaError(
                    "ibv_close_device", EBUSY,
                    "cannot close context with %d open child resource(s)"
                    % self._children,
                )
            if _v_close_device(self._ctx) != 0:
                _fail("ibv_close_device")
            self._ctx = NULL

    def __dealloc__(self):
        if self._ctx is not NULL and self._children == 0:
            _v_close_device(self._ctx)
            self._ctx = NULL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        return "Context(name=%r)" % self.name


cdef class AsyncEvent:
    """An asynchronous device event from :meth:`Context.get_async_event`."""

    cdef c.ibv_async_event _ev
    cdef bint _acked
    cdef readonly int event_type

    @property
    def event_type_str(self) -> str:
        return _v_event_type_str(self.event_type).decode()

    def __repr__(self):
        return "AsyncEvent(%s)" % self.event_type_str


# --------------------------------------------------------------------------- #
# Protection domain / memory region
# --------------------------------------------------------------------------- #
cdef class PD:
    """A protection domain (``ibv_pd``)."""

    cdef c.ibv_pd *_pd
    cdef readonly Context context

    @staticmethod
    cdef PD _wrap(c.ibv_pd *pd, Context ctx):
        cdef PD self = PD.__new__(PD)
        self._pd = pd
        self.context = ctx
        ctx._add_child()
        return self

    cdef int _ensure(self) except -1:
        if self._pd is NULL:
            raise EfaError("pd is closed", EBADF)
        return 0

    def reg_mr(self, addr, length, int access) -> "MR":
        """Register a memory region over ``[addr, addr+length)``.

        ``addr`` is an integer virtual address; it may be a host pointer or a
        CUDA device pointer (GPUDirect via nvidia_peermem).
        """
        self._ensure()
        cdef object py_addr = int(addr)
        cdef object py_length = int(length)
        if py_addr < 0 or py_addr > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("addr must be between 0 and 2**64 - 1")
        if py_length <= 0 or py_length > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("length must be between 1 and 2**64 - 1")
        if py_addr + py_length > 0x10000000000000000:
            raise ValueError("memory region address range exceeds 2**64")
        cdef uint64_t a = <uint64_t>py_addr
        cdef size_t ln = <size_t>py_length
        cdef unsigned int acc = <unsigned int>access
        cdef c.ibv_mr *mr
        with nogil:
            mr = _v_reg_mr_iova2(self._pd, <void*><uintptr_t>a, ln, a, acc)
        if mr is NULL:
            _fail("ibv_reg_mr")
        return MR._wrap(mr, self)

    def reg_dmabuf_mr(self, offset, length, iova, int fd, int access) -> "MR":
        """Register a dma-buf backed region (modern GPUDirect path)."""
        self._ensure()
        if _v_reg_dmabuf_mr is NULL:
            raise RuntimeError(
                "ibv_reg_dmabuf_mr is unavailable: this libibverbs predates "
                "rdma-core 34. Use reg_mr with nvidia_peermem instead.")
        cdef object py_offset = int(offset)
        cdef object py_length = int(length)
        cdef object py_iova = int(iova)
        if py_offset < 0 or py_offset > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("offset must be between 0 and 2**64 - 1")
        if py_length <= 0 or py_length > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("length must be between 1 and 2**64 - 1")
        if py_iova < 0 or py_iova > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("iova must be between 0 and 2**64 - 1")
        cdef c.ibv_mr *mr
        cdef uint64_t off = <uint64_t>py_offset
        cdef size_t ln = <size_t>py_length
        cdef uint64_t iv = <uint64_t>py_iova
        with nogil:
            mr = _v_reg_dmabuf_mr(self._pd, off, ln, iv, fd, access)
        if mr is NULL:
            _fail("ibv_reg_dmabuf_mr")
        return MR._wrap(mr, self)

    def create_qp(self, init_attr) -> "QP":
        """Create an SRD (default) or UD queue pair.

        SRD QPs go through ``efadv_create_qp_ex``; UD QPs through
        ``ibv_create_qp_ex``. Both are created with the extended send-ops
        work-request API enabled (see :class:`QPInitAttr`).
        """
        self._ensure()
        cdef c.ibv_qp_init_attr_ex a
        memset(&a, 0, sizeof(a))
        scq_obj = init_attr.send_cq
        rcq_obj = init_attr.recv_cq
        cdef c.ibv_cq *scq_p = _cq_ptr(scq_obj)
        cdef c.ibv_cq *rcq_p = _cq_ptr(rcq_obj)
        if (<_CQBase>scq_obj).context is not self.context or \
                (<_CQBase>rcq_obj).context is not self.context:
            raise ValueError("queue-pair CQs must belong to the PD's context")
        a.send_cq = scq_p
        a.recv_cq = rcq_p
        cdef int qp_type = init_attr.qp_type
        a.sq_sig_all = 1 if init_attr.sq_sig_all else 0
        a.cap.max_send_wr = init_attr.max_send_wr
        a.cap.max_recv_wr = init_attr.max_recv_wr
        a.cap.max_send_sge = init_attr.max_send_sge
        a.cap.max_recv_sge = init_attr.max_recv_sge
        a.cap.max_inline_data = init_attr.max_inline_data
        a.comp_mask = _QP_INIT_ATTR_PD | _QP_INIT_ATTR_SEND_OPS_FLAGS
        a.pd = self._pd
        cdef uint64_t ops
        if init_attr.send_ops_flags is not None:
            ops = <uint64_t>init_attr.send_ops_flags
        else:
            ops = _OPS_SEND | _OPS_SEND_WITH_IMM
            if qp_type == _QPT_SRD:
                ops |= _OPS_RDMA_READ | _OPS_RDMA_WRITE | _OPS_RDMA_WRITE_WITH_IMM
        a.send_ops_flags = ops

        cdef c.rdma4py_efadv_qp_init_attr ea
        cdef c.ibv_qp *qp = NULL
        if qp_type == _QPT_SRD:
            a.qp_type = _QPT_DRIVER
            memset(&ea, 0, sizeof(ea))
            ea.driver_qp_type = 0  # EFADV_QP_DRIVER_TYPE_SRD
            ea.sl = <uint8_t>init_attr.sl
            if init_attr.unsolicited_write_recv:
                ea.flags = 1  # EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV
            qp = _e_create_qp_ex(self.context._ctx, &a, &ea, sizeof(ea))
            if qp is NULL:
                _fail("efadv_create_qp_ex")
        elif qp_type == _QPT_UD:
            a.qp_type = _QPT_UD
            # UD supports only SEND ops; a default mask with RDMA bits would
            # be rejected by the provider.
            if init_attr.send_ops_flags is None:
                a.send_ops_flags = _OPS_SEND | _OPS_SEND_WITH_IMM
            qp = _create_qp_ex(self.context._ctx, &a)
            if qp is NULL:
                _fail("ibv_create_qp_ex")
        else:
            raise ValueError("EFA supports qp_type SRD or UD")
        return QP._wrap(qp, self, scq_obj, rcq_obj, qp_type, a.send_ops_flags)

    def create_ah(self, addr, int sgid_index=0, int port_num=1) -> "AH":
        """Create an address handle for a remote EFA device.

        ``addr`` may be an :class:`AHAttr`, a :class:`Gid`, or 16 raw GID
        bytes.
        """
        self._ensure()
        cdef AHAttr attr
        if isinstance(addr, AHAttr):
            attr = <AHAttr>addr
        elif isinstance(addr, Gid):
            attr = AHAttr((<Gid>addr).raw, sgid_index=sgid_index,
                          port_num=port_num)
        else:
            attr = AHAttr(bytes(addr), sgid_index=sgid_index,
                          port_num=port_num)
        cdef c.ibv_ah_attr a
        _fill_ah_attr(&a, attr)
        cdef c.ibv_ah *ah = _v_create_ah(self._pd, &a)
        if ah is NULL:
            _fail("ibv_create_ah")
        return AH._wrap(ah, self)

    def close(self):
        cdef int rc
        if self._pd is not NULL:
            rc = _v_dealloc_pd(self._pd)
            if rc != 0:
                _fail_rc("ibv_dealloc_pd", rc)
            self._pd = NULL
            self.context._release_child()

    def __dealloc__(self):
        if self._pd is not NULL:
            if _v_dealloc_pd(self._pd) == 0:
                self.context._release_child()
            self._pd = NULL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


cdef class MRAttr:
    """EFA MR attributes from :meth:`MR.query_efa` (``efadv_query_mr``)."""

    cdef readonly int ic_id_validity
    cdef readonly int recv_ic_id
    cdef readonly int rdma_read_ic_id
    cdef readonly int rdma_recv_ic_id

    def __repr__(self):
        return ("MRAttr(ic_id_validity=0x%x, recv_ic_id=%d, "
                "rdma_read_ic_id=%d, rdma_recv_ic_id=%d)") % (
            self.ic_id_validity, self.recv_ic_id, self.rdma_read_ic_id,
            self.rdma_recv_ic_id)


cdef class MR:
    """A registered memory region (``ibv_mr``)."""

    cdef c.ibv_mr *_mr
    cdef readonly PD pd
    cdef readonly object owner

    @staticmethod
    cdef MR _wrap(c.ibv_mr *mr, PD pd):
        cdef MR self = MR.__new__(MR)
        self._mr = mr
        self.pd = pd
        self.owner = None
        return self

    cdef int _ensure(self) except -1:
        if self._mr is NULL:
            raise EfaError("memory region is closed", EBADF)
        return 0

    @property
    def closed(self) -> bool:
        return self._mr is NULL

    @property
    def addr(self) -> int:
        self._ensure()
        return <uint64_t><uintptr_t>self._mr.addr

    @property
    def length(self) -> int:
        self._ensure()
        return self._mr.length

    @property
    def lkey(self) -> int:
        self._ensure()
        return self._mr.lkey

    @property
    def rkey(self) -> int:
        self._ensure()
        return self._mr.rkey

    @property
    def handle(self) -> int:
        self._ensure()
        return self._mr.handle

    def sge(self, length=None, offset=0):
        """Return an :class:`SGE` for a bounded range of this MR."""
        self._ensure()
        cdef object py_offset = int(offset)
        if py_offset < 0 or py_offset > self._mr.length:
            raise ValueError("SGE offset is outside the memory region")
        if length is None:
            length = self._mr.length - py_offset
        return SGE(self, length, offset=py_offset)

    def query_efa(self) -> MRAttr:
        """EFA-specific MR attributes (``efadv_query_mr``, rdma-core >= 44)."""
        self._ensure()
        if _e_query_mr is NULL:
            raise RuntimeError(
                "efadv_query_mr is unavailable: this libefa predates EFA 1.3")
        cdef c.rdma4py_efadv_mr_attr a
        memset(&a, 0, sizeof(a))
        cdef int rc = _e_query_mr(self._mr, &a, sizeof(a))
        if rc != 0:
            _fail_rc("efadv_query_mr", rc)
        cdef MRAttr r = MRAttr.__new__(MRAttr)
        r.ic_id_validity = a.ic_id_validity
        r.recv_ic_id = a.recv_ic_id
        r.rdma_read_ic_id = a.rdma_read_ic_id
        r.rdma_recv_ic_id = a.rdma_recv_ic_id
        return r

    def _keepalive(self, owner):
        """Retain the object that owns this MR's backing allocation."""
        self.owner = owner
        return self

    def close(self):
        cdef int rc
        if self._mr is not NULL:
            rc = _v_dereg_mr(self._mr)
            if rc != 0:
                _fail_rc("ibv_dereg_mr", rc)
            self._mr = NULL
            self.owner = None

    def __dealloc__(self):
        if self._mr is not NULL:
            _v_dereg_mr(self._mr)
            self._mr = NULL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        if self._mr is NULL:
            return "MR(closed=True)"
        return "MR(addr=0x%x, length=%d, lkey=0x%x, rkey=0x%x)" % (
            self.addr, self.length, self.lkey, self.rkey)


# --------------------------------------------------------------------------- #
# Completion channel / queues
# --------------------------------------------------------------------------- #
cdef class CompChannel:
    """A completion event channel (``ibv_comp_channel``)."""

    cdef c.ibv_comp_channel *_chan
    cdef readonly Context context

    @staticmethod
    cdef CompChannel _wrap(c.ibv_comp_channel *chan, Context ctx):
        cdef CompChannel self = CompChannel.__new__(CompChannel)
        self._chan = chan
        self.context = ctx
        ctx._add_child()
        return self

    @property
    def fd(self) -> int:
        self._ensure()
        return self._chan.fd

    cdef int _ensure(self) except -1:
        if self._chan is NULL:
            raise EfaError("completion channel is closed", EBADF)
        return 0

    def get_cq_event(self):
        """Block until a CQ event arrives; return the associated CQ object.

        The event must later be acknowledged with ``ack_events()`` on the
        returned :class:`CQ` / :class:`CQEx`.
        """
        self._ensure()
        cdef c.ibv_cq *cq = NULL
        cdef void *ctx = NULL
        cdef int rc
        with nogil:
            rc = _v_get_cq_event(self._chan, &cq, &ctx)
        if rc != 0:
            _fail("ibv_get_cq_event")
        cdef _CQBase obj = <_CQBase>ctx
        obj._unacked += 1
        return obj

    def close(self):
        cdef int rc
        if self._chan is not NULL:
            rc = _v_destroy_comp_channel(self._chan)
            if rc != 0:
                _fail_rc("ibv_destroy_comp_channel", rc)
            self._chan = NULL
            self.context._release_child()

    def __dealloc__(self):
        if self._chan is not NULL:
            if _v_destroy_comp_channel(self._chan) == 0:
                self.context._release_child()
            self._chan = NULL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


cdef class _CQBase:
    """Shared plumbing for :class:`CQ` and :class:`CQEx`."""

    cdef readonly Context context
    cdef readonly CompChannel channel
    cdef int _unacked

    cdef c.ibv_cq *_ptr(self):
        return NULL

    cdef int _ensure(self) except -1:
        if self._ptr() is NULL:
            raise EfaError("completion queue is closed", EBADF)
        return 0

    @property
    def cqe(self) -> int:
        self._ensure()
        return self._ptr().cqe

    def req_notify(self, bint solicited_only=False):
        """Request a completion notification on the CQ's channel."""
        self._ensure()
        cdef int rc = c.ibv_req_notify_cq(
            self._ptr(), 1 if solicited_only else 0)
        if rc != 0:
            _fail_rc("ibv_req_notify_cq", rc)

    def ack_events(self, unsigned int nevents=1):
        """Acknowledge ``nevents`` events delivered via the channel."""
        self._ensure()
        if nevents == 0:
            raise ValueError("nevents must be positive")
        if <int>nevents > self._unacked:
            raise ValueError("cannot acknowledge more CQ events than were received")
        _v_ack_cq_events(self._ptr(), nevents)
        self._unacked -= <int>nevents

    def close(self):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


cdef CQAttr _query_efa_cq(c.ibv_cq *cq):
    if _e_query_cq is NULL:
        raise RuntimeError(
            "efadv_query_cq is unavailable: this libefa predates EFA 1.4")
    cdef c.rdma4py_efadv_cq_attr a
    memset(&a, 0, sizeof(a))
    cdef int rc = _e_query_cq(cq, &a, sizeof(a))
    if rc != 0:
        _fail_rc("efadv_query_cq", rc)
    cdef CQAttr r = CQAttr.__new__(CQAttr)
    r.buffer_addr = <uint64_t><uintptr_t>a.buffer
    r.entry_size = a.entry_size
    r.num_entries = a.num_entries
    r.doorbell_addr = <uint64_t><uintptr_t>a.doorbell
    return r


cdef c.ibv_cq *_cq_ptr(obj) except? NULL:
    """The underlying ibv_cq pointer of a CQ or CQEx (with ensure)."""
    cdef _CQBase base = <_CQBase?>obj
    base._ensure()
    return base._ptr()


cdef class CQ(_CQBase):
    """A classic completion queue (``ibv_cq``)."""

    cdef c.ibv_cq *_cq

    cdef c.ibv_cq *_ptr(self):
        return self._cq

    def poll(self, int num_entries) -> list:
        """Poll up to ``num_entries`` completions; return a list of :class:`WC`."""
        self._ensure()
        if num_entries <= 0:
            raise ValueError("num_entries must be positive")
        cdef c.ibv_wc *wcs = <c.ibv_wc*>calloc(num_entries, sizeof(c.ibv_wc))
        if wcs is NULL:
            raise MemoryError()
        cdef int n
        cdef int i
        try:
            with nogil:
                n = c.ibv_poll_cq(self._cq, num_entries, wcs)
            if n < 0:
                raise EfaError("ibv_poll_cq", -n if n < -1 else (
                    errno if errno != 0 else EIO))
            out = []
            for i in range(n):
                out.append(_wc_from_c(&wcs[i]))
            return out
        finally:
            free(wcs)

    def query_efa(self) -> CQAttr:
        """Return the provider's direct CQ layout (EFA 1.4 or newer)."""
        self._ensure()
        return _query_efa_cq(self._cq)

    def close(self):
        cdef int rc
        if self._cq is not NULL:
            if self._unacked > 0:
                _v_ack_cq_events(self._cq, <unsigned int>self._unacked)
                self._unacked = 0
            rc = _v_destroy_cq(self._cq)
            if rc != 0:
                _fail_rc("ibv_destroy_cq", rc)
            self._cq = NULL
            self.context._release_child()

    def __dealloc__(self):
        if self._cq is not NULL:
            if self._unacked > 0:
                _v_ack_cq_events(self._cq, <unsigned int>self._unacked)
            if _v_destroy_cq(self._cq) == 0:
                self.context._release_child()
            self._cq = NULL


cdef class CQEx(_CQBase):
    """An extended EFA completion queue (``efadv_create_cq``).

    Functionally a :class:`CQ`, plus optional sender-GID reporting
    (``sgid=True``): SRD/UD receivers can identify previously unknown peers
    straight from the completion (see :attr:`WC.sgid`).
    """

    cdef c.ibv_cq_ex *_cqx
    cdef c.rdma4py_efadv_cq *_ecq
    cdef bint _sgid
    cdef bint _unsolicited

    cdef c.ibv_cq *_ptr(self):
        if self._cqx is NULL:
            return NULL
        return c.ibv_cq_ex_to_cq(self._cqx)

    @property
    def sgid_enabled(self) -> bool:
        return self._sgid

    @property
    def unsolicited_enabled(self) -> bool:
        return self._unsolicited

    def poll(self, int num_entries) -> list:
        """Poll up to ``num_entries`` completions; return a list of :class:`WC`."""
        self._ensure()
        if num_entries <= 0:
            raise ValueError("num_entries must be positive")
        cdef c.ibv_poll_cq_attr pattr
        memset(&pattr, 0, sizeof(pattr))
        cdef int rc
        with nogil:
            rc = c.ibv_start_poll(self._cqx, &pattr)
        out = []
        if rc == ENOENT:
            return out
        if rc != 0:
            _fail_rc("ibv_start_poll", rc)
        cdef WC w
        cdef c.ibv_gid sg
        try:
            while True:
                w = WC.__new__(WC)
                w.wr_id = self._cqx.wr_id
                w.status = self._cqx.status
                w.opcode = c.ibv_wc_read_opcode(self._cqx)
                w.vendor_err = c.ibv_wc_read_vendor_err(self._cqx)
                w.byte_len = c.ibv_wc_read_byte_len(self._cqx)
                w.imm_data = c.ntohl(c.ibv_wc_read_imm_data(self._cqx))
                w.qp_num = c.ibv_wc_read_qp_num(self._cqx)
                w.src_qp = c.ibv_wc_read_src_qp(self._cqx)
                w.wc_flags = c.ibv_wc_read_wc_flags(self._cqx)
                if self._ecq is not NULL:
                    # Only reported for receive completions whose sender has
                    # no local AH; ENOENT otherwise.
                    if self._sgid and c.rdma4py_efadv_wc_read_sgid(
                        self._ecq, &sg
                    ) == 0:
                        w.sgid = bytes(bytearray(sg.raw[:16]))
                    if self._unsolicited:
                        w.unsolicited = c.rdma4py_efadv_wc_is_unsolicited(
                            self._ecq
                        )
                out.append(w)
                if len(out) >= num_entries:
                    break
                rc = c.ibv_next_poll(self._cqx)
                if rc == ENOENT:
                    break
                if rc != 0:
                    _fail_rc("ibv_next_poll", rc)
        finally:
            c.ibv_end_poll(self._cqx)
        return out

    def query_efa(self) -> CQAttr:
        """Return the provider's direct CQ layout (EFA 1.4 or newer)."""
        self._ensure()
        return _query_efa_cq(c.ibv_cq_ex_to_cq(self._cqx))

    def close(self):
        cdef int rc
        cdef c.ibv_cq *cq
        if self._cqx is not NULL:
            cq = c.ibv_cq_ex_to_cq(self._cqx)
            if self._unacked > 0:
                _v_ack_cq_events(cq, <unsigned int>self._unacked)
                self._unacked = 0
            rc = _v_destroy_cq(cq)
            if rc != 0:
                _fail_rc("ibv_destroy_cq", rc)
            self._cqx = NULL
            self._ecq = NULL
            self.context._release_child()

    def __dealloc__(self):
        cdef c.ibv_cq *cq
        if self._cqx is not NULL:
            cq = c.ibv_cq_ex_to_cq(self._cqx)
            if self._unacked > 0:
                _v_ack_cq_events(cq, <unsigned int>self._unacked)
            if _v_destroy_cq(cq) == 0:
                self.context._release_child()
            self._cqx = NULL
            self._ecq = NULL


cdef c.ibv_qp *_create_qp_ex(c.ibv_context *ctx,
                             c.ibv_qp_init_attr_ex *attr) noexcept nogil:
    # Equivalent of the header's static inline ibv_create_qp_ex, minus its
    # compat path (which references the exported ibv_create_qp symbol and
    # would defeat the dlopen-only linkage). We always pass more than
    # IBV_QP_INIT_ATTR_PD in comp_mask, so the op-table path is the right one.
    cdef c.verbs_context *vctx = c.verbs_get_ctx(ctx)
    if vctx is NULL:
        return NULL
    return vctx.create_qp_ex(ctx, attr)


cdef WC _wc_from_c(c.ibv_wc *w):
    cdef WC r = WC.__new__(WC)
    r.wr_id = w.wr_id
    r.status = w.status
    r.opcode = w.opcode
    r.vendor_err = w.vendor_err
    r.byte_len = w.byte_len
    r.imm_data = c.ntohl(w.imm_data)
    r.qp_num = w.qp_num
    r.src_qp = w.src_qp
    r.wc_flags = w.wc_flags
    return r


# --------------------------------------------------------------------------- #
# Queue pair
# --------------------------------------------------------------------------- #
# QP attr-mask bits needed by the state-machine helpers (see enums.QPAttrMask).
cdef enum:
    _QP_STATE = 1 << 0
    _QP_PKEY_INDEX = 1 << 4
    _QP_PORT = 1 << 5
    _QP_QKEY = 1 << 6
    _QP_RNR_RETRY = 1 << 11
    _QP_SQ_PSN = 1 << 16

# ibv_qp_type (SRD is our own tag mapped to IBV_QPT_DRIVER at create).
cdef enum:
    _QPT_UD = 4
    _QPT_DRIVER = 0xFF
    _QPT_SRD = 0xFF

# ibv_qp_state
cdef enum:
    _QPS_INIT = 1
    _QPS_RTR = 2
    _QPS_RTS = 3

# ibv_qp_init_attr_mask
cdef enum:
    _QP_INIT_ATTR_PD = 1 << 0
    _QP_INIT_ATTR_SEND_OPS_FLAGS = 1 << 6

# ibv_qp_create_send_ops_flags
cdef enum:
    _OPS_RDMA_WRITE = 1 << 0
    _OPS_RDMA_WRITE_WITH_IMM = 1 << 1
    _OPS_SEND = 1 << 2
    _OPS_SEND_WITH_IMM = 1 << 3
    _OPS_RDMA_READ = 1 << 4

# ibv_wr_opcode (the subset EFA supports)
cdef enum:
    _WR_RDMA_WRITE = 0
    _WR_RDMA_WRITE_WITH_IMM = 1
    _WR_SEND = 2
    _WR_SEND_WITH_IMM = 3
    _WR_RDMA_READ = 4

# ibv_send_flags
cdef enum:
    _SEND_INLINE = 1 << 3


cdef class WQAttr:
    """A work queue's layout from :meth:`QP.query_wqs` (``efadv_query_qp_wqs``)."""

    cdef readonly uint64_t buffer_addr
    cdef readonly uint32_t entry_size
    cdef readonly uint32_t num_entries
    cdef readonly uint64_t doorbell_addr
    cdef readonly uint32_t max_batch

    def __repr__(self):
        return ("WQAttr(buffer_addr=0x%x, entry_size=%d, num_entries=%d, "
                "max_batch=%d)") % (self.buffer_addr, self.entry_size,
                                    self.num_entries, self.max_batch)


cdef class QP:
    """An EFA queue pair (SRD or UD).

    Sends go through the extended work-request API under the hood, so every
    supported opcode, including RDMA read/write on SRD, which the classic
    ``ibv_post_send`` cannot express on this provider, posts uniformly via
    :meth:`post_send`.
    """

    cdef c.ibv_qp *_qp
    cdef c.ibv_qp_ex *_qpx
    cdef readonly PD pd
    cdef readonly object send_cq
    cdef readonly object recv_cq
    cdef readonly int qp_type
    cdef readonly uint64_t send_ops_flags

    cdef int _ensure(self) except -1:
        if self._qp is NULL:
            raise EfaError("queue pair is closed", EBADF)
        return 0

    @staticmethod
    cdef QP _wrap(c.ibv_qp *qp, PD pd, scq, rcq, int qp_type,
                  uint64_t send_ops_flags):
        cdef QP self = QP.__new__(QP)
        self._qp = qp
        self._qpx = _v_qp_to_qp_ex(qp)
        if self._qpx is NULL:
            _v_destroy_qp(qp)
            self._qp = NULL
            _fail("ibv_qp_to_qp_ex")
        self.pd = pd
        self.send_cq = scq
        self.recv_cq = rcq
        self.qp_type = qp_type
        self.send_ops_flags = send_ops_flags
        return self

    @property
    def qp_num(self) -> int:
        self._ensure()
        return self._qp.qp_num

    @property
    def state(self) -> int:
        """The queue pair's current state (authoritative, via query)."""
        self._ensure()
        cdef c.ibv_qp_attr a
        cdef c.ibv_qp_init_attr ia
        memset(&a, 0, sizeof(a))
        cdef int rc = _v_query_qp(self._qp, &a, _QP_STATE, &ia)
        if rc != 0:
            _fail_rc("ibv_query_qp", rc)
        return a.qp_state

    def query(self):
        """Return ``(attrs, cap)`` for the queue pair as plain dict + QPCap."""
        self._ensure()
        cdef c.ibv_qp_attr a
        cdef c.ibv_qp_init_attr ia
        memset(&a, 0, sizeof(a))
        memset(&ia, 0, sizeof(ia))
        # STATE | PKEY_INDEX | PORT | QKEY | RNR_RETRY | SQ_PSN | CAP:
        # the attributes the EFA kernel driver can report (others -> EOPNOTSUPP).
        cdef int rc = _v_query_qp(self._qp, &a, 0x90871, &ia)
        if rc != 0:
            _fail_rc("ibv_query_qp", rc)
        cdef QPCap cap = QPCap.__new__(QPCap)
        cap.max_send_wr = ia.cap.max_send_wr
        cap.max_recv_wr = ia.cap.max_recv_wr
        cap.max_send_sge = ia.cap.max_send_sge
        cap.max_recv_sge = ia.cap.max_recv_sge
        cap.max_inline_data = ia.cap.max_inline_data
        attrs = {
            "qp_state": a.qp_state,
            "qkey": a.qkey,
            "sq_psn": a.sq_psn,
            "port_num": a.port_num,
            "pkey_index": a.pkey_index,
            "rnr_retry": a.rnr_retry,
        }
        return attrs, cap

    def modify(self, int attr_mask, **fields):
        """Low-level ``ibv_modify_qp``.

        ``attr_mask`` is an ORed :class:`~efa.enums.QPAttrMask`. Scalar
        attributes are passed as keywords (e.g. ``qp_state=``, ``qkey=``).
        """
        self._ensure()
        unknown = set(fields) - {
            "qp_state", "cur_qp_state", "qkey", "rq_psn", "sq_psn",
            "pkey_index", "port_num", "rnr_retry",
        }
        if unknown:
            raise TypeError("unknown QP attribute(s): %s" %
                            ", ".join(sorted(unknown)))
        cdef c.ibv_qp_attr a
        memset(&a, 0, sizeof(a))
        if "qp_state" in fields: a.qp_state = fields["qp_state"]
        if "cur_qp_state" in fields: a.cur_qp_state = fields["cur_qp_state"]
        if "qkey" in fields: a.qkey = fields["qkey"]
        if "rq_psn" in fields: a.rq_psn = fields["rq_psn"]
        if "sq_psn" in fields: a.sq_psn = fields["sq_psn"]
        if "pkey_index" in fields: a.pkey_index = fields["pkey_index"]
        if "port_num" in fields: a.port_num = fields["port_num"]
        if "rnr_retry" in fields: a.rnr_retry = fields["rnr_retry"]
        cdef int rc = _v_modify_qp(self._qp, &a, attr_mask)
        if rc != 0:
            _fail_rc("ibv_modify_qp", rc)

    # -- SRD/UD state machine --------------------------------------------- #
    def to_init(self, qkey, int port=1, int pkey_index=0):
        """Transition RESET -> INIT, binding the qkey."""
        self._ensure()
        cdef c.ibv_qp_attr a
        memset(&a, 0, sizeof(a))
        a.qp_state = _QPS_INIT
        a.pkey_index = <uint16_t>pkey_index
        a.port_num = <uint8_t>port
        a.qkey = <uint32_t>int(qkey)
        cdef int rc = _v_modify_qp(
            self._qp, &a, _QP_STATE | _QP_PKEY_INDEX | _QP_PORT | _QP_QKEY)
        if rc != 0:
            _fail_rc("ibv_modify_qp(->INIT)", rc)

    def to_rtr(self):
        """Transition INIT -> RTR (connectionless: no remote info needed)."""
        self._ensure()
        cdef c.ibv_qp_attr a
        memset(&a, 0, sizeof(a))
        a.qp_state = _QPS_RTR
        cdef int rc = _v_modify_qp(self._qp, &a, _QP_STATE)
        if rc != 0:
            _fail_rc("ibv_modify_qp(->RTR)", rc)

    def to_rts(self, psn=0, rnr_retry=None):
        """Transition RTR -> RTS.

        ``rnr_retry`` (0-7, 7 = infinite) is EFA's receiver-not-ready retry
        count; needs :attr:`~efa.enums.EfaDeviceCaps.RNR_RETRY`.
        """
        self._ensure()
        cdef c.ibv_qp_attr a
        memset(&a, 0, sizeof(a))
        a.qp_state = _QPS_RTS
        a.sq_psn = <uint32_t>int(psn)
        cdef int mask = _QP_STATE | _QP_SQ_PSN
        if rnr_retry is not None:
            a.rnr_retry = <uint8_t>int(rnr_retry)
            mask |= _QP_RNR_RETRY
        cdef int rc = _v_modify_qp(self._qp, &a, mask)
        if rc != 0:
            _fail_rc("ibv_modify_qp(->RTS)", rc)

    def prepare(self, qkey, int port=1, psn=0, rnr_retry=None):
        """Drive RESET -> INIT -> RTR -> RTS in one call.

        SRD/UD are connectionless, so a QP is fully usable after this without
        any remote information; destinations are named per-send.
        """
        self.to_init(qkey, port=port)
        self.to_rtr()
        self.to_rts(psn=psn, rnr_retry=rnr_retry)
        return self

    # -- data path ---------------------------------------------------------- #
    def post_send(self, wrs):
        """Post one or more :class:`SendWR` as a single batch.

        All work requests are queued between ``wr_start``/``wr_complete``, so
        either the whole batch is submitted or none of it is.

        EFA requires every send WR to carry
        :attr:`~efa.enums.SendFlags.SIGNALED` (or the QP must be created with
        ``sq_sig_all=True``); an unsignaled WR fails the batch with EINVAL.
        """
        self._ensure()
        if isinstance(wrs, SendWR):
            wrs = [wrs]
        elif not isinstance(wrs, list):
            wrs = list(wrs)
        _post_send_ex(self._qpx, wrs, self.send_ops_flags)

    def post_recv(self, wrs):
        """Post one or more :class:`RecvWR` to the receive queue."""
        self._ensure()
        if isinstance(wrs, RecvWR):
            wrs = [wrs]
        elif not isinstance(wrs, list):
            wrs = list(wrs)
        _post_recv_qp(self._qp, wrs)

    def query_wqs(self):
        """Return ``(sq, rq)`` :class:`WQAttr` (``efadv_query_qp_wqs``, EFA >= 1.4)."""
        self._ensure()
        if _e_query_qp_wqs is NULL:
            raise RuntimeError(
                "efadv_query_qp_wqs is unavailable: this libefa predates EFA 1.4")
        cdef c.rdma4py_efadv_wq_attr sq
        cdef c.rdma4py_efadv_wq_attr rq
        memset(&sq, 0, sizeof(sq))
        memset(&rq, 0, sizeof(rq))
        cdef int rc = _e_query_qp_wqs(self._qp, &sq, &rq, sizeof(sq))
        if rc != 0:
            _fail_rc("efadv_query_qp_wqs", rc)
        cdef WQAttr s = WQAttr.__new__(WQAttr)
        s.buffer_addr = <uint64_t><uintptr_t>sq.buffer
        s.entry_size = sq.entry_size
        s.num_entries = sq.num_entries
        s.doorbell_addr = <uint64_t><uintptr_t>sq.doorbell
        s.max_batch = sq.max_batch
        cdef WQAttr r = WQAttr.__new__(WQAttr)
        r.buffer_addr = <uint64_t><uintptr_t>rq.buffer
        r.entry_size = rq.entry_size
        r.num_entries = rq.num_entries
        r.doorbell_addr = <uint64_t><uintptr_t>rq.doorbell
        r.max_batch = rq.max_batch
        return s, r

    def close(self):
        cdef int rc
        if self._qp is not NULL:
            rc = _v_destroy_qp(self._qp)
            if rc != 0:
                _fail_rc("ibv_destroy_qp", rc)
            self._qp = NULL
            self._qpx = NULL

    def __dealloc__(self):
        if self._qp is not NULL:
            _v_destroy_qp(self._qp)
            self._qp = NULL
            self._qpx = NULL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        if self._qp is NULL:
            return "QP(closed=True)"
        return "QP(qp_num=%d, qp_type=%s)" % (
            self._qp.qp_num, "SRD" if self.qp_type == _QPT_SRD else "UD")


# --------------------------------------------------------------------------- #
# Work-request posting
# --------------------------------------------------------------------------- #
# Maps each supported WR opcode to the send-ops bit the QP must have been
# created with. Posting an op the provider did not install would call a NULL
# function pointer (crash), so this is checked before dispatch.
cdef uint64_t _op_required_bit(int opcode):
    if opcode == _WR_SEND:
        return _OPS_SEND
    if opcode == _WR_SEND_WITH_IMM:
        return _OPS_SEND_WITH_IMM
    if opcode == _WR_RDMA_READ:
        return _OPS_RDMA_READ
    if opcode == _WR_RDMA_WRITE:
        return _OPS_RDMA_WRITE
    if opcode == _WR_RDMA_WRITE_WITH_IMM:
        return _OPS_RDMA_WRITE_WITH_IMM
    return 0


cdef int _post_send_ex(c.ibv_qp_ex *qpx, list wrs,
                       uint64_t allowed_ops) except -1:
    """Build and submit a batch through the extended WR API.

    Python-level structures are converted while the batch is open; any error
    aborts the whole batch (``ibv_wr_abort``) before the exception propagates.
    """
    cdef int n = len(wrs)
    cdef uint64_t need
    if n == 0:
        return 0
    cdef c.ibv_sge sges[64]
    cdef c.ibv_data_buf bufs[64]
    cdef SendWR w
    cdef SGE s
    cdef AH ah
    cdef int i
    cdef int nsge
    cdef int j
    cdef int rc
    c.ibv_wr_start(qpx)
    try:
        for i in range(n):
            w = <SendWR>wrs[i]
            if w.ah is None:
                raise ValueError(
                    "EFA sends are connectionless: each SendWR needs a "
                    "destination (dest=Peer or ah/remote_qpn/remote_qkey)")
            ah = <AH>w.ah
            ah._ensure()
            nsge = len(w.sg_list)
            if nsge > 64:
                raise ValueError("at most 64 SGEs per work request")
            need = _op_required_bit(w.opcode)
            if need == 0:
                raise ValueError(
                    "EFA supports SEND, SEND_WITH_IMM, RDMA_READ, RDMA_WRITE "
                    "and RDMA_WRITE_WITH_IMM (got opcode %d)" % w.opcode)
            if not (allowed_ops & need):
                # The provider only installs wr_* ops named in
                # send_ops_flags; dispatching an uninstalled op would call a
                # NULL pointer.
                raise ValueError(
                    "opcode %d is not enabled in this QP's send_ops_flags"
                    % w.opcode)
            qpx.wr_id = w.wr_id
            qpx.wr_flags = w.send_flags
            if w.opcode == _WR_SEND:
                c.ibv_wr_send(qpx)
            elif w.opcode == _WR_SEND_WITH_IMM:
                c.ibv_wr_send_imm(qpx, c.htonl(w.imm_data))
            elif w.opcode == _WR_RDMA_READ:
                c.ibv_wr_rdma_read(qpx, w.rkey, w.remote_addr)
            elif w.opcode == _WR_RDMA_WRITE:
                c.ibv_wr_rdma_write(qpx, w.rkey, w.remote_addr)
            elif w.opcode == _WR_RDMA_WRITE_WITH_IMM:
                c.ibv_wr_rdma_write_imm(qpx, w.rkey, w.remote_addr,
                                        c.htonl(w.imm_data))
            else:
                raise ValueError(
                    "EFA supports SEND, SEND_WITH_IMM, RDMA_READ, RDMA_WRITE "
                    "and RDMA_WRITE_WITH_IMM (got opcode %d)" % w.opcode)
            c.ibv_wr_set_ud_addr(qpx, ah._ah, w.remote_qpn, w.remote_qkey)
            if w.send_flags & _SEND_INLINE:
                for j in range(nsge):
                    s = <SGE>w.sg_list[j]
                    bufs[j].addr = <void*><uintptr_t>s.addr
                    bufs[j].length = s.length
                c.ibv_wr_set_inline_data_list(qpx, nsge, &bufs[0])
            else:
                for j in range(nsge):
                    s = <SGE>w.sg_list[j]
                    sges[j].addr = s.addr
                    sges[j].length = s.length
                    sges[j].lkey = s.lkey
                c.ibv_wr_set_sge_list(qpx, nsge, &sges[0])
    except BaseException:
        c.ibv_wr_abort(qpx)
        raise
    with nogil:
        rc = c.ibv_wr_complete(qpx)
    if rc != 0:
        raise EfaError("ibv_wr_complete", rc)
    return 0


cdef _post_recv_qp(c.ibv_qp *qp, list wrs):
    cdef int n = len(wrs)
    if n == 0:
        return
    cdef int total = 0
    for wr in wrs:
        total += len(wr.sg_list)
    cdef c.ibv_recv_wr *cwrs = <c.ibv_recv_wr*>calloc(n, sizeof(c.ibv_recv_wr))
    cdef c.ibv_sge *csges = <c.ibv_sge*>calloc(total if total else 1, sizeof(c.ibv_sge))
    if cwrs is NULL or csges is NULL:
        free(cwrs); free(csges)
        raise MemoryError()
    cdef c.ibv_recv_wr *bad = NULL
    cdef int rc
    cdef int i
    cdef int sge_off = 0
    cdef int nsge
    cdef int j
    cdef RecvWR w
    cdef SGE s
    try:
        for i in range(n):
            w = <RecvWR>wrs[i]
            nsge = len(w.sg_list)
            cwrs[i].wr_id = w.wr_id
            cwrs[i].num_sge = nsge
            if nsge:
                for j in range(nsge):
                    s = <SGE>w.sg_list[j]
                    csges[sge_off + j].addr = s.addr
                    csges[sge_off + j].length = s.length
                    csges[sge_off + j].lkey = s.lkey
                cwrs[i].sg_list = &csges[sge_off]
                sge_off += nsge
            cwrs[i].next = &cwrs[i + 1] if i + 1 < n else NULL
        with nogil:
            rc = c.ibv_post_recv(qp, &cwrs[0], &bad)
        if rc != 0:
            raise EfaError("ibv_post_recv", rc)
    finally:
        free(cwrs)
        free(csges)


# --------------------------------------------------------------------------- #
# Address handle
# --------------------------------------------------------------------------- #
cdef class AH:
    """An address handle naming a remote EFA device (by GID)."""

    cdef c.ibv_ah *_ah
    cdef readonly PD pd

    @staticmethod
    cdef AH _wrap(c.ibv_ah *ah, PD pd):
        cdef AH self = AH.__new__(AH)
        self._ah = ah
        self.pd = pd
        return self

    cdef int _ensure(self) except -1:
        if self._ah is NULL:
            raise EfaError("address handle is closed", EBADF)
        return 0

    @property
    def ahn(self) -> int:
        """The EFA address handle number (``efadv_query_ah``)."""
        self._ensure()
        cdef c.rdma4py_efadv_ah_attr a
        memset(&a, 0, sizeof(a))
        cdef int rc = _e_query_ah(self._ah, &a, sizeof(a))
        if rc != 0:
            _fail_rc("efadv_query_ah", rc)
        return a.ahn

    def close(self):
        cdef int rc
        if self._ah is not NULL:
            rc = _v_destroy_ah(self._ah)
            if rc != 0:
                _fail_rc("ibv_destroy_ah", rc)
            self._ah = NULL

    def __dealloc__(self):
        if self._ah is not NULL:
            _v_destroy_ah(self._ah)
            self._ah = NULL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


__all__ = [
    "EfaError", "CompletionError", "Gid", "DeviceAttr", "EfaDeviceAttr",
    "PortAttr", "WC", "CQAttr", "SGE", "SendWR", "RecvWR", "QPCap", "QPInitAttr",
    "AHAttr", "MRAttr", "WQAttr", "Device", "get_device_list",
    "get_efa_device_list", "Context", "AsyncEvent", "PD", "MR", "CompChannel",
    "CQ", "CQEx", "QP", "AH",
]
