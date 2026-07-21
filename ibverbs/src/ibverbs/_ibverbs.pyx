# cython: language_level=3, embedsignature=True, binding=True
"""Low-level Cython bindings for libibverbs.

Every RDMA resource is a RAII ``cdef class`` that owns its C handle and frees
it in ``__dealloc__``. Children keep a Python reference to their parent so the
garbage collector cannot free a parent before its children. Hot paths
(``post_send``/``post_recv``/``poll``/``get_cq_event``) release the GIL.
"""

import os

from libc.errno cimport EBADF, EBUSY, EIO, errno
from libc.stddef cimport size_t
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, uintptr_t
from libc.stdlib cimport calloc, free
from libc.string cimport memset, memcpy, strcmp

cimport ibverbs._libverbs as c
cimport ibverbs._librdmacm as cm


# --------------------------------------------------------------------------- #
# Runtime binding to libibverbs (dlopen/dlsym)
# --------------------------------------------------------------------------- #
# The exported (non-inline) verbs are resolved at import time from
# libibverbs.so.1 via dlopen instead of being hard-linked. This means the
# extension has no external NEEDED dependency (trivially manylinux-compliant),
# gives a clean ImportError when libibverbs is absent, and tolerates an older
# libibverbs by treating newer verbs (e.g. ibv_reg_dmabuf_mr) as optional. The
# data-path fast paths (post_send/poll_cq/...) remain compiled inline and
# dispatch through the provider op table, so they need no symbol resolution.

cdef extern from "dlfcn.h" nogil:
    void *dlopen(const char *filename, int flag)
    void *dlsym(void *handle, const char *symbol)
    char *dlerror()
    int RTLD_NOW
    int RTLD_GLOBAL
    int RTLD_LOCAL

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
ctypedef c.ibv_qp *(*fp_create_qp)(c.ibv_pd *, c.ibv_qp_init_attr *) noexcept nogil
ctypedef int (*fp_destroy_qp)(c.ibv_qp *) noexcept nogil
ctypedef int (*fp_modify_qp)(c.ibv_qp *, c.ibv_qp_attr *, int) noexcept nogil
ctypedef int (*fp_query_qp)(c.ibv_qp *, c.ibv_qp_attr *, int, c.ibv_qp_init_attr *) noexcept nogil
ctypedef c.ibv_ah *(*fp_create_ah)(c.ibv_pd *, c.ibv_ah_attr *) noexcept nogil
ctypedef int (*fp_destroy_ah)(c.ibv_ah *) noexcept nogil
ctypedef c.ibv_srq *(*fp_create_srq)(c.ibv_pd *, c.ibv_srq_init_attr *) noexcept nogil
ctypedef int (*fp_destroy_srq)(c.ibv_srq *) noexcept nogil
ctypedef int (*fp_modify_srq)(c.ibv_srq *, c.ibv_srq_attr *, int) noexcept nogil
ctypedef int (*fp_query_srq)(c.ibv_srq *, c.ibv_srq_attr *) noexcept nogil
ctypedef int (*fp_get_async_event)(c.ibv_context *, c.ibv_async_event *) noexcept nogil
ctypedef void (*fp_ack_async_event)(c.ibv_async_event *) noexcept nogil
ctypedef const char *(*fp_str_from_int)(int) noexcept nogil

cdef void *_libhandle = NULL
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
cdef fp_create_qp _v_create_qp = NULL
cdef fp_destroy_qp _v_destroy_qp = NULL
cdef fp_modify_qp _v_modify_qp = NULL
cdef fp_query_qp _v_query_qp = NULL
cdef fp_create_ah _v_create_ah = NULL
cdef fp_destroy_ah _v_destroy_ah = NULL
cdef fp_create_srq _v_create_srq = NULL
cdef fp_destroy_srq _v_destroy_srq = NULL
cdef fp_modify_srq _v_modify_srq = NULL
cdef fp_query_srq _v_query_srq = NULL
cdef fp_get_async_event _v_get_async_event = NULL
cdef fp_ack_async_event _v_ack_async_event = NULL
cdef fp_str_from_int _v_event_type_str = NULL
cdef fp_str_from_int _v_wc_status_str = NULL


# librdmacm is optional for base verbs users and loaded only when CM is used.
ctypedef int (*fp_cm_getaddrinfo)(const char *, const char *, cm.rdma_addrinfo *, cm.rdma_addrinfo **) noexcept nogil
ctypedef void (*fp_cm_freeaddrinfo)(cm.rdma_addrinfo *) noexcept nogil
ctypedef int (*fp_cm_create_ep)(cm.rdma_cm_id **, cm.rdma_addrinfo *, c.ibv_pd *, c.ibv_qp_init_attr *) noexcept nogil
ctypedef void (*fp_cm_destroy_ep)(cm.rdma_cm_id *) noexcept nogil
ctypedef int (*fp_cm_create_qp)(cm.rdma_cm_id *, c.ibv_pd *, c.ibv_qp_init_attr *) noexcept nogil
ctypedef void (*fp_cm_destroy_qp)(cm.rdma_cm_id *) noexcept nogil
ctypedef int (*fp_cm_connect)(cm.rdma_cm_id *, cm.rdma_conn_param *) noexcept nogil
ctypedef int (*fp_cm_disconnect)(cm.rdma_cm_id *) noexcept nogil

cdef void *_cm_libhandle = NULL
cdef fp_cm_getaddrinfo _cm_getaddrinfo = NULL
cdef fp_cm_freeaddrinfo _cm_freeaddrinfo = NULL
cdef fp_cm_create_ep _cm_create_ep = NULL
cdef fp_cm_destroy_ep _cm_destroy_ep = NULL
cdef fp_cm_create_qp _cm_create_qp = NULL
cdef fp_cm_destroy_qp _cm_destroy_qp = NULL
cdef fp_cm_connect _cm_connect = NULL
cdef fp_cm_disconnect _cm_disconnect = NULL


cdef void *_req_sym(void *h, const char *name) except NULL:
    cdef void *sym = dlsym(h, name)
    if sym is NULL:
        raise ImportError("libibverbs is missing required symbol %s"
                          % (<bytes>name).decode())
    return sym


cdef int _load_libibverbs() except -1:
    global _libhandle
    global _v_get_device_list, _v_free_device_list, _v_get_device_name
    global _v_get_device_guid, _v_open_device, _v_close_device, _v_query_device
    global _v_query_port, _v_query_gid, _v_alloc_pd, _v_dealloc_pd
    global _v_reg_mr_iova2, _v_reg_dmabuf_mr, _v_dereg_mr
    global _v_create_comp_channel, _v_destroy_comp_channel, _v_create_cq
    global _v_destroy_cq, _v_get_cq_event, _v_ack_cq_events, _v_create_qp
    global _v_destroy_qp, _v_modify_qp, _v_query_qp, _v_create_ah, _v_destroy_ah
    global _v_create_srq, _v_destroy_srq, _v_modify_srq, _v_query_srq
    global _v_get_async_event, _v_ack_async_event, _v_event_type_str
    global _v_wc_status_str
    cdef void *h
    cdef char *e
    h = dlopen(b"libibverbs.so.1", RTLD_NOW | RTLD_GLOBAL)
    if h is NULL:
        e = dlerror()
        raise ImportError(
            "could not load libibverbs.so.1 - install rdma-core / libibverbs "
            "(%s)" % ((<bytes>e).decode() if e is not NULL else "not found"))
    _libhandle = h
    _v_get_device_list = <fp_get_device_list>_req_sym(h, b"ibv_get_device_list")
    _v_free_device_list = <fp_free_device_list>_req_sym(h, b"ibv_free_device_list")
    _v_get_device_name = <fp_get_device_name>_req_sym(h, b"ibv_get_device_name")
    _v_get_device_guid = <fp_get_device_guid>_req_sym(h, b"ibv_get_device_guid")
    _v_open_device = <fp_open_device>_req_sym(h, b"ibv_open_device")
    _v_close_device = <fp_close_device>_req_sym(h, b"ibv_close_device")
    _v_query_device = <fp_query_device>_req_sym(h, b"ibv_query_device")
    _v_query_port = <fp_query_port>_req_sym(h, b"ibv_query_port")
    _v_query_gid = <fp_query_gid>_req_sym(h, b"ibv_query_gid")
    _v_alloc_pd = <fp_alloc_pd>_req_sym(h, b"ibv_alloc_pd")
    _v_dealloc_pd = <fp_dealloc_pd>_req_sym(h, b"ibv_dealloc_pd")
    _v_reg_mr_iova2 = <fp_reg_mr_iova2>_req_sym(h, b"ibv_reg_mr_iova2")
    _v_dereg_mr = <fp_dereg_mr>_req_sym(h, b"ibv_dereg_mr")
    _v_create_comp_channel = <fp_create_comp_channel>_req_sym(h, b"ibv_create_comp_channel")
    _v_destroy_comp_channel = <fp_destroy_comp_channel>_req_sym(h, b"ibv_destroy_comp_channel")
    _v_create_cq = <fp_create_cq>_req_sym(h, b"ibv_create_cq")
    _v_destroy_cq = <fp_destroy_cq>_req_sym(h, b"ibv_destroy_cq")
    _v_get_cq_event = <fp_get_cq_event>_req_sym(h, b"ibv_get_cq_event")
    _v_ack_cq_events = <fp_ack_cq_events>_req_sym(h, b"ibv_ack_cq_events")
    _v_create_qp = <fp_create_qp>_req_sym(h, b"ibv_create_qp")
    _v_destroy_qp = <fp_destroy_qp>_req_sym(h, b"ibv_destroy_qp")
    _v_modify_qp = <fp_modify_qp>_req_sym(h, b"ibv_modify_qp")
    _v_query_qp = <fp_query_qp>_req_sym(h, b"ibv_query_qp")
    _v_create_ah = <fp_create_ah>_req_sym(h, b"ibv_create_ah")
    _v_destroy_ah = <fp_destroy_ah>_req_sym(h, b"ibv_destroy_ah")
    _v_create_srq = <fp_create_srq>_req_sym(h, b"ibv_create_srq")
    _v_destroy_srq = <fp_destroy_srq>_req_sym(h, b"ibv_destroy_srq")
    _v_modify_srq = <fp_modify_srq>_req_sym(h, b"ibv_modify_srq")
    _v_query_srq = <fp_query_srq>_req_sym(h, b"ibv_query_srq")
    _v_get_async_event = <fp_get_async_event>_req_sym(h, b"ibv_get_async_event")
    _v_ack_async_event = <fp_ack_async_event>_req_sym(h, b"ibv_ack_async_event")
    _v_event_type_str = <fp_str_from_int>_req_sym(h, b"ibv_event_type_str")
    _v_wc_status_str = <fp_str_from_int>_req_sym(h, b"ibv_wc_status_str")
    # Optional: added in rdma-core 34 (IBVERBS_1.12). Absent on older systems.
    _v_reg_dmabuf_mr = <fp_reg_dmabuf_mr>dlsym(h, b"ibv_reg_dmabuf_mr")
    return 0


cdef int _load_librdmacm() except -1:
    global _cm_libhandle, _cm_getaddrinfo, _cm_freeaddrinfo
    global _cm_create_ep, _cm_destroy_ep, _cm_create_qp, _cm_destroy_qp
    global _cm_connect, _cm_disconnect
    if _cm_libhandle is not NULL:
        return 0
    cdef void *h = dlopen(b"librdmacm.so.1", RTLD_NOW | RTLD_LOCAL)
    cdef char *e
    if h is NULL:
        e = dlerror()
        raise RuntimeError(
            "could not load librdmacm.so.1 - install rdma-core / librdmacm "
            "(%s)" % ((<bytes>e).decode() if e is not NULL else "not found"))
    _cm_getaddrinfo = <fp_cm_getaddrinfo>_req_sym(h, b"rdma_getaddrinfo")
    _cm_freeaddrinfo = <fp_cm_freeaddrinfo>_req_sym(h, b"rdma_freeaddrinfo")
    _cm_create_ep = <fp_cm_create_ep>_req_sym(h, b"rdma_create_ep")
    _cm_destroy_ep = <fp_cm_destroy_ep>_req_sym(h, b"rdma_destroy_ep")
    _cm_create_qp = <fp_cm_create_qp>_req_sym(h, b"rdma_create_qp")
    _cm_destroy_qp = <fp_cm_destroy_qp>_req_sym(h, b"rdma_destroy_qp")
    _cm_connect = <fp_cm_connect>_req_sym(h, b"rdma_connect")
    _cm_disconnect = <fp_cm_disconnect>_req_sym(h, b"rdma_disconnect")
    _cm_libhandle = h
    return 0


_load_libibverbs()


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class VerbsError(OSError):
    """Raised when a libibverbs call fails.

    Subclasses :class:`OSError`, so ``.errno`` and ``.strerror`` are populated.
    ``.operation`` names the verb that failed.
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
    raise VerbsError(op, errno if errno != 0 else EIO)


cdef int _fail_rc(str op, int rc) except -1:
    raise VerbsError(op, _error_from_rc(rc))


# --------------------------------------------------------------------------- #
# Linkage smoke test (kept for the import test)
# --------------------------------------------------------------------------- #
def _linked() -> bool:
    """Return True if libibverbs was loaded at import (always True if imported)."""
    return _libhandle is not NULL


def _has_dmabuf() -> bool:
    """Return True if this libibverbs provides ``ibv_reg_dmabuf_mr`` (rdma-core >= 34)."""
    return _v_reg_dmabuf_mr is not NULL


def _has_rdmacm() -> bool:
    """Return whether librdmacm can be loaded on this host."""
    try:
        _load_librdmacm()
    except RuntimeError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Result value objects
# --------------------------------------------------------------------------- #
cdef class Gid:
    """A 16-byte GID (RoCE/InfiniBand address)."""

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
    """Device capabilities from :meth:`Context.query_device`."""

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
    cdef readonly int max_qp_rd_atom
    cdef readonly int max_ah
    cdef readonly int max_srq
    cdef readonly int max_srq_wr
    cdef readonly int max_srq_sge
    cdef readonly int max_pkeys
    cdef readonly int phys_port_cnt

    def __repr__(self):
        return "DeviceAttr(fw_ver=%r, max_qp=%d, max_cqe=%d, phys_port_cnt=%d)" % (
            self.fw_ver, self.max_qp, self.max_cqe, self.phys_port_cnt)


cdef class PortAttr:
    """Port attributes from :meth:`Context.query_port`."""

    cdef readonly int state
    cdef readonly int max_mtu
    cdef readonly int active_mtu
    cdef readonly int gid_tbl_len
    cdef readonly uint32_t port_cap_flags
    cdef readonly uint32_t max_msg_sz
    cdef readonly int pkey_tbl_len
    cdef readonly int lid
    cdef readonly int sm_lid
    cdef readonly int lmc
    cdef readonly int active_width
    cdef readonly int active_speed
    cdef readonly int link_layer

    def __repr__(self):
        return "PortAttr(state=%d, active_mtu=%d, lid=%d, link_layer=%d)" % (
            self.state, self.active_mtu, self.lid, self.link_layer)


cdef class WC:
    """A work completion returned by :meth:`CQ.poll`."""

    cdef readonly uint64_t wr_id
    cdef readonly int status
    cdef readonly int opcode
    cdef readonly uint32_t vendor_err
    cdef readonly uint32_t byte_len
    cdef readonly uint32_t imm_data
    cdef readonly uint32_t qp_num
    cdef readonly uint32_t src_qp
    cdef readonly unsigned int wc_flags

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


class CompletionError(VerbsError):
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
    """A send work request."""

    cdef public uint64_t wr_id
    cdef public list sg_list
    cdef public int opcode
    cdef public unsigned int send_flags
    cdef public uint64_t remote_addr
    cdef public uint32_t rkey
    cdef public uint32_t imm_data
    cdef public uint64_t compare_add
    cdef public uint64_t swap
    cdef public object ah
    cdef public uint32_t remote_qpn
    cdef public uint32_t remote_qkey

    def __init__(self, wr_id=0, sg_list=None, opcode=0, send_flags=0,
                 remote_addr=0, rkey=0, imm_data=0, compare_add=0, swap=0,
                 ah=None, remote_qpn=0, remote_qkey=0):
        self.wr_id = wr_id
        self.sg_list = list(sg_list) if sg_list is not None else []
        self.opcode = opcode
        self.send_flags = send_flags
        self.remote_addr = remote_addr
        self.rkey = rkey
        self.imm_data = imm_data
        self.compare_add = compare_add
        self.swap = swap
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
    """Parameters for :meth:`PD.create_qp`."""

    def __init__(self, send_cq, recv_cq, qp_type=2, max_send_wr=128,
                 max_recv_wr=128, max_send_sge=1, max_recv_sge=1,
                 max_inline_data=0, srq=None, sq_sig_all=False):
        self.send_cq = send_cq
        self.recv_cq = recv_cq
        self.qp_type = int(qp_type)
        self.max_send_wr = max_send_wr
        self.max_recv_wr = max_recv_wr
        self.max_send_sge = max_send_sge
        self.max_recv_sge = max_recv_sge
        self.max_inline_data = max_inline_data
        self.srq = srq
        self.sq_sig_all = bool(sq_sig_all)


cdef class AHAttr:
    """Address-handle attributes (also used to reach RTR for RC/UD)."""

    cdef public bytes dgid
    cdef public uint32_t flow_label
    cdef public uint8_t sgid_index
    cdef public uint8_t hop_limit
    cdef public uint8_t traffic_class
    cdef public uint16_t dlid
    cdef public uint8_t sl
    cdef public uint8_t src_path_bits
    cdef public uint8_t static_rate
    cdef public uint8_t is_global
    cdef public uint8_t port_num

    def __init__(self, dgid=None, sgid_index=0, dlid=0, is_global=1,
                 port_num=1, hop_limit=1, traffic_class=0, sl=0, flow_label=0,
                 src_path_bits=0, static_rate=0):
        self.dgid = bytes(dgid) if dgid is not None else b"\x00" * 16
        if len(self.dgid) != 16:
            raise ValueError("dgid must be 16 bytes")
        self.sgid_index = sgid_index
        self.dlid = dlid
        self.is_global = is_global
        self.port_num = port_num
        self.hop_limit = hop_limit
        self.traffic_class = traffic_class
        self.sl = sl
        self.flow_label = flow_label
        self.src_path_bits = src_path_bits
        self.static_rate = static_rate


cdef void _fill_ah_attr(c.ibv_ah_attr *dst, AHAttr src):
    memset(dst, 0, sizeof(c.ibv_ah_attr))
    dst.is_global = src.is_global
    dst.dlid = src.dlid
    dst.sl = src.sl
    dst.src_path_bits = src.src_path_bits
    dst.static_rate = src.static_rate
    dst.port_num = src.port_num
    if src.is_global:
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
            raise VerbsError("ibv_open_device(%s)" % self.name, errno)
        return Context._wrap(ctx, self.name)

    def __repr__(self):
        return "Device(name=%r, guid=0x%x)" % (self.name, self.guid)


def get_device_list() -> list:
    """Return the list of :class:`Device` objects present on this host."""
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


cdef class Context:
    """An open device context (``ibv_context``)."""

    cdef c.ibv_context *_ctx
    cdef readonly str name
    cdef unsigned int _children
    cdef bint _owned

    def __cinit__(self):
        self._ctx = NULL
        self._children = 0
        self._owned = True

    @staticmethod
    cdef Context _wrap(c.ibv_context *ctx, str name):
        cdef Context self = Context.__new__(Context)
        self._ctx = ctx
        self.name = name
        self._owned = True
        return self

    @staticmethod
    cdef Context _wrap_borrowed(c.ibv_context *ctx, str name):
        cdef Context self = Context.__new__(Context)
        self._ctx = ctx
        self.name = name
        self._owned = False
        return self

    cdef int _ensure(self) except -1:
        if self._ctx is NULL:
            raise VerbsError("context is closed", EBADF)
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
        r.max_qp_rd_atom = a.max_qp_rd_atom
        r.max_ah = a.max_ah
        r.max_srq = a.max_srq
        r.max_srq_wr = a.max_srq_wr
        r.max_srq_sge = a.max_srq_sge
        r.max_pkeys = a.max_pkeys
        r.phys_port_cnt = a.phys_port_cnt
        return r

    def query_port(self, int port_num) -> PortAttr:
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
        r.sm_lid = a.sm_lid
        r.lmc = a.lmc
        r.active_width = a.active_width
        r.active_speed = a.active_speed
        r.link_layer = a.link_layer
        return r

    def query_gid(self, int port_num, int index) -> Gid:
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
            if not self._owned:
                raise VerbsError(
                    "ibv_close_device", EBUSY,
                    "context is owned by an RDMA CM endpoint",
                )
            if self._children != 0:
                raise VerbsError(
                    "ibv_close_device", EBUSY,
                    "cannot close context with %d open child resource(s)"
                    % self._children,
                )
            if _v_close_device(self._ctx) != 0:
                _fail("ibv_close_device")
            self._ctx = NULL

    def __dealloc__(self):
        if self._owned and self._ctx is not NULL and self._children == 0:
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
            raise VerbsError("pd is closed", EBADF)
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
        self._ensure()
        cdef c.ibv_qp_init_attr a
        memset(&a, 0, sizeof(a))
        cdef CQ scq = <CQ>init_attr.send_cq
        cdef CQ rcq = <CQ>init_attr.recv_cq
        cdef SRQ srq = None
        scq._ensure()
        rcq._ensure()
        if scq.context is not self.context or rcq.context is not self.context:
            raise ValueError("queue-pair CQs must belong to the PD's context")
        a.send_cq = scq._cq
        a.recv_cq = rcq._cq
        if init_attr.srq is not None:
            srq = <SRQ>init_attr.srq
            srq._ensure()
            if srq.pd is not self:
                raise ValueError("queue-pair SRQ must belong to this PD")
            a.srq = srq._srq
        a.qp_type = init_attr.qp_type
        a.sq_sig_all = 1 if init_attr.sq_sig_all else 0
        a.cap.max_send_wr = init_attr.max_send_wr
        a.cap.max_recv_wr = init_attr.max_recv_wr
        a.cap.max_send_sge = init_attr.max_send_sge
        a.cap.max_recv_sge = init_attr.max_recv_sge
        a.cap.max_inline_data = init_attr.max_inline_data
        cdef c.ibv_qp *qp = _v_create_qp(self._pd, &a)
        if qp is NULL:
            _fail("ibv_create_qp")
        return QP._wrap(qp, self, scq, rcq, srq)

    def create_ah(self, AHAttr attr) -> "AH":
        self._ensure()
        cdef c.ibv_ah_attr a
        _fill_ah_attr(&a, attr)
        cdef c.ibv_ah *ah = _v_create_ah(self._pd, &a)
        if ah is NULL:
            _fail("ibv_create_ah")
        return AH._wrap(ah, self)

    def create_srq(self, max_wr=128, max_sge=1, srq_limit=0) -> "SRQ":
        self._ensure()
        cdef c.ibv_srq_init_attr a
        memset(&a, 0, sizeof(a))
        a.attr.max_wr = max_wr
        a.attr.max_sge = max_sge
        a.attr.srq_limit = srq_limit
        cdef c.ibv_srq *srq = _v_create_srq(self._pd, &a)
        if srq is NULL:
            _fail("ibv_create_srq")
        return SRQ._wrap(srq, self)

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
            raise VerbsError("memory region is closed", EBADF)
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
# Completion channel / queue
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
            raise VerbsError("completion channel is closed", EBADF)
        return 0

    def get_cq_event(self) -> "CQ":
        """Block until a CQ event arrives; return the associated :class:`CQ`.

        The event must later be acknowledged with :meth:`CQ.ack_events`.
        """
        self._ensure()
        cdef c.ibv_cq *cq = NULL
        cdef void *ctx = NULL
        cdef int rc
        with nogil:
            rc = _v_get_cq_event(self._chan, &cq, &ctx)
        if rc != 0:
            _fail("ibv_get_cq_event")
        cdef CQ obj = <CQ>ctx
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


cdef class CQ:
    """A completion queue (``ibv_cq``)."""

    cdef c.ibv_cq *_cq
    cdef readonly Context context
    cdef readonly CompChannel channel
    cdef int _unacked

    cdef int _ensure(self) except -1:
        if self._cq is NULL:
            raise VerbsError("completion queue is closed", EBADF)
        return 0

    @property
    def cqe(self) -> int:
        self._ensure()
        return self._cq.cqe

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
                raise VerbsError("ibv_poll_cq", -n if n < -1 else (
                    errno if errno != 0 else EIO))
            out = []
            for i in range(n):
                out.append(_wc_from_c(&wcs[i]))
            return out
        finally:
            free(wcs)

    def req_notify(self, bint solicited_only=False):
        """Request a completion notification on the CQ's channel."""
        self._ensure()
        cdef int rc = c.ibv_req_notify_cq(
            self._cq, 1 if solicited_only else 0)
        if rc != 0:
            _fail_rc("ibv_req_notify_cq", rc)

    def ack_events(self, unsigned int nevents=1):
        """Acknowledge ``nevents`` events delivered via the channel."""
        self._ensure()
        if nevents == 0:
            raise ValueError("nevents must be positive")
        if <int>nevents > self._unacked:
            raise ValueError("cannot acknowledge more CQ events than were received")
        _v_ack_cq_events(self._cq, nevents)
        self._unacked -= <int>nevents

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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


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
    _QP_ACCESS_FLAGS = 1 << 3
    _QP_PKEY_INDEX = 1 << 4
    _QP_PORT = 1 << 5
    _QP_QKEY = 1 << 6
    _QP_AV = 1 << 7
    _QP_PATH_MTU = 1 << 8
    _QP_TIMEOUT = 1 << 9
    _QP_RETRY_CNT = 1 << 10
    _QP_RNR_RETRY = 1 << 11
    _QP_RQ_PSN = 1 << 12
    _QP_MAX_QP_RD_ATOMIC = 1 << 13
    _QP_MIN_RNR_TIMER = 1 << 15
    _QP_SQ_PSN = 1 << 16
    _QP_MAX_DEST_RD_ATOMIC = 1 << 17
    _QP_DEST_QPN = 1 << 20

# ibv_qp_type
cdef enum:
    _QPT_RC = 2
    _QPT_UC = 3
    _QPT_UD = 4

# ibv_qp_state
cdef enum:
    _QPS_INIT = 1
    _QPS_RTR = 2
    _QPS_RTS = 3

# ibv_wr_opcode (subset needed to pick the send-WR union member)
cdef enum:
    _WR_RDMA_WRITE = 0
    _WR_RDMA_WRITE_WITH_IMM = 1
    _WR_RDMA_READ = 4
    _WR_ATOMIC_CMP_AND_SWP = 5
    _WR_ATOMIC_FETCH_AND_ADD = 6


cdef class QP:
    """A queue pair (``ibv_qp``)."""

    cdef c.ibv_qp *_qp
    cdef readonly PD pd
    cdef readonly CQ send_cq
    cdef readonly CQ recv_cq
    cdef readonly object srq
    cdef int _port
    cdef cm.rdma_cm_id *_cm_id
    cdef object _cm_owner

    cdef int _ensure(self) except -1:
        if self._qp is NULL:
            raise VerbsError("queue pair is closed", EBADF)
        return 0

    @staticmethod
    cdef QP _wrap(c.ibv_qp *qp, PD pd, CQ scq, CQ rcq, SRQ srq):
        cdef QP self = QP.__new__(QP)
        self._qp = qp
        self.pd = pd
        self.send_cq = scq
        self.recv_cq = rcq
        self.srq = srq
        self._port = 1
        self._cm_id = NULL
        self._cm_owner = None
        return self

    @property
    def qp_num(self) -> int:
        self._ensure()
        return self._qp.qp_num

    @property
    def qp_type(self) -> int:
        self._ensure()
        return self._qp.qp_type

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
        cdef int rc = _v_query_qp(self._qp, &a, 0x1FFFFFF, &ia)
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
            "path_mtu": a.path_mtu,
            "qkey": a.qkey,
            "rq_psn": a.rq_psn,
            "sq_psn": a.sq_psn,
            "dest_qp_num": a.dest_qp_num,
            "qp_access_flags": a.qp_access_flags,
            "port_num": a.port_num,
            "timeout": a.timeout,
            "retry_cnt": a.retry_cnt,
            "rnr_retry": a.rnr_retry,
            "min_rnr_timer": a.min_rnr_timer,
            "max_rd_atomic": a.max_rd_atomic,
            "max_dest_rd_atomic": a.max_dest_rd_atomic,
        }
        return attrs, cap

    def modify(self, int attr_mask, ah_attr=None, **fields):
        """Low-level ``ibv_modify_qp``.

        ``attr_mask`` is an ORed :class:`~ibverbs.enums.QPAttrMask`. Scalar
        attributes are passed as keywords (e.g. ``qp_state=``, ``port_num=``);
        the address vector is passed as an :class:`AHAttr` via ``ah_attr=``.
        """
        self._ensure()
        unknown = set(fields) - {
            "qp_state", "cur_qp_state", "path_mtu", "qkey", "rq_psn",
            "sq_psn", "dest_qp_num", "qp_access_flags", "pkey_index",
            "port_num", "timeout", "retry_cnt", "rnr_retry",
            "min_rnr_timer", "max_rd_atomic", "max_dest_rd_atomic",
        }
        if unknown:
            raise TypeError("unknown QP attribute(s): %s" %
                            ", ".join(sorted(unknown)))
        cdef c.ibv_qp_attr a
        memset(&a, 0, sizeof(a))
        if "qp_state" in fields: a.qp_state = fields["qp_state"]
        if "cur_qp_state" in fields: a.cur_qp_state = fields["cur_qp_state"]
        if "path_mtu" in fields: a.path_mtu = fields["path_mtu"]
        if "qkey" in fields: a.qkey = fields["qkey"]
        if "rq_psn" in fields: a.rq_psn = fields["rq_psn"]
        if "sq_psn" in fields: a.sq_psn = fields["sq_psn"]
        if "dest_qp_num" in fields: a.dest_qp_num = fields["dest_qp_num"]
        if "qp_access_flags" in fields: a.qp_access_flags = fields["qp_access_flags"]
        if "pkey_index" in fields: a.pkey_index = fields["pkey_index"]
        if "port_num" in fields: a.port_num = fields["port_num"]
        if "timeout" in fields: a.timeout = fields["timeout"]
        if "retry_cnt" in fields: a.retry_cnt = fields["retry_cnt"]
        if "rnr_retry" in fields: a.rnr_retry = fields["rnr_retry"]
        if "min_rnr_timer" in fields: a.min_rnr_timer = fields["min_rnr_timer"]
        if "max_rd_atomic" in fields: a.max_rd_atomic = fields["max_rd_atomic"]
        if "max_dest_rd_atomic" in fields: a.max_dest_rd_atomic = fields["max_dest_rd_atomic"]
        if ah_attr is not None:
            _fill_ah_attr(&a.ah_attr, <AHAttr>ah_attr)
        cdef int rc = _v_modify_qp(self._qp, &a, attr_mask)
        if rc != 0:
            _fail_rc("ibv_modify_qp", rc)

    # -- RC/UD state-machine helpers -----------------------------------------
    def to_init(self, int port, int access=0, int pkey_index=0, int qkey=0):
        """Transition RESET -> INIT."""
        self._ensure()
        cdef c.ibv_qp_attr a
        cdef int mask
        memset(&a, 0, sizeof(a))
        a.qp_state = _QPS_INIT
        a.pkey_index = pkey_index
        a.port_num = <uint8_t>port
        self._port = port
        if self._qp.qp_type == _QPT_UD:
            a.qkey = qkey
            mask = _QP_STATE | _QP_PKEY_INDEX | _QP_PORT | _QP_QKEY
        else:
            a.qp_access_flags = access
            mask = _QP_STATE | _QP_PKEY_INDEX | _QP_PORT | _QP_ACCESS_FLAGS
        cdef int rc = _v_modify_qp(self._qp, &a, mask)
        if rc != 0:
            _fail_rc("ibv_modify_qp(->INIT)", rc)

    def to_rtr(self, remote, int sgid_index, mtu=None, int hop_limit=1,
               int min_rnr_timer=12, int max_dest_rd_atomic=1, int sl=0,
               int traffic_class=0):
        """Transition INIT -> RTR using a remote :class:`~ibverbs.helpers.QPInfo`."""
        self._ensure()
        cdef c.ibv_qp_attr a
        cdef int mask
        cdef int rc
        cdef bytes dgid
        memset(&a, 0, sizeof(a))
        a.qp_state = _QPS_RTR
        if self._qp.qp_type == _QPT_UD:
            rc = _v_modify_qp(self._qp, &a, _QP_STATE)
            if rc != 0:
                _fail_rc("ibv_modify_qp(->RTR)", rc)
            return
        a.path_mtu = int(mtu) if mtu is not None else int(remote.mtu)
        a.dest_qp_num = remote.qp_num
        a.rq_psn = remote.psn
        a.max_dest_rd_atomic = max_dest_rd_atomic
        a.min_rnr_timer = min_rnr_timer
        dgid = bytes(remote.gid)
        if len(dgid) != 16:
            raise ValueError("remote gid must be exactly 16 bytes")
        a.ah_attr.is_global = 1
        memcpy(&a.ah_attr.grh.dgid.raw[0], <char*>dgid, 16)
        a.ah_attr.grh.sgid_index = <uint8_t>sgid_index
        a.ah_attr.grh.hop_limit = <uint8_t>hop_limit
        a.ah_attr.grh.traffic_class = <uint8_t>traffic_class
        a.ah_attr.dlid = <uint16_t>remote.lid
        a.ah_attr.sl = <uint8_t>sl
        a.ah_attr.src_path_bits = 0
        a.ah_attr.port_num = <uint8_t>self._port
        mask = (_QP_STATE | _QP_AV | _QP_PATH_MTU | _QP_DEST_QPN | _QP_RQ_PSN
                | _QP_MAX_DEST_RD_ATOMIC | _QP_MIN_RNR_TIMER)
        rc = _v_modify_qp(self._qp, &a, mask)
        if rc != 0:
            _fail_rc("ibv_modify_qp(->RTR)", rc)

    def to_rts(self, int psn, int timeout=14, int retry_cnt=7, int rnr_retry=7,
               int max_rd_atomic=1):
        """Transition RTR -> RTS."""
        self._ensure()
        cdef c.ibv_qp_attr a
        cdef int mask
        memset(&a, 0, sizeof(a))
        a.qp_state = _QPS_RTS
        a.sq_psn = psn
        if self._qp.qp_type == _QPT_UD:
            mask = _QP_STATE | _QP_SQ_PSN
        else:
            a.timeout = <uint8_t>timeout
            a.retry_cnt = <uint8_t>retry_cnt
            a.rnr_retry = <uint8_t>rnr_retry
            a.max_rd_atomic = <uint8_t>max_rd_atomic
            mask = (_QP_STATE | _QP_TIMEOUT | _QP_RETRY_CNT | _QP_RNR_RETRY
                    | _QP_SQ_PSN | _QP_MAX_QP_RD_ATOMIC)
        cdef int rc = _v_modify_qp(self._qp, &a, mask)
        if rc != 0:
            _fail_rc("ibv_modify_qp(->RTS)", rc)

    def post_send(self, wrs):
        """Post one or more :class:`SendWR` to the send queue."""
        self._ensure()
        if isinstance(wrs, SendWR):
            wrs = [wrs]
        elif not isinstance(wrs, list):
            wrs = list(wrs)
        _post_send(self._qp, wrs)

    def post_recv(self, wrs):
        """Post one or more :class:`RecvWR` to the receive queue."""
        self._ensure()
        if isinstance(wrs, RecvWR):
            wrs = [wrs]
        elif not isinstance(wrs, list):
            wrs = list(wrs)
        _post_recv_qp(self._qp, wrs)

    def close(self):
        cdef int rc
        if self._qp is not NULL:
            if self._cm_id is not NULL:
                _cm_destroy_qp(self._cm_id)
                self._cm_id = NULL
                self._cm_owner = None
            else:
                rc = _v_destroy_qp(self._qp)
                if rc != 0:
                    _fail_rc("ibv_destroy_qp", rc)
            self._qp = NULL

    def __dealloc__(self):
        if self._qp is not NULL:
            if self._cm_id is not NULL:
                _cm_destroy_qp(self._cm_id)
                self._cm_id = NULL
            else:
                _v_destroy_qp(self._qp)
            self._qp = NULL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        if self._qp is NULL:
            return "QP(closed=True)"
        return "QP(qp_num=%d, qp_type=%d)" % (self._qp.qp_num, self._qp.qp_type)


# --------------------------------------------------------------------------- #
# Work-request posting (shared by QP and SRQ)
# --------------------------------------------------------------------------- #
cdef int _count_sges(list wrs) except -1:
    cdef int total = 0
    for wr in wrs:
        total += len(wr.sg_list)
    return total


cdef void _fill_sges(c.ibv_sge *dst, list sg_list):
    cdef int i = 0
    cdef SGE s
    for item in sg_list:
        s = <SGE>item
        dst[i].addr = s.addr
        dst[i].length = s.length
        dst[i].lkey = s.lkey
        i += 1


cdef _post_send(c.ibv_qp *qp, list wrs):
    cdef int n = len(wrs)
    if n == 0:
        return
    cdef int total = _count_sges(wrs)
    cdef c.ibv_send_wr *cwrs = <c.ibv_send_wr*>calloc(n, sizeof(c.ibv_send_wr))
    cdef c.ibv_sge *csges = <c.ibv_sge*>calloc(total if total else 1, sizeof(c.ibv_sge))
    if cwrs is NULL or csges is NULL:
        free(cwrs); free(csges)
        raise MemoryError()
    cdef c.ibv_send_wr *bad = NULL
    cdef int rc
    cdef int i
    cdef int sge_off = 0
    cdef int nsge
    cdef SendWR w
    cdef AH ah
    try:
        for i in range(n):
            w = <SendWR>wrs[i]
            nsge = len(w.sg_list)
            cwrs[i].wr_id = w.wr_id
            cwrs[i].opcode = w.opcode
            cwrs[i].send_flags = w.send_flags
            cwrs[i].num_sge = nsge
            if nsge:
                _fill_sges(&csges[sge_off], w.sg_list)
                cwrs[i].sg_list = &csges[sge_off]
                sge_off += nsge
            # imm_data is sent in network byte order.
            cwrs[i].imm_data = c.htonl(w.imm_data)
            # ``wr`` is a C union: populate ONLY the member for this opcode,
            # otherwise a later member's write aliases (clobbers) an earlier
            # member (e.g. atomic.compare_add over rdma.rkey).
            if w.ah is not None:
                ah = <AH>w.ah
                cwrs[i].wr.ud.ah = ah._ah
                cwrs[i].wr.ud.remote_qpn = w.remote_qpn
                cwrs[i].wr.ud.remote_qkey = w.remote_qkey
            elif w.opcode == _WR_ATOMIC_CMP_AND_SWP or w.opcode == _WR_ATOMIC_FETCH_AND_ADD:
                cwrs[i].wr.atomic.remote_addr = w.remote_addr
                cwrs[i].wr.atomic.compare_add = w.compare_add
                cwrs[i].wr.atomic.swap = w.swap
                cwrs[i].wr.atomic.rkey = w.rkey
            elif (w.opcode == _WR_RDMA_WRITE or w.opcode == _WR_RDMA_WRITE_WITH_IMM
                  or w.opcode == _WR_RDMA_READ):
                cwrs[i].wr.rdma.remote_addr = w.remote_addr
                cwrs[i].wr.rdma.rkey = w.rkey
            cwrs[i].next = &cwrs[i + 1] if i + 1 < n else NULL
        with nogil:
            rc = c.ibv_post_send(qp, &cwrs[0], &bad)
        if rc != 0:
            raise VerbsError("ibv_post_send", rc)
    finally:
        free(cwrs)
        free(csges)


cdef _post_recv_chain(c.ibv_qp *qp, c.ibv_srq *srq, list wrs):
    cdef int n = len(wrs)
    if n == 0:
        return
    cdef int total = _count_sges(wrs)
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
    cdef RecvWR w
    try:
        for i in range(n):
            w = <RecvWR>wrs[i]
            nsge = len(w.sg_list)
            cwrs[i].wr_id = w.wr_id
            cwrs[i].num_sge = nsge
            if nsge:
                _fill_sges(&csges[sge_off], w.sg_list)
                cwrs[i].sg_list = &csges[sge_off]
                sge_off += nsge
            cwrs[i].next = &cwrs[i + 1] if i + 1 < n else NULL
        with nogil:
            if srq is not NULL:
                rc = c.ibv_post_srq_recv(srq, &cwrs[0], &bad)
            else:
                rc = c.ibv_post_recv(qp, &cwrs[0], &bad)
        if rc != 0:
            raise VerbsError("ibv_post_recv", rc)
    finally:
        free(cwrs)
        free(csges)


cdef _post_recv_qp(c.ibv_qp *qp, list wrs):
    _post_recv_chain(qp, NULL, wrs)


# --------------------------------------------------------------------------- #
# Address handle / shared receive queue
# --------------------------------------------------------------------------- #
cdef class AH:
    """An address handle (``ibv_ah``) for UD sends."""

    cdef c.ibv_ah *_ah
    cdef readonly PD pd

    @staticmethod
    cdef AH _wrap(c.ibv_ah *ah, PD pd):
        cdef AH self = AH.__new__(AH)
        self._ah = ah
        self.pd = pd
        return self

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


cdef class SRQ:
    """A shared receive queue (``ibv_srq``)."""

    cdef c.ibv_srq *_srq
    cdef readonly PD pd

    cdef int _ensure(self) except -1:
        if self._srq is NULL:
            raise VerbsError("shared receive queue is closed", EBADF)
        return 0

    @staticmethod
    cdef SRQ _wrap(c.ibv_srq *srq, PD pd):
        cdef SRQ self = SRQ.__new__(SRQ)
        self._srq = srq
        self.pd = pd
        return self

    def post_recv(self, wrs):
        self._ensure()
        if isinstance(wrs, RecvWR):
            wrs = [wrs]
        elif not isinstance(wrs, list):
            wrs = list(wrs)
        _post_recv_chain(NULL, self._srq, wrs)

    def query(self):
        self._ensure()
        cdef c.ibv_srq_attr a
        memset(&a, 0, sizeof(a))
        cdef int rc = _v_query_srq(self._srq, &a)
        if rc != 0:
            _fail_rc("ibv_query_srq", rc)
        return {"max_wr": a.max_wr, "max_sge": a.max_sge, "srq_limit": a.srq_limit}

    def modify(self, max_wr=None, srq_limit=None):
        self._ensure()
        cdef c.ibv_srq_attr a
        cdef int mask = 0
        memset(&a, 0, sizeof(a))
        if max_wr is not None:
            a.max_wr = max_wr
            mask |= 1  # IBV_SRQ_MAX_WR
        if srq_limit is not None:
            a.srq_limit = srq_limit
            mask |= 2  # IBV_SRQ_LIMIT
        cdef int rc = _v_modify_srq(self._srq, &a, mask)
        if rc != 0:
            _fail_rc("ibv_modify_srq", rc)

    def close(self):
        cdef int rc
        if self._srq is not NULL:
            rc = _v_destroy_srq(self._srq)
            if rc != 0:
                _fail_rc("ibv_destroy_srq", rc)
            self._srq = NULL

    def __dealloc__(self):
        if self._srq is not NULL:
            _v_destroy_srq(self._srq)
            self._srq = NULL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# --------------------------------------------------------------------------- #
# RDMA connection manager (optional librdmacm)
# --------------------------------------------------------------------------- #
cdef class CMID:
    """A synchronously resolved RDMA-CM endpoint.

    :meth:`resolve` performs address and route resolution. The returned
    :attr:`context` is borrowed from RDMA-CM and must not be closed directly;
    allocate the PD and CQs from it, then use :meth:`create_qp` so CM owns the
    QP state transitions and destruction.
    """

    cdef cm.rdma_cm_id *_id
    cdef readonly Context context
    cdef readonly str host
    cdef readonly int port
    cdef readonly object source
    cdef bint _connected

    def __cinit__(self):
        self._id = NULL
        self.context = None
        self.source = None
        self._connected = False

    @classmethod
    def resolve(cls, str host, port=4420, source=None):
        """Resolve ``host:port``, optionally binding a source IP address."""
        if not host or "\x00" in host:
            raise ValueError("host must be a non-empty string without NUL bytes")
        if source is not None:
            if not isinstance(source, str) or not source or "\x00" in source:
                raise ValueError(
                    "source must be a non-empty string without NUL bytes"
                )
        port = int(port)
        if port <= 0 or port > 65535:
            raise ValueError("port must be between 1 and 65535")
        _load_librdmacm()
        cdef bytes node = host.encode("idna")
        cdef bytes service = str(port).encode("ascii")
        cdef const char *node_ptr = node
        cdef const char *service_ptr = service
        cdef bytes source_node
        cdef bytes source_service = b"0"
        cdef const char *source_node_ptr = NULL
        cdef const char *source_service_ptr = source_service
        cdef cm.rdma_addrinfo hints
        cdef cm.rdma_addrinfo source_hints
        cdef cm.rdma_addrinfo *result = NULL
        cdef cm.rdma_addrinfo *source_result = NULL
        cdef cm.rdma_cm_id *endpoint = NULL
        cdef int rc
        memset(&hints, 0, sizeof(hints))
        hints.ai_qp_type = _QPT_RC
        hints.ai_port_space = 0x0106  # RDMA_PS_TCP
        if source is not None:
            source_node = (<str>source).encode("idna")
            source_node_ptr = source_node
            memset(&source_hints, 0, sizeof(source_hints))
            source_hints.ai_qp_type = _QPT_RC
            source_hints.ai_port_space = 0x0106
            with nogil:
                rc = _cm_getaddrinfo(
                    source_node_ptr,
                    source_service_ptr,
                    &source_hints,
                    &source_result,
                )
            if rc != 0:
                _fail_rc("rdma_getaddrinfo(source)", rc)
            hints.ai_family = source_result.ai_family
            hints.ai_flags = 0x00000008  # RAI_FAMILY
            hints.ai_src_len = source_result.ai_dst_len
            hints.ai_src_addr = source_result.ai_dst_addr
        try:
            with nogil:
                rc = _cm_getaddrinfo(node_ptr, service_ptr, &hints, &result)
            if rc != 0:
                _fail_rc("rdma_getaddrinfo", rc)
            try:
                with nogil:
                    rc = _cm_create_ep(&endpoint, result, NULL, NULL)
                if rc != 0:
                    _fail_rc("rdma_create_ep", rc)
            finally:
                _cm_freeaddrinfo(result)
        finally:
            if source_result is not NULL:
                _cm_freeaddrinfo(source_result)
        if endpoint is NULL or endpoint.verbs is NULL:
            if endpoint is not NULL:
                _cm_destroy_ep(endpoint)
            raise VerbsError("rdma_create_ep", EIO, "resolved endpoint has no device")
        cdef CMID self = CMID.__new__(CMID)
        self._id = endpoint
        self.host = host
        self.port = port
        self.source = source
        self.context = Context._wrap_borrowed(
            endpoint.verbs, _v_get_device_name(endpoint.verbs.device).decode())
        return self

    cdef int _ensure(self) except -1:
        if self._id is NULL:
            raise VerbsError("rdma_cm_id is closed", EBADF)
        return 0

    @property
    def closed(self) -> bool:
        return self._id is NULL

    @property
    def connected(self) -> bool:
        return bool(self._connected)

    def create_qp(self, PD pd, init_attr) -> QP:
        """Create a CM-managed RC QP using ``pd`` and ``init_attr``."""
        self._ensure()
        pd._ensure()
        if self._id.qp is not NULL:
            raise VerbsError("rdma_create_qp", EBUSY, "endpoint already has a QP")
        if pd.context is not self.context:
            raise ValueError("protection domain must belong to this CM context")
        cdef CQ scq = <CQ>init_attr.send_cq
        cdef CQ rcq = <CQ>init_attr.recv_cq
        cdef SRQ srq = None
        scq._ensure()
        rcq._ensure()
        if scq.context is not self.context or rcq.context is not self.context:
            raise ValueError("queue-pair CQs must belong to this CM context")
        cdef c.ibv_qp_init_attr a
        memset(&a, 0, sizeof(a))
        a.send_cq = scq._cq
        a.recv_cq = rcq._cq
        if init_attr.srq is not None:
            srq = <SRQ>init_attr.srq
            srq._ensure()
            if srq.pd is not pd:
                raise ValueError("queue-pair SRQ must belong to this PD")
            a.srq = srq._srq
        if int(init_attr.qp_type) != _QPT_RC:
            raise ValueError("RDMA-CM endpoint requires an RC queue pair")
        a.qp_type = _QPT_RC
        a.sq_sig_all = 1 if init_attr.sq_sig_all else 0
        a.cap.max_send_wr = init_attr.max_send_wr
        a.cap.max_recv_wr = init_attr.max_recv_wr
        a.cap.max_send_sge = init_attr.max_send_sge
        a.cap.max_recv_sge = init_attr.max_recv_sge
        a.cap.max_inline_data = init_attr.max_inline_data
        cdef int rc = _cm_create_qp(self._id, pd._pd, &a)
        if rc != 0:
            _fail_rc("rdma_create_qp", rc)
        cdef QP qp = QP._wrap(self._id.qp, pd, scq, rcq, srq)
        qp._cm_id = self._id
        qp._cm_owner = self
        return qp

    def connect(self, private_data=b"", *, responder_resources=1,
                initiator_depth=1, retry_count=7, rnr_retry_count=7) -> bytes:
        """Connect the QP and return the peer's RDMA-CM private data."""
        self._ensure()
        if self._id.qp is NULL:
            raise VerbsError("rdma_connect", EBADF, "create a QP before connecting")
        cdef bytes data = bytes(private_data)
        if len(data) > 255:
            raise ValueError("RDMA-CM private data cannot exceed 255 bytes")
        for name, value in (
            ("responder_resources", responder_resources),
            ("initiator_depth", initiator_depth),
            ("retry_count", retry_count),
            ("rnr_retry_count", rnr_retry_count),
        ):
            if int(value) < 0 or int(value) > 255:
                raise ValueError("%s must be between 0 and 255" % name)
        cdef cm.rdma_conn_param param
        memset(&param, 0, sizeof(param))
        if data:
            param.private_data = <const void *><const char *>data
            param.private_data_len = <uint8_t>len(data)
        param.responder_resources = <uint8_t>responder_resources
        param.initiator_depth = <uint8_t>initiator_depth
        param.flow_control = 1
        param.retry_count = <uint8_t>retry_count
        param.rnr_retry_count = <uint8_t>rnr_retry_count
        cdef int rc
        with nogil:
            rc = _cm_connect(self._id, &param)
        if rc != 0:
            _fail_rc("rdma_connect", rc)
        self._connected = True
        cdef cm.rdma_cm_event *event = self._id.event
        cdef uint8_t n
        if event is NULL or event.param.conn.private_data is NULL:
            return b""
        n = event.param.conn.private_data_len
        return (<char *>event.param.conn.private_data)[:n]

    def disconnect(self):
        """Synchronously disconnect an established endpoint."""
        self._ensure()
        if not self._connected:
            return
        cdef int rc
        with nogil:
            rc = _cm_disconnect(self._id)
        if rc != 0:
            _fail_rc("rdma_disconnect", rc)
        self._connected = False

    def close(self):
        if self._id is not NULL:
            if self._id.qp is not NULL:
                raise VerbsError(
                    "rdma_destroy_ep", EBUSY, "close the CM-managed QP first")
            if self.context is not None and self.context._children != 0:
                raise VerbsError(
                    "rdma_destroy_ep", EBUSY,
                    "cannot close endpoint with %d open context resource(s)"
                    % self.context._children,
                )
            if self.context is not None:
                self.context._ctx = NULL
            _cm_destroy_ep(self._id)
            self._id = NULL
            self._connected = False

    def __dealloc__(self):
        if self._id is not NULL:
            if self._id.qp is not NULL:
                _cm_destroy_qp(self._id)
            if self.context is not None:
                self.context._ctx = NULL
            _cm_destroy_ep(self._id)
            self._id = NULL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        if self._id is NULL:
            return "CMID(closed=True)"
        return "CMID(host=%r, port=%d, connected=%r)" % (
            self.host, self.port, bool(self._connected))


__all__ = [
    "VerbsError", "CompletionError", "Gid", "DeviceAttr", "PortAttr", "WC",
    "SGE", "SendWR",
    "RecvWR", "QPCap", "QPInitAttr", "AHAttr", "Device", "get_device_list",
    "Context", "AsyncEvent", "PD", "MR", "CompChannel", "CQ", "QP", "AH", "SRQ",
    "CMID",
]
