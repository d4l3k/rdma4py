# cython: language_level=3, embedsignature=True, binding=True
"""Low-level Cython bindings for libibverbs.

Every RDMA resource is a RAII ``cdef class`` that owns its C handle and frees
it in ``__dealloc__``. Children keep a Python reference to their parent so the
garbage collector cannot free a parent before its children. Hot paths
(``post_send``/``post_recv``/``poll``/``get_cq_event``) release the GIL.
"""

import os

from libc.errno cimport errno
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, uintptr_t
from libc.stdlib cimport calloc, free
from libc.string cimport memset, memcpy, strcmp

cimport ibverbs._libverbs as c


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class VerbsError(OSError):
    """Raised when a libibverbs call fails.

    Subclasses :class:`OSError`, so ``.errno`` and ``.strerror`` are populated.
    ``.operation`` names the verb that failed.
    """

    def __init__(self, str operation, int err):
        self.operation = operation
        msg = "%s failed: %s" % (operation, os.strerror(err))
        super().__init__(err, msg)


cdef int _fail(str op) except -1:
    raise VerbsError(op, errno)


# --------------------------------------------------------------------------- #
# Linkage smoke test (kept for the import test)
# --------------------------------------------------------------------------- #
def _linked() -> bool:
    cdef int num = 0
    cdef c.ibv_device **lst = c.ibv_get_device_list(&num)
    if lst is not NULL:
        c.ibv_free_device_list(lst)
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
        return c.ibv_wc_status_str(self.status).decode()

    def raise_for_status(self):
        """Raise :class:`VerbsError` if this completion did not succeed."""
        if self.status != 0:
            raise VerbsError("work completion (%s)" % self.status_str, 0)

    def __repr__(self):
        return ("WC(wr_id=%d, status=%d (%s), opcode=%d, byte_len=%d, "
                "qp_num=%d)") % (self.wr_id, self.status, self.status_str,
                                 self.opcode, self.byte_len, self.qp_num)


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

    def __init__(self, target, length, lkey=0, offset=0):
        cdef MR mr
        if isinstance(target, MR):
            mr = <MR>target
            self.addr = (<uint64_t><uintptr_t>mr._mr.addr) + <uint64_t>offset
            self.lkey = mr._mr.lkey if lkey == 0 else <uint32_t>lkey
        else:
            self.addr = (<uint64_t>int(target)) + <uint64_t>offset
            self.lkey = <uint32_t>lkey
        self.length = <uint32_t>length

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
        cdef c.ibv_device **lst = c.ibv_get_device_list(&num)
        cdef c.ibv_context *ctx = NULL
        cdef bytes want = self.name.encode()
        cdef int i
        if lst is NULL:
            _fail("ibv_get_device_list")
        try:
            for i in range(num):
                if strcmp(c.ibv_get_device_name(lst[i]), <char*>want) == 0:
                    ctx = c.ibv_open_device(lst[i])
                    break
        finally:
            c.ibv_free_device_list(lst)
        if ctx is NULL:
            raise VerbsError("ibv_open_device(%s)" % self.name, errno)
        return Context._wrap(ctx, self.name)

    def __repr__(self):
        return "Device(name=%r, guid=0x%x)" % (self.name, self.guid)


def get_device_list() -> list:
    """Return the list of :class:`Device` objects present on this host."""
    cdef int num = 0
    cdef c.ibv_device **lst = c.ibv_get_device_list(&num)
    cdef int i
    if lst is NULL:
        _fail("ibv_get_device_list")
    out = []
    try:
        for i in range(num):
            name = c.ibv_get_device_name(lst[i]).decode()
            guid = c.ibv_get_device_guid(lst[i])
            out.append(Device(name, guid))
    finally:
        c.ibv_free_device_list(lst)
    return out


cdef class Context:
    """An open device context (``ibv_context``)."""

    cdef c.ibv_context *_ctx
    cdef readonly str name

    def __cinit__(self):
        self._ctx = NULL

    @staticmethod
    cdef Context _wrap(c.ibv_context *ctx, str name):
        cdef Context self = Context.__new__(Context)
        self._ctx = ctx
        self.name = name
        return self

    cdef int _ensure(self) except -1:
        if self._ctx is NULL:
            raise VerbsError("context is closed", 0)
        return 0

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
        if c.ibv_query_device(self._ctx, &a) != 0:
            _fail("ibv_query_device")
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
        if c.___ibv_query_port(self._ctx, <uint8_t>port_num, &a) != 0:
            _fail("ibv_query_port")
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
        if c.ibv_query_gid(self._ctx, <uint8_t>port_num, index, &g) != 0:
            _fail("ibv_query_gid")
        return Gid(bytes(bytearray(g.raw[:16])))

    def alloc_pd(self) -> "PD":
        self._ensure()
        cdef c.ibv_pd *pd = c.ibv_alloc_pd(self._ctx)
        if pd is NULL:
            _fail("ibv_alloc_pd")
        return PD._wrap(pd, self)

    def create_comp_channel(self) -> "CompChannel":
        self._ensure()
        cdef c.ibv_comp_channel *ch = c.ibv_create_comp_channel(self._ctx)
        if ch is NULL:
            _fail("ibv_create_comp_channel")
        return CompChannel._wrap(ch, self)

    def create_cq(self, int cqe, channel=None, int comp_vector=0) -> "CQ":
        self._ensure()
        cdef c.ibv_comp_channel *chp = NULL
        cdef CompChannel ch = None
        if channel is not None:
            ch = <CompChannel>channel
            chp = ch._chan
        cdef CQ cq = CQ.__new__(CQ)
        cq.context = self
        cq.channel = ch
        cq._cq = c.ibv_create_cq(self._ctx, cqe, <void*>cq, chp, comp_vector)
        if cq._cq is NULL:
            _fail("ibv_create_cq")
        return cq

    def get_async_event(self) -> "AsyncEvent":
        self._ensure()
        cdef AsyncEvent ev = AsyncEvent.__new__(AsyncEvent)
        cdef int rc
        with nogil:
            rc = c.ibv_get_async_event(self._ctx, &ev._ev)
        if rc != 0:
            _fail("ibv_get_async_event")
        ev._acked = False
        ev.event_type = ev._ev.event_type
        return ev

    def ack_async_event(self, AsyncEvent ev):
        if not ev._acked:
            c.ibv_ack_async_event(&ev._ev)
            ev._acked = True

    def close(self):
        if self._ctx is not NULL:
            c.ibv_close_device(self._ctx)
            self._ctx = NULL

    def __dealloc__(self):
        if self._ctx is not NULL:
            c.ibv_close_device(self._ctx)
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
        return c.ibv_event_type_str(self.event_type).decode()

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
        return self

    cdef int _ensure(self) except -1:
        if self._pd is NULL:
            raise VerbsError("pd is closed", 0)
        return 0

    def reg_mr(self, addr, length, int access) -> "MR":
        """Register a memory region over ``[addr, addr+length)``.

        ``addr`` is an integer virtual address; it may be a host pointer or a
        CUDA device pointer (GPUDirect via nvidia_peermem).
        """
        self._ensure()
        cdef uint64_t a = <uint64_t>int(addr)
        cdef size_t ln = <size_t>length
        cdef unsigned int acc = <unsigned int>access
        cdef c.ibv_mr *mr
        with nogil:
            mr = c.ibv_reg_mr_iova2(self._pd, <void*><uintptr_t>a, ln, a, acc)
        if mr is NULL:
            _fail("ibv_reg_mr")
        return MR._wrap(mr, self)

    def reg_dmabuf_mr(self, offset, length, iova, int fd, int access) -> "MR":
        """Register a dma-buf backed region (modern GPUDirect path)."""
        self._ensure()
        cdef c.ibv_mr *mr
        cdef uint64_t off = <uint64_t>int(offset)
        cdef size_t ln = <size_t>length
        cdef uint64_t iv = <uint64_t>int(iova)
        with nogil:
            mr = c.ibv_reg_dmabuf_mr(self._pd, off, ln, iv, fd, access)
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
        a.send_cq = scq._cq
        a.recv_cq = rcq._cq
        if init_attr.srq is not None:
            srq = <SRQ>init_attr.srq
            a.srq = srq._srq
        a.qp_type = init_attr.qp_type
        a.sq_sig_all = 1 if init_attr.sq_sig_all else 0
        a.cap.max_send_wr = init_attr.max_send_wr
        a.cap.max_recv_wr = init_attr.max_recv_wr
        a.cap.max_send_sge = init_attr.max_send_sge
        a.cap.max_recv_sge = init_attr.max_recv_sge
        a.cap.max_inline_data = init_attr.max_inline_data
        cdef c.ibv_qp *qp = c.ibv_create_qp(self._pd, &a)
        if qp is NULL:
            _fail("ibv_create_qp")
        return QP._wrap(qp, self, scq, rcq, srq)

    def create_ah(self, AHAttr attr) -> "AH":
        self._ensure()
        cdef c.ibv_ah_attr a
        _fill_ah_attr(&a, attr)
        cdef c.ibv_ah *ah = c.ibv_create_ah(self._pd, &a)
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
        cdef c.ibv_srq *srq = c.ibv_create_srq(self._pd, &a)
        if srq is NULL:
            _fail("ibv_create_srq")
        return SRQ._wrap(srq, self)

    def close(self):
        if self._pd is not NULL:
            c.ibv_dealloc_pd(self._pd)
            self._pd = NULL

    def __dealloc__(self):
        if self._pd is not NULL:
            c.ibv_dealloc_pd(self._pd)
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

    @staticmethod
    cdef MR _wrap(c.ibv_mr *mr, PD pd):
        cdef MR self = MR.__new__(MR)
        self._mr = mr
        self.pd = pd
        return self

    @property
    def addr(self) -> int:
        return <uint64_t><uintptr_t>self._mr.addr

    @property
    def length(self) -> int:
        return self._mr.length

    @property
    def lkey(self) -> int:
        return self._mr.lkey

    @property
    def rkey(self) -> int:
        return self._mr.rkey

    @property
    def handle(self) -> int:
        return self._mr.handle

    def close(self):
        if self._mr is not NULL:
            c.ibv_dereg_mr(self._mr)
            self._mr = NULL

    def __dealloc__(self):
        if self._mr is not NULL:
            c.ibv_dereg_mr(self._mr)
            self._mr = NULL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
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
        return self

    @property
    def fd(self) -> int:
        return self._chan.fd

    def get_cq_event(self) -> "CQ":
        """Block until a CQ event arrives; return the associated :class:`CQ`.

        The event must later be acknowledged with :meth:`CQ.ack_events`.
        """
        cdef c.ibv_cq *cq = NULL
        cdef void *ctx = NULL
        cdef int rc
        with nogil:
            rc = c.ibv_get_cq_event(self._chan, &cq, &ctx)
        if rc != 0:
            _fail("ibv_get_cq_event")
        cdef CQ obj = <CQ>ctx
        obj._unacked += 1
        return obj

    def close(self):
        if self._chan is not NULL:
            c.ibv_destroy_comp_channel(self._chan)
            self._chan = NULL

    def __dealloc__(self):
        if self._chan is not NULL:
            c.ibv_destroy_comp_channel(self._chan)
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

    @property
    def cqe(self) -> int:
        return self._cq.cqe

    def poll(self, int num_entries) -> list:
        """Poll up to ``num_entries`` completions; return a list of :class:`WC`."""
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
                raise VerbsError("ibv_poll_cq", errno)
            out = []
            for i in range(n):
                out.append(_wc_from_c(&wcs[i]))
            return out
        finally:
            free(wcs)

    def req_notify(self, bint solicited_only=False):
        """Request a completion notification on the CQ's channel."""
        if c.ibv_req_notify_cq(self._cq, 1 if solicited_only else 0) != 0:
            _fail("ibv_req_notify_cq")

    def ack_events(self, unsigned int nevents=1):
        """Acknowledge ``nevents`` events delivered via the channel."""
        c.ibv_ack_cq_events(self._cq, nevents)
        if <int>nevents <= self._unacked:
            self._unacked -= <int>nevents
        else:
            self._unacked = 0

    def close(self):
        if self._cq is not NULL:
            if self._unacked > 0:
                c.ibv_ack_cq_events(self._cq, <unsigned int>self._unacked)
                self._unacked = 0
            c.ibv_destroy_cq(self._cq)
            self._cq = NULL

    def __dealloc__(self):
        if self._cq is not NULL:
            if self._unacked > 0:
                c.ibv_ack_cq_events(self._cq, <unsigned int>self._unacked)
            c.ibv_destroy_cq(self._cq)
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

    @staticmethod
    cdef QP _wrap(c.ibv_qp *qp, PD pd, CQ scq, CQ rcq, SRQ srq):
        cdef QP self = QP.__new__(QP)
        self._qp = qp
        self.pd = pd
        self.send_cq = scq
        self.recv_cq = rcq
        self.srq = srq
        self._port = 1
        return self

    @property
    def qp_num(self) -> int:
        return self._qp.qp_num

    @property
    def qp_type(self) -> int:
        return self._qp.qp_type

    @property
    def state(self) -> int:
        """The queue pair's current state (authoritative, via query)."""
        cdef c.ibv_qp_attr a
        cdef c.ibv_qp_init_attr ia
        memset(&a, 0, sizeof(a))
        if c.ibv_query_qp(self._qp, &a, _QP_STATE, &ia) != 0:
            _fail("ibv_query_qp")
        return a.qp_state

    def query(self):
        """Return ``(attrs, cap)`` for the queue pair as plain dict + QPCap."""
        cdef c.ibv_qp_attr a
        cdef c.ibv_qp_init_attr ia
        memset(&a, 0, sizeof(a))
        memset(&ia, 0, sizeof(ia))
        if c.ibv_query_qp(self._qp, &a, 0x1FFFFFF, &ia) != 0:
            _fail("ibv_query_qp")
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
        if c.ibv_modify_qp(self._qp, &a, attr_mask) != 0:
            _fail("ibv_modify_qp")

    # -- RC/UD state-machine helpers -----------------------------------------
    def to_init(self, int port, int access=0, int pkey_index=0, int qkey=0):
        """Transition RESET -> INIT."""
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
        if c.ibv_modify_qp(self._qp, &a, mask) != 0:
            _fail("ibv_modify_qp(->INIT)")

    def to_rtr(self, remote, int sgid_index, mtu=None, int hop_limit=1,
               int min_rnr_timer=12, int max_dest_rd_atomic=1, int sl=0,
               int traffic_class=0):
        """Transition INIT -> RTR using a remote :class:`~ibverbs.helpers.QPInfo`."""
        cdef c.ibv_qp_attr a
        cdef int mask
        cdef bytes dgid
        memset(&a, 0, sizeof(a))
        a.qp_state = _QPS_RTR
        if self._qp.qp_type == _QPT_UD:
            if c.ibv_modify_qp(self._qp, &a, _QP_STATE) != 0:
                _fail("ibv_modify_qp(->RTR)")
            return
        a.path_mtu = int(mtu) if mtu is not None else int(remote.mtu)
        a.dest_qp_num = remote.qp_num
        a.rq_psn = remote.psn
        a.max_dest_rd_atomic = max_dest_rd_atomic
        a.min_rnr_timer = min_rnr_timer
        dgid = bytes(remote.gid)
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
        if c.ibv_modify_qp(self._qp, &a, mask) != 0:
            _fail("ibv_modify_qp(->RTR)")

    def to_rts(self, int psn, int timeout=14, int retry_cnt=7, int rnr_retry=7,
               int max_rd_atomic=1):
        """Transition RTR -> RTS."""
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
        if c.ibv_modify_qp(self._qp, &a, mask) != 0:
            _fail("ibv_modify_qp(->RTS)")

    def post_send(self, wrs):
        """Post one or more :class:`SendWR` to the send queue."""
        if isinstance(wrs, SendWR):
            wrs = [wrs]
        _post_send(self._qp, wrs)

    def post_recv(self, wrs):
        """Post one or more :class:`RecvWR` to the receive queue."""
        if isinstance(wrs, RecvWR):
            wrs = [wrs]
        _post_recv_qp(self._qp, wrs)

    def close(self):
        if self._qp is not NULL:
            c.ibv_destroy_qp(self._qp)
            self._qp = NULL

    def __dealloc__(self):
        if self._qp is not NULL:
            c.ibv_destroy_qp(self._qp)
            self._qp = NULL

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
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
        if self._ah is not NULL:
            c.ibv_destroy_ah(self._ah)
            self._ah = NULL

    def __dealloc__(self):
        if self._ah is not NULL:
            c.ibv_destroy_ah(self._ah)
            self._ah = NULL


cdef class SRQ:
    """A shared receive queue (``ibv_srq``)."""

    cdef c.ibv_srq *_srq
    cdef readonly PD pd

    @staticmethod
    cdef SRQ _wrap(c.ibv_srq *srq, PD pd):
        cdef SRQ self = SRQ.__new__(SRQ)
        self._srq = srq
        self.pd = pd
        return self

    def post_recv(self, wrs):
        if isinstance(wrs, RecvWR):
            wrs = [wrs]
        _post_recv_chain(NULL, self._srq, wrs)

    def query(self):
        cdef c.ibv_srq_attr a
        memset(&a, 0, sizeof(a))
        if c.ibv_query_srq(self._srq, &a) != 0:
            _fail("ibv_query_srq")
        return {"max_wr": a.max_wr, "max_sge": a.max_sge, "srq_limit": a.srq_limit}

    def modify(self, max_wr=None, srq_limit=None):
        cdef c.ibv_srq_attr a
        cdef int mask = 0
        memset(&a, 0, sizeof(a))
        if max_wr is not None:
            a.max_wr = max_wr
            mask |= 1  # IBV_SRQ_MAX_WR
        if srq_limit is not None:
            a.srq_limit = srq_limit
            mask |= 2  # IBV_SRQ_LIMIT
        if c.ibv_modify_srq(self._srq, &a, mask) != 0:
            _fail("ibv_modify_srq")

    def close(self):
        if self._srq is not NULL:
            c.ibv_destroy_srq(self._srq)
            self._srq = NULL

    def __dealloc__(self):
        if self._srq is not NULL:
            c.ibv_destroy_srq(self._srq)
            self._srq = NULL


__all__ = [
    "VerbsError", "Gid", "DeviceAttr", "PortAttr", "WC", "SGE", "SendWR",
    "RecvWR", "QPCap", "QPInitAttr", "AHAttr", "Device", "get_device_list",
    "Context", "AsyncEvent", "PD", "MR", "CompChannel", "CQ", "QP", "AH", "SRQ",
]
