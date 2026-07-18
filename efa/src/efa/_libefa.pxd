# cython: language_level=3
"""Declarations of the libibverbs + libefa (rdma-core) C API used by the bindings.

Only the fields we touch are declared for each struct; Cython trusts the real
``<infiniband/verbs.h>`` / ``<infiniband/efadv.h>`` for the full layout, so
partial declarations are safe. Enum-typed fields/params are modelled as plain
``int`` (C converts implicitly) to keep this file small and version-robust.
"""

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stddef cimport size_t


cdef extern from "arpa/inet.h" nogil:
    uint32_t htonl(uint32_t hostlong)
    uint32_t ntohl(uint32_t netlong)


cdef extern from "infiniband/verbs.h" nogil:
    # --- opaque handles (only ever used behind pointers) ---
    cdef struct ibv_device:
        pass
    cdef struct ibv_ah:
        pass
    cdef struct ibv_srq:
        pass
    cdef struct ibv_xrcd:
        pass

    # --- context / device ---
    cdef struct ibv_context:
        ibv_device *device
        int async_fd
        int num_comp_vectors

    cdef struct ibv_device_attr:
        char fw_ver[64]
        uint64_t node_guid
        uint64_t sys_image_guid
        uint64_t max_mr_size
        uint32_t vendor_id
        uint32_t vendor_part_id
        uint32_t hw_ver
        int max_qp
        int max_qp_wr
        int max_sge
        int max_cq
        int max_cqe
        int max_mr
        int max_pd
        int max_qp_rd_atom
        int max_ah
        uint16_t max_pkeys
        uint8_t phys_port_cnt

    cdef struct ibv_port_attr:
        int state
        int max_mtu
        int active_mtu
        int gid_tbl_len
        uint32_t port_cap_flags
        uint32_t max_msg_sz
        uint16_t pkey_tbl_len
        uint16_t lid
        uint16_t sm_lid
        uint8_t lmc
        uint8_t active_width
        uint8_t active_speed
        uint8_t link_layer

    cdef union ibv_gid:
        uint8_t raw[16]

    # --- protection domain / memory region ---
    cdef struct ibv_pd:
        ibv_context *context
        uint32_t handle

    cdef struct ibv_mr:
        ibv_context *context
        ibv_pd *pd
        void *addr
        size_t length
        uint32_t handle
        uint32_t lkey
        uint32_t rkey

    # --- completion queue / channel ---
    cdef struct ibv_comp_channel:
        ibv_context *context
        int fd

    cdef struct ibv_cq:
        ibv_context *context
        ibv_comp_channel *channel
        uint32_t handle
        int cqe

    cdef struct ibv_wc:
        uint64_t wr_id
        int status
        int opcode
        uint32_t vendor_err
        uint32_t byte_len
        uint32_t imm_data
        uint32_t qp_num
        uint32_t src_qp
        unsigned int wc_flags
        uint16_t pkey_index
        uint16_t slid
        uint8_t sl
        uint8_t dlid_path_bits

    # --- extended completion queue ---
    cdef struct ibv_poll_cq_attr:
        uint32_t comp_mask

    cdef struct ibv_cq_ex:
        ibv_context *context
        ibv_comp_channel *channel
        void *cq_context
        uint32_t handle
        int cqe
        uint32_t comp_mask
        int status
        uint64_t wr_id

    cdef struct ibv_cq_init_attr_ex:
        uint32_t cqe
        void *cq_context
        ibv_comp_channel *channel
        uint32_t comp_vector
        uint64_t wc_flags
        uint32_t comp_mask
        uint32_t flags
        ibv_pd *parent_domain

    # static inline poll helpers (dispatch via the cq_ex op table; no symbol).
    ibv_cq *ibv_cq_ex_to_cq(ibv_cq_ex *cq)
    int ibv_start_poll(ibv_cq_ex *cq, ibv_poll_cq_attr *attr)
    int ibv_next_poll(ibv_cq_ex *cq)
    void ibv_end_poll(ibv_cq_ex *cq)
    int ibv_wc_read_opcode(ibv_cq_ex *cq)
    uint32_t ibv_wc_read_vendor_err(ibv_cq_ex *cq)
    uint32_t ibv_wc_read_byte_len(ibv_cq_ex *cq)
    uint32_t ibv_wc_read_imm_data(ibv_cq_ex *cq)
    uint32_t ibv_wc_read_qp_num(ibv_cq_ex *cq)
    uint32_t ibv_wc_read_src_qp(ibv_cq_ex *cq)
    unsigned int ibv_wc_read_wc_flags(ibv_cq_ex *cq)

    # --- scatter/gather + work requests ---
    cdef struct ibv_sge:
        uint64_t addr
        uint32_t length
        uint32_t lkey

    cdef struct ibv_data_buf:
        void *addr
        size_t length

    cdef struct _wr_rdma:
        uint64_t remote_addr
        uint32_t rkey
    cdef struct _wr_ud:
        ibv_ah *ah
        uint32_t remote_qpn
        uint32_t remote_qkey
    cdef union _wr_union:
        _wr_rdma rdma
        _wr_ud ud

    cdef struct ibv_send_wr:
        uint64_t wr_id
        ibv_send_wr *next
        ibv_sge *sg_list
        int num_sge
        int opcode
        unsigned int send_flags
        uint32_t imm_data
        _wr_union wr

    cdef struct ibv_recv_wr:
        uint64_t wr_id
        ibv_recv_wr *next
        ibv_sge *sg_list
        int num_sge

    # --- queue pair ---
    cdef struct ibv_qp_cap:
        uint32_t max_send_wr
        uint32_t max_recv_wr
        uint32_t max_send_sge
        uint32_t max_recv_sge
        uint32_t max_inline_data

    cdef struct ibv_qp_init_attr:
        void *qp_context
        ibv_cq *send_cq
        ibv_cq *recv_cq
        ibv_srq *srq
        ibv_qp_cap cap
        int qp_type
        int sq_sig_all

    cdef struct ibv_qp_init_attr_ex:
        void *qp_context
        ibv_cq *send_cq
        ibv_cq *recv_cq
        ibv_srq *srq
        ibv_qp_cap cap
        int qp_type
        int sq_sig_all
        uint32_t comp_mask
        ibv_pd *pd
        uint64_t send_ops_flags

    cdef struct ibv_global_route:
        ibv_gid dgid
        uint32_t flow_label
        uint8_t sgid_index
        uint8_t hop_limit
        uint8_t traffic_class

    cdef struct ibv_ah_attr:
        ibv_global_route grh
        uint16_t dlid
        uint8_t sl
        uint8_t src_path_bits
        uint8_t static_rate
        uint8_t is_global
        uint8_t port_num

    cdef struct ibv_qp_attr:
        int qp_state
        int cur_qp_state
        int path_mtu
        uint32_t qkey
        uint32_t rq_psn
        uint32_t sq_psn
        uint32_t dest_qp_num
        unsigned int qp_access_flags
        ibv_qp_cap cap
        ibv_ah_attr ah_attr
        uint16_t pkey_index
        uint8_t max_rd_atomic
        uint8_t max_dest_rd_atomic
        uint8_t min_rnr_timer
        uint8_t port_num
        uint8_t timeout
        uint8_t retry_cnt
        uint8_t rnr_retry

    cdef struct ibv_qp:
        ibv_context *context
        ibv_pd *pd
        ibv_cq *send_cq
        ibv_cq *recv_cq
        ibv_srq *srq
        uint32_t handle
        uint32_t qp_num
        int state
        int qp_type

    # --- extended (qp_ex) work-request API ---
    # ``wr_id``/``wr_flags`` are set directly; the wr_* helpers below are
    # ``static inline`` dispatching through the op table in ibv_qp_ex.
    cdef struct ibv_qp_ex:
        ibv_qp qp_base
        uint64_t comp_mask
        uint64_t wr_id
        unsigned int wr_flags

    # The header's static inline ibv_create_qp_ex references the *exported*
    # ibv_create_qp symbol on its compat path, which would break the
    # dlopen-only linkage model. Instead we reach the extended create through
    # the provider op table: verbs_get_ctx is static inline and references no
    # exported symbol.
    ctypedef ibv_qp *(*fp_verbs_create_qp_ex)(
        ibv_context *context,
        ibv_qp_init_attr_ex *qp_init_attr_ex) noexcept nogil

    cdef struct verbs_context:
        fp_verbs_create_qp_ex create_qp_ex

    verbs_context *verbs_get_ctx(ibv_context *ctx)

    void ibv_wr_start(ibv_qp_ex *qp)
    int ibv_wr_complete(ibv_qp_ex *qp)
    void ibv_wr_abort(ibv_qp_ex *qp)
    void ibv_wr_send(ibv_qp_ex *qp)
    void ibv_wr_send_imm(ibv_qp_ex *qp, uint32_t imm_data)
    void ibv_wr_rdma_read(ibv_qp_ex *qp, uint32_t rkey, uint64_t remote_addr)
    void ibv_wr_rdma_write(ibv_qp_ex *qp, uint32_t rkey, uint64_t remote_addr)
    void ibv_wr_rdma_write_imm(ibv_qp_ex *qp, uint32_t rkey,
                               uint64_t remote_addr, uint32_t imm_data)
    void ibv_wr_set_ud_addr(ibv_qp_ex *qp, ibv_ah *ah, uint32_t remote_qpn,
                            uint32_t remote_qkey)
    void ibv_wr_set_sge_list(ibv_qp_ex *qp, size_t num_sge,
                             const ibv_sge *sg_list)
    void ibv_wr_set_inline_data_list(ibv_qp_ex *qp, size_t num_buf,
                                     const ibv_data_buf *buf_list)

    # --- async events ---
    cdef union _async_element:
        ibv_cq *cq
        ibv_qp *qp
        ibv_srq *srq
        int port_num
    cdef struct ibv_async_event:
        _async_element element
        int event_type

    # --- data-path fast paths ---
    # ``static inline`` in verbs.h: they dispatch through the provider op
    # table and reference NO exported symbol, so they compile directly into
    # this extension. Everything else is resolved at runtime via dlopen/dlsym
    # (see _efa.pyx) so the extension does not hard-link libibverbs/libefa.
    int ibv_poll_cq(ibv_cq *cq, int num_entries, ibv_wc *wc)
    int ibv_req_notify_cq(ibv_cq *cq, int solicited_only)
    int ibv_post_send(ibv_qp *qp, ibv_send_wr *wr, ibv_send_wr **bad_wr)
    int ibv_post_recv(ibv_qp *qp, ibv_recv_wr *wr, ibv_recv_wr **bad_wr)


cdef extern from * nogil:
    """
    #include <stdbool.h>
    #include <stdint.h>
    #include <infiniband/efadv.h>

    /*
     * Keep provider ABI layouts local so the extension can compile against
     * older rdma-core headers. New fields consume reserved bytes, and all
     * provider entry points are resolved with dlsym at runtime.
     */
    struct rdma4py_efadv_device_attr {
        uint64_t comp_mask;
        uint32_t max_sq_wr;
        uint32_t max_rq_wr;
        uint16_t max_sq_sge;
        uint16_t max_rq_sge;
        uint16_t inline_buf_size;
        uint8_t reserved[2];
        uint32_t device_caps;
        uint32_t max_rdma_size;
    };

    struct rdma4py_efadv_ah_attr {
        uint64_t comp_mask;
        uint16_t ahn;
        uint8_t reserved[6];
    };

    struct rdma4py_efadv_qp_init_attr {
        uint64_t comp_mask;
        uint32_t driver_qp_type;
        uint16_t flags;
        uint8_t sl;
        uint8_t reserved;
    };

    struct rdma4py_efadv_cq_init_attr {
        uint64_t comp_mask;
        uint64_t wc_flags;
    };

    struct rdma4py_efadv_wq_attr {
        uint64_t comp_mask;
        uint8_t *buffer;
        uint32_t entry_size;
        uint32_t num_entries;
        uint32_t *doorbell;
        uint32_t max_batch;
        uint8_t reserved[4];
    };

    struct rdma4py_efadv_cq_attr {
        uint64_t comp_mask;
        uint8_t *buffer;
        uint32_t entry_size;
        uint32_t num_entries;
        uint32_t *doorbell;
    };

    struct rdma4py_efadv_mr_attr {
        uint64_t comp_mask;
        uint16_t ic_id_validity;
        uint16_t recv_ic_id;
        uint16_t rdma_read_ic_id;
        uint16_t rdma_recv_ic_id;
    };

    struct rdma4py_efadv_cq {
        uint64_t comp_mask;
        int (*wc_read_sgid)(struct rdma4py_efadv_cq *, union ibv_gid *);
        bool (*wc_is_unsolicited)(struct rdma4py_efadv_cq *);
    };

    static inline int rdma4py_efadv_wc_read_sgid(
        struct rdma4py_efadv_cq *cq, union ibv_gid *sgid)
    {
        return cq->wc_read_sgid(cq, sgid);
    }

    static inline int rdma4py_efadv_wc_is_unsolicited(
        struct rdma4py_efadv_cq *cq)
    {
        return cq->wc_is_unsolicited(cq);
    }
    """

    cdef struct rdma4py_efadv_device_attr:
        uint64_t comp_mask
        uint32_t max_sq_wr
        uint32_t max_rq_wr
        uint16_t max_sq_sge
        uint16_t max_rq_sge
        uint16_t inline_buf_size
        uint32_t device_caps
        uint32_t max_rdma_size

    cdef struct rdma4py_efadv_ah_attr:
        uint64_t comp_mask
        uint16_t ahn

    cdef struct rdma4py_efadv_qp_init_attr:
        uint64_t comp_mask
        uint32_t driver_qp_type
        uint16_t flags
        uint8_t sl

    cdef struct rdma4py_efadv_cq_init_attr:
        uint64_t comp_mask
        uint64_t wc_flags

    cdef struct rdma4py_efadv_wq_attr:
        uint64_t comp_mask
        uint8_t *buffer
        uint32_t entry_size
        uint32_t num_entries
        uint32_t *doorbell
        uint32_t max_batch

    cdef struct rdma4py_efadv_cq_attr:
        uint64_t comp_mask
        uint8_t *buffer
        uint32_t entry_size
        uint32_t num_entries
        uint32_t *doorbell

    cdef struct rdma4py_efadv_mr_attr:
        uint64_t comp_mask
        uint16_t ic_id_validity
        uint16_t recv_ic_id
        uint16_t rdma_read_ic_id
        uint16_t rdma_recv_ic_id

    cdef struct rdma4py_efadv_cq:
        uint64_t comp_mask

    int rdma4py_efadv_wc_read_sgid(rdma4py_efadv_cq *cq, ibv_gid *sgid)
    bint rdma4py_efadv_wc_is_unsolicited(rdma4py_efadv_cq *cq)
