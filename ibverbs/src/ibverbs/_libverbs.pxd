# cython: language_level=3
"""Declarations of the libibverbs (rdma-core) C API used by the bindings.

Only the fields we touch are declared for each struct; Cython trusts the real
``<infiniband/verbs.h>`` for the full layout, so partial declarations are safe.
Enum-typed fields/params are modelled as plain ``int`` (C converts implicitly)
to keep this file small and version-robust.
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
    cdef struct ibv_wq:
        pass
    cdef struct ibv_dm:
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
        int max_qp_init_rd_atom
        int max_ah
        int max_srq
        int max_srq_wr
        int max_srq_sge
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

    # --- scatter/gather + work requests ---
    cdef struct ibv_sge:
        uint64_t addr
        uint32_t length
        uint32_t lkey

    cdef struct _wr_rdma:
        uint64_t remote_addr
        uint32_t rkey
    cdef struct _wr_atomic:
        uint64_t remote_addr
        uint64_t compare_add
        uint64_t swap
        uint32_t rkey
    cdef struct _wr_ud:
        ibv_ah *ah
        uint32_t remote_qpn
        uint32_t remote_qkey
    cdef union _wr_union:
        _wr_rdma rdma
        _wr_atomic atomic
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
        int path_mig_state
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

    # --- shared receive queue ---
    cdef struct ibv_srq_attr:
        uint32_t max_wr
        uint32_t max_sge
        uint32_t srq_limit

    cdef struct ibv_srq_init_attr:
        void *srq_context
        ibv_srq_attr attr

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
    # These are ``static inline`` in verbs.h: they dispatch through the
    # provider function-pointer table on the context and reference NO exported
    # symbol, so they compile directly into this extension. Everything else is
    # resolved at runtime via dlopen/dlsym (see _ibverbs.pyx) so the extension
    # does not hard-link libibverbs.
    int ibv_poll_cq(ibv_cq *cq, int num_entries, ibv_wc *wc)
    int ibv_req_notify_cq(ibv_cq *cq, int solicited_only)
    int ibv_post_send(ibv_qp *qp, ibv_send_wr *wr, ibv_send_wr **bad_wr)
    int ibv_post_recv(ibv_qp *qp, ibv_recv_wr *wr, ibv_recv_wr **bad_wr)
    int ibv_post_srq_recv(ibv_srq *srq, ibv_recv_wr *recv_wr,
                          ibv_recv_wr **bad_recv_wr)


cdef extern from "_mlx5dv_bridge.h" nogil:
    cdef struct rdma4py_mlx5_qp_info:
        void *sq_buf
        uint32_t sq_wqe_cnt
        uint32_t sq_stride
        void *rq_buf
        uint32_t rq_wqe_cnt
        uint32_t rq_stride
        uint32_t *sq_dbrec
        uint32_t *rq_dbrec
        uint64_t *uar
        uint32_t uar_size

    cdef struct rdma4py_mlx5_cq_info:
        void *buf
        uint32_t *dbrec
        uint32_t cqe_cnt
        uint32_t cqe_size
        uint32_t cqn

    int rdma4py_mlx5dv_qp_info(void *init_obj_fn, ibv_qp *qp,
                                rdma4py_mlx5_qp_info *info)
    int rdma4py_mlx5dv_cq_info(void *init_obj_fn, ibv_cq *cq,
                                rdma4py_mlx5_cq_info *info)
