# cython: language_level=3
"""Minimal librdmacm declarations used by the optional CM binding."""

from libc.stdint cimport uint8_t, uint32_t
from libc.stddef cimport size_t

cimport ibverbs._libverbs as v


cdef extern from "rdma/rdma_cma.h" nogil:
    cdef struct rdma_event_channel:
        int fd

    cdef struct rdma_conn_param:
        const void *private_data
        uint8_t private_data_len
        uint8_t responder_resources
        uint8_t initiator_depth
        uint8_t flow_control
        uint8_t retry_count
        uint8_t rnr_retry_count
        uint8_t srq
        uint32_t qp_num

    cdef union rdma_cm_event_param:
        rdma_conn_param conn

    cdef struct rdma_cm_event:
        rdma_cm_id *id
        rdma_cm_id *listen_id
        int event
        int status
        rdma_cm_event_param param

    cdef struct rdma_cm_id:
        v.ibv_context *verbs
        rdma_event_channel *channel
        void *context
        v.ibv_qp *qp
        int ps
        uint8_t port_num
        rdma_cm_event *event
        v.ibv_pd *pd
        int qp_type

    cdef struct rdma_addrinfo:
        int ai_flags
        int ai_family
        int ai_qp_type
        int ai_port_space
        size_t ai_src_len
        size_t ai_dst_len
        void *ai_src_addr
        void *ai_dst_addr
        char *ai_src_canonname
        char *ai_dst_canonname
        size_t ai_route_len
        void *ai_route
        size_t ai_connect_len
        void *ai_connect
        rdma_addrinfo *ai_next
