# cython: language_level=3
# Minimal declarations of the libibverbs C API needed to prove linkage.
# Expanded with the full surface in later tasks.

cdef extern from "infiniband/verbs.h" nogil:
    cdef struct ibv_device:
        pass

    ibv_device **ibv_get_device_list(int *num_devices)
    void ibv_free_device_list(ibv_device **list)
