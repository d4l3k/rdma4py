# cython: language_level=3
"""Low-level Cython bindings for libibverbs."""

cimport ibverbs._libverbs as c


def _linked() -> bool:
    """Return True if the extension is linked against a working libibverbs.

    Calls into the library (enumerating devices) purely to prove the symbols
    resolve at load time; the result of the enumeration is discarded.
    """
    cdef int num = 0
    cdef c.ibv_device **lst = c.ibv_get_device_list(&num)
    if lst is not NULL:
        c.ibv_free_device_list(lst)
    return True
