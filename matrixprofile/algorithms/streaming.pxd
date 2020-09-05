from cython.view cimport array


# This isn't a formal interface, but I tend to think append, resize, reserve, and repack
# should be public interface components of all streaming buffers, since these describe
# operations that are used for aggregated buffer management. 
#

cdef class auto_params:
    cdef:
        array _ts
        array _mu
        array _invn
        array _r_bwd
        array _c_bwd
        array _r_fwd
        array _c_fwd
        Py_ssize_t tslen # number of valid stored time series entries
        append(self, double[::1] dat, Py_ssize_t drop_below=?)
        resize(self, Py_ssize_t sz, Py_ssize_t drop_below=?)
        reserve(self, Py_ssize_t sz)
        repack(self, Py_ssize_t drop_below=?)
     
    cdef readonly:
        Py_ssize_t minidx
        Py_ssize_t rlen  # abbreviation for retained len
        Py_ssize_t signal_len(self)
        row_diffs(self, Py_ssize_t begin=?)
        col_diffs(self, Py_ssize_t begin=?)
  

cdef class mpstream:
    cdef:
        auto_params tssect
        array _mp
        array _mpi
        append(self, double[::1] ts)

    cdef readonly:
        Py_ssize_t bufferlen
        Py_ssize_t profilelen
        Py_ssize_t sseqlen
        Py_ssize_t minsep
        Py_ssize_t maxsep
   
     
