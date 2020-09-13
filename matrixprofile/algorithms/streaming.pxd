from cython.view cimport array


# This isn't a formal interface, but I tend to think append, resize, reserve, and repack
# should be public interface components of all streaming buffers, since these describe
# operations that are used for aggregated buffer management. 
#


cdef windowed_mean(double [::1] ts, double[::1] mu, Py_ssize_t windowlen)
cdef windowed_invcnorm(double[::1] ts, double[::1] mu, double[::1] sig, Py_ssize_t windowlen)
cdef normalize_one(double[::1] out, double[::1] ts, double mu, double sig)
cdef crosscov(double[::1] out, double[::1] ts, double[::1] mu, double[::1] sig, double[::1] cmpseq)


cdef class TSParams:
    cdef:
        array _ts
        array _mu
        array _invn
        inline double[::1] ts(self)
        inline double[::1] mu(self)
        inline double[::1] invn(self)
        dropleading(self, Py_ssize_t dropct)
        repack(self)
        append(self, double[::1] dat, Py_ssize_t dropct=?)
        resize_buffer(self, Py_ssize_t updatedlen)
        Py_ssize_t max_append_end(self)

    cdef readonly:
        inline Py_ssize_t globmax_ts_index(self)
        inline Py_ssize_t globmax_ss_index(self)
        Py_ssize_t globmin_index                      
        Py_ssize_t eltct 
        inline Py_ssize_t subseqct(self)
  

cdef class MProfile:
    cdef:
        TSParams ts
        array _mp
        array _mpi
        append(self, double[::1] ts)

    cdef readonly:
        inline Py_ssize_t bufferlen(self)
        inline Py_ssize_t profilelen(self)
        Py_ssize_t sseqlen
        Py_ssize_t minsep
        Py_ssize_t maxsep
   
     
