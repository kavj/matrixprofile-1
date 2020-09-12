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
        inline Py_ssize_t buffered_len(self)
        inline Py_ssize_t buffered_subseqct(self)
        repack(self, Py_ssize_t dropct=?)
        append(self, double[::1] dat, Py_ssize_t dropct=?)
     
    cdef readonly:
        inline Py_ssize_t maxindex(self)
        inline Py_ssize_t maxsubseqindex(self)
        Py_ssize_t minindex                      
        inline Py_ssize_t buffered_len(self)        
        inline Py_ssize_t buffered_subseqct(self)
  

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
   
     
