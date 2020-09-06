from cython.view cimport array


# This isn't a formal interface, but I tend to think append, resize, reserve, and repack
# should be public interface components of all streaming buffers, since these describe
# operations that are used for aggregated buffer management. 
#


cdef windowed_mean(double [::1] ts, double[::1] mu)
cdef windowed_invcent_norm(double[::1] ts, double[::1] mu, double[::1] sig)
cdef normalize_one(double[::1] out, double[::1] ts, double mu, double sig)
cdef normalize_one(double[::1] out, double[::1] ts, double mu, double sig)
cdef crosscov(double[::1] out, double[::1] ts, double[::1] mu, double[::1] sig, double[::1] cmpseq)
cdef mpx_step_eqns(double[::1] ts, double[::1] mu, double[::1] mu_s, double[::1] rbwd, double[::1] cbwd, double[::1] rfwd, double[::1] cfwd)


cdef class AutoParams:
    cdef:
        array ts
        array mu
        array invn
        array r_bwd
        array c_bwd
        array r_fwd
        array c_fwd
        append(self, double[::1] dat, Py_ssize_t dropct=?)
        resize(self, Py_ssize_t sz, Py_ssize_t dropct=?)
        reserve(self, Py_ssize_t sz)
        repack(self, Py_ssize_t dropct)
        inline Py_ssize_t last_abs_idx(self)
        inline Py_ssize_t sseqct(self)
     
    cdef readonly:
        Py_ssize_t minidx                  # first index of the in memory portion
        Py_ssize_t total_signal_len(self)  # total length of the signal at this point, including out of memory sections
        Py_ssize_t total_sseq_ct(self)     # total number of subsequences, including the out of memory portion
        Py_ssize_t tslen                   # number of time series elements retained in memory
        row_diffs(self, Py_ssize_t begin=?)
        col_diffs(self, Py_ssize_t begin=?)
  

cdef class MpStream:
    cdef:
        AutoParams tsp
        array _mp
        array _mpi
        append(self, double[::1] ts)

    cdef readonly:
        inline Py_ssize_t bufferlen(self)
        inline Py_ssize_t profilelen
        Py_ssize_t sseqlen
        Py_ssize_t minsep
        Py_ssize_t maxsep
   
     
