cimport numpy as np


cpdef compute_cross_cov(cc, ts, mu, cmpseq)

cpdef compute_self_compare(mp, mpi, cov, df, dg, sig, w, minlag, index_offset=*)

cpdef compute_ab_compare(mp_a, mp_b, mpi_a, mpi_b, cov, df_a, df_b, dg_a, dg_b, sig_a, sig_b, offset_a=*, offset_b=*)

cdef void difference_equations(double[::1] df, double[::1] dg, double[::1] ts, double[::1] mu, Py_ssize_t w) nogil

cdef void cross_cov(double[::1] out, double[::1] ts, double[::1] mu, double[::1] cmpseq) nogil

cdef void self_compare(double[::1] mp, np.int_t[::1] mpi, double [::1] cov, double[::1] df, double[::1] dg, double[::1] sig, Py_ssize_t subseqlen, Py_ssize_t minsep, Py_ssize_t index_offset) nogil
    
cdef void ab_compare(double[::1] mp_a, double[::1] mp_b, np.int_t[::1] mpi_a, np.int_t[::1] mpi_b, double[::1] cov, double[::1] df_a, double[::1] df_b, double[::1] dg_a, double[::1] dg_b, double[::1] sig_a, double[::1] sig_b, Py_ssize_t offset_a, Py_ssize_t offset_b) nogil
