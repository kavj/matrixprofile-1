# -*- coding: utf-8 -*-
#cython: boundscheck=False, cdivision=True, wraparound=False

from libc.math cimport sqrt
from matrixprofile.cycore import muinvn


cdef windowed_mean(double [::1] ts, double[::1] mu): 
    cdef Py_ssize_t tslen = ts.shape[0]
    cdef Py_ssize_t windowlen = tslen - mu.shape[0] + 1
    if tslen < windowlen:
        raise ValueError, "Window length exceeds the number of elements in the time series"
    cdef double accum = ts[0];
    cdef double resid = 0;
    cdef Py_ssize_t i
    cdef double m, n, p, q, r, s
    for i in range(1, windowlen):
        m = ts[i]
        p = accum
        accum += m
        q = accum - p
        resid += ((p - (accum - q)) + (m - q))
    mu[0] = (accum + resid) / windowlen
    for i in range(windowlen, tslen):
        m = ts[i - windowlen]
        n = ts[i]
        p = accum - m
        q = p - accum
        r = resid + ((accum - (p - q)) - (m + q))
        accum = p + n
        s = accum - p
        resid = r + ((p - (accum - s)) + (n - s))
        mu[i - windowlen + 1] = (accum + resid) / windowlen
    return mu


cdef windowed_cnorm(double [::1] ts, double[::1] mu, double[::1] sig):
    cdef Py_ssize_t tslen = ts.shape[0]
    cdef Py_ssize_t windowcount = mu.shape[0]
    if windowcount < 1 or windowcount > tslen: 
        raise ValueError

    cdef Py_ssize_t windowlen = tslen - windowcount + 1
    cdef double accum = 0
    cdef double m_
    cdef Py_ssize_t i, j

    for i in range(windowcount):
        m_ = mu[i]
        accum = 0
        for j in range(i, i + windowlen):
            accum += (ts[j] - m_)**2
        sig[i] = sqrt(accum)
    return sig


# Todo: refactor all mpx logic and moment calculations to a sharable C interface that does not use numpy's namespace

cdef normalize(double [:, ::1] ss, double[::1] ts, double[::1] mu, double[::1] sig, Py_ssize_t start):
    cdef Py_ssize_t windowcount = ss.shape[0]
    cdef Py_ssize_t windowlen = ss.shape[1]
    windowcount = mu.shape[0]
    windowlen = ts.shape[0] - mu.shape[0] + 1
    # ts, mu, and sig must represent the same number of available subsequences
    # start dictates the starting index within these, as this avoids redundant slicing of multiple components
    # ss.shape[0] exposes the number of subsequences that should be normalized
    # 

    cdef Py_ssize_t i,j
    cdef double m_, s_
    for i in range(windowcount):
        accum = 0.0
        m_ = mu[i]
        s_ = sig[i]
        for j in range(windowlen):
            ss[i, j] = (ts[i + j] - m_) / s_


# factored out from mpx method, should probably go somewhere else
cdef mpx_step_eqns(double [::1] ts, double [::1] mu, double[::1] mu_s, double[::1] rbwd, double[::1] cbwd, double[::1] rfwd, double[::1] cfwd):    
    cdef Py_ssize_t sseqct = mu.shape[0]
    cdef Py_ssize_t sseqlen = ts.shape[0] - mu.shape[0] + 1
    cdef Py_ssize_t i

    for i in range(sseqct-1):
        rbwd[i] = ts[i] - mu[i]
        cbwd[i] = ts[i] - mu_s[i+1]
        rfwd[i] = ts[i+sseqlen] - mu[i+1]
        cfwd[i] = ts[i+sseqlen] - mu_s[i+1]


cdef class auto_params:

    def __cinit__(self, double[::1] ts, Py_ssize_t sseqlen):
        cdef init_buffer_len = 4096 if ts.shape[0] <= 4096 else ts.shape[0]
        self._ts = array(shape=(init_buffer_len,), itemsize=sizeof(double), format='d')
        self._mu = array(shape=(init_buffer_len,), itemsize=sizeof(double), format='d')
        self._invn = array(shape=(init_buffer_len,), itemsize=sizeof(double), format='d')
        self._r_bwd = array(shape=(init_buffer_len-1,), itemsize=sizeof(double), format='d')
        self._c_bwd = array(shape=(init_buffer_len-1,), itemsize=sizeof(double), format='d')
        self._r_fwd = array(shape=(init_buffer_len-1,), itemsize=sizeof(double), format='d')
        self._c_fwd = array(shape=(init_buffer_len-1,), itemsize=sizeof(double), format='d')
        self.rlen = ts.shape[0]
 
    cdef row_diffs(self, Py_ssize_t begin=-1):
        if begin == -1:
            return self._r_bwd.get_memview(), self._r_fwd.get_memview()
        elif begin < self.minidx:
            raise ValueError('index too low')
        elif begin >= self.minidx + self.rlen:
            raise ValueError('index too high')
        cdef Py_ssize_t beginpos = begin - self.minidx
        return self._r_bwd[beginpos:self.rlen-1], self._r_fwd[beginpos:self.rlen-1]
    
    cdef col_diffs(self, Py_ssize_t begin=-1):
        if begin == -1:  
            return self._c_bwd.get_memview(), self._c_fwd.get_memview()
        elif begin < self.minidx:
            raise ValueError('index too low')
        elif begin >= self.minidx + self.rlen:
            raise ValueError('index too high')
        cdef Py_ssize_t beginpos = begin - self.minidx
        return self._c_bwd[beginpos:self.rlen-1], self._c_fwd[beginpos:self.rlen-1]

    cdef Py_ssize_t signal_len(self):
        return self.minidx + self.rlen
        
    cdef resize(self, Py_ssize_t sz, Py_ssize_t drop_below=0):
        if drop_below < 0:
            raise ValueError
        cdef strtpos = drop_below - self.minidx if drop_below > self.minidx else 0
        cdef retainct = self.rlen - strtpos if strtpos <= self.rlen else 0
        if retainct > sz:
            raise ValueError('size is too small to accommodate the number of retained entries')
        cdef ts = array(shape=(sz,), itemsize=sizeof(double), format='d')
        cdef mu = array(shape=(sz,), itemsize=sizeof(double), format='d')
        cdef invn = array(shape=(sz,), itemsize=sizeof(double), format='d')
        cdef rbwd = array(shape=(sz,), itemsize=sizeof(double), format='d')
        cdef rfwd = array(shape=(sz,), itemsize=sizeof(double), format='d')
        cdef cbwd = array(shape=(sz,), itemsize=sizeof(double), format='d')
        cdef cfwd = array(shape=(sz,), itemsize=sizeof(double), format='d')  
        if retainct > 0:
            ts[:retainct] = self._ts[strtpos:self.rlen]
            mu[:retainct] = self._mu[strtpos:self.rlen]
            invn[:retainct] = self._invn[strtpos:self.rlen]
        if retainct > 1:
            rbwd[:retainct-1] = self._r_bwd[strtpos:]
            rfwd[:retainct-1] = self._r_fwd[strtpos:]
            cbwd[:retainct-1] = self._c_bwd[strtpos:]
            cfwd[:retainct-1] = self._c_fwd[strtpos:]
        self._ts = ts
        self._mu = mu
        self._invn = invn
        self._r_bwd = rbwd
        self._r_fwd = rfwd
        self._c_bwd = cbwd
        self._c_fwd = cfwd
        self.rlen = retainct
      
    cdef reserve(self, Py_ssize_t sz):
        if self._ts.shape[0] < sz:
            self.resize(sz)

    cdef repack(self, Py_ssize_t drop_below=0):
        cdef Py_ssize_t shiftby = drop_below - self.minidx
        if shiftby < 0:
            return
        cdef Py_ssize_t updf = self.rlen - shiftby if shiftby < self.rlen else 0
        if updf < self.rlen:
            self._ts[:updf] = self._ts[shiftby:self.rlen]
            self._mu[:updf] = self._ts[shiftby:self.rlen]
            self._invn[:updf] = self._invn[shiftby:self.rlen]
            self._r_bwd[:updf] = self._r_bwd[shiftby:self.rlen]
            self._r_fwd[:updf] = self._r_fwd[shiftby:self.rlen]
            self._c_bwd[:updf] = self._c_bwd[shiftby:self.rlen]
            self._c_fwd[:updf] = self._c_fwd[shiftby:self.rlen]
        self.rlen = updf

    cdef append(self, double[::1] dat, Py_ssize_t drop_below=0):
        cdef Py_ssize_t startpos = 0 if drop_below <= self.minidx else self.minidx - drop_below
        cdef Py_ssize_t updsz = self.rlen - startpos + dat.shape[0]
        if updsz > self.ts.shape[0]:
            self.resize(2*self._ts.shape[0] if updsz <= 2*self._ts.shape[0] else 2*updsz, drop_below)
        # append here


cdef class mpstream:

    def __cinit__(self, Py_ssize_t sseqlen, Py_ssize_t minsep, Py_ssize_t maxsep, Py_ssize_t init_buffer_len=4096):
        if sseqlen < 4:
            raise ValueError
        if minsep <= 0:
            raise ValueError(f'minsep must be strictly positive, received a value of {minsep}')
        elif maxsep <= 0 or maxsep <= minsep:  # should have some default
            raise ValueError('max separation must be positive and strictly larger than minsep, received minsep:{minsep}, maxsep:{maxep}')
        elif init_buffer_len <= 0:
            raise ValueError
        self.minsep = minsep
        self.maxsep = maxsep
        self.profilelen = 0
 
    cdef append(self, auto_params sect):
        pass
 

