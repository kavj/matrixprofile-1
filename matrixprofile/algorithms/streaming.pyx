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


cdef windowed_cent_norm(double [::1] ts, double[::1] mu, double[::1] sig):
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

    def __cinit__(self, double[::1] ts, Py_ssize_t sseqlen, Py_ssize_t offset=0):
        cdef init_buffer_len = 4096 if ts.shape[0] <= 4096 else 2 * ts.shape[0]
        cdef ss_buffer_len = init_buffer_len - sseqlen + 1
        self.ts = array(shape=(init_buffer_len,), itemsize=sizeof(double), format='d')
        self.mu = array(shape=(ss_buffer_len,), itemsize=sizeof(double), format='d')
        self.invn = array(shape=(ss_buffer_len,), itemsize=sizeof(double), format='d')
        self.r_bwd = array(shape=(ss_buffer_len-1,), itemsize=sizeof(double), format='d')
        self.c_bwd = array(shape=(ss_buffer_len-1,), itemsize=sizeof(double), format='d')
        self.r_fwd = array(shape=(ss_buffer_len-1,), itemsize=sizeof(double), format='d')
        self.c_fwd = array(shape=(ss_buffer_len-1,), itemsize=sizeof(double), format='d')
        self.tslen = ts.shape[0]
        self.sseqct = ts.shape[0] - sseqlen + 1
        self.sseqlen = sseqlen
        self.minidx = offset
 
    
    cdef inline Py_ssize_t total_signal_len(self):
        return self.minidx + self.tslen


    cdef inline Py_ssize_t total_sseq_ct(self):
        return self.total_signal_len() - self.sseqlen + 1
 
    cdef inline Py_ssize_t last_abs_idx(self):
        return self.sseqlen + self.minidx 


    cdef row_diffs(self, Py_ssize_t begin=-1):
        if begin == -1:
            return self._r_bwd.get_memview(), self._r_fwd.get_memview()
        elif begin < self.minidx:
            raise ValueError('index too low')
        elif begin >= self.minidx + self.ret_tslen:
            raise ValueError('index too high')
        cdef Py_ssize_t begpos = begin - self.minidx
        return self._r_bwd[begpos:self.sseqct-1], self._r_fwd[begpos:self.sseqct-1]

    
    cdef col_diffs(self, Py_ssize_t begin=-1):
        if begin == -1:  
            return self._c_bwd.get_memview(), self._c_fwd.get_memview()
        elif begin < self.minidx:
            raise ValueError('index too low')
        elif begin >= self.minidx + self.tslen:
            raise ValueError('index too high')
        cdef Py_ssize_t begpos = begin - self.minidx
        return self._c_bwd[begpos:self.sseqct-1], self._c_fwd[begpos:self.sseqct-1]


    cdef resize(self, Py_ssize_t sz, Py_ssize_t dropct=0):
        if dropct > self.tslen:
            raise ValueError
        elif self.tslen - dropct > sz:
            raise ValueError
        cdef Py_ssize_t ss_sz = sz - self.sseqlen + 1 if sz >= self.sseqlen else 0
        cdef Py_ssize_t dif_sz = ss_sz - 1 if ss_sz > 0 else 0
        cdef array ts = array(shape=(sz,), itemsize=sizeof(double), format='d')
        cdef array mu = array(shape=(ss_sz,), itemsize=sizeof(double), format='d')
        cdef array invn = array(shape=(ss_sz,), itemsize=sizeof(double), format='d')
        cdef array rbwd = array(shape=(dif_sz,), itemsize=sizeof(double), format='d')
        cdef array rfwd = array(shape=(dif_sz,), itemsize=sizeof(double), format='d')
        cdef array cbwd = array(shape=(dif_sz,), itemsize=sizeof(double), format='d')
        cdef array cfwd = array(shape=(dif_sz,), itemsize=sizeof(double), format='d')  
        
        cdef Py_ssize_t old_tslen = self.tslen        
        cdef Py_ssize_t old_sseqct = self.sseqct

        if dropct < old_tslen:
            self.tslen -= dropct
            ts[:self.tslen] = self.ts[dropct:old_tslen]
        else:
            self.tslen = 0
        if dropct < self.sseqct:
            self.sseqct -= dropct
            mu[:self.sseqct] = self.mu[dropct:old_sseqct]
            invn[:self.sseqct] = self.invn[dropct:old_sseqct]
        else:
            self.sseqct = 0
        if self.sseqct > 1:
            rbwd[:self.sseqct-1] = self.r_bwd[dropct:old_sseqct-1]
            rfwd[:self.sseqct-1] = self.r_fwd[dropct:old_sseqct-1]
            cbwd[:self.sseqct-1] = self.c_bwd[dropct:old_sseqct-1]
            cfwd[:self.sseqct-1] = self.c_fwd[dropct:old_sseqct-1]
        self.ts = ts
        self.mu = mu
        self.invn = invn
        self.r_bwd = rbwd
        self.r_fwd = rfwd
        self.c_bwd = cbwd
        self.c_fwd = cfwd
      

    cdef reserve(self, Py_ssize_t sz):
        if self._ts.shape[0] < sz:
            self.resize(sz)


    cdef repack(self, Py_ssize_t dropct):
        if dropct <= 0:
           raise ValueError
        elif dropct > self.tslen:
            raise ValueError
        cdef Py_ssize_t old_tslen = self.tslen
        cdef Py_ssize_t old_sseqct = self.sseqct
        if dropct >= self.sseqct:
            self.tslen = self.tslen - dropct if self.tslen > dropct else 0
            self.sseqct = 0
            self.minidx += dropct
            return
        self.ts[:self.tslen] = self.ts[dropct:old_tslen]
        if self.sseqct > 0:
            self.mu[:self.sseqct] = self.mu[dropct:old_sseqct]
            self.invn[:self.sseqct] = self.invn[dropct:old_sseqct]
        if self.sseqct > 1:
            self.r_bwd[:self.sseqct-1] = self.r_bwd[dropct:old_sseqct-1]
            self.r_fwd[:self.sseqct-1] = self.r_fwd[dropct:old_sseqct-1]
            self.c_bwd[:self.sseqct-1] = self.c_bwd[dropct:old_sseqct-1]
            self.c_fwd[:self.sseqct-1] = self.c_fwd[dropct:old_sseqct-1]


    cdef append(self, double[::1] dat, Py_ssize_t dropct=0):
        if dropct > self.tslen or dropct < 0:
            raise ValueError
        cdef Py_ssize_t minsz = self.tslen - dropct + dat.shape[0]
        if minsz > self.ts.shape[0]:
            self.resize(2*minsz, dropct)
        elif dropct > 0:
            self.repack(dropct)
        if minsz > 0:
            self.ts[self.tslen:self.tslen+dat.shape[0]] = dat
            windowed_mean(self.ts[self.tslen:self.tslen+dat.shape[0]], self.sseqlen) 
            self.tslen += self.ts.shape[0]


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
 

    cdef append(self, double[::1] ts):
        self.ts.append(ts)
         

