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


cdef windowed_cent_norm(double[::1] ts, double[::1] mu, double[::1] sig):
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


cdef normalize_one(double[::1] out, double[::1] ts, double mu, double sig):
    cdef Py_ssize_t i,j
    for i in range(out.shape[0]):
        out[i] = (ts[i] - mu) / sig


cdef crosscov(double[::1] out, double[::1] ts, double[::1] mu, double[::1] sig, double[::1] cmpseq):
    cdef Py_ssize_t sseqct = out.shape[0]
    cdef double accum, m_
    if sseqct != mu.shape[0] or sseqct != sig.shape[0]:
        raise ValueError
    elif cmpseq.shape[0] != ts.shape[0] - sseqct + 1:
        raise ValueError
    cdef Py_ssize_t i, j
    for i in range(sseqct):
        accum = 0.0
        m_ = mu[i]
        for j in range(cmpseq.shape[0]):
            accum += (ts[i + j] - m_) * cmpseq[j]
        out[i] = accum


# factored out from mpx method, should probably go somewhere else
cdef mpx_step_eqns(double[::1] ts, double[::1] mu, double[::1] mu_s, double[::1] rbwd, double[::1] cbwd, double[::1] rfwd, double[::1] cfwd):    
    cdef Py_ssize_t sseqct = mu.shape[0]
    cdef Py_ssize_t sseqlen = ts.shape[0] - mu.shape[0] + 1
    cdef Py_ssize_t i

    for i in range(sseqct-1):
        rbwd[i] = ts[i] - mu[i]
        cbwd[i] = ts[i] - mu_s[i+1]
        rfwd[i] = ts[i+sseqlen] - mu[i+1]
        cfwd[i] = ts[i+sseqlen] - mu_s[i+1]


cdef class AutoParams:

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


    cdef inline Py_ssize_t sseqct(self):
        return self.tslen - self.sseqlen + 1
    
    cdef inline Py_ssize_t total_signal_len(self):
        return self.minidx + self.tslen


    cdef inline Py_ssize_t total_sseq_ct(self):
        return self.total_signal_len() - self.sseqlen + 1


    cdef inline Py_ssize_t last_abs_idx(self):
        return self.sseqlen + self.minidx 


    cdef row_diffs(self, Py_ssize_t begin=-1):
        if begin == -1:
            return self.r_bwd.get_memview(), self.r_fwd.get_memview()
        elif begin < self.minidx:
            raise ValueError('index too low')
        elif begin >= self.minidx + self.ret_tslen:
            raise ValueError('index too high')
        cdef Py_ssize_t begpos = begin - self.minidx
        return self.r_bwd[begpos:self.sseqct-1], self.r_fwd[begpos:self.sseqct-1]

    
    cdef col_diffs(self, Py_ssize_t begin=-1):
        if begin == -1:  
            return self.c_bwd.get_memview(), self.c_fwd.get_memview()
        elif begin < self.minidx:
            raise ValueError('index too low')
        elif begin >= self.minidx + self.tslen:
            raise ValueError('index too high')
        cdef Py_ssize_t begpos = begin - self.minidx
        return self.c_bwd[begpos:self.sseqct-1], self.c_fwd[begpos:self.sseqct-1]


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


    cdef _append_inplace(self, double[::1] dat):
        if dat.shape[0] == 0:
            raise ValueError
        # cdef Py_ssize_t old_tslen = self.tslen
        # cdef Py_ssize_t old_sseqct = self.sseqct()
        cdef Py_ssize_t addct
        cdef old_tslen = self.tslen
        cdef old_ssct = self.sseqct()
        if old_tslen < self.sseqlen:
            addct = self.tslen - self.sseqlen + 1 if self.tslen >= self.sseqlen else 0
        else:
            addct = dat.shape[0]
        self.tslen = old_tslen + dat.shape[0]
        self.ts[old_tslen:self.tslen] = dat
        cdef Py_ssize_t dif_addct = addct if old_ssct > 0 else addct - 1
        moving_mean(ts[self.tslen-ss_add_ct-self.sseqlen+1:self.tslen], self.mu[old_sseqct:self.sseqct])
        #

    cdef append(self, double[::1] dat, Py_ssize_t dropct=0):
        if dat.shape[0] == 0:
            raise ValueError
        if dropct > self.tslen or dropct < 0:
            raise ValueError
        cdef Py_ssize_t minsz = self.tslen - dropct + dat.shape[0]
        cdef double[::1] mu_s
        cdef Py_ssize_t strt
        # shift if necessary
        if minsz < self.ts.shape[0]:
            if dropct > 0:
                self.repack(dropct)
        else:
            self.resize(2*minsz, dropct)
        cdef Py_ssize_t addct
        cdef old_tslen = self.tslen
        cdef old_ssct = self.sseqct()
        if old_tslen < self.sseqlen:
            addct = self.tslen - self.sseqlen + 1 if self.tslen >= self.sseqlen else 0
        else:
            addct = dat.shape[0]
        self.tslen = old_tslen + dat.shape[0]
        self.ts[old_tslen:self.tslen] = dat
        cdef Py_ssize_t dif_addct = addct if old_ssct > 0 else addct - 1
        windowed_mean(ts[self.tslen-ss_addct-self.sseqlen+1:self.tslen], self.mu[old_sseqct:self.sseqct])
        mu_s = array(shape=(dif_add_ct,), itemsize=sizeof(double), format='d')
        windowed_mean(self.ts[self.tslen-ss_addct:self.tslen-1], mu_s[old_sseqct:self.sseqct-1])
        # update difference equations
        

cdef class MpStream:

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
         

