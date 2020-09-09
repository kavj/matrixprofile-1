# -*- coding: utf-8 -*-
#cython: boundscheck=False, cdivision=True, wraparound=False

from libc.math cimport sqrt
from matrixprofile.cycore import muinvn

# These are here for convenience purposes for now
# They can be factored out, as this module should primarily contain the buffering specific codes
#

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


cdef windowed_invcent_norm(double[::1] ts, double[::1] mu, double[::1] invn):
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
        invn[i] = 1/sqrt(accum)
    return invn


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
    """ Descriptor for time series matrix profile calculations using method mpx.

    """

    def __cinit__(self, double[::1] ts, Py_ssize_t sseqlen, Py_ssize_t minreslen = 4096, Py_ssize_t offset=0):
        cdef init_len
        if minreslen >= 2 * ts.shape[0]:
            init_len = minreslen
        else:
            init_len = 2 * ts.shape[0]
        cdef ss_buf_ct = init_len - sseqlen + 1
        self.ts = array(shape=(init_len,), itemsize=sizeof(double), format='d')
        self.mu = array(shape=(ss_buf_ct,), itemsize=sizeof(double), format='d')
        self.invn = array(shape=(ss_buf_ct,), itemsize=sizeof(double), format='d')
        self.r_bwd = array(shape=(ss_buf_ct-1,), itemsize=sizeof(double), format='d')
        self.c_bwd = array(shape=(ss_buf_ct-1,), itemsize=sizeof(double), format='d')
        self.r_fwd = array(shape=(ss_buf_ct-1,), itemsize=sizeof(double), format='d')
        self.c_fwd = array(shape=(ss_buf_ct-1,), itemsize=sizeof(double), format='d')
        self.tslen = ts.shape[0]
        self.sseqlen = sseqlen
        self.minidx = offset
        windowed_mean(self.ts, self.mu)


    cdef inline Py_ssize_t sseqct(self):
        return self.tslen - self.sseqlen + 1 if self.tslen >= self.sseqlen else 0


    cdef inline Py_ssize_t signal_len(self):
        return self.minidx + self.tslen


    cdef inline Py_ssize_t signal_sseqct(self):
        return self.signal_len() - self.sseqlen + 1


    cdef inline Py_ssize_t last_abs_idx(self):
        return self.sseqlen + self.minidx 


    cdef row_diffs(self, Py_ssize_t begin=0):
        cdef Py_ssize_t endpos = self.sseqct()-1
        if endpos < 0:
            endpos = 0
        return self.r_bwd[begin:endpos], self.r_fwd[begin:endpos]


    cdef col_diffs(self, Py_ssize_t begin=-1):
        cdef Py_ssize_t endpos = self.sseqct()-1
        if endpos < 0:
            endpos = 0
        return self.c_bwd[begin:endpos], self.c_fwd[begin:endpos]


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
        cdef Py_ssize_t old_ssct = self.sseqct()

        if dropct < old_tslen:
            self.tslen -= dropct
            ts[:self.tslen] = self.ts[dropct:old_tslen]
        else:
            self.tslen = 0
        cdef Py_ssize_t ssct = self.sseqct()
        if dropct < old_ssct:
            mu[:ssct] = self.mu[dropct:old_ssct]
            invn[:ssct] = self.invn[dropct:old_ssct]
        if ssct > 1:
            rbwd[:ssct-1] = self.r_bwd[dropct:old_ssct-1]
            rfwd[:ssct-1] = self.r_fwd[dropct:old_ssct-1]
            cbwd[:ssct-1] = self.c_bwd[dropct:old_ssct-1]
            cfwd[:ssct-1] = self.c_fwd[dropct:old_ssct-1]
        self.ts = ts
        self.mu = mu
        self.invn = invn
        self.r_bwd = rbwd
        self.r_fwd = rfwd
        self.c_bwd = cbwd
        self.c_fwd = cfwd
      

    cdef reserve(self, Py_ssize_t sz):
        if self.ts.shape[0] < sz:
            self.resize(sz)


    cdef repack(self, Py_ssize_t dropct):
        if dropct <= 0:
           raise ValueError
        elif dropct > self.tslen:
            raise ValueError
        cdef Py_ssize_t old_tslen = self.tslen
        cdef Py_ssize_t old_ssct = self.sseqct()
        self.tslen -= dropct
        self.ts[:self.tslen] = self.ts[dropct:old_tslen]
        cdef Py_ssize_t ssct = self.sseqct()
        if ssct > 0:
            self.mu[:ssct] = self.mu[dropct:old_ssct]
            self.invn[:ssct] = self.invn[dropct:old_ssct]
        if ssct > 1:
            self.r_bwd[:ssct-1] = self.r_bwd[dropct:old_ssct-1]
            self.r_fwd[:ssct-1] = self.r_fwd[dropct:old_ssct-1]
            self.c_bwd[:ssct-1] = self.c_bwd[dropct:old_ssct-1]
            self.c_fwd[:ssct-1] = self.c_fwd[dropct:old_ssct-1]


    cdef append(self, double[::1] dat, Py_ssize_t dropct=0):
        if dat.shape[0] == 0:
            raise ValueError
        if dropct > self.tslen or dropct < 0:
            raise ValueError
        cdef Py_ssize_t minsz = self.tslen - dropct + dat.shape[0]
        cdef double[::1] mu_s
        cdef Py_ssize_t strt
        # shift if necessary
        if minsz <= self.ts.shape[0]:
            if dropct > 0:
                self.repack(dropct)
        else:
            self.resize(2 * minsz, dropct)
        cdef old_tslen = self.tslen
        cdef old_ssct = self.sseqct()
        self.tslen += dat.shape[0]
        self.ts[old_tslen:self.tslen] = dat
        cdef Py_ssize_t ssct = self.sseqct()
        cdef Py_ssize_t addct = ssct if old_ssct == 0 else dat.shape[0]
        cdef Py_ssize_t dif_addct = addct - 1 if old_ssct == 0 else addct
        cdef Py_ssize_t tsbeg = 0 if old_ssct == 0 else old_tslen - self.sseqlen + 1
        cdef Py_ssize_t ssbeg = ssct - addct
        cdef Py_ssize_t difbeg = ssct - dif_addct - 1
        cdef Py_ssize_t difct = ssct - 1
        windowed_mean(self.ts[tsbeg:self.tslen], self.mu[tsbeg:ssct])
        mu_s = array(shape=(dif_addct,), itemsize=sizeof(double), format='d')
        windowed_mean(self.ts[tsbeg:self.tslen-1], mu_s[:dif_addct])
        mpx_step_eqns(self.ts[tsbeg:self.tslen],
                      self.mu[ssbeg:ssct],
                      mu_s,
                      self.r_bwd[difbeg:difct],
                      self.c_bwd[difbeg:difct],
                      self.r_fwd[difbeg:difct],
                      self.c_fwd[difbeg:difct])


cdef class AutoMProfile:
    """ auto profile indicates that our comparisons use normalized cross correlation between 2 sections of the same
        time series
    """

    def __cinit__(self, Py_ssize_t sseqlen, Py_ssize_t minsep, Py_ssize_t maxsep, double[::1] ts, Py_ssize_t ts_reserve_len=4096):
        if sseqlen < 4:
            raise ValueError
        if minsep <= 0:
            raise ValueError(f'minsep must be strictly positive, received a value of {minsep}')
        elif maxsep <= 0 or maxsep <= minsep:  # should have some default
            raise ValueError('max separation must be positive and strictly larger than minsep, received minsep:{minsep}, maxsep:{maxep}')
        elif ts_reserve_len <= 0:
            raise ValueError
        elif minsep == maxsep:
            raise ValueError
        self.minsep = minsep
        self.maxsep = maxsep
        self.tsp = AutoParams(ts, sseqlen, ts_reserve_len)


    cdef inline Py_ssize_t bufferlen(self):
        return self.ts.shape[0]


    cdef inline Py_ssize_t profilelen(self):
        return (self.tslen - self.sseqlen + 1 if self.sseqlen <= self.tslen else 0) if self.tslen > self.minsep else 0


    cdef append(self, double[::1] ts):
        cdef Py_ssize_t dropct = 0
        if self.profilelen() >= self.maxsep:
            # drop anything that cannot be compared with the first element of
            pass
        self.ts.append(ts)
        if ts.signal_len() > self.maxsep:
            pass

