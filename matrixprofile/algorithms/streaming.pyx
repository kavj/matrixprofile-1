# -*- coding: utf-8 -*-
#cython: boundscheck=True, cdivision=True, wraparound=True
from cython.view cimport array
from libc.math cimport sqrt
from matrixprofile.cycore import muinvn

# These are here for convenience purposes for now
# They can be factored out, as this module should primarily contain the buffering specific codes
#

cdef windowed_mean(double [::1] ts, double[::1] mu, Py_ssize_t windowlen): 
    if ts.shape[0] < windowlen:
        raise ValueError(f"Window length exceeds the number of elements in the time series")
    # safer to test this explicitly than infer the last parameter 
    cdef Py_ssize_t windowct = ts.shape[0] - windowlen + 1
    if windowct != mu.shape[0]:
        raise ValueError(f"subsequence count {windowct} does not match output shape {mu.shape[0]}")
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
    for i in range(windowlen, ts.shape[0]):
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


cdef windowed_invcnorm(double[::1] ts, double[::1] mu, double[::1] invn, Py_ssize_t windowlen):
    cdef Py_ssize_t windowct = ts.shape[0] - windowlen + 1
    if not (windowct == mu.shape[0] == invn.shape[0]):
        raise ValueError(f"window count {windowct} does not match output shapes {mu.shape[0]} and {invn.shape[0]}") 

    cdef double accum = 0.0
    cdef double m_
    cdef Py_ssize_t i, j

    for i in range(windowct):
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


cdef mpx_difeq(double [::1] out, double[::1] ts, double[::1] mu):
    if not (ts.shape[0] == mu.shape[0] == out.shape[0]):
        raise ValueError(f'time series of shape {ts.shape[0]} is incompatible with mean vector mu of shape {mu.shape[0]} and output shape {out.shape[0]}')
    cdef Py_ssize_t i
    for i in range(ts.shape[0]):
        out[i] = ts[i] - mu[i]


cdef class TSParams:
    """ Descriptor for time series matrix profile calculations using method mpx.

    """

    def __cinit__(self, Py_ssize_t subseqlen, Py_ssize_t subseqbuflen):
        cdef Py_ssize_t eltsz = sizeof(double)
        cdef Py_ssize_t tsbuflen = subseqbuflen + subseqlen - 1
        self._ts = array(shape=(tsbuflen,), itemsize=eltsz, format='d')
        self._mu = array(shape=(subseqbuflen,), itemsize=eltsz, format='d')
        self._invn = array(shape=(subseqbuflen,), itemsize=eltsz, format='d')
        self.subseqlen = subseqlen
        self.globmin_index = 0
        self.beginpos = 0
        self.eltct = 0  

    cdef inline Py_ssize_t subseqct(self):
        return self.eltct - self.sseqlen + 1 if self.eltct > self.sseqlen else 0

    cdef inline double[::1] ts(self):
        return self._ts[self.beginpos:self.beginpos + self.eltct]

    cdef inline double[::1] mu(self):
        return self._mu[self.beginpos:self.beginpos+self.subseqct()]

    cdef inline double[::1] invn(self):
        return self._invn[self.beginpos:self.beginpos+self.subseqct()]

    cdef Py_ssize_t max_append_end(self):
        return self._ts.shape[0] - self.beginpos - self.eltct

    cdef dropleading(self, Py_ssize_t dropct):
        if dropct > self.elct:
            raise ValueError(f"cannot drop {dropct} elements from {self.eltct} elements")
        self.minidx += dropct
        self.beginpos += dropct
        self.eltct -= dropct

    cdef repack(self):
        cdef Py_ssize_t ssct = self.subseqct()
        self._ts[:self.eltct] = self.ts()
        self._mu[:ssct] = self.mu()
        self._invn[:ssct] = self.invn()

    cdef resize_buffer(self, Py_ssize_t updatedlen):
        if updatedlen < self.eltct:
            raise ValueError("updated buffer size is too small to accommodate all elements")
        elif updatedlen == self._ts.shape[0]:
            return
        
        cdef double[::1] ts_ = self.ts()
        cdef double[::1] mu_ = self.mu()
        cdef double[::1] invn_ = self.invn()
       
        cdef Py_ssize_t updssbuflen = updatedlen - self.subseqlen + 1 if updatedlen >= self.subseqlen + 1 else 0
        self._ts = array(shape=(updatedlen,), itemsize=sizeof(double), format='d')
        self._mu = array(shape=(updssbuflen,), itemsize=sizeof(double), format='d')
        self._invn = array(shape=(updssbuflen,), itemsize=sizeof(double), format='d')
        
        self.beginpos = 0
        self.ts()[:] = ts_
        self.mu()[:] = mu_
        self.invn()[:] = invn_

    cdef inline Py_ssize_t globmax_ts_index(self):
        return self.minindex + self.eltct

    cdef inline Py_ssize_t globmax_ss_index(self):
        return self.minindex + self.subseqct()

    cdef append(self, double[::1] dat, Py_ssize_t dropct=0):
        if dat.shape[0] == 0:
            raise ValueError('Cannot append from an empty block')
        elif not (0 < dropct < self.buffered_len()):
            raise ValueError(f'drop count {dropct} is incompatible with buffered time series length {self.buffered_len()}')
        cdef Py_ssize_t updsz = self.buffered_len() - dropct + dat.shape[0]
        if updsz > self._ts.shape[0]:
            raise ValueError(f'buffer of size {self._ts.shape[0]} is incompatible with required size {updsz}')
        if self._ts.shape[0] - self.beginpos < self.updsz:
            self.repack() #update
        else:
            self.beginpos += dropct
        cdef updend = self.endpos + dat.shape[0]
        self._ts[self.endpos : updend] = dat[:]
        self.endpos = updend
        windowed_mean(self._ts, self._mu, self.subseqlen) 


cdef class MProfile:
    """ auto profile indicates that our comparisons use normalized cross correlation between 2 sections of the same
        time series
    """

    def __cinit__(self, Py_ssize_t sseqlen, Py_ssize_t minsep, Py_ssize_t maxsep, double[::1] ts, Py_ssize_t ts_reserve_len=4096):
        if sseqlen < 4:
            raise ValueError('subsequence lengths below 4 are not supported')
        if minsep <= 0:
            raise ValueError(f'negative minsep {minsep} is unsupported')
        elif maxsep <= 0 or maxsep <= minsep:  # should have some default
            raise ValueError('max separation must be positive and strictly larger than minsep, received minsep:{minsep}, maxsep:{maxep}')
        elif ts_reserve_len <= 0:
            raise ValueError
        elif minsep == maxsep:
            raise ValueError
        self.minsep = minsep
        self.maxsep = maxsep
        self.tsp = TSParams(ts, sseqlen, ts_reserve_len)

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
            pass
        self.ts.append(ts)
        if ts.signal_len() > self.maxsep:
            pass

