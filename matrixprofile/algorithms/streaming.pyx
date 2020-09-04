# -*- coding: utf-8 -*-
#cython: boundscheck=False, cdivision=True, wraparound=False

from matrixprofile.cycore import muinvn


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
        
        cdef Py_ssize_t updf = self.rlen - shiftby
        self._ts[:updf] = self._ts[shiftby:self.rlen]
        self._mu[:updf] = self._ts[shiftby:self.rlen]
        self._invn[:updf] = self._invn[shiftby:self.rlen]
        self._r_bwd[:updf] = self._r_bwd[shiftby:self.rlen]
        self._r_fwd[:updf] = self._r_fwd[shiftby:self.rlen]
        self._c_bwd[:updf] = self._c_bwd[shiftby:self.rlen]
        self._c_fwd[:updf] = self._c_fwd[shiftby:self.rlen]
        self.rlen -= shiftby

    cdef append(self, double[::1] dat, Py_ssize_t drop_below=0):
        cdef Py_ssize_t start = 0 if drop_below <= self.minidx else self.minidx - drop_below
        cdef Py_ssize_t updsz = self.rlen - start + dat.shape[0]
        if updsz > self.ts.shape[0]:
            self.resize(2*self._ts.shape[0] if updsz <= 2*self._ts.shape[0] else 2*updsz)
            self.rlen -= start        

cdef class mpstream:

    def __cinit__(self, Py_ssize_t sseqlen, Py_ssize_t minlag, Py_ssize_t maxlag, Py_ssize_t init_buffer_len=4096):
        if sseqlen < 4:
            raise ValueError
        if minlag <= 0:
            raise ValueError
        elif maxlag <= 0:  # should have some default
            raise ValueError
        elif init_buffer_len <= 0:
            raise ValueError
        self.minlag = minlag
        self.maxlag = maxlag
        self.profilelen = 0
 
    cdef append(self, auto_params sect):
        pass
 
        

