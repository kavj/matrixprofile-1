import numpy as np
from collections import namedtuple


class BufferedArray:
    """ an extremely basic buffered array """

    def __init__(self, size, dtype='d', minindex=0):
        self.size = size
        self.minindex = minindex
        self._data = np.empty(size, dtype=dtype)
        self.count = 0
        self.beginpos = 0

    @property
    def data(self):
        return self._data[self.beginpos:self.beginpos + self.count]

    @property
    def maxindex(self):
        """ the absolute time based index of the last buffer element in memory"""
        return self.minindex + self.count - 1 if self.count != 0 else None

    @property
    def maxfill(self):
        """ number of unused entries"""
        return self._data.shape[0] - self.count

    @property
    def maxappend(self):
        """ number of unused entries located at the end of the buffer """
        return self._data.shape[0] - self.count - self.beginpos

    # dunder methods are added for debugging convenience.
    # Be warned, these create views on every call

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        """ number of in memory elements
            note: len is supposed to match the number of elements returned by iter
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def normalize_buffer(self):
        """ normalize buffer layout so that retained data spans the positional range
            0 to size - 1.
            Since this is an inplace op, it can invalidate python iterators to this object.
            It is intentionally left as an explicit op.
        """
        if self.count > 0 and self.beginpos != 0:
            self._data[:self.count] = self.data
            self.beginpos = 0

    def fill(self, x, count):
        if count > self.maxfill:
            raise ValueError
        elif count > self.maxappend:
            self.normalize_buffer()
        self.count += count
        self.data[-count:] = x

    def shiftby(self, count, normalize=False, fillval=None):
        """ discard ct elements from memory starting from the current beginning of the in memory
            portion
        """
        if count > self.size:
            raise ValueError('cannot shift by more than total buffer length')
        elif count > self.count:
            if fillval is None:
                raise ValueError('cannot shift by an amount greater than the number of live elements without a '
                                 'fill value')
            else:
                self.count = count - self.count
                self.beginpos = 0
                self.data[:] = fillval
                self.minindex += self.count
        elif count == self.count:
            self.beginpos = 0
            self.minindex += self.count
            self.count = 0
        else:
            self.beginpos += count
            self.count -= count
            if normalize:
                self.normalize_buffer()
        # This is needed for cases aggregating buffers of different lengths. In particular, if we have a time series
        # and a windowed mean function, where the windowed mean function is over full windows of k elements,
        # a shift by m: length(timeseries) - k + 1 < m < length(timeseries) we would be left with 0 live elements in
        # our mean buffer but length(timeseries) - m elements in our time series. In both cases we need to shift the
        # index by m in order to correctly maintain consistency.
        self.minindex += count

    def resize(self, sz):
        """ resize underlying buffer, raise an error if it would truncate live data """
        if sz < self.count:
            raise ValueError(f'buffer size {sz} is too small to accommodate {self.count} live elements')
        dat = self.data
        self._data = np.empty(sz, dtype=self._data.dtype)
        self.data[:] = dat

    def fillable(self, sz, allow_resize=True):
        """ increment the number of live elements by sz without initialization and return a view
            to those elements. If allow_resize is True, then the underlying buffer will be resized
            if necessary. If the buffer does not have sufficient space at the end of the array, it will
            shift all live data to the beginning of the array.
        """
        if sz > self.maxfill:
            if allow_resize:
                if 2*self.size >= self.count + sz:
                    self.resize(2*self.size)
                else: # kick this back if it could mess up size padding. 
                    raise RuntimeError("Cannot implicitly resize by more than a factor of 2")
            else:
                raise ValueError
        elif sz > self.maxappend:
            self.normalize_buffer()
        self.count += sz
        return self.data[-sz:]         

    def append(self, dat, allow_resize=True):
        """ append to buffer """
        reqspace = dat.size
        if self.maxfill < reqspace:
            if allow_resize:
                self.resize(2*(self.count + reqspace))
            else:
                raise ValueError(f'appending {dat.size} elements would overflow available buffer space: {self.maxfill}')
        elif self.maxappend < reqspace:
            self.normalize_buffer()
        oldct = self.count
        self.count += reqspace 
        self.data[oldct:] = dat


mpxautobufs = namedtuple('mpxautobufs', ['mp', 'mpi', 'cov', 'ts', 'mu', 'invn', 'rbwd', 'rfwd', 'cbwd', 'cfwd'])



def xcov(ts, mu, firstrow, out=None):
    ssct = ts.shape[0] - w + 1
    if ssct > 0 and out is None:
        out = np.empty(ssct - minsep, dtype='d')
    mps.crosscov(out, ts[minsep:], mu[minsep:], firstrow)
    return out


def mpx(ts, w):
    ssct = ts.shape[0] - w + 1
    mu = np.empty(ssct, dtype='d')
    mu_s = np.empty(ssct, dtype='d') # window length w - 1 skipping first and last
    invn = np.empty(ssct, dtype='d')
    minsep = w // 4

    mps.windowed_mean(ts, mu, w)
    mps.windowed_mean(ts[:-1], mu_s, w-1)
    mps.windowed_invcnorm(ts, mu, invn, w)
    rbwd = ts[:ssct-1] - mu[:-1]
    cbwd = ts[:ssct-1] - mu_s[1:]
    rfwd = ts[w:] - mu[1:]
    cfwd = ts[w:] - mu_s[1:]

    mp = np.full(ssct, -1, dtype='d')
    mpi = np.full(ssct, -1, dtype='i')

    first_row = ts[:w] - mu[0]
    cov = np.empty(ssct - minsep, dtype='d')   
 
    mps.crosscov(cov, ts[minsep:],  mu[minsep:], first_row)

    # roffset only applies when this is tiled by row, since otherwise the index would be wrong
    mps.mpx_inner(cov, rbwd, rfwd, cbwd, cfwd, invn, mp, mpi, minsep, 0) 

    return mp, mpi


class MPXstream:
    """ auto profile indicates that our comparisons use normalized cross correlation between 2 sections of the same
        time series

        may refactor later to share things between profile types
    """

    def __init__(self, sseqlen, minsep, maxsep=None, minbufsz =None):
        if sseqlen < 4:
            raise ValueError('subsequence lengths below 4 are not supported')
        elif maxsep is not None:
            if not (0 < minsep < maxsep):
                raise ValueError(f" minsep must be a positive value between 0 and maxsep if maxsep is provided, received minsep: {minsep}, maxsep: {maxsep}")
        elif minsep < 0:
            raise ValueError(f'negative minsep {minsep} is unsupported')
        # This object indexes by subsequence, not by time series element
        self.sseqlen = sseqlen
        self.minsep = minsep
        self.maxsep = maxsep
        if minbufsz is None:
            if maxsep is None:
                minbufsz = max(2 * sseqlen, 4096)
            else:
                minbufsz = (maxsep - minsep) + sseqlen - 1
        
        # These are actually different sizes, but it's much easier to allocate them uniformly
        # This way if they are very large, they can be aligned to a multiple of page size
        #
        # This works fine, since they each explicitly track the number of live elements at a given time      
 
        self.mp=BufferedArray(minbufsz)
        self.mpi=BufferedArray(minbufsz, dtype='q')
        self.ts=BufferedArray(minbufsz)
        self.mu=BufferedArray(minbufsz)
        self.invn=BufferedArray(minbufsz)
        self.rbwd=BufferedArray(minbufsz)
        self.rfwd=BufferedArray(minbufsz)
        self.cbwd=BufferedArray(minbufsz)
        self.cfwd=BufferedArray(minbufsz)
        self.cov=BufferedArray(minbufsz)
 
        # cov and first_row are treated as scratch space that is only ever 
        # read or overwritten,
        # so we don't include them in the object's buffers
        self.first_row = None 

    @property
    def buffers(self):
        return mpxautobufs(self.mp, self.mpi, self.ts, self.mu, self.invn, self.rbwd, self.rfwd, self.cbwd, self.cfwd)


    @property
    def count(self):
        return self.mp.count

    @property
    def buffer_size(self):
        """ buffer size """
        return self.mp.size

    def drop(self, ct):
        """ discard ct elements from memory starting from the current beginning of the in memory
            portion

            I may need to adjust buffers
        """
        buffers = self.buffers
        for buf in self.buffers:
            buf.drop(ct)

    def normalize_buffers(self):
        """ normalize buffer layout so that retained data spans the positional range
            0 to size - 1
        """

        for buf in self.buffers:
            buf.repack()

    def resize(self, sz):
        """ resize underlying buffer """
        if self.count > sz:
            raise ValueError(f"a resized buffer of size {sz} is too small to retain {self.ts.count} saved elements")
        elif sz == self.size:
            self.normalize_buffer()
        else:
            # these need a per buffer adjustment factor which I will add
            for buf in self.buffers:
                buf.resize(sz)

    def append(ts):
        self.ts.append(ts)
        sz = ts.shape[0]
        prevct = max(ts.count - self.sseqlen + 1, 0)
        if prevct == 0:
            addct = max(self.ts.count + ts.shape[0] - self.sseqlen + 1, 0)
            ts_ = self.ts
        else:
            addct = self.ts.shape[0]
            ts_ = self.ts[1-ts.shape[0]-self.sseqlen:]
        if addct != 0:
            difct = addct if prevct != 0 else addct - 1
            mu_ = self.mu.fillable(addct)
            mps.moving_mean(ts_, mu_, self.sseqlen)
            xcov(ts_, mu_, self.firstrow, out=self.cov.data)
            if difct != 0:
                rbwd_ = self.rbwd.fillable(difct)
                rfwd_ = self.rfwd.fillable(difct)
                cbwd_ = self.cbwd.fillable(difct)
                cfwd_ = self.cfwd.fillable(difct)
                mu_s = np.empty(difct)
                mps.windowed_mean(ts[:1], mu_s, w-1)
                np.subtract(ts[:difct], mu_[:-1], out=rbwd_)
                np.subtract(ts[self.sseqlen:], mu_[1:], out=rfwd_)
                np.subtract(ts[:difct], mu_s, out=cbwd_)
                np.subtract(ts[self.sseqlen:], mu_s, out=cfwd_)
                self.mp.fill(-1.0, addct)
                self.mpi.fill(-1, addct)
            if addct > self.minsep:
                self.cov.reset()
                cov_ = self.cov.fillable(addct - self.minsep)
                np.subtract(self.ts.data[:self.sseqlen], self.mu.data[0], out=self.first_row)
                mps.crosscov(cov, self.ts[minsep:],  self.mu[minsep:], first_row)
                mps.mpx_inner(cov_, self.rbwd, self.rfwd, self.cbwd, self.cfwd, invn, mp, mpi, minsep, self.mp.minindex) 
            

