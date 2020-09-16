import numpy as np
from collections import namedtuple


class BufferedArray:
    """ an extremely basic buffered array """

    def __init__(self, size, dtype='d', minindex=0):
        self.size = size
        self.minindex = minindex
        self._seq = np.empty(size, dtype=dtype)
        self.count = 0
        self.beginpos = 0

    @property
    def seq(self):
        return self._seq[self.beginpos:self.beginpos + self.count]

    @property
    def maxindex(self):
        """ the absolute time based index of the last buffer element in memory"""
        return self.minindex + self.count - 1 if self.count != 0 else None

    @property
    def maxfill(self):
        """ number of unused entries"""
        return self._seq.shape[0] - self.count

    @property
    def maxappend(self):
        """ number of unused entries located at the end of the buffer """
        return self._seq.shape[0] - self.count - self.beginpos

    # dunder methods are added for debugging convenience.
    # Be warned, these create views on every call

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        """ number of in memory elements
            note: len is supposed to match the number of elements returned by iter
        """
        return self.seq.shape[0]

    def __getitem__(self, item):
        return self.seq[item]

    def __setitem__(self, key, value):
        self.seq[key] = value

    def normalize_buffer(self):
        """ normalize buffer layout so that retained data spans the positional range
            0 to size - 1.
            Since this is an inplace op, it can invalidate python iterators to this object.
            It is intentionally left as an explicit op.
        """
        if self.count > 0 and self.beginpos != 0:
            self._seq[:self.count] = self.seq
            self.beginpos = 0

    def fill(self, x, count):
        if count > self.maxfill:
            raise ValueError
        elif count > self.maxappend:
            self.normalize_buffer()
        self.count += count
        self.seq[-count:] = x

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
                self.seq[:] = fillval
        elif count == self.count:
            self.beginpos = 0
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
        dat = self.seq
        self._seq = np.empty(sz, dtype=self._seq.dtype)
        self.seq[:] = dat

    def fillable(self, sz, allow_resize=True):
        if sz > self.maxfill:
            if allow_resize:
                self.resize(2*(self.count + sz))
            else:
                raise ValueError
        elif sz > self.maxappend:
            self.normalize_buffer()
        self.count += sz
        return self.seq[-sz:]         

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
        self.seq[oldct:] = dat


mpxautobufs = namedtuple('mpxautobufs', ['mp', 'mpi', 'cov', 'ts', 'mu', 'invn', 'rbwd', 'rfwd', 'cbwd', 'cfwd'])


def dif_eqs(ts, mu, w):
    ssct = ts.shape[0] - w + 1
    if ssct != mu.shape[0]:
        raise ValueError
    mu_s = np.empty(ssct)
    mps.windowed_mean(ts, mu_s, w-1)
    rbwd = ts[:ssct-1] - mu[:-1]
    cbwd = ts[:ssct-1] - mu_s[1:]
    rfwd = ts[w:] - mu[1:]
    cfwd = ts[w:] - mu_s[1:]
    return rbwd, cbwd, rfwd, cfwd


def xcov(ts, mu, firstrow, minlag):
    ssct = ts.shape[0] - w + 1
    cov = np.empty(ssct - minlag, dtype='d')
    mps.crosscov(cov, ts[minlag:], mu[minlag:], firstrow)
    return cov


def mpx(ts, w):
    ssct = ts.shape[0] - w + 1
    mu = np.empty(ssct, dtype='d')
    mu_s = np.empty(ssct, dtype='d') # window length w - 1 skipping first and last
    invn = np.empty(ssct, dtype='d')
    minlag = w // 4

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
    cov = np.empty(ssct - minlag, dtype='d')   
 
    mps.crosscov(cov, ts[minlag:],  mu[minlag:], first_row)

    # roffset only applies when this is tiled by row, since otherwise the index would be wrong
    mps.mpx_inner(cov, rbwd, rfwd, cbwd, cfwd, invn, mp, mpi, minlag, 0) 

    return mp, mpi


class MPXstream:
    """ auto profile indicates that our comparisons use normalized cross correlation between 2 sections of the same
        time series

        may refactor later to share things between profile types
    """

    def __init__(self, sseqlen, minsep, maxsep, minbufsz =None):
        if sseqlen < 4:
            raise ValueError('subsequence lengths below 4 are not supported')
        elif not (0 < minsep < maxsep):
            raise ValueError(f'negative minsep {minsep} is unsupported')
        # This object indexes by subsequence, not by time series element
        self.sseqlen = sseqlen
        self.minsep = minsep
        self.maxsep = maxsep
        if minbufsz is None:
            minbufsz = (maxsep - minsep) + sseqlen - 1
        # this makes it easy to iterate over all buffers and apply something like a shift operation
        # I don't currently inherit from named tuple, because it adds restrictions and exposes too many operations.
        # making this class a data class would make more sense here, but I don't think it has been
        # backported to anything newer than 3.8. 
        self.buffers = mpxautobufs(mp=BufferedArray(minbufsz),
                                   mpi=BufferedArray(minbufsz, dtype='q'),
                                   ts=BufferedArray(minbufsz + sseqlen - 1),
                                   mu=BufferedArray(minbufsz),
                                   invn=BufferedArray(minbufsz),
                                   rbwd=BufferedArray(minbufsz),
                                   rfwd=BufferedArray(minbufsz),
                                   cbwd=BufferedArray(minbufsz),
                                   cfwd=BufferedArray(minbufsz))

    @property
    def count(self):
        return self.buffers.mp.count

    @property
    def cov(self):
        return self.buffers.cov

    @property
    def ts(self):
        return self.buffers.ts

    @property
    def mu(self):
        return self.buffers.mu

    @property
    def rfwd(self):
        return self.buffers.rfwd

    @property
    def rbwd(self):
        return self.buffers.rbwd

    @property
    def cfwd(self):
        return self.buffers.cfwd

    @property
    def cbwd(self):
        return self.buffers.cbwd

    @property
    def mp(self):
        return self.buffers.mp

    @property
    def mpi(self):
        return self.buffers.mpi

    @property
    def size(self):
        """ buffer size """
        return self.buffers.mp.size

    def drop(self, ct):
        """ discard ct elements from memory starting from the current beginning of the in memory
            portion

            I may need to adjust buffers
        """
        buffers = self.buffers
        for buf in self.buffers:
            buf.drop(ct)

    def repack(self):
        """ normalize buffer layout so that retained data spans the positional range
            0 to size - 1
        """

        for buf in self.buffers:
            buf.repack()

    def resize(self, sz):
        """ resize underlying buffer """
        if self.count > sz:
            raise ValueError
        elif sz == self.size:
            self.normalize_buffer()
        else:
            # these need a per buffer adjustment factor which I will add
            for buf in self.buffers:
                buf.resize(sz)


    def seed(ts):
        if self.mp.sseqct > 0:
            raise ValueError('cannot seed non empty profile')
        elif ts.size == 0:
            raise ValueError('empty seed array')
        self.ts.append(ts)
        mps.moving_mean(ts, self.mu.seq, self.sseqlen)
        xcov(self.ts, self.mu, firstrow, minlag)
        rbwd, cbwd, rfwd, cfwd = dif_eqs(self.ts.seq, self.mu.seq, self.sseqlen)
        self.rbwd.append(rbwd)
        self.rfwd.append(rfwd)
        self.cbwd.append(cbwd)
        self.cfwd.append(cfwd)
        # cov is just overwritten whenever we append
        # whereas the others are still required later
        cov = np.empty(ts.shape[0] - self.sseqlen - self.minlag + 1)
        first_row = self.ts.seq[:self.sseqlen] - self.mu.seq[0]
        mps.crosscov(cov, ts[minlag:],  mu[minlag:], first_row)
        mps.mpx_inner(cov, rbwd, rfwd, cbwd, cfwd, invn, mp, mpi, minlag, 0) 

    def append(self, dat, allow_resize=False):
        if 0 < dat.size:
            if dat.size <= self.maxappend:
                if self.size - self.count < dat.size:
                    self.repack()
            elif allow_resize:
                minreq = self.count + dat.size
                self.resize(2 * minreq)
            else:
                raise ValueError
            # ignore maxsep for now
            
            # check starting subsequence count, as it serves as minlag here
            oldssct = max(self.ts.count - self.sseqlen + 1, 0)
            minlag = self.minlag if ssct < minlag else oldssct
            sectbegin = max(0, self.ts.count - self.subseqlen + 1)
            addct = (dat.shape[0] if self.ts.count >= self.subseqlen 
                     else self.ts.count + dat.shape[0] - self.subseqlen + 1)
            self.mp.fill(-1.0, dat.shape[0])
            self.mpi.fill(-1, dat.shape[0])
            self.ts.append(dat)
            # update moving mean 
            
            # we use up to subsequence length - 1 existing entries
            rbwd, cbwd, rfwd, cfwd = dif_eqs(self.ts.seq[sectbegin:], self.mu.seq[sectbegin:], self.sseqlen)
            self.rbwd.append(rbwd)
            self.rfwd.append(rfwd)
            self.cbwd.append(cbwd)
            self.cfwd.append(cfwd)
            cov = np.empty(oldssct + addct - minlag)
             
            mps.windowed_mean(ts, mu_s, w-1)
            # mps.crosscov(cov, ts[minlag:], 





         


