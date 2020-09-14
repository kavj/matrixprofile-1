import numpy as np
from collections import namedtuple


class BufferedArray:
    """ an extremely basic buffered array """

    def __init__(self, size, dtype='d', minindex=0):
        self.size = bufferlen
        self.minindex = minindex
        self._seq = np.empty(size, dtype=dtype)
        self.count = 0
        self.beginpos = 0

    @property
    def seq(self):
        return self.seq[self.beginpos:self.beginpos:self.beginpos + self.count]

    @property
    def maxindex(self):
        """ the absolute time based index of the last buffer element in memory"""
        return self.minindex + self.count if self.count != 0 else None

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
            self._seq[:self.count] = self._seq
            self.beginpos = 0

    def shiftby(self, count, normalize=False, fillval=None):
        """ discard ct elements from memory starting from the current beginning of the in memory
            portion
        """
        if count > self.size:
            raise ValueError('cannot shift by more than total buffer length')
        if count >= self.count:
            if count > self.count:
                # shift by
                if fillval is None:
                    raise ValueError('cannot shift by an amount greater than the number of live elements without a '
                                     'fill value')
                self.count = count - self.count
                self.beginpos = 0
                self.seq[:] = fillval
            else:
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
            raise ValueError
        dat = self.seq
        self._seq = np.empty(sz, dtype=self._seq.dtype)
        self.seq[:] = dat

    def append(self, dat):
        """ append to buffer """
        if (self.maxappend < dat.size < self.maxfill):
            self.normalize_buffer()
        elif self.maxfill < dat.size:
            raise ValueError(f'appending {dat.size} elements would overflow available buffer space: {self.maxfill}')
        oldct = self.count
        self.count += dat.size
        self.seq[oldct:] = dat


mpxautobufs = namedtuple('mpxautobufs', ['mp', 'mpi', 'ts', 'mu', 'invn', 'rbwd', 'rfwd', 'cbwd', 'cfwd'])


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
        self.minsep = minsep
        self.maxsep = maxsep
        if minbufsz is None:
            minbufsz = (maxsep - minsep) + sseqlen - 1
        # this makes it easy to iterate over all buffers and apply something like a shift operation
        # I don't currently inherit from named tuple, because it adds restrictions and exposes too many operations.
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
    def mp(self):
        return self.buffers.mp.seq

    @property
    def mpi(self):
        return self.buffers.mpi.seq

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
            self.repack()
        else:
            # these need a per buffer adjustment factor which I will add
            for buf in self.buffers:
                buf.resize(sz)

    def append(self, dat, allow_resize=False):
        if 0 < dat.size:
            if dat.size <= self.maxappendable:
                if self.size - self.count < dat.size:
                    self.repack()
            elif allow_resize:
                minreq = self.count + dat.size
                self.resize(2 * minreq)
            else:
                raise ValueError
            self.mp[sectbegin:] = -1
            self.mpi[sectbegin:] = -1

