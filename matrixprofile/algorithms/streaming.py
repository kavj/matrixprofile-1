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

    def shiftby(self, count):
        """ discard ct elements from memory starting from the current beginning of the in memory
            portion
        """
        # This is needed to aggregate buffers of different lengths for some cases
        # In particular, we may wish to shift a time series with n elements using a subsequence length of m by
        # k elements, where n - m + 1 < k < n.
        # As a result, we wouldn't have any complete subsequences, but the time series itself would retain some.
        # the amount shifted would still need to reflect the amount dropped by the longest sequence.

        if count >= self.count:
            # avoid resetting leading position, in case this is aggregated with other arrays
            self.beginpos = self._seq.shape[0]
            self.count = 0
            self.minindex += count
        else:
            self.beginpos += count
            self.count -= count
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
        if dat.size > self.maxappend:
            raise ValueError
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
