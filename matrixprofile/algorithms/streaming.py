import numpy as np
from abc import ABC, abstractmethod


class streamed(ABC):
    """
       This is a reference example of a streaming buffer interface.
       It does not need to be explicitly inherited from. It's here to allow critique of basic
       interface elements that will be common to all streaming classes.

    """

    @property
    @abstractmethod
    def minindex(self):
        """ the absolute time based index of the first buffer element in memory """
        raise NotImplementedError

    @property
    @abstractmethod
    def maxindex(self):
        """ the absolute time based index of the last buffer element in memory"""
        raise NotImplementedError

    @property
    @abstractmethod
    def maxfillable(self):
        """ the maximum number of entries that may be appended without resizing."""
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self):
        """ buffer size """
        raise NotImplementedError

    @property
    @abstractmethod
    def count(self):
        """ number of in memory elements """
        raise NotImplementedError

    @abstractmethod
    def repack(self):
        """ normalize buffer layout so that retained data spans the positional range
            0 to size - 1
        """
        raise NotImplementedError

    @abstractmethod
    def dropleading(self, ct):
        """ discard ct elements from memory starting from the current beginning of the in memory
            portion
        """
        raise NotImplementedError

    @abstractmethod
    def resize(self, updatedsz):
        """ resize underlying buffer """
        raise NotImplementedError

    @abstractmethod
    def append(self, dat, allow_resize=False):
        """ append to buffer """
        raise NotImplementedError


class TSParams:
    """ Descriptor for time series matrix profile calculations using method mpx.
    """

    def __init__(self, subseqlen, bufferlen):
        self.subseqlen = subseqlen
        self.minindex = 0  # global min
        self.count = 0
        self.beginpos = 0
        ssbuflen = max(bufferlen - subseqlen + 1, 0)
        self._ts = np.empty(bufferlen, dtype='d')
        self._mu = np.empty(ssbuflen, dtype='d')
        self._invn = np.empty(ssbuflen, dtype='d')


    @property
    def ts(self):
        return self._ts[self.beginpos:self.beginpos + self.count]

    @property
    def mu(self):
        return self._mu[self.beginpos:self.beginpos + self.subseqct]

    @property
    def invn(self):
        return self._invn[self.beginpos:self.beginpos + self.subseqct]

    @property
    def maxindex(self):
        """ the absolute time based index of the last buffer element in memory"""
        return self.minindex + self.size

    @property
    def maxfillable(self):
        """ the maximum number of free entries that may be  appended without resizing."""

        # This might still require repacking.
        return self.size - self.count

    @property
    def size(self):
        """ buffer size """
        return self._ts.shape[0]

    @property
    def subseqct(self):
        return max(self.count - self.subseqlen + 1, 0)

    def dropleading(self, ct):
        """ discard ct elements from memory starting from the current beginning of the in memory
            portion
        """
        if dropct > self.count:
            raise ValueError(f"cannot drop {dropct} elements from {self.count} elements")
        self.minindex += dropct
        self.beginpos += dropct
        self.count -= dropct

    def repack(self):
        """ normalize buffer layout so that retained data spans the positional range
            0 to size - 1
        """

        if self.count > 0 and self.beginpos != 0:
            self._ts[:self.count] = self.ts
            ssct = self.subseqct
            self._mu[:ssct] = self.mu
            self._invn[:ssct] = self.invn
            self.beginpos = 0

    def resize(self, sz):
        """ resize underlying buffer """
        if self.count > sz:
            raise ValueError
        elif sz != self.size:
            ts_ = self.ts
            mu_ = self.mu
            invn_ = self.invn
            self.beginpos = 0
            ssbufsz = max(sz - self.sseqlen + 1, 0)
            self._ts = np.empty(sz, dtype='d')
            self._mu = np.empty(ssbufsz, dtype='d')
            self._invn = np.empty(ssbufsz, dtype='d')
            self.ts[:] = ts_
            self.mu[:] = mu_
            self.invn[:] = invn_
        elif self.beginpos != 0:
            self.repack()

    @property
    def maxssindex(self):
        return self.minindex + self.subseqct

    def append(self, dat, allow_resize=False):
        if 0 < dat.shape[0]:
            if dat.shape[0] <= self.maxfillable:
                if self.size - self.count < dat.shape[0]:
                    self.repack()
            elif allow_resize:
                minreq = self.count + dat.shape[0]
                self.resize(2 * minreq)
            else:
                raise ValueError
            sectbegin = self.sseqct
            self.count += dat.shape[0]
            windowed_mean(self.ts[sectbegin:], self.mu[sectbegin:])
            windowed_invcnorm(self.ts[sectbegin:], self.mu[sectbegin:], self.invn[sectbegin:])


class MProfile:
    """ auto profile indicates that our comparisons use normalized cross correlation between 2 sections of the same
        time series
    """

    def __init__(self, sseqlen, minsep, maxsep, minbufferlen=None):
        if sseqlen < 4:
            raise ValueError('subsequence lengths below 4 are not supported')
        elif not (0 < minsep < maxsep):
            raise ValueError(f'negative minsep {minsep} is unsupported')
        # This object indexes by subsequence, not by time series element
        self.minsep = minsep
        self.maxsep = maxsep
        self.count = 0
        self.beginpos = 0
        self.minindex = 0
        if minbufferlen is None:
            minbufferlen = (maxsep - minsep) + sseqlen - 1
        self.timeseries = TSParams(sseqlen, bufferlen=minbufferlen+sseqlen-1)
        self._mp = np.empty(minbufferlen, dtype='d')
        self._mpi = np.empty(minbufferlen, dtype='q')

    @property
    def mp(self):
        return self._mp[self.beginpos:self.beginpos+self.count]

    @property
    def mpi(self):
        return self._mpi[self.beginpos:self.beginpos+self.count]

    @property
    def maxindex(self):
        """ the absolute time based index of the last buffer element in memory"""
        return self.minindex + self.size

    @property
    def maxfillable(self):
        """ the maximum number of free entries that may be  appended without resizing."""
        return self.size - self.count

    @property
    def size(self):
        """ buffer size """
        return self._mp.shape[0]

    def dropleading(self, ct):
        """ discard ct elements from memory starting from the current beginning of the in memory
            portion
        """
        if dropct > self.count:
            raise ValueError(f"cannot drop {ct} elements from {self.count} elements")
        self.minindex += dropct
        self.beginpos += dropct
        self.count -= dropct

    def repack(self):
        """ normalize buffer layout so that retained data spans the positional range
            0 to size - 1
        """

        if self.count > 0 and self.beginpos != 0:
            self._ts[:self.count] = self.ts
            ssct = self.subseqct
            self._mu[:ssct] = self.mu
            self._invn[:ssct] = self.invn
            self.beginpos = 0

    def resize(self, sz):
        """ resize underlying buffer """
        if self.count > sz:
            raise ValueError
        elif sz != self.size:
            mp_ = self.mp
            mpi_ = self.mpi
            self.beginpos = 0
            self.mp[:] = mp_
            self.mpi[:] = mpi_
        elif self.beginpos != 0:
            self.repack()

    def append(self, dat, allow_resize=False):
        if 0 < dat.shape[0]:
            if dat.shape[0] <= self.maxfillable:
                if self.size - self.count < dat.shape[0]:
                    self.repack()
            elif allow_resize:
                minreq = self.count + dat.shape[0]
                self.resize(2 * minreq)
            else:
                raise ValueError
            sectbegin = self.count
            self.count += ts.shape[0]
            self.mp[sectbegin:] = -1
            self.mpi[sectbegin:] = -1
