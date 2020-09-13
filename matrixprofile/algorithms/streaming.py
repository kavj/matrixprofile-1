import numpy as np


class TSParams:
    """ Descriptor for time series matrix profile calculations using method mpx.
    """

    def __init__(self, subseqlen, bufferlen):
        self.subseqlen = subseqlen
        self.minindex = 0  # global min
        self.beginpos = 0
        self.eltct = 0
        ssbuflen = max(bufferlen - subseqlen + 1, 0)
        self._ts = np.empty(bufferlen, dtype='d')
        self._mu = np.empty(ssbuflen, dtype='d')
        self._invn = np.empty(ssbuflen, dtype='d')

    @property
    def bufferlen(self):
        return self._ts.shape[0]

    @property
    def maxfillable(self):
        """ maximum number of elements that can be added without repacking or resizing"""
        return max(self.bufferlen - self.beginpos - self.eltct, 0)

    # global rather than local positions
    @property
    def subseqct(self):
        return max(self.eltct - self.subseqlen + 1, 0)

    @property
    def maxindex(self):
        return self.minindex + self.subseqct

    @property
    def ts(self):
        return self._ts[self.beginpos:self.beginpos + self.eltct]

    @property
    def mu(self):
        return self._mu[self.beginpos:self.beginpos + self.subseqct]

    @property
    def invn(self):
        return self._invn[self.beginpos:self.beginpos + self.subseqct]

    def dropleading(self, dropct):
        if dropct > self.elct:
            raise ValueError(f"cannot drop {dropct} elements from {self.eltct} elements")
        self.minindex += dropct
        self.beginpos += dropct
        self.eltct -= dropct

    def repack(self):
        self._ts[:self.eltct] = self.ts
        self._mu[:self.ssct] = self.mu
        self._invn[:self.ssct] = self.invn
        self.beginpos = 0

    def resize_buffer(self, updatedsz):
        if updatedlen < self.eltct:
            raise ValueError("updated buffer size is too small to accommodate all elements")
        elif updatedlen != self._ts.shape[0]:
            ts_ = self.ts()
            mu_ = self.mu()
            invn_ = self.invn()
            self.beginpos = 0
            updothersz = max(updatedsz - self.sseqlen + 1, 0)
            self._ts = np.empty(updatedz, dtype='d')
            self._mu = np.empty(updothersz, dtype='d')
            self._invn = np.empty(updothersz, dtype='d')
            self.ts[:] = ts_
            self.mu[:] = mu_
            self.invn[:] = invn_

    def append(self, allow_resize=False):
        if dat.shape[0] > self.bufferlen - self.eltct:
            if allow_resize:
                minreq = self.eltct + dat.shape[0]
                self.resize_buffer(2 * minreq)
            else:
                raise ValueError
        elif dat.shape[0] > self.maxfillable:
            self.repack()
        ssectbegin = max(self.eltct - self.sseqlen + 1, 0)
        self.eltct += ts.shape[0]
        windowed_mean(self.ts[ssectbegin:], self.mu[ssectbegin:])
        windowed_invcnorm(self.ts[ssectbegin:], self.mu[ssectbegin:], self.invn[ssectbegin:])


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
        self.profilelen = 0
        self.beginpos = 0
        self.minindex = 0
        if minbufferlen is None:
            minbufferlen = (maxsep - minsep) + sseqlen - 1
        self.timeseries = TSParams(sseqlen, bufferlen=minbufferlen)

    def dropleading(self, count):
        if not (0 <= count <= self.profilelen):
            raise ValueError
        self.minindex += count
        self.beginpos += count
        self.profilelen -= count

    # should be accompanied by a method to output finished sections
    def trimleading(self):
        if self.profilelen >= maxsep:
            pass

    # ideally we would be able to write completed sections to a file

    def append(self, buffered, allow_resize=False):
        if self.profilelen >= self.maxsep:
            # For a single time series input, we only need to know how many things can be compared
            # to the first added subsequence, based on maxlag. The rest falls in line well enough.
            pass
        else:
            pass

