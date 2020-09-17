import numpy as np


class BufferedArray:
    """ An extremely basic buffered array, made with streaming data in mind

        Since this is written primarily with streaming data in mind, resizing requires an explicit operation.
        This ensures aggregated groups of arrays that are supposed to maintain similar length remain that way.

        Arrays are maintained as contiguous 1D sections in memory. Appended elements are added to the end of this and
        do not affect the starting position of the array in memory unless an insufficient number of elements are
        available at the end. These cases are handled by normalizing the buffer, performed by copying the array back
        to the first position of the buffer.

        Since normalization clobbers any existing iterators and data references, there is a property meant to expose
        whether it's necessary. maxappend indicates the number of entries available at the end of the buffer, whereas
        maxfill indicates the total number of unused entries. If the buffer is normalized, these are the same.

        Shift operations adjust the minimum index and the starting position of the array in memory.
        shift operations adjust the starting position of the array in memory over the same underlying buffer,
        up to the length of the array.

        Shifts are done implicitly, by changing the starting position of the array, the number of live elements it
        currently contains, and the minimum index, which specifies the index of the first element of the array with
        respect to the underlying stream, which may be of arbitrary length.

        Slicing or calling .array provides a view of the underlying array implementation without extra machinery.
        This approach is used, because slice notation is virtually never used on buffers, and this avoids forcing
        people to use an extra round of indirection just to get the appropriate view.

        Typing somebufferedarray.array[...] instead of somebufferedarray[...] gets old fast.

        iter and len dundermethods are usually provided together. These can be removed if necessary, but they are here
        in part to accompany slicing behavior. This way

        for elt in somearray[start:end:step]:
            ....

        and

        for elt in somearray:
            ....

        both work as expected in that they generate a view on the presently used portion of the array
        Avoiding possible invalidation or clobbering of iterators and slices should be done by making a copy.
        That can be added here via optional argument or added inline.

        Right now the underlying array implementation uses numpy. This can be changed. My biggest concern is how
        well the interface works.

    """

    def __init__(self, size, dtype='d', minindex=0):
        self.size = size
        self.minindex = minindex
        self._array = np.empty(size, dtype=dtype)
        self.count = 0
        self.beginpos = 0

    @property
    def array(self):
        return self._array[self.beginpos:self.beginpos + self.count]

    @property
    def maxindex(self):
        """ the absolute time based index of the last buffer element in memory"""
        return self.minindex + self.count - 1 if self.count != 0 else None

    @property
    def maxfill(self):
        """ number of presently unused entries"""
        return self._array.shape[0] - self.count

    @property
    def maxappend(self):
        """ number of unused entries located at the end of the buffer """
        return self._array.shape[0] - self.count - self.beginpos

    # dunder methods are added for debugging convenience.
    # Be warned, these create views on every call

    def __iter__(self):
        return iter(self.array)

    def __len__(self):
        """ number of in memory elements
            note: len is supposed to match the number of elements returned by iter
        """
        return self.array.size

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.array[key] = value

    def normalize_buffer(self):
        """ normalize buffer layout so that retained data spans the positional range
            0 to size - 1.
            Since this is an inplace op, it can invalidate python iterators to this object.
            It is intentionally left as an explicit op.
        """
        if self.beginpos != 0:
            self._array[:self.count] = self.array
            self.beginpos = 0

    def append(self, fillval, count):
        """
        """
        if not (0 < count < self.maxfill):
            raise ValueError(f"fill count must be between 0 and current buffer size {self.maxfill}, {count} received")
        elif count > self.maxappend:
            self.normalize_buffer()
        self.count += count
        self.array[-count:] = fillval

    def shiftby(self, count):
        """ discard ct elements from memory starting from the current beginning of the in memory
            portion
        """
        if count < 0:
            raise ValueError("shift count cannot be negative")
        elif count > self.count:
            # This can arise when aggregating multiple sequences of non-uniform length.
            # One sequence may depend on > 1 elements of another sequence, so a shift by some amount
            # greater than what is contained here right now indicates no data was provided. For that reason
            # we apply the entire shift to minindex, implicitly normalize the buffer, and set the number of live
            # elements to 0 (so array.data returns an empty view)

            self.beginpos = 0
            self.count = 0
        else:
            self.beginpos += count
            self.count -= count
        self.minindex += count

    def resize(self, sz):
        """ resize underlying buffer, raise an error if it would truncate live data """
        if sz < self.count:
            raise ValueError(f"buffer size {sz} is too small to accommodate {self.count} live elements")
        dat = self.array
        self._array = np.empty(sz, dtype=self._data.dtype)
        self.array[:] = dat

    def fillable(self, sz):
        """ increment the number of live elements by sz without initialization and return a view
            to those elements. If allow_resize is True, then the underlying buffer will be resized
            if necessary. If the buffer does not have sufficient space at the end of the array, it will
            shift all live data to the beginning of the array.
        """
        if sz > self.maxfill:
            raise ValueError("fillable size requested exceeds the number of free elements in the current buffer")
        elif sz > self.maxappend:
            self.normalize_buffer()
        self.count += sz
        return self.data[-sz:]

    def extend(self, dat):
        """ append to buffer. In most python interfaces, extend is used with iterables. This naming convention
            maintains consistency with that, without checking explicit inheritance from Iterable, which numpy may
            not adhere to strictly.
        """
        reqspace = dat.size
        if self.maxfill < reqspace:
            raise ValueError(f"appending {dat.size} elements would overflow available buffer space: {self.maxfill}")
        elif self.maxappend < reqspace:
            self.normalize_buffer()
        prevct = self.count
        self.count += reqspace
        self.array[prevct:] = dat


def xcov(ts, mu, firstrow, out=None):
    ssct = ts.shape[0] - w + 1
    if ssct > 0 and out is None:
        out = np.empty(ssct - minsep, dtype='d')
    mps.crosscov(out, ts[minsep:], mu[minsep:], firstrow)
    return out


def mpx(ts, w):
    ssct = ts.shape[0] - w + 1
    mu = np.empty(ssct, dtype='d')
    mu_s = np.empty(ssct, dtype='d')  # window length w - 1 skipping first and last
    invn = np.empty(ssct, dtype='d')
    minsep = w // 4

    mps.windowed_mean(ts, mu, w)
    mps.windowed_mean(ts[:-1], mu_s, w - 1)
    mps.windowed_invcnorm(ts, mu, invn, w)
    rbwd = ts[:ssct - 1] - mu[:-1]
    cbwd = ts[:ssct - 1] - mu_s[1:]
    rfwd = ts[w:] - mu[1:]
    cfwd = ts[w:] - mu_s[1:]

    mp = np.full(ssct, -1, dtype='d')
    mpi = np.full(ssct, -1, dtype='i')

    first_row = ts[:w] - mu[0]
    cov = np.empty(ssct - minsep, dtype='d')

    mps.crosscov(cov, ts[minsep:], mu[minsep:], first_row)

    # roffset only applies when this is tiled by row, since otherwise the index would be wrong
    mps.mpx_inner(cov, rbwd, rfwd, cbwd, cfwd, invn, mp, mpi, minsep, 0)

    return mp, mpi


class MPXstream:
    """
       This provides a simple implementation based on the buffered array class.

    """

    def __init__(self, sseqlen, minsep, maxsep=None, minbufsz=None):
        if sseqlen < 2:
            raise ValueError("subsequence lengths less than 2 do not admit a normalized representation")
        elif maxsep is not None:
            if not (0 < minsep < maxsep):
                raise ValueError(
                    f" minsep must be a positive value between 0 and maxsep if maxsep is provided, received minsep: {minsep}, maxsep: {maxsep}")
        elif minsep <= 0:
            raise ValueError("zero or negative minsep is not well defined")
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

        # I'm using a leading underscore, which indicates that something should be private, to refer
        # to the buffer data structure here and no underscore to refer to the live section of the array
        self._mp = BufferedArray(minbufsz)
        self._mpi = BufferedArray(minbufsz, dtype='q')
        self._ts = BufferedArray(minbufsz)
        self._mu = BufferedArray(minbufsz)
        self._invn = BufferedArray(minbufsz)
        self._rbwd = BufferedArray(minbufsz)
        self._rfwd = BufferedArray(minbufsz)
        self._cbwd = BufferedArray(minbufsz)
        self._cfwd = BufferedArray(minbufsz)
        # unlike the others, cov is just scratch space

        # so we don't include them in the object's buffers
        self.first_row = None

    @property
    def mp(self):
        return self._mp.array

    @property
    def mpi(self):
        return self._mpi.array

    @property
    def ts(self):
        return self._ts.array

    @property
    def mu(self):
        return self._mu.array

    @property
    def invn(self):
        return self._invn.array

    @property
    def rbwd(self):
        return self._rbwd.array

    @property
    def rfwd(self):
        return self._cbwd.array

    @property
    def cbwd(self):
        return self._cfwd.array

    @property
    def astuple(self):
        return self.mp, self.mpi, self.ts, self.mu, self.invn, self.rbwd, self.rfwd, self.cbwd, self.cfwd

    @property
    def count(self):
        return self.ts.count

    @property
    def sseqct(self):
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
        for seq in self.astuple:
            seq.shiftby(ct)

    def normalize_buffers(self):
        """ normalize buffer layout so that retained data spans the positional range
            0 to size - 1
        """
        for seq in self.astuple:
            seq.normalize_buffer()

    def resize(self, sz):
        """ resize underlying buffer. sz must accommodate the time series length rather than subsequence length.
        """
        if self.ts.count > sz:
            raise ValueError(f"a resized buffer of size {sz} is too small to retain a time series with {self.ts.count} "
                             f"live elements")
        elif sz == self.size:
            self.normalize_buffers()
        else:
            # Buffers are allocated uniformly, because they're typically close enough in length and this allows for
            # allocating by powers of 2 if necessary
            for buf in self.buffers:
                buf.resize(sz)

    def append(ts):
        self._ts.append(ts)
        prevct = self.sseqct
        self._ts.append(ts)
        updct = max(self.ts.count - self.sseqlen + 1, 0)
        if updct == prevct:
            return
        if updct > self.ts.size:
            self.resize(2*updct)
        addct = updct - prevct
        difct = addct if prevct != 0 else addct - 1
        ts_ = self.ts[prevct:]
        mu_ = self._mu.fillable(addct)
        mps.moving_mean(ts_, mu_, self.sseqlen)

        mu_s = np.empty(difct)
        mps.windowed_mean(ts_[:1], mu_s, w - 1)

        # This can be done using fillable as well, which might make fewer copies.
        # It's just less readable. I would probably prefer to move the fill function for these
        # back to cython if we need to go that rather than use something weird here.

        self.rbwd.extend(ts_[:difct] - mu_[-1])
        self.rfwd.extend(ts_[self.sseqlen:] - mu_[1:])
        self.cbwd.extend(ts_[:difct])
        self.cfwd.extend(ts_[self.sseqlen:] - mu_s)

        # extend these even if no comparisons are possible right now to maintain a consistent state
        self.mp.append(initval=-1.0, count=addct)
        self.mpi.append(initval=-1, count=addct)

        if updct <= self.minsep:
            return
        elif prevct < self.minsep:
            trim = self.minsep - prevct
            ts_ = ts_[trim:]
            mu_ = mu_[trim:]

        cov_ = np.empty(mu_.shape[0], dtype='d')
        xcov(ts_, mu_, self.firstrow, out=cov_)

        minsep_ = max(prevct, self.minsep)

        mps.mpx_inner(cov_,
                      self.rbwd,
                      self.rfwd,
                      self.cbwd,
                      self.cfwd,
                      self.invn,
                      self.mp,
                      self.mpi,
                      minsep_,
                      self.mp.minindex)
