from abc import abstractmethod, ABC
import numpy as np


class StreamArrayBase(ABC):
    """ A base class containing the intended interface for an array class specialized
        for streaming data.

        The specialization of a streaming array is an attempt to provide a queue like interface which is compatible
        with a pointer like iterator.

        This makes it easy enough to reuse allocated space when trivially copyable data is added to the end of or
        removed from the beginning of the array without reallocating the entire buffer at each step. It also alleviates
        the issue of explicitly computing the slice corresponding to a subrange in user code in conjunction with
        a secondary slice operation.

    """

    @property
    @abstractmethod
    def array(self):
        """ return a C contiguous view to the currently used section of the underlying array or buffer """
        raise NotImplementedError

    @property
    @abstractmethod
    def unused_count(self):
        """ number of presently unused entries"""
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    # This is intended to propagate slicing so that we can write streamed[...]
    # rather than streamed.array[...], which brings it close to the verbosity of
    # inline slicing of subranges
    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key, value):
        raise NotImplementedError

    @abstractmethod
    def extend(self, count, fill_value=None):
        """ Extend array boundary and optionally initialize with fill_value
            This is inconsistent with Python's internal use of the word "extend", which
            explicitly takes an iterable, but it's still clearer than other options.
        """
        raise NotImplementedError

    @abstractmethod
    def drop_leading(self, count):
        """ discard count elements from the beginning of the array
        """
        raise NotImplementedError

    @abstractmethod
    def resize_buffer(self, size):
        """ resize the underlying buffer or array. This may raise an error if the updated size is too small to
            accommodate the currently used or "live" portion of the current buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def append(self, data):
        """ extend the current array, and copy data to the extended portion. I'm using the term "extend" because

        """
        raise NotImplementedError


class StreamArray(StreamArrayBase):
    """
    """

    def __init__(self, size, dtype='d', min_index=0):
        self.size = size
        self.min_index = min_index
        self._array = np.empty(size, dtype=dtype)
        self.count = 0
        self.begin_pos = 0

    @property
    def array(self):
        return self._array[self.begin_pos:self.begin_pos + self.count]

    @property
    def max_index(self):
        """ the absolute time based index of the last buffer element in memory"""
        return self.min_index + self.count - 1 if self.count != 0 else None

    @property
    def unused_count(self):
        """ number of presently unused entries"""
        return self._array.shape[0] - self.count

    @property
    def max_appendable(self):
        """ number of unused entries located at the end of the buffer """
        return self._array.shape[0] - self.count - self.begin_pos

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

    def extend(self, count, fill_value=None):
        """
        try to extend the current array by count positions, assign fill_value to these if one is provided.
        """
        if not (0 < count < self.maxfill):
            raise ValueError(f"fill count must be between 0 and current buffer size {self.maxfill}, {count} received")
        elif count > self.maxappend:
            self.normalize_buffer()
        self.count += count
        if fill_value is not None:
            self.array[-count:].fill(fill_value)

    def drop_leading(self, count):
        """ discard ct elements from memory starting from the current beginning of the in memory
            portion
        """
        if count < 0:
            raise ValueError("shift count cannot be negative")
        elif count > self.count:
            # This can arise when aggregating multiple sequences of non-uniform length.
            # One sequence may depend on > 1 elements of another sequence, so a shift by some amount
            # greater than what is contained here right now indicates no data was provided. For that reason
            # we apply the entire shift to min_index, implicitly normalize the buffer, and set the number of live
            # elements to 0 (so array.data returns an empty view)

            self.begin_pos = 0
            self.count = 0
        else:
            self.begin_pos += count
            self.count -= count
        self.min_index += count

    def resize_buffer(self, size):
        """ resize underlying buffer, raise an error if it would truncate live data """
        if size < self.count:
            raise ValueError(f"buffer size {size} is too small to accommodate {self.count} live elements")
        data = self.array
        self._array = np.empty(size, dtype=self._array.dtype)
        self.array[:] = data

    def append(self, data):
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

    def normalize_buffer(self):
        """ normalize buffer layout so that retained data spans the positional range
            0 to size - 1.
            Since this is an inplace op, it can invalidate python iterators to this object.
            It is intentionally left as an explicit op.
        """
        if self.begin_pos != 0:
            self._array[:self.count] = self.array
            self.begin_pos = 0


def xcov(ts, mu, cmpto, out=None):
    sseqct = ts.shape[0] - cmpto.shape[0] + 1
    if sseqct > 0 and out is None:
        out = np.empty(sseqct - minsep, dtype='d')
    mps.crosscov(out, ts[minsep:], mu[minsep:], cmpto)
    return out


def mpx(ts, w):
    sseqct = ts.shape[0] - w + 1
    mu = np.empty(sseqct, dtype='d')
    mu_s = np.empty(sseqct, dtype='d')  # window length w - 1 skipping first and last
    invn = np.empty(sseqct, dtype='d')
    minsep = w // 4

    mps.windowed_mean(ts, mu, w)
    mps.windowed_mean(ts[:-1], mu_s, w - 1)
    mps.windowed_invcnorm(ts, mu, invn, w)
    rbwd = ts[:sseqct - 1] - mu[:-1]
    cbwd = ts[:sseqct - 1] - mu_s[1:]
    rfwd = ts[w:] - mu[1:]
    cfwd = ts[w:] - mu_s[1:]

    mp = np.full(sseqct, -1, dtype='d')
    mpi = np.full(sseqct, -1, dtype='i')

    first_row = ts[:w] - mu[0]
    cov = np.empty(sseqct - minsep, dtype='d')

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
                raise ValueError(f"minsep must be a positive value between 0 and maxsep if maxsep is provided, "
                                 f"received minsep: {minsep}, maxsep: {maxsep}")
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
        self.mp = StreamArray(minbufsz)
        self.mpi = StreamArray(minbufsz, dtype='q')
        self.ts = StreamArray(minbufsz)
        self.mu = StreamArray(minbufsz)
        self.invn = StreamArray(minbufsz)
        self.rbwd = StreamArray(minbufsz)
        self.rfwd = StreamArray(minbufsz)
        self.cbwd = StreamArray(minbufsz)
        self.cfwd = StreamArray(minbufsz)
        # unlike the others, cov is just scratch space

        # so we don't include them in the object's buffers
        self.first_row = None

    # data classes have this as a method in 3.8+
    # It's used here to cleanly call a method on all persistent sequences used by mpx
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
            seq.drop_leading(ct)

    def normalize_buffers(self):
        """ normalize buffer layout so that retained data spans the positional range
            0 to size - 1
        """
        for seq in self.astuple:
            seq.normalize_buffer()

    def resize(self, size):
        """ resize underlying buffer. size must accommodate the time series length rather than subsequence length.
        """
        if self.ts.count > size:
            raise ValueError(f"a resized buffer of size {size} is too small to retain a time series with {self.ts.count} "
                             f"live elements")
        elif size == self.size:
            self.normalize_buffers()
        else:
            # Buffers are allocated uniformly, because they're typically close enough in length and this allows for
            # allocating by powers of 2 if necessary
            for buf in self.buffers:
                buf.resize_buffer(size)

    def append(ts):
        self.ts.extend(ts)
        prevct = self.sseqct
        self.ts.extend(ts)
        updct = max(self.ts.count - self.sseqlen + 1, 0)
        if updct == prevct:
            return
        if updct > self.ts.size:
            self.resize(2 * updct)
        addct = updct - prevct
        difct = addct if prevct != 0 else addct - 1
        ts_ = self.ts[prevct:]
        self.mu.append(addct)
        mu_ = self.mu[-addct:]
        mps.moving_mean(ts_, mu_, self.sseqlen)

        mu_s = np.empty(difct)
        mps.windowed_mean(ts_[:1], mu_s, w - 1)

        self.rbwd.extend(ts_[:difct] - mu_[:-1])
        self.rfwd.extend(ts_[self.sseqlen:] - mu_[1:])
        self.cbwd.extend(ts_[:difct])
        self.cfwd.extend(ts_[self.sseqlen:] - mu_s)

        # extend these even if no comparisons are possible right now to maintain a consistent state
        self.mp.append(count=addct, initval=-1.0)
        self.mpi.append(count=addct, initval=-1)

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
                      self.mp.min_index)
