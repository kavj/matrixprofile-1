from matrixprofile.cycore import muinvn
from matrixprofile.algorithms.cympx_inner import compute_difference_equations as difference_equations, compute_cross_cov as cross_cov, compute_self_compare as self_compare
import numpy as np


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
    def free_count(self):
        """ number of presently unused entries"""
        return self._array.shape[0] - self.count

    @property
    def max_size(self):
        return self._array.shape[0]

    # iter and len consider only the viewable portion

    def __iter__(self):
        return iter(self.array)

    def __len__(self):
        return self.array.size

    # note: handling of wraparounds, multi-slicing, etc. whether well defined or not here, is
    #       propagated to the currently live view of the underlying array implementation, partly
    #       because there are too many cases to handle with a small class like this.
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
        if self.begin_pos != 0:
            self._array[:self.count] = self.array
            self.begin_pos = 0

    def drop_leading(self, count):
        """ discard ct elements from memory starting from the current beginning of the in memory
            portion
        """
        if count < 0:
            raise ValueError("shift count cannot be negative")
        elif count >= self.count:
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
        if size < self.count:
            raise ValueError(f"buffer size {size} is too small to accommodate {self.count} live elements")
        data = self.array
        self._array = np.empty(size, dtype=self._array.dtype)
        self.begin_pos = 0
        self.array[:] = data

    def extend(self, count, fill_value=None):
        prevct = self.count
        reqct = prevct + count
        if reqct > self.max_size:
            self.resize_buffer(2 * reqct, fill_value)
        elif reqct > self.free_count - self.begin_pos:
            # shift array contents
            self.normalize_buffer()
        self.count += count
        if fill_value is not None:
            self.array[prevct:].fill(fill_value)

    def append(self, data):
        """ append to buffer. In most python interfaces, extend is used with iterables. This naming convention
            maintains consistency with that, without checking explicit inheritance from Iterable, which numpy may
            not adhere to strictly.
        """
        prevct = self.count
        reqct = data.size
        if reqspace > self.free_count:
            self.resize_buffer(2 * reqct)
        elif reqct > self.free_count - self.begin_pos:
            self.normalize_buffer()
        self.count += reqspace
        self.array[prevct:] = data


class MpxStream:
    """
       This provides a simple implementation based on the buffered array class.

    """

    def __init__(self, subseqlen, minsep, minbufsz=None):
        if subseqlen < 4:
            raise ValueError("subsequence lengths less than 2 do not admit a normalized representation")
        elif maxsep is not None:
            if not (0 < minsep < maxsep):
                raise ValueError(f"minsep must be a positive value between 0 and maxsep if maxsep is provided, "
                                 f"received minsep: {minsep}, maxsep: {maxsep}")
        elif minsep < 1:
            raise ValueError("non-positive minsep is not well defined")
        # This object indexes by subsequence, not by time series element
        self.subseqlen = subseqlen
        self.minsep = minsep
        self.maxsep = maxsep
        if minbufsz is None:
            minbufsz = minsep + subseqlen - 1
        self.mp = StreamArray(minbufsz)
        self.mpi = StreamArray(minbufsz, dtype=np.int_)
        self.ts = StreamArray(minbufsz)
        self.mu = StreamArray(minbufsz)
        self.invn = StreamArray(minbufsz)
        self.df = StreamArray(minbufsz)
        self.dg = StreamArray(minbufsz)
        self.leadseq = None

    @property
    def count(self):
        return self.ts.count

    @property
    def subseqct(self):
        return max(self.ts.count - self.subseqlen + 1, 0)

    def drop(self, ct):
        """ discard ct elements from memory starting from the current beginning of the in memory
            portion

            I may need to adjust buffers
        """
        for seq in self.astuple:
            seq.drop_leading(ct)

    def append(self, data):
        prevct = self.subseqct
        self.ts.append(data)
        addct = self.subseqct - prevct
        if addct == 0:
            return
        # always maintain a consistent state for the
        # time series, profile, and index
        self.mp.extend(addct, -1.0)
        self.mpi.extend(addct, -1)
        if self.minsep < self.subseqct
            if prevct < minsep:
                # For simplicity, ignore these for subseqct <= minsep. 
                mu, invn = muinvn(self.ts.array, self.subseqlen)
                self.mu.append(mu)
                self.invn.append(invn)
                self.df.extend(self.subseqct)
                self.dg.extend(self.subseqct)
                self.leadseq = self.ts[:self.subseqlen] - mu[0]
                # This needs to avoid the leading zero, which I planned to do away with anyway
                difference_equations(self.ts.array, self.mu.array, self.df[:-1], self.dg[:-1])
                cc = np.empty(self.subseqct)
                cross_cov(cc, self.ts.array, self.mu.array, self.leadseq)

            elif minsep < self.subseqct:
                ts_ = self.ts[prevct-w+1:]
                mu, invn = muinvn(ts_, self.subseqlen)
                self.mu.append(mu)
                self.invn.append(invn)
                mu_ = self.mu[prevct:]
                invn_ = self.invn[prevct:]
                self.df.extend(addct)
                self.dg.extend(addct)
                df_ = self.df[prevct:-1]
                dg_ = self.dg[prevct:-1]
                difference_equations(ts_, mu_, df_, dg_)

                cross_cov(cc, ts_, mu_)
                # minsep = prevct

                # Here we are extending something that was already long enough to admit comparisons
                # We do it by increasing minsep to omit the parts that were already computed
            


