import numpy as np
import matrixprofile.streaming as mps


def mpx(ts, w):
    ssct = ts.shape[0] - w + 1
    mu = np.empty(ssct)
    mu_s = np.empty(ssct-1)
    invn = np.empty(ssct)
    minlag = w // 4

    mps.windowed_mean(ts, mu, w)
    mps.windowed_mean(ts[1:], mu_s, w-1)
    mps.windowed_invcnorm(ts, mu, invn, w)
    rbwd = ts[:ssct] - mu
    cbwd = ts[:ssct] - mu_s
    rfwd = ts[w:] - mu[1:]
    cfwd = ts[w:] - mu_s

    mp = np.full(ssct, -1)
    mpi = np.full(ssct, -1)

    first_row = ts[:w] - mu[0]
    
    crosscov(cov, ts[minlag:],  mu[minlag:], first_row)

    # roffset only applies when this is tiled by row, since otherwise the index would be wrong
    # (offsetting at the end would also be acceptable)
    # appending data can be handled by using a high minlag and computing the appropriate initial co-moment

    mpx_inner(cov, rbwd, rfwd, cbwd, cfwd, invn, mp, mpi, minlag, 0) 


