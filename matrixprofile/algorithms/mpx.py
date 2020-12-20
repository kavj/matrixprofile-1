#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile import core
from matrixprofile.algorithms.cympx import mpx_parallel as cympx_parallel
from matrixprofile.algorithms.cympx import mpx_ab_parallel as cympx_ab_parallel


def mpx_self_compare_dispatch(ts, w, cross_correlation=0, n_jobs=1):
    """

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix profile for.
    w : int
        The window size.
    cross_correlation : bint
        Flag (0, 1) to determine if cross_correlation distance should be
        returned. It defaults to Euclidean Distance (0).
    n_jobs : int, Default = 1
        Number of cpu cores to use.
    
    Returns
    -------
    (array_like, array_like) :
        The matrix profile (distance profile, profile index).

    """

    n =  ts.shape[0]
    profile_len = n - w + 1
    minlag = w // 4
    mp = np.full(profile_len, -1.0)
    mpi = np.full(profile_len, -1, dtype=np.int_)

    if profile_len - minlag <= 0:
        warn("Time series is too short to perform any comparisons.")
        return mp, mpi
    
    ts, mu, invn, first, last, isvalidwindow = mpx_preprocessing(ts, w)
    
    # Raise an error rather instead of recursively shrinking window range
    if not isvalidwindow[0]:
        raise ValueError

    span = ts.shape[0] - w + 1
    diagcount = span - minlag

    if diagcount <= 0:
        warn("Zero valid comparisons could be performed due to missing data")
        return mp, mpi

    if not isvalidwindow[minlag]:
        options, = np.where(isvalidwindow[minlag:])
        if options.shape[0] == 0:
            warn(f"Zero valid comparisons could be performed minlag width: {minlag} or more apart")
            return mp, mpi

    cov = np.empty(span - minlag)
    # At this point, at least one comparison here is valid
    # and ts[:w] - mu[0] has a valid normalized form. 
    # Nothing else is guaranteed.
    cross_cov(cov, ts, mu, ts[:w] - mu[0])
    
    # If the remaining problem is one contiguous chunk
    # we can compute the actual comparisons naively.
    missing_count = span - np.count_nonzero(isvalidwindow)
    if missing_count == 0:
        # something like
        self_compare(mp[first:last+1], mpi[first:last+1], cov, df, dg, sig, w, minlag, index_offset=0)
    else:
        masked_self_compare_masked(mp, mpi, cov, df, dg, sig, w, minlag, index_offset=0)
    
    return mp, mpi


def mpx(ts, w, query=None, cross_correlation=False, n_jobs=1):
    """
    The MPX algorithm computes the matrix profile without using the FFT.

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix profile for.
    w : int
        The window size.
    query : array_like
        Optionally a query series.
    cross_correlation : bool, Default=False
        Setermine if cross_correlation distance should be returned. It defaults
        to Euclidean Distance.
    n_jobs : int, Default = 1
        Number of cpu cores to use.
    
    Returns
    -------
    dict : profile
        A MatrixProfile data structure.
        
        >>> {
        >>>     'mp': The matrix profile,
        >>>     'pi': The matrix profile 1NN indices,
        >>>     'rmp': The right matrix profile,
        >>>     'rpi': The right matrix profile 1NN indices,
        >>>     'lmp': The left matrix profile,
        >>>     'lpi': The left matrix profile 1NN indices,
        >>>     'metric': The distance metric computed for the mp,
        >>>     'w': The window size used to compute the matrix profile,
        >>>     'ez': The exclusion zone used,
        >>>     'join': Flag indicating if a similarity join was computed,
        >>>     'sample_pct': Percentage of samples used in computing the MP,
        >>>     'data': {
        >>>         'ts': Time series data,
        >>>         'query': Query data if supplied
        >>>     }
        >>>     'class': "MatrixProfile"
        >>>     'algorithm': "mpx"
        >>> }

    """
    ts = np.asarray(ts, dtype='d')
    if ts.ndim != 1:
        raise ValueError("mpx requires a 1D input")
    n_jobs = core.valid_n_jobs(n_jobs)
    is_join = False

    if core.is_array_like(query):
        query = core.to_np_array(query).astype('d')
        is_join = True
        mp, mpi, mpb, mpib = cympx_ab_parallel(ts, query, w, cross_correlation, n_jobs)
    else:
        mp, mpi = cympx_parallel(ts, w, cross_correlation, n_jobs)

    distance_metric = 'euclidean'
    if cross_correlation:
        distance_metric = 'cross_correlation'

    return {
        'mp': mp,
        'pi': mpi,
        'rmp': None,
        'rpi': None,
        'lmp': None,
        'lpi': None,
        'metric': distance_metric,
        'w': w,
        'ez': int(np.floor(w / 4)),
        'join': is_join,
        'sample_pct': 1,
        'data': {
            'ts': ts,
            'query': query
        },
        'class': 'MatrixProfile',
        'algorithm': 'mpx'
    }
