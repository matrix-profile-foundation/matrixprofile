#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import math

import numpy as np

from matrixprofile import core
from matrixprofile.algorithms.cympx import mpx_parallel as cympx_parallel
from matrixprofile.algorithms.cympx import mpx_ab_parallel as cympx_ab_parallel


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
        Determine if cross_correlation distance should be returned. It defaults
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
    ts = core.to_np_array(ts).astype('d')
    n_jobs = core.valid_n_jobs(n_jobs)
    is_join = False

    if core.is_array_like(query):
        query = core.to_np_array(query).astype('d')
        is_join = True
        mp, mpi, mpb, mpib = cympx_ab_parallel(ts, query, w, 
            int(cross_correlation), n_jobs)
    else:
        mp, mpi = cympx_parallel(ts, w, int(cross_correlation), n_jobs)

    mp = np.asarray(mp)
    mpi = np.asarray(mpi)
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
        'ez': int(np.ceil(w / 4.0)) if is_join else 0,
        'join': is_join,
        'sample_pct': 1,
        'data': {
            'ts': ts,
            'query': query
        },
        'class': 'MatrixProfile',
        'algorithm': 'mpx'
    }
