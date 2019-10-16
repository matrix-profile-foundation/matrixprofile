# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

from libc.math cimport pow
from libc.math cimport floor
from libc.math cimport ceil
from libc.math cimport sqrt

from cython.parallel import prange

from numpy cimport ndarray
cimport numpy as np
cimport cython

import numpy as np

from matrixprofile.cycore import muinvn


# cpdef int minInt(int x, int y):
#     return y ^ ((x ^ y) & -(x < y));

# cpdef double minDouble(double x, double y):
#     return y ^ ((x ^ y) & -(x < y));


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef mpx_ab(double[:] ts, double[:] query, unsigned int w, int cross_correlation, int n_jobs):
    """
    The MPX algorithm computes the matrix profile without using the FFT. Right
    now it only supports single dimension self joins.

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix profile for.
    query : array_like
        The query o compute the matrix profile for.
    w : int
        The window size.
    cross_correlation : int
        Flag (0, 1) to determine if cross_correlation distance should be
        returned. It defaults to Euclidean Distance (0).
    n_jobs : int, Default = all
        Number of cpu cores to use. Defaults to using all.
    
    Returns
    -------
    (array_like, array_like) :
        The matrix profile (distance profile, profile index).
    """
    cdef unsigned int i, j, k, mx
    cdef unsigned int n = ts.shape[0]
    cdef unsigned int qn = query.shape[0]
    cdef double cov_, corr_

    cdef unsigned int profile_len = n - w + 1
    cdef unsigned int profile_lenb = qn - w + 1

    stats_a = muinvn(ts, w)
    cdef double[:] mua = stats_a[0]
    cdef double[:] siga = stats_a[1]

    stats_b = muinvn(query, w)
    cdef double[:] mub = stats_b[0]
    cdef double[:] sigb = stats_b[1]
    
    cdef double[:] diff_fa = np.empty(profile_len, dtype='d')
    cdef double[:] diff_ga = np.empty(profile_len, dtype='d')
    cdef double[:] diff_fb = np.empty(profile_lenb, dtype='d')
    cdef double[:] diff_gb = np.empty(profile_lenb, dtype='d')

    cdef np.ndarray[np.double_t, ndim=1] mp = np.full(profile_len, -1, dtype='d')
    cdef np.ndarray[np.int_t, ndim=1] mpi = np.full(profile_len, np.nan, dtype='int')
    cdef np.ndarray[np.double_t, ndim=1] mpb = np.full(profile_lenb, -1, dtype='d')
    cdef np.ndarray[np.int_t, ndim=1] mpib = np.full(profile_lenb, np.nan, dtype='int')
    
    diff_ga[0] = 0
    diff_fb[0] = 0
    diff_gb[0] = 0

    # # this is where we compute the diagonals and later the matrix profile
    diff_fa[0] = 0    
    for i in prange(w, n, num_threads=n_jobs, nogil=True):
        diff_fa[i - w + 1] = (0.5 * (ts[i] - ts[i - w]))

    diff_fb[0] = 0    
    for i in prange(w, qn, num_threads=n_jobs, nogil=True):
        diff_fa[i - w + 1] = (0.5 * (query[i] - query[i - w]))
    
    diff_ga[0] = 0
    for i in prange(w, n, num_threads=n_jobs, nogil=True):
        diff_ga[i - w + 1] = (ts[i] - mua[i - w + 1]) + (ts[i - w] - mua[i - w])

    diff_gb[0] = 0
    for i in prange(w, qn, num_threads=n_jobs, nogil=True):
        diff_ga[i - w + 1] = (query[i] - mub[i - w + 1]) + (query[i - w] - mub[i - w])


    for i in prange(profile_len, num_threads=n_jobs, nogil=True):
        mx = (profile_len - i) if (profile_len - i) < profile_lenb else profile_lenb

        cov_ = 0
        for j in range(i, i + w):
            cov_ = cov_ + ((ts[i] - mua[i]) * (query[j-i] - mub[0]))

        for j in range(mx):
            cov_ = cov_ + diff_fa[j + i] * diff_gb[j] + diff_ga[j + i] * diff_fb[j]
            corr_ = cov_ * siga[j + i] * sigb[j]

            if corr_ > mp[j + i]:
                mp[j + i] = corr_ if corr_ < 1.0 else 1.0
                mpi[j + i] = j

            if corr_ > mpb[j]:
                mpb[j] = corr_ if corr_ < 1.0 else 1.0
                mpib[j] = j + i


    # convert normalized cross correlation to euclidean distance
    if cross_correlation == 0:
        for i in range(profile_len):
            mp[i] = sqrt(2 * w * (1 - mp[i]))

        for i in range(profile_lenb):
            mpb[i] = sqrt(2 * w * (1 - mpb[i]))
    
    return (mp, mpi, mpb, mpib)