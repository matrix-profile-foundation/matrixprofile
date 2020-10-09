# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

from libc.math cimport floor
from libc.math cimport ceil
from libc.math cimport sqrt

from cython.parallel import prange

from numpy cimport ndarray
cimport numpy as np
cimport cython
cimport openmp
from numpy.math cimport INFINITY

import numpy as np

from matrixprofile.cycore import muinvn


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef mpx_parallel(double[::1] ts, int w, bint cross_correlation=0, int n_jobs=1):
    """
    The MPX algorithm computes the matrix profile without using the FFT. Right
    now it only supports single dimension self joins.

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
    cdef int i, j, diag, offset, threadnum, col
    cdef int n = ts.shape[0]

    # the original implementation allows the minlag to be manually set
    # here it is always w / 4 similar to SCRIMP++
    cdef int minlag = int(ceil(w / 4.0))
    cdef int profile_len = n - w + 1
    
    cdef double c, c_cmp

    stats = muinvn(ts, w)
    cdef double[::1] mu = stats[0]
    cdef double[::1] sig = stats[1]
    
    cdef double[::1] df = np.empty(profile_len, dtype='d')
    cdef double[::1] dg = np.empty(profile_len, dtype='d')
    cdef np.ndarray[np.double_t, ndim=1] mp = np.full(profile_len, -1.0, dtype='d')
    cdef np.ndarray[np.int_t, ndim=1] mpi = np.full(profile_len, -1, dtype='int')
    
    cdef double[:,::1] tmp_mp = np.full((n_jobs, profile_len), -1.0, dtype='d')
    cdef np.int_t[:,::1] tmp_mpi = np.full((n_jobs, profile_len), -1, dtype='int')
    
    # this is where we compute the diagonals and later the matrix profile
    df[0] = 0
    dg[0] = 0
    for i in prange(w, n, num_threads=n_jobs, nogil=True):
        df[i - w + 1] = (0.5 * (ts[i] - ts[i - w]))
        dg[i - w + 1] = (ts[i] - mu[i - w + 1]) + (ts[i - w] - mu[i - w])    

    for diag in prange(minlag + 1, profile_len, num_threads=n_jobs, nogil=True):
        c = 0
        threadnum = openmp.omp_get_thread_num()
        for i in range(diag, diag + w):
            c = c + ((ts[i] - mu[diag]) * (ts[i-diag] - mu[0]))

        for offset in range(n - w - diag + 1):
            col = offset + diag
            c = c + df[offset] * dg[col] + df[col] * dg[offset]
            c_cmp = c * sig[offset] * sig[col]
            
            # update the distance profile and profile index
            if c_cmp > tmp_mp[threadnum, offset]:
                tmp_mp[threadnum, offset] = c_cmp
                tmp_mpi[threadnum, offset] = col
            
            if c_cmp > tmp_mp[threadnum, col]:
                if c_cmp > 1.0:
                    c_cmp = 1.0
                tmp_mp[threadnum, col] = c_cmp
                tmp_mpi[threadnum, col] = offset
    
    # combine parallel results...
    for i in range(tmp_mp.shape[0]):
        for j in range(tmp_mp.shape[1]):
            if tmp_mp[i,j] > mp[j]:
                if tmp_mp[i, j] > 1.0:
                    mp[j] = 1.0
                else:
                    mp[j] = tmp_mp[i, j]
                mpi[j] = tmp_mpi[i, j]
    
    # convert normalized cross correlation to euclidean distance
    if cross_correlation == 0:
        for i in range(profile_len):
            mp[i] = sqrt(2.0 * w * (1.0 - mp[i]))
    
    return (mp, mpi)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef mpx_ab_parallel(double[::1] ts, double[::1] query, int w, bint cross_correlation=0, int n_jobs=1):
    """
    The MPX algorithm computes the matrix profile without using the FFT. This
    specific implementation includes similarity join (AB join).

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix profile for.
    query : array_like
        The query o compute the matrix profile for.
    w : int
        The window size.
    cross_correlation : bint
        Flag (0, 1) to determine if cross_correlation distance should be
        returned. It defaults to Euclidean Distance (0).
    n_jobs : int, Default = 1
        Number of cpu cores to use.
    
    Returns
    -------
    (array_like, array_like, array_like, array_like) :
        The matrix profile (distance profile, profile index, dist..b, prof..b).

    """
    cdef int i, j, k, mx, threadnum
    cdef int n = ts.shape[0]
    cdef int qn = query.shape[0]
    cdef double cov_, corr_, eucdist, mxdist

    cdef int profile_len = n - w + 1
    cdef int profile_lenb = qn - w + 1

    stats_a = muinvn(ts, w)
    cdef double[::1] mua = stats_a[0]
    cdef double[::1] siga = stats_a[1]

    stats_b = muinvn(query, w)
    cdef double[::1] mub = stats_b[0]
    cdef double[::1] sigb = stats_b[1]
    
    cdef double[::1] diff_fa = np.empty(profile_len, dtype='d')
    cdef double[::1] diff_ga = np.empty(profile_len, dtype='d')
    cdef double[::1] diff_fb = np.empty(profile_lenb, dtype='d')
    cdef double[::1] diff_gb = np.empty(profile_lenb, dtype='d')

    cdef np.ndarray[np.double_t, ndim=1] mp = np.full(profile_len, -1.0, dtype='d')
    cdef np.ndarray[np.int_t, ndim=1] mpi = np.full(profile_len, -1, dtype='int')
    cdef np.ndarray[np.double_t, ndim=1] mpb = np.full(profile_lenb, -1.0, dtype='d')
    cdef np.ndarray[np.int_t, ndim=1] mpib = np.full(profile_lenb, -1, dtype='int')
    
    cdef double[:,::1] tmp_mp = np.full((n_jobs, profile_len), -1.0, dtype='d')
    cdef np.int_t[:,::1] tmp_mpi = np.full((n_jobs, profile_len), -1, dtype='int')
    cdef double[:,::1] tmp_mpb = np.full((n_jobs, profile_lenb), -1.0, dtype='d')
    cdef np.int_t[:,::1] tmp_mpib = np.full((n_jobs, profile_lenb), -1, dtype='int')
    
    # # this is where we compute the diagonals and later the matrix profile
    diff_fa[0] = 0
    diff_ga[0] = 0
    for i in prange(w, n, num_threads=n_jobs, nogil=True):
        diff_fa[i - w + 1] = (0.5 * (ts[i] - ts[i - w]))
        diff_ga[i - w + 1] = (ts[i] - mua[i - w + 1]) + (ts[i - w] - mua[i - w])

    diff_fb[0] = 0
    diff_gb[0] = 0
    for i in prange(w, qn, num_threads=n_jobs, nogil=True):
        diff_fb[i - w + 1] = (0.5 * (query[i] - query[i - w]))
        diff_gb[i - w + 1] = (query[i] - mub[i - w + 1]) + (query[i - w] - mub[i - w])

    # AB JOIN
    for i in prange(profile_len, num_threads=n_jobs, nogil=True):
        threadnum = openmp.omp_get_thread_num()
        mx = (profile_len - i) if (profile_len - i) < profile_lenb else profile_lenb

        cov_ = 0
        for j in range(i, i + w):
            cov_ = cov_ + ((ts[j] - mua[i]) * (query[j-i] - mub[0]))

        for j in range(mx):
            k = j + i
            cov_ = cov_ + diff_fa[k] * diff_gb[j] + diff_ga[k] * diff_fb[j]
            corr_ = cov_ * siga[k] * sigb[j]

            if corr_ > tmp_mp[threadnum, k]:
                tmp_mp[threadnum, k] = corr_
                tmp_mpi[threadnum, k] = j

            if corr_ > tmp_mpb[threadnum, j]:
                tmp_mpb[threadnum, j] = corr_
                tmp_mpib[threadnum, j] = k


    # BA JOIN
    for i in prange(profile_lenb, num_threads=n_jobs, nogil=True):
        threadnum = openmp.omp_get_thread_num()
        mx = (profile_lenb - i) if (profile_lenb - i) < profile_len else profile_len

        cov_ = 0
        for j in range(i, i + w):
            cov_ = cov_ + ((query[j] - mub[i]) * (ts[j-i] - mua[0]))

        for j in range(mx):
            k = j + i
            cov_ = cov_ + diff_fb[k] * diff_ga[j] + diff_gb[k] * diff_fa[j]
            corr_ = cov_ * sigb[k] * siga[j]

            if corr_ > tmp_mpb[threadnum, k]:
                tmp_mpb[threadnum, k] = corr_
                tmp_mpib[threadnum, k] = j

            if corr_ > tmp_mp[threadnum, j]:
                tmp_mp[threadnum, j] = corr_
                tmp_mpi[threadnum, j] = k
                                
    # reduce results
    for i in range(tmp_mp.shape[0]):
        for j in range(tmp_mp.shape[1]):
            if tmp_mp[i,j] > mp[j]:
                if tmp_mp[i, j] > 1.0:
                    mp[j] = 1.0
                else:
                    mp[j] = tmp_mp[i, j]
                
                mpi[j] = tmp_mpi[i, j]
    
    for i in range(tmp_mpb.shape[0]):
        for j in range(tmp_mpb.shape[1]):
            if tmp_mpb[i,j] > mpb[j]:
                if tmp_mpb[i, j] > 1.0:
                    mpb[j] = 1.0
                else:
                    mpb[j] = tmp_mpb[i, j]
                
                mpib[j] = tmp_mpib[i, j]

    # convert normalized cross correlation to euclidean distance
    mxdist = 2.0 * sqrt(w)
    if cross_correlation == 0:
        for i in range(profile_len):
            if mp[i] == -1.0:
                mp[i] = INFINITY
            else:
                mp[i] = sqrt(2.0 * w * (1.0 - mp[i]))

        for i in range(profile_lenb):
            if mpb[i] == -1.0:
                mpb[i] = INFINITY
            else:
                mpb[i] = sqrt(2.0 * w * (1.0 - mpb[i]))
            eucdist = sqrt(2.0 * w * (1.0 - mpb[i]))
    else:
        for i in range(profile_len):
            if mp[i] > 1.0:
                mp[i] = 1.0

        for i in range(profile_lenb):
            if mpb[i] > 1.0:
                mpb[i] = 1.0
    
    return (mp, mpi, mpb, mpib)
