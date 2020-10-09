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
cpdef mpx_parallel(double[:] ts, int w, bint cross_correlation=0, int n_jobs=1):
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
    cdef int i, j, diag, offset
    cdef int n = ts.shape[0]

    # the original implementation allows the minlag to be manually set
    # here it is always w / 4 similar to SCRIMP++
    cdef int minlag = int(floor(w / 4))
    cdef int profile_len = n - w + 1
    
    cdef double c, c_cmp

    stats = muinvn(ts, w)
    cdef double[:] mu = stats[0]
    cdef double[:] sig = stats[1]
    
    cdef double[:] df = np.empty(profile_len, dtype='d')
    cdef double[:] dg = np.empty(profile_len, dtype='d')
    cdef np.ndarray[np.double_t, ndim=1] mp = np.full(profile_len, -1, dtype='d')
    cdef np.ndarray[np.int_t, ndim=1] mpi = np.full(profile_len, np.nan, dtype='int')
    
    cdef double[:,:] tmp_mp = np.full((profile_len, n_jobs), -1, dtype='d')
    cdef np.int_t[:,:] tmp_mpi = np.full((profile_len, n_jobs), np.nan, dtype='int')
    
    # this is where we compute the diagonals and later the matrix profile
    df[0] = 0
    dg[0] = 0
    for i in prange(w, n, num_threads=n_jobs, nogil=True):
        df[i - w + 1] = (0.5 * (ts[i] - ts[i - w]))
        dg[i - w + 1] = (ts[i] - mu[i - w + 1]) + (ts[i - w] - mu[i - w])    

    for diag in prange(minlag, profile_len, num_threads=n_jobs, nogil=True):
        c = 0
        for i in range(diag, diag + w):
            c = c + ((ts[i] - mu[diag]) * (ts[i-diag] - mu[0]))

        for offset in range(n - w - diag + 1):
            c = c + df[offset] * dg[offset + diag] + df[offset + diag] * dg[offset]
            c_cmp = c * sig[offset] * sig[offset + diag]
            
            # update the distance profile and profile index
            if c_cmp > tmp_mp[offset, openmp.omp_get_thread_num()]:
                tmp_mp[offset, openmp.omp_get_thread_num()] = c_cmp
                tmp_mpi[offset, openmp.omp_get_thread_num()] = offset + diag
            
            if c_cmp > tmp_mp[offset + diag, openmp.omp_get_thread_num()]:
                if c_cmp > 1:
                    c_cmp = 1
                tmp_mp[offset + diag, openmp.omp_get_thread_num()] = c_cmp
                tmp_mpi[offset + diag, openmp.omp_get_thread_num()] = offset
    
    # combine parallel results...
    for i in range(tmp_mp.shape[0]):
        for j in range(tmp_mp.shape[1]):
            if tmp_mp[i,j] > mp[i]:
                if tmp_mp[i, j] > 1:
                    mp[i] = 1
                else:
                    mp[i] = tmp_mp[i, j]
                mpi[i] = tmp_mpi[i, j]
    
    # convert normalized cross correlation to euclidean distance
    if cross_correlation == 0:
        for i in range(profile_len):
            mp[i] = sqrt(2 * w * (1 - mp[i]))
    
    return (mp, mpi)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef mpx_ab_parallel(double[:] ts, double[:] query, int w, bint cross_correlation=0, int n_jobs=1):
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
    cdef int i, j, k, mx
    cdef int n = ts.shape[0]
    cdef int qn = query.shape[0]
    cdef double cov_, corr_, eucdist, mxdist

    cdef int profile_len = n - w + 1
    cdef int profile_lenb = qn - w + 1

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
    
    cdef double[:,:] tmp_mp = np.full((profile_len, n_jobs), -1, dtype='d')
    cdef np.int_t[:,:] tmp_mpi = np.full((profile_len, n_jobs), np.nan, dtype='int')
    cdef double[:,:] tmp_mpb = np.full((profile_lenb, n_jobs), -1, dtype='d')
    cdef np.int_t[:,:] tmp_mpib = np.full((profile_lenb, n_jobs), np.nan, dtype='int')
    
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
        mx = (profile_len - i) if (profile_len - i) < profile_lenb else profile_lenb

        cov_ = 0
        for j in range(i, i + w):
            cov_ = cov_ + ((ts[j] - mua[i]) * (query[j-i] - mub[0]))

        for j in range(mx):
            cov_ = cov_ + diff_fa[j + i] * diff_gb[j] + diff_ga[j + i] * diff_fb[j]
            corr_ = cov_ * siga[j + i] * sigb[j]

            if corr_ > tmp_mp[j + i, openmp.omp_get_thread_num()]:
                tmp_mp[j + i, openmp.omp_get_thread_num()] = corr_
                tmp_mpi[j + i, openmp.omp_get_thread_num()] = j

            if corr_ > tmp_mpb[j, openmp.omp_get_thread_num()]:
                tmp_mpb[j, openmp.omp_get_thread_num()] = corr_
                tmp_mpib[j, openmp.omp_get_thread_num()] = j + i


    # BA JOIN
    for i in prange(profile_lenb, num_threads=n_jobs, nogil=True):
        mx = (profile_lenb - i) if (profile_lenb - i) < profile_len else profile_len

        cov_ = 0
        for j in range(i, i + w):
            cov_ = cov_ + ((query[j] - mub[i]) * (ts[j-i] - mua[0]))

        for j in range(mx):
            cov_ = cov_ + diff_fb[j + i] * diff_ga[j] + diff_gb[j + i] * diff_fa[j]
            corr_ = cov_ * sigb[j + i] * siga[j]

            if corr_ > tmp_mpb[j + i, openmp.omp_get_thread_num()]:
                tmp_mpb[j + i, openmp.omp_get_thread_num()] = corr_
                tmp_mpib[j + i, openmp.omp_get_thread_num()] = j

            if corr_ > tmp_mp[j, openmp.omp_get_thread_num()]:
                tmp_mp[j, openmp.omp_get_thread_num()] = corr_
                tmp_mpi[j, openmp.omp_get_thread_num()] = j + i
                                
    # reduce results
    for i in range(tmp_mp.shape[0]):
        for j in range(tmp_mp.shape[1]):
            if tmp_mp[i,j] > mp[i]:
                if tmp_mp[i, j] > 1:
                    mp[i] = 1
                else:
                    mp[i] = tmp_mp[i, j]
                
                mpi[i] = tmp_mpi[i, j]
    
    for i in range(tmp_mpb.shape[0]):
        for j in range(tmp_mpb.shape[1]):
            if tmp_mpb[i,j] > mpb[i]:
                if tmp_mpb[i, j] > 1:
                    mpb[i] = 1
                else:
                    mpb[i] = tmp_mpb[i, j]
                
                mpib[i] = tmp_mpib[i, j]

    # convert normalized cross correlation to euclidean distance
    mxdist = 2 * sqrt(w)
    if cross_correlation == 0:
        for i in range(profile_len):
            eucdist = sqrt(2 * w * (1 - mp[i]))
            if eucdist < 0:
                eucdist = 0

            if eucdist == mxdist:
                eucdist = INFINITY
            mp[i] = eucdist

        for i in range(profile_lenb):
            eucdist = sqrt(2 * w * (1 - mpb[i]))
            if eucdist < 0:
                eucdist = 0

            if eucdist == mxdist:
                eucdist = INFINITY
            mpb[i] = eucdist
    else:
        for i in range(profile_len):
            if mp[i] > 1:
                mp[i] = 1

        for i in range(profile_lenb):
            if mpb[i] > 1:
                mpb[i] = 1
    
    return (mp, mpi, mpb, mpib)
