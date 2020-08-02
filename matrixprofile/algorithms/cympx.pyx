# -*- coding: utf-8 -*-
#cython: boundscheck=False, cdivision=True, wraparound=False

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
from cython.view cimport array as cvarray
from numpy cimport ndarray
cimport numpy as np
cimport cython
cimport openmp
from numpy.math cimport INFINITY

import numpy as np

from matrixprofile.cycore import muinvn


cpdef mpx_parallel(double[::1] ts, int w, int cross_correlation, int n_jobs):
    """
    The MPX algorithm computes the matrix profile without using the FFT. Right
    now it only supports single dimension self joins. 
 
    The experimental version uses a slightly simpler factorization in an effort to 
    avoid cases of very ill conditioned products on time series with streams of zeros
    or missing data.

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix profile for.
    w : int
        The window size.
    cross_correlation : int
        Flag (0, 1) to determine if cross_correlation distance should be
        returned. It defaults to Euclidean Distance (0).
    n_jobs : int, Default = 1
        Number of cpu cores to use.
    
    Returns
    -------
    (array_like, array_like) :
        The matrix profile (distance profile, profile index).

    """
    if w < 2: 
        raise ValueError('subsequence length is too short to admit a normalized representation')
    
    cdef int i, j, diag, row, col
    cdef double cov_, corr_
    
    cdef int minlag = w // 4
    cdef int subseqcount = ts.size - w + 1

    if subseqcount < 1 + minlag:  
        raise ValueError('time series is too short relative to subsequence length w')
    
    cdef double[::1] mu, mu_s, invnorm
    mu, invnorm  = muinvn(ts, w)
    mu_s, _ = muinvn(ts[:ts.size-1], w-1)
    mprof_ = np.empty(subseqcount, dtype='d')
    mprofidx_ = np.empty(subseqcount, dtype='i')
    cdef double[::1] mprof = mprof_
    cdef int[::1] mprofidx = mprofidx_ 

    for i in range(subseqcount):
        mprof[i] = -1.0
        mprofidx[i] = -1
    
    cdef double[::1] r_bwd = cvarray(shape=(subseqcount-1,), itemsize=sizeof(double), format='d')
    cdef double[::1] c_bwd = cvarray(shape=(subseqcount-1,), itemsize=sizeof(double), format='d')
    cdef double[::1] r_fwd = cvarray(shape=(subseqcount-1,), itemsize=sizeof(double), format='d')
    cdef double[::1] c_fwd = cvarray(shape=(subseqcount-1,), itemsize=sizeof(double), format='d')
   
    for i in range(subseqcount-1):
        r_bwd[i] = ts[i] - mu[i]
        c_bwd[i] = ts[i] - mu_s[i+1]
        r_fwd[i] = ts[i+w] - mu[i+1]
        c_fwd[i] = ts[i+w] - mu_s[i+1]

    cdef double[::1] first_row = cvarray(shape=(w,), itemsize=sizeof(double), format='d')
    cdef double m_ = mu[0]
    for i in range(w):
        first_row[i] = ts[i] - m_     

    for diag in range(minlag, subseqcount):
        cov_ = 0 
        for i in range(diag, diag + w):
            cov_ += (ts[i] - mu[diag]) * first_row[i-diag]

        for row in range(subseqcount - diag):
            col = diag + row
            if row > 0: 
                cov_ -= r_bwd[row-1] * c_bwd[col-1] 
                cov_ += r_fwd[row-1] * c_fwd[col-1]
            corr_ = cov_ * invnorm[row] * invnorm[col]
            if corr_ > 1.0:
                corr_ = 1.0
            if corr_ > mprof[row]:
                mprof[row] = corr_ 
                mprofidx[row] = col 
            if corr_ > mprof[col]:
                mprof[col] = corr_
                mprofidx[col] = row
    
    if cross_correlation == 0:
        for i in range(subseqcount):
            mprof[i] = sqrt(2 * w * (1 - mprof[i]))
    
    return mprof_, mprofidx_


cpdef mpx_ab_parallel(double[:] ts, double[:] query, int w, int cross_correlation, int n_jobs):
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
    cross_correlation : int
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
    elif cross_correlation == 1:
        for i in range(profile_len):
            if mp[i] > 1:
                mp[i] = 1

        for i in range(profile_lenb):
            if mpb[i] > 1:
                mpb[i] = 1
    
    return (mp, mpi, mpb, mpib)
