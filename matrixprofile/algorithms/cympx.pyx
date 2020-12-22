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
from .cympx_inner cimport cross_cov, self_compare, ab_compare, difference_equations

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
    cdef int i, j, diag, offset, threadnum, col, cov_begin, blockwid, diagct
    cdef int n = ts.shape[0]

    # the original implementation allows the minlag to be manually set
    # here it is always w / 4 similar to SCRIMP++
    cdef int minlag = int(floor(w / 4))
    cdef int profile_len = n - w + 1
    
    stats = muinvn(ts, w)
    cdef double[::1] mu = stats[0]
    cdef double[::1] sig = stats[1]
    
    cdef double[::1] df = np.empty(profile_len-1, dtype='d')
    cdef double[::1] dg = np.empty(profile_len-1, dtype='d')
    cdef np.ndarray[np.double_t, ndim=1] mp = np.full(profile_len, -1.0, dtype='d')
    cdef np.ndarray[np.int_t, ndim=1] mpi = np.full(profile_len, -1, dtype='int')
    
    cdef double[:,::1] tmp_mp = np.full((n_jobs, profile_len), -1.0, dtype='d')
    cdef np.int_t[:,::1] tmp_mpi = np.full((n_jobs, profile_len), -1, dtype='int')
    cdef double[::1] cov = np.empty(profile_len - minlag, dtype='d')   
    cdef double[::1] first_seq = np.empty(w, dtype='d')

    for i in range(w):
        first_seq[i] = ts[i] - mu[0]
    cross_cov(cov, ts[minlag:], mu[minlag:], first_seq)
    
    # this is where we compute the diagonals and later the matrix profile
    difference_equations(df, dg, ts, mu, w)

    diagct = profile_len - minlag
    blockwid = diagct // n_jobs

    if blockwid == 0:
        # fall back to sequential in this case, since it indicates a small problem anyway
        blockwid = diagct
        n_jobs = 1

    for diag in prange(minlag, profile_len, blockwid, num_threads=n_jobs, nogil=True):
        cov_begin = diag - minlag
        threadnum = openmp.omp_get_thread_num()
        self_compare(tmp_mp[threadnum, :], tmp_mpi[threadnum,:], cov[cov_begin:cov_begin+blockwid], df, dg, sig, w, diag, 0)

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
    
    return mp, mpi


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
    cdef double cov_, corr_, eucdist, mxdist

    cdef int profile_len_a = ts.shape[0] - w + 1
    cdef int profile_len_b = query.shape[0] - w + 1

    stats_a = muinvn(ts, w)
    cdef double[::1] mu_a = stats_a[0]
    cdef double[::1] sig_a = stats_a[1]

    stats_b = muinvn(query, w)
    cdef double[::1] mu_b = stats_b[0]
    cdef double[::1] sig_b = stats_b[1]
    
    cdef double[::1] df_a = np.empty(profile_len_a-1, dtype='d')
    cdef double[::1] dg_a = np.empty(profile_len_a-1, dtype='d')
    cdef double[::1] df_b = np.empty(profile_len_b-1, dtype='d')
    cdef double[::1] dg_b = np.empty(profile_len_b-1, dtype='d')

    cdef np.ndarray[np.double_t, ndim=1] mp_a = np.full(profile_len_a, -1.0, dtype='d')
    cdef np.ndarray[np.int_t, ndim=1] mpi_a = np.full(profile_len_a, -1, dtype='int')
    cdef np.ndarray[np.double_t, ndim=1] mp_b = np.full(profile_len_b, -1.0, dtype='d')
    cdef np.ndarray[np.int_t, ndim=1] mpi_b = np.full(profile_len_b, -1, dtype='int')
    
    cdef double[:,::1] tmp_mp_a = np.full((n_jobs, profile_len_a), -1.0, dtype='d')
    cdef np.int_t[:,::1] tmp_mpi_a = np.full((n_jobs, profile_len_a), -1, dtype='int')
    cdef double[:,::1] tmp_mp_b = np.full((n_jobs, profile_len_b), -1.0, dtype='d')
    cdef np.int_t[:,::1] tmp_mpi_b = np.full((n_jobs, profile_len_b), -1, dtype='int')

    cdef double[::1] cov = np.empty(profile_len_a, dtype='d')
    cdef double[::1] covb = np.empty(profile_len_b, dtype='d')
    cdef double[::1] first_seq = np.empty(w, dtype='d')

    for i in range(w):
        first_seq[i] = query[i] - mu_b[0]
    
    cross_cov(cov, ts, mu_a, first_seq)

    for i in range(w):
        first_seq[i] = ts[i] - mu_a[0]

    cross_cov(covb, query, mu_b, first_seq)
    
    # # this is where we compute the diagonals and later the matrix profile
    difference_equations(df_a, dg_a, ts, mu_a, w)
    difference_equations(df_b, dg_b, query, mu_b, w)

    # Todo: a non-uniform partitioning of diagonals would better balance workloads across threads
    #       but this requires sub-routines capable of processing scheduling information
    #       and outputting an index sequence for the starting position of each block. 
    cdef int blockwid = profile_len_a // n_jobs

    if blockwid == 0:
        # fall back to sequential in this case, since it indicates a small problem anyway
        blockwid = profile_len_a
        n_jobs = 1

    # AB JOIN
    for i in prange(0, profile_len_a, blockwid, num_threads=n_jobs, nogil=True):
        threadnum = openmp.omp_get_thread_num()
        ab_compare(tmp_mp_a[threadnum, i:], 
               tmp_mp_b[threadnum, :], 
               tmp_mpi_a[threadnum, i:], 
               tmp_mpi_b[threadnum, :], 
               cov[i:i+blockwid], 
               df_a[i:], 
               df_b, 
               dg_a[i:], 
               dg_b, 
               sig_a[i:], 
               sig_b, 
               i, 
               0)
    
    blockwid = profile_len_b // n_jobs
    
    if blockwid == 0:
        # fall back to sequential in this case, since it indicates a small problem anyway
        blockwid = profile_len_a
        n_jobs = 1

    # BA JOIN
    for i in prange(0, profile_len_b, blockwid, num_threads=n_jobs, nogil=True):
        threadnum = openmp.omp_get_thread_num()
        ab_compare(tmp_mp_b[threadnum, i:], 
               tmp_mp_a[threadnum, :], 
               tmp_mpi_b[threadnum, i:], 
               tmp_mpi_a[threadnum, :], 
               covb[i:i+blockwid], 
               df_b[i:], 
               df_a, 
               dg_b[i:], 
               dg_a, 
               sig_b[i:], 
               sig_a, 
               i, 
               0)
                                
    # reduce results
    for i in range(tmp_mp_a.shape[0]):
        for j in range(tmp_mp_a.shape[1]):
            if tmp_mp_a[i,j] > mp_a[j]:
                if tmp_mp_a[i, j] > 1.0:
                    mp_a[j] = 1.0
                else:
                    mp_a[j] = tmp_mp_a[i, j]
                
                mpi_a[j] = tmp_mpi_a[i, j]
    
    for i in range(tmp_mp_b.shape[0]):
        for j in range(tmp_mp_b.shape[1]):
            if tmp_mp_b[i,j] > mp_b[j]:
                if tmp_mp_b[i, j] > 1.0:
                    mp_b[j] = 1.0
                else:
                    mp_b[j] = tmp_mp_b[i, j]
                
                mpi_b[j] = tmp_mpi_b[i, j]

    # convert normalized cross correlation to euclidean distance
    mxdist = 2.0 * sqrt(w)
    if cross_correlation == 0:
        for i in range(profile_len_a):
            if mp_a[i] == -1.0:
                mp_a[i] = INFINITY
            else:
                mp_a[i] = sqrt(2.0 * w * (1.0 - mp_a[i]))

        for i in range(profile_len_b):
            if mp_b[i] == -1.0:
                mp_b[i] = INFINITY
            else:
                mp_b[i] = sqrt(2.0 * w * (1.0 - mp_b[i]))
            eucdist = sqrt(2.0 * w * (1.0 - mp_b[i]))
    else:
        for i in range(profile_len_a):
            if mp_a[i] > 1.0:
                mp_a[i] = 1.0

        for i in range(profile_len_b):
            if mp_b[i] > 1.0:
                mp_b[i] = 1.0
    
    return mp_a, mpi_a, mp_b, mpi_b
