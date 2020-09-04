# -*- coding: utf-8 -*-
#cython: boundscheck=True, cdivision=True, wraparound=True
import numpy as np
from cython.view cimport array
from libc.math cimport sqrt
from matrixprofile.cycore import muinvn

# These are here for convenience purposes for now
# They can be factored out, as this module should primarily contain the buffering specific codes
#

cpdef windowed_mean(double [::1] ts, double[::1] mu, Py_ssize_t windowlen):
     
    # this could be inferred, but it's safer this way
    if ts.shape[0] < windowlen:
        raise ValueError(f"Window length {windowlen} exceeds the number of elements in the time series {ts.shape[0]}")
    # safer to test this explicitly than infer the last parameter 
    cdef Py_ssize_t windowct = ts.shape[0] - windowlen + 1
    if windowct != mu.shape[0]:
        raise ValueError(f"subsequence count {windowct} does not match output shape {mu.shape[0]}")
    cdef double accum = ts[0];
    cdef double resid = 0;
    cdef Py_ssize_t i
    cdef double m, n, p, q, r, s
    for i in range(1, windowlen):
        m = ts[i]
        p = accum
        accum += m
        q = accum - p
        resid += ((p - (accum - q)) + (m - q))
    mu[0] = (accum + resid) / windowlen
    for i in range(windowlen, ts.shape[0]):
        m = ts[i - windowlen]
        n = ts[i]
        p = accum - m
        q = p - accum
        r = resid + ((accum - (p - q)) - (m + q))
        accum = p + n
        s = accum - p
        resid = r + ((p - (accum - s)) + (n - s))
        mu[i - windowlen + 1] = (accum + resid) / windowlen
    return mu


cpdef windowed_invcnorm(double[::1] ts, double[::1] mu, double[::1] invn, Py_ssize_t windowlen):
    cdef Py_ssize_t windowct = ts.shape[0] - windowlen + 1
    if not (windowct == mu.shape[0] == invn.shape[0]):
        raise ValueError(f"window count {windowct} does not match output shapes {mu.shape[0]} and {invn.shape[0]}") 

    cdef double accum = 0.0
    cdef double m_
    cdef Py_ssize_t i, j

    for i in range(windowct):
        m_ = mu[i]
        accum = 0
        for j in range(i, i + windowlen):
            accum += (ts[j] - m_)**2
        invn[i] = 1/sqrt(accum)
    return invn


cpdef normalize_one(double[::1] out, double[::1] ts, double mu, double sig):
    cdef Py_ssize_t i,j
    for i in range(out.shape[0]):
        out[i] = (ts[i] - mu) / sig


cpdef crosscov(double[::1] out, double[::1] ts, double[::1] mu, double[::1] cmpseq):
    cdef Py_ssize_t sseqct = out.shape[0]
    cdef double accum, m_
    if sseqct != mu.shape[0]:
        raise ValueError
    elif cmpseq.shape[0] != ts.shape[0] - sseqct + 1:
        raise ValueError
    cdef Py_ssize_t i, j
    for i in range(sseqct):
        accum = 0.0
        m_ = mu[i]
        for j in range(cmpseq.shape[0]):
            accum += (ts[i + j] - m_) * cmpseq[j]
        out[i] = accum


cpdef mpx_difeq(double [::1] out, double[::1] ts, double[::1] mu):
    if not (ts.shape[0] == mu.shape[0] == out.shape[0]):
        raise ValueError(f'time series of shape {ts.shape[0]} is incompatible with mean vector mu of shape {mu.shape[0]} and output shape {out.shape[0]}')
    cdef Py_ssize_t i
    for i in range(ts.shape[0]):
        out[i] = ts[i] - mu[i]


cpdef mpx_inner(double[::1] cov,
                double[::1] r_bwd,
                double[::1] r_fwd,
                double[::1] c_bwd,
                double[::1] c_fwd,
                double[::1] invn,
                double[::1] mp,
                int[::1] mpi,
                int minlag,
                int roffset):
    cdef int i, j, diag, row, col, cvpos
    cdef double cv, corr_
    cdef int subseqct = mp.shape[0]
    # check full requirements for shape mismatches
    if not ((cov.shape[0] + minlag) == mp.shape[0] == mpi.shape[0] == invn.shape[0]):
        raise ValueError(f"these should match, cov-minlag:{cov.shape[0]+minlag}, mp:{mp.shape[0]}, mpi:{mpi.shape[0]}, invn:{invn.shape[0]}")
    elif minlag < 1:
        raise ValueError(f"minlag must be a positive integer, received {minlag}")
    for diag in range(minlag, subseqct):
        cvpos = diag - minlag
        cv = cov[cvpos]
        for row in range(subseqct - diag):
            col = diag + row
            if row > 0: 
                cv -= r_bwd[row-1] * c_bwd[col-1]
                cv += r_fwd[row-1] * c_fwd[col-1]
            corr_ = cv * invn[row] * invn[col]
            if corr_ > 1.0:
                corr_ = 1.0
            if corr_ > mp[row]:
                mp[row] = corr_ 
                mpi[row] = col + roffset
            if corr_ > mp[col]:
                mp[col] = corr_
                mpi[col] = row + roffset
    

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

    cdef double[::1] r_bwd = array(shape=(subseqcount-1,), itemsize=sizeof(double), format='d')
    cdef double[::1] c_bwd = array(shape=(subseqcount-1,), itemsize=sizeof(double), format='d')
    cdef double[::1] r_fwd = array(shape=(subseqcount-1,), itemsize=sizeof(double), format='d')
    cdef double[::1] c_fwd = array(shape=(subseqcount-1,), itemsize=sizeof(double), format='d')
   
    for i in range(subseqcount-1):
        r_bwd[i] = ts[i] - mu[i]
        c_bwd[i] = ts[i] - mu_s[i+1]
        r_fwd[i] = ts[i+w] - mu[i+1]
        c_fwd[i] = ts[i+w] - mu_s[i+1]

    cdef double[::1] first_row = array(shape=(w,), itemsize=sizeof(double), format='d')
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


