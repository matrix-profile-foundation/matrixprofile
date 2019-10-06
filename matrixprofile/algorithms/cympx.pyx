# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

from libcpp cimport bool
from libc.math cimport pow
from libc.math cimport floor
cdef extern from "math.h":
    double sqrt(double m)

from numpy cimport ndarray
cimport numpy as np
cimport cython

import numpy as np

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef mpx(double[:] ts, unsigned int w, int cross_correlation):
    """
    The MPX algorithm computes the matrix profile without using the FFT. Right
    now it only supports single dimension self joins.

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix profile for.
    w : int
        The window size.
    cross_correlation : int
        Flag (0, 1) to determine if cross_correlation distance should be
        returned. It defaults to Euclidean Distance (0).
    
    Returns
    -------
    (array_like, array_like) :
        The matrix profile (distance profile, profile index).
    """
    cdef unsigned int i, j, diag, offset
    cdef unsigned int n = ts.shape[0]

    # the original implementation allows the minlag to be manually set
    # here it is always w / 4 similar to SCRIMP++
    cdef unsigned int minlag = int(floor(w / 4))
    cdef unsigned int profile_len = n - w + 1
    
    cdef double p, s, x, z, c, a1, a2, a3, mu_a, c_cmp
    cdef double[:] h = np.empty(n, dtype='d')
    cdef double[:] r = np.empty(n, dtype='d')
    cdef double[:] mu = np.empty(profile_len, dtype='d')
    cdef double[:] sig = np.empty(profile_len, dtype='d')
    
    cdef double[:] df = np.empty(profile_len, dtype='d')
    cdef double[:] dg = np.empty(profile_len, dtype='d')
    cdef np.ndarray[np.double_t, ndim=1] mp = np.full(profile_len, -1, dtype='d')
    cdef np.ndarray[np.int_t, ndim=1] mpi = np.full(profile_len, np.nan, dtype='int')
    
    # mean and std calculations below use the following approach
    # Ogita et al, Accurate Sum and Dot Product
    # results here are a moving average and stable inverse centered norm based
    # on Accurate Sum and Dot Product, Ogita et al
    # compute moving mean
    p = ts[0]
    s = 0

    for i in range(1, w):
        x = p + ts[i]
        z = x - p
        s = s + ((p - (x - z)) + (ts[i] - z))
        p = x
    
    mu[0] = (p + s) / w
    for i in range(w, n):
        x = p - ts[i - w + 1]
        z = x - p
        s = s + ((p - (x - z)) - (ts[i - w] + z))
        p = x

        x = p + ts[i]
        z = x - p
        s = s + ((p - (x - z)) + (ts[i] - z))
        p = x

        mu[i - w + 1] = (p + s) / w
    
    # compute moving standard deviation    
    for i in range(profile_len):
        for j in range(i, i + w):
            mu_a = ts[j] - mu[i]
            h[j] = mu_a * mu_a

            c = (pow(2, 27) + 1) * mu_a
            a1 = (c - (c - mu_a))
            a2 = (mu_a - a1)
            a3 = a1 * a2
            r[j] = a2 * a2 - (((h[j] - a1 * a1) - a3) - a3)

        p = h[i]
        s = r[i]
        for j in range(i + 1, i + w):
            x = p + h[j]
            z = x - p
            s = s + (((p - (x - z)) + (h[j] - z)) + r[j])
            p = x

        sig[i] = 1 / sqrt(p + s)
    
    # this is where we compute the diagonals and later the matrix profile
    df[0] = 0
    for i in range(w, n):
        df[i - w + 1] = (0.5 * (ts[i] - ts[i - w]))
    
    dg[0] = 0
    for i in range(w, n):
        dg[i - w + 1] = (ts[i] - mu[i - w + 1]) + (ts[i - w] - mu[i - w])
    
    for diag in range(minlag, profile_len):
        c = 0
        for i in range(diag, diag + w):
            c = c + ((ts[i] - mu[diag]) * (ts[i-diag] - mu[0]))        
        
        for offset in range(n - w - diag + 1):
            c = c + df[offset] * dg[offset + diag] + df[offset + diag] * dg[offset]
            c_cmp = c * sig[offset] * sig[offset + diag]
            
            # update the distance profile and profile index
            if c_cmp > mp[offset]:
                if c_cmp > 1:
                    c_cmp = 1
                mp[offset] = c_cmp
                mpi[offset] = offset + diag
            
            if c_cmp > mp[offset + diag]:
                if c_cmp > 1:
                    c_cmp = 1
                mp[offset + diag] = c_cmp
                mpi[offset + diag] = offset
    
    # convert normalized cross correlation to euclidean distance
    if cross_correlation == 0:
        for i in range(profile_len):
            mp[i] = sqrt(2 * w * (1 - mp[i]))
    
    return (mp, mpi)