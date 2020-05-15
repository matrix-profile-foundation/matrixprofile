# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate


from libc.math cimport pow
cdef extern from "math.h":
    double sqrt(double m)

from numpy cimport ndarray
cimport numpy as np
cimport cython

import numpy as np


@cython.boundscheck(False)
@cython.cdivision(True)
def muinvn(double[:] a, unsigned int w):
    """
    Computes the moving average and standard deviation over the provided
    array. This approach uses Welford's method. It leads to more precision
    in the results.

    Parameters
    ----------
    a : array_like
        The array to compute statistics on.
    w : int
        The window size.
    
    Returns
    -------
    (array_like, array_like) :
        The (mu, sigma) arrays respectively.

    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = a.shape[0]
    cdef double p, s, x, z, c, a1, a2, a3, mu_a
    cdef Py_ssize_t profile_len = n - w + 1
    cdef double[:] h = np.empty(n, dtype='d')
    cdef double[:] r = np.empty(n, dtype='d')
    cdef double[:] mu = np.empty(profile_len, dtype='d')
    cdef double[:] sig = np.empty(profile_len, dtype='d')
    
    # compute moving mean
    p = a[0]
    s = 0
    for i in range(1, w):
        x = p + a[i]
        z = x - p
        s = s + ((p - (x - z)) + (a[i] - z))
        p = x
    
    mu[0] = (p + s) / w
    for i in range(w, n):
        x = p - a[i - w + 1]
        z = x - p
        s = s + ((p - (x - z)) - (a[i - w] + z))
        p = x

        x = p + a[i]
        z = x - p
        s = s + ((p - (x - z)) + (a[i] - z))
        p = x

        mu[i - w + 1] = (p + s) / w
    
    # compute moving standard deviation    
    for i in range(profile_len):
        for j in range(i, i + w):
            mu_a = a[j] - mu[i]
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

        if p + s == 0:
            sig[i] = 0
        else:
            sig[i] = 1 / sqrt(p + s)
    
    return (mu, sig)


@cython.boundscheck(False)
@cython.cdivision(True)
def moving_avg_std(double[:] a, unsigned int w):
    """
    Computes the moving average and standard deviation over the provided
    array.

    Parameters
    ----------
    a : array_like
        The array to compute statistics on.
    w : int
        The window size.
    
    Returns
    -------
    (array_like, array_like) :
        The (mu, sigma) arrays respectively.

    """
    cdef Py_ssize_t i
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t ws = w
    cdef Py_ssize_t profile_len = n - w + 1
    cdef double[:] cumsum = np.empty(n, dtype='d')
    cdef double[:] sq_cumsum = np.empty(n, dtype='d')
    cdef double[:] sums = np.empty(profile_len, dtype='d')
    cdef double[:] sq_sums = np.empty(profile_len, dtype='d')
    cdef double[:] mu = np.empty(profile_len, dtype='d')
    cdef double[:] sig_sq = np.empty(profile_len, dtype='d')
    cdef double[:] sig = np.empty(profile_len, dtype='d')
    
    cumsum[0] = a[0]
    sq_cumsum[0] = a[0] * a[0]
    for i in range(1, n):
        cumsum[i] = a[i] + cumsum[i - 1]
        sq_cumsum[i] = a[i] * a[i] + sq_cumsum[i - 1]
    
    sums[0] = cumsum[w - 1]
    sq_sums[0] = sq_cumsum[w - 1]
    for i in range(n - w):
        sums[i + 1] = cumsum[w + i] - cumsum[i]
        sq_sums[i + 1] = sq_cumsum[w + i] - sq_cumsum[i]
    
    for i in range(profile_len):
        mu[i] = sums[i] / w
        sig_sq[i] = sq_sums[i] / w - mu[i] * mu[i]

        if sig_sq[i] < 0:
            sig[i] = 0
        else:
            sig[i] = sqrt(sig_sq[i])
    
    return (mu, sig)