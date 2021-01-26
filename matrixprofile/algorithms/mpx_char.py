# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:48:43 2021

@author: awilkins
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
#import math
from numba import jit, prange


#@jit
def mpx_single_char(ts, w, cross_correlation = 0, n_jobs = 1):
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
    #cdef int i, j, diag, offset, threadnum, col
    n = ts.shape[0]

    # the original implementation allows the minlag to be manually set
    # here it is always w / 4 similar to SCRIMP++
    #minlag = int(np.ceil(w / 4.0))
    profile_len = n - w + 1
    
    #df = np.empty(profile_len, dtype = 'd')
    #dg = np.empty(profile_len, dtype = 'd')
    mp = np.full(profile_len, -1.0, dtype = 'd')
    mpi = np.full(profile_len, -1, dtype = 'int')
    
    #tmp_mp = np.full((n_jobs, profile_len), -1.0, dtype = 'd')
    #tmp_mpi = np.full((n_jobs, profile_len), -1, dtype = 'int')
    
    # this is where we compute the diagonals and later the matrix profile
    #df[0] = 0
    #dg[0] = 0
    
    # Iterate over every starting location
    for i in range(w, n + 1):
        # Select the next 'w' indices starting at ts[i - w]
        source = ts[i - w: i]
        dist = np.inf
        # Slide the starting location
        for j in range(w, n + 1):
            # Make sure not to compare with itself
            if j == i:
                continue
            # Select the next 'w' indices starting at ts[j - w]
            target = ts[j - w: j]
            # Measure the Hamming distance
            tmp_dist = editDistance(target, source)
            # If it beats the best so far, update mp and mpi
            if tmp_dist < dist:
                dist = tmp_dist
                mp[i - w] = dist
                mpi[i - w] = j - w
                # Add an early stopping criteria
                if dist == 0:
                    break
    
    
    # for i in prange(w, n, num_threads=n_jobs, nogil=True):
    #     df[i - w + 1] = (0.5 * (ts[i] - ts[i - w]))
    #     dg[i - w + 1] = (ts[i] - mu[i - w + 1]) + (ts[i - w] - mu[i - w])    

    # for diag in prange(minlag + 1, profile_len, num_threads=n_jobs, nogil=True):
    #     c = 0
    #     threadnum = openmp.omp_get_thread_num()
    #     for i in range(diag, diag + w):
    #         c = c + ((ts[i] - mu[diag]) * (ts[i-diag] - mu[0]))

    #     for offset in range(n - w - diag + 1):
    #         col = offset + diag
    #         c = c + df[offset] * dg[col] + df[col] * dg[offset]
    #         c_cmp = c * sig[offset] * sig[col]
            
    #         # update the distance profile and profile index
    #         if c_cmp > tmp_mp[threadnum, offset]:
    #             tmp_mp[threadnum, offset] = c_cmp
    #             tmp_mpi[threadnum, offset] = col
            
    #         if c_cmp > tmp_mp[threadnum, col]:
    #             if c_cmp > 1.0:
    #                 c_cmp = 1.0
    #             tmp_mp[threadnum, col] = c_cmp
    #             tmp_mpi[threadnum, col] = offset
    
    # # combine parallel results...
    # for i in range(tmp_mp.shape[0]):
    #     for j in range(tmp_mp.shape[1]):
    #         if tmp_mp[i,j] > mp[j]:
    #             if tmp_mp[i, j] > 1.0:
    #                 mp[j] = 1.0
    #             else:
    #                 mp[j] = tmp_mp[i, j]
    #             mpi[j] = tmp_mpi[i, j]
    
    # # convert normalized cross correlation to euclidean distance
    # if cross_correlation == 0:
    #     for i in range(profile_len):
    #         mp[i] = sqrt(2.0 * w * (1.0 - mp[i]))
    
    return (mp, mpi)



def editDistance(target, source):
    """
    Returns the Levenshtein distance between two strings or vectors of strings.

    Parameters
    ----------
    target : str, np.ndarray
        String or vector to be compared against source.
    source : str, np.ndarray
        String or vector to be compared to target.  len(source) should be
        greater than or equal to len(target).

    Returns
    -------
    distance : int
        Levenshtein distance between target and source.

    """
    
    # Ensure source is the longer of the two strings
    if len(source) < len(target):
        return editDistance(target, source)
    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the added optimization
    # that we only need the last two rows of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]