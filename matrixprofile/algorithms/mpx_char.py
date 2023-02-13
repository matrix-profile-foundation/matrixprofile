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
def mpx_single_char(ts, w):
    """
    The MPX algorithm computes the matrix profile using Hamming distance.

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix profile for.
    w : int
        The window size.
    
    Returns
    -------
    (array_like, array_like) :
        The matrix profile (distance profile, profile index).

    """
    n = ts.shape[0]

    profile_len = n - w + 1
    
    mp = np.full(profile_len, -1.0, dtype = 'd')
    mpi = np.full(profile_len, -1, dtype = 'int')
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