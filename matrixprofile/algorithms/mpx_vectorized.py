# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate


import math

from matrixprofile import core
from matrixprofile import cycore

import numpy as np

def mpx_vectorized(ts, w):
    """
    Computes the matrix profile using an optimized vectorization approach. This
    algorithm only supports single dimension self joins at this time.

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
    ts = core.to_np_array(ts)
    n = len(ts)
    minlag = int(math.floor(w / 4))
    
    mu, sig = cycore.muinvn(ts, w)
    
    df = np.append([0,], (1 / 2) * (ts[w:n] - ts[0:(n - w)]))
    dg = np.append([0,], (ts[w:n] - mu[1:(n - w + 1)]) + (ts[0:(n - w)] - mu[0:(n - w)]))
    diagmax = n - w + 1
    
    mp = np.full(diagmax, -1, dtype='float64')
    mpi = np.full(diagmax, np.nan, dtype=int)
    
    seq_diag = np.arange(minlag, diagmax)
    seq_diag = np.random.choice(seq_diag, size=len(seq_diag), replace=False)
    
    for diag in seq_diag:
        c = np.sum((ts[diag:(diag + w)] - mu[diag]) * (ts[0:w] - mu[0]))
        
        offset = np.arange(0, n - w - diag + 1)
        off_diag = offset + diag
        d = df[offset] * dg[off_diag] + df[off_diag] * dg[offset]
        d[0] = d[0] + c
        d = np.cumsum(d)
        
        d_cmp = d * sig[offset] * sig[off_diag]
        mask = d_cmp > mp[offset]
        indices = np.append(mask, np.full(diag, 0, dtype='bool'))
        mp[indices] = d_cmp[mask]
        mpi[indices] = off_diag[mask]
        
        mask = d_cmp > mp[off_diag]
        indices = np.append(np.full(diag, 0, dtype='bool'), mask)
        mp[indices] = d_cmp[mask]
        mpi[indices] = off_diag[mask]
    
    mp = np.sqrt(2 * w * (1 - mp))
    
    return (mp, mpi)