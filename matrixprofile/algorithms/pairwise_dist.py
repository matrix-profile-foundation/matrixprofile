#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile import core
from matrixprofile.algorithms.mpdist import mpdist

def compute_dist(args):
    """
    Helper function to parallelize pairwise distance calculation.

    Parameters
    ----------
    args : tuple
        The arguments to pass to the mpdist calculation.
    
    Returns
    -------
    values : tuple
        The kth index and distance.
    """
    k = args[0]
    distance = mpdist(args[1], args[2], args[3], threshold=args[4])

    return (k, distance)


def pairwise_dist(X, window_size, threshold=0.05, n_jobs=1):
    """
    Utility function to compute all pairwise distances between the timeseries
    using MPDist. 
    
    Note
    ----
    scipy.spatial.distance.pdist cannot be used because they
    do not allow for jagged arrays, however their code was used as a reference
    in creating this function.
    https://github.com/scipy/scipy/blob/master/scipy/spatial/distance.py#L2039

    Parameters
    ----------
    X : array_like
        An array_like object containing time series to compute distances for.
    window_size : int
        The window size to use in computing the MPDist.
    threshold : float
        The threshold used to compute MPDist.
    n_jobs : int
        Number of CPU cores to use during computation.
    
    Returns
    -------
    Y : np.ndarray
        Returns a condensed distance matrix Y.  For
        each :math:`i` and :math:`j` (where :math:`i<j<m`),where m is the 
        number of original observations. The metric ``dist(u=X[i], v=X[j])``
        is computed and stored in entry ``ij``.
    """
    if not core.is_array_like(X):
        raise ValueError('X must be array_like!')
    
    # identify shape based on iterable or np.ndarray.shape
    m = 0
    
    if isinstance(X, np.ndarray) and len(X.shape) == 2:
        m = X.shape[0]
    else:
        m = len(X)
    
    dm = np.empty((m * (m - 1)) // 2, dtype=np.double)
    k = 0

    if n_jobs == 1:
        for i in range(0, m - 1):
            for j in range(i + 1, m):
                dm[k] = mpdist(X[i], X[j], window_size, threshold=threshold, 
                            n_jobs=n_jobs)
                k = k + 1
    else:
        args = []
        for i in range(0, m - 1):
            for j in range(i + 1, m):
                args.append((k, X[i], X[j], window_size, threshold))
                k = k + 1
        
        with core.mp_pool()(n_jobs) as pool:
            results = pool.map(compute_dist, args)
        
        # put results in the matrix
        for result in results:
            dm[result[0]] = result[1]
    
    return dm
