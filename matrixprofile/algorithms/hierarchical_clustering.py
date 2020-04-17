#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, inconsistent, fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from matrixprofile import core
from matrixprofile.algorithms.mpdist import mpdist

def hierarchical_clusters(X, window_size, t, n_jobs=1, method='single', 
                          depth=2, criterion='distance'):
    """
    Cluster N time series into hierarchical clusters using agglomerative
    approach. This function is more or less a convenience wrapper around 
    SciPy's scipy.cluster.hierarchy functions, but uses the MPDist algorithm
    to compute distances between each pair of time series.
    
    Parameters
    ----------
    X : array_like
        An N x M matrix where N is the time series and M is the observations at
        a given time.
    window_size : int
        The window size used to compute the MPDist.
    t : float
        SciPy ....
    n_jobs : int, Default 1
        The number of cpu cores used to compute the MPDist.
    method : str, Default single
        Options: {single, complete, average, weighted}
        SciPy ....
    depth : int, Default 2
        A non-negative value more than 0 to specify the number of levels below
        a non-singleton cluster to allow.
    criterion : str, Default distance
        Options: {inconsistent, distance, monocrit, maxclust}
        SciPy ......
    
    Returns
    -------
    dict :
        Clustering statistics, distances and labels.
        
        >>> {
        >>>     pairwise_distances: MPDist between pairs of time series as 
        >>>                         np.ndarray,
        >>>     linkage_matrix: clustering linkage matrix as np.ndarray,
        >>>     inconsistency_statistics: inconsistency stats as np.ndarray,
        >>>     assignments: cluster label associated with input X location as
        >>>                  np.ndarray,
        >>>     cophenet: float the cophenet statistic,
        >>>     cophenet_distances: cophenet distances between pairs of time 
        >>>                         series as np.ndarray
        >>> }
    """
    # TODO: add percentile for MPDist as function argument

    # valid SciPy clustering options to work with custom distance metric
    valid_methods = set(['single', 'complete', 'average', 'weighted'])
    valid_criterions = set([
        'inconsistent', 'distance', 'monocrit', 'maxclust'
    ])
    method = method.lower()
    criterion = criterion.lower()

    # error handling
    if not core.is_array_like(X):
        raise ValueError('X must be array like!')

    X = core.to_np_array(X)
    if len(X.shape) != 2:
        raise ValueError('X must be an N x M 2D matrix!')

    # TODO: t
    
    if not isinstance(depth, int) or depth < 1:
        raise ValueError('depth must be an integer greater than 0')

    if method not in valid_methods:
        opts_str = ', '.join(valid_methods)
        raise ValueError('method may only be one of: ' + opts_str)
    
    if criterion not in valid_criterions:
        opts_str = ', '.join(valid_criterions)
        raise ValueError('criterion may only be one of: ' + opts_str)
    
    def compute_distance(ts, ts_b):
        return mpdist(ts, ts_b, window_size, n_jobs=n_jobs)

    Y = pdist(X, metric=compute_distance)
    Z = linkage(Y, method=method)
    R = inconsistent(Z, d=depth)
    c, coph_dists = cophenet(Z, Y)
    T = fcluster(Z, criterion=criterion, depth=depth, R=R, t=t)
    
    return {
        'pairwise_distances': Y,
        'linkage_matrix': Z,
        'inconsistency_statistics': R,
        'assignments': T,
        'cophenet': c,
        'cophenet_distances': coph_dists,
    }
