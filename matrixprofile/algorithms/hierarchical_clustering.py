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

from matrixprofile import core
from matrixprofile.algorithms.mpdist import mpdist


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
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            dm[k] = mpdist(X[i], X[j], window_size, threshold=threshold, 
                           n_jobs=n_jobs)
            k = k + 1
    
    return dm


def hierarchical_clusters(X, window_size, t, threshold=0.05, method='single', 
                          depth=2, criterion='distance', n_jobs=1):
    """
    Cluster M time series into hierarchical clusters using agglomerative
    approach. This function is more or less a convenience wrapper around 
    SciPy's scipy.cluster.hierarchy functions, but uses the MPDist algorithm
    to compute distances between each pair of time series.

    Note
    ----
    Memory usage could potentially high depending on the length of your
    time series and how many distances are computed!
    
    Parameters
    ----------
    X : array_like
        An M x N matrix where M is the time series and N is the observations at
        a given time.
    window_size : int
        The window size used to compute the MPDist.
    t : float
        SciPy ....
    threshold : float, Default 0.05
        The percentile in which the MPDist is taken from. By default it is
        set to 0.05 based on empircal research results from the paper. 
        Generally, you should not change this unless you know what you are
        doing! This value must be a float greater than 0 and less than 1.
    method : str, Default single
        Options: {single, complete, average, weighted}
        SciPy ....
    depth : int, Default 2
        A non-negative value more than 0 to specify the number of levels below
        a non-singleton cluster to allow.
    criterion : str, Default distance
        Options: {inconsistent, distance, monocrit, maxclust}
        SciPy ......
    n_jobs : int, Default 1
        The number of cpu cores used to compute the MPDist.
    
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
        >>>     class: hclusters
        >>> }
    """
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

    # TODO: t

    if not isinstance(threshold, float) or threshold <= 0 or threshold >= 1:
        raise ValueError('threshold must be a float greater than 0 and less'\
            ' than 1')
    
    if not isinstance(depth, int) or depth < 1:
        raise ValueError('depth must be an integer greater than 0')

    if method not in valid_methods:
        opts_str = ', '.join(valid_methods)
        raise ValueError('method may only be one of: ' + opts_str)
    
    if criterion not in valid_criterions:
        opts_str = ', '.join(valid_criterions)
        raise ValueError('criterion may only be one of: ' + opts_str)

    Y = pairwise_dist(X, window_size, threshold=threshold, n_jobs=n_jobs)
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
        'class': 'hclusters'
    }
