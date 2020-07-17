#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

from scipy.cluster.hierarchy import linkage, inconsistent, fcluster
from scipy.cluster.hierarchy import cophenet

from matrixprofile import core
from matrixprofile.algorithms.pairwise_dist import pairwise_dist


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
    t : scalar
        For criteria 'inconsistent', 'distance' or 'monocrit', this is the 
        threshold to apply when forming flat clusters.
        For 'maxclust' criteria, this would be max number of clusters 
        requested.
    threshold : float, Default 0.05
        The percentile in which the MPDist is taken from. By default it is
        set to 0.05 based on empircal research results from the paper. 
        Generally, you should not change this unless you know what you are
        doing! This value must be a float greater than 0 and less than 1.
    method : str, Default single
        The linkage algorithm to use.
        Options: {single, complete, average, weighted}
    depth : int, Default 2
        A non-negative value more than 0 to specify the number of levels below
        a non-singleton cluster to allow.
    criterion : str, Default distance
        Options: {inconsistent, distance, maxclust, monocrit}
        The criterion to use in forming flat clusters.
          ``inconsistent`` :
              If a cluster node and all its
              descendants have an inconsistent value less than or equal
              to `t`, then all its leaf descendants belong to the
              same flat cluster. When no non-singleton cluster meets
              this criterion, every node is assigned to its own
              cluster. (Default)
          ``distance`` :
              Forms flat clusters so that the original
              observations in each flat cluster have no greater a
              cophenetic distance than `t`.
          ``maxclust`` :
              Finds a minimum threshold ``r`` so that
              the cophenetic distance between any two original
              observations in the same flat cluster is no more than
              ``r`` and no more than `t` flat clusters are formed.
          ``monocrit`` :
              Forms a flat cluster from a cluster node c
              with index i when ``monocrit[j] <= t``.
              For example, to threshold on the maximum mean distance
              as computed in the inconsistency matrix R with a
              threshold of 0.8 do::
                  MR = maxRstat(Z, R, 3)
                  cluster(Z, t=0.8, criterion='monocrit', monocrit=MR)
    n_jobs : int, Default 1
        The number of cpu cores used to compute the MPDist.
    
    Returns
    -------
    clusters : dict
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

    if not isinstance(t, (float, int)):
        raise ValueError('t must be a scalar (int or float)')

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
