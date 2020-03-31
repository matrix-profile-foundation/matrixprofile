# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate


import numpy as np

from matrixprofile import core


def pmp_top_k_discords(profile, exclusion_zone=None, k=3):
    """
    Computes the top K discords for the given Pan-MatrixProfile. The return
    values is a list of row by col indices.

    Notes
    -----
    This algorithm is written to work with Euclidean distance. If you submit
    a PMP of Pearson metrics, then it is first converted to Euclidean.

    Parameters
    ----------
    profile : dict
        Data structure from a PMP algorithm.
    exclusion_zone : int, Default window / 2
        The zone to exclude around the found discords to reduce trivial
        findings. By default we use the row-wise window / 2.
    k : int
        Maximum number of discords to find.

    Returns
    -------
    dict : profile
        A 2D array of indices. The first column corresponds to the row index
        and the second column corresponds to the column index of the 
        submitted PMP. It is placed back on the original object passed in as
        'discords' key.

    """
    if not core.is_pmp_obj(profile):
        raise ValueError('Expecting PMP data structure!')

    # this function requires euclidean distance
    # convert if the metric is pearson
    metric = profile.get('metric', None)
    pmp = profile.get('pmp', None)
    windows = profile.get('windows', None)
    
    tmp = None
    if metric == 'pearson':
        tmp = core.pearson_to_euclidean(pmp, windows)
    else:
        tmp = np.copy(pmp).astype('d')        
    
    # replace nan and infs with -infinity
    # for whatever reason numpy argmax finds infinity as max so
    # this is a way to get around it by converting to -infinity
    tmp[core.nan_inf_indices(tmp)] = -np.inf
            
    # iterate finding the max value k times or until negative
    # infinity is obtained
    found = []
    
    for _ in range(k):
        max_idx = np.unravel_index(np.argmax(tmp), tmp.shape)
        window = windows[max_idx[0]]
        
        if tmp[max_idx] == -np.inf:
            break
        
        found.append(max_idx)
        
        # apply exclusion zone
        # the exclusion zone is based on 1/2 of the window size
        # used to compute that specific matrix profile
        n = tmp[max_idx[0]].shape[0]
        if exclusion_zone is None:
            exclusion_zone = int(np.floor(window / 2))

        ez_start = np.max([0, max_idx[1] - exclusion_zone])
        ez_stop = np.min([n, max_idx[1] + exclusion_zone])
        tmp[max_idx[0]][ez_start:ez_stop] = -np.inf
    
    profile['discords'] = np.array(found)

    return profile


def mp_top_k_discords(profile, exclusion_zone=None, k=3):
    """
    Find the top K number of discords (anomalies) given a matrix profile,
    exclusion zone and the desired number of discords. The exclusion zone
    nullifies entries on the left and right side of the first and subsequent
    discords to remove non-trivial matches. More specifically, a discord found
    at location X will more than likely have additional discords to the left or
    right of it.

    Parameters
    ----------
    profile : dict
        The MatrixProfile data structure.
    exclusion_zone : int, Default mp algorithm ez
        Desired number of values to exclude on both sides of the anomaly.
    k : int
        Desired number of discords to find.

    Returns
    -------
    dict : profile
        The original input profile with an additional "discords" key containing
        the a np.ndarray of discord indices.

    """
    if not core.is_mp_obj(profile):
        raise ValueError('Expecting MP data structure!')

    found = []
    tmp = np.copy(profile.get('mp', None)).astype('d')
    n = len(tmp)

    # TODO: this is based on STOMP standards when this motif finding algorithm
    # originally came out. Should we default this to 4.0 instead? That seems
    # to be the common value now per new research.
    window_size = profile.get('w', None)
    if exclusion_zone is None:
        exclusion_zone = profile.get('ez', None)
    
    # obtain indices in ascending order
    indices = np.argsort(tmp)
    
    # created flipped view for discords
    indices = indices[::-1]

    for idx in indices:
        if not np.isinf(tmp[idx]):
            found.append(idx)

            # apply exclusion zone
            if exclusion_zone > 0:
                exclusion_zone_start = np.max([0, idx - exclusion_zone])
                exclusion_zone_end = np.min([n, idx + exclusion_zone])
                tmp[exclusion_zone_start:exclusion_zone_end] = np.inf

        if len(found) >= k:
            break


    profile['discords'] = np.array(found, dtype='int')

    return profile


def top_k_discords(profile, exclusion_zone=None, k=3):
    """
    Find the top K number of discords (anomalies) given a mp or pmp,
    exclusion zone and the desired number of discords. The exclusion zone
    nullifies entries on the left and right side of the first and subsequent
    discords to remove non-trivial matches. More specifically, a discord found
    at location X will more than likely have additional discords to the left or
    right of it.

    Parameters
    ----------
    profile : dict
        A MatrixProfile or Pan-MatrixProfile structure.
    exclusion_zone : int, Default mp algorithm ez
        Desired number of values to exclude on both sides of the anomaly.
    k : int
        Desired number of discords to find.

    Returns
    -------
    dict : profile
        The original profile object with an additional 'discords' key. Take
        note that a MatrixProfile discord contains a single value while the
        Pan-MatrixProfile contains a row and column index.

    """
    if not core.is_mp_or_pmp_obj(profile):
        raise ValueError('Expecting MP or PMP data structure!')

    cls = profile.get('class', None)
    func = None

    if cls == 'MatrixProfile':
        func = mp_top_k_discords
    elif cls == 'PMP':
        func = pmp_top_k_discords
    else:
        raise ValueError('Unsupported data structure!')

    return func(
        profile,
        exclusion_zone=exclusion_zone,
        k=k,
    )
