# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate


import numpy as np


def top_k_discords(profile, exclusion_zone, k=3):
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
        The matrix profile computed from the compute function.
    exclusion_zone : int
        Desired number of values to exclude on both sides of the anomaly.
    k : int
        Desired number of discords to find.

    Returns
    -------
    List of indices where the discords were found in the matrix profile.
    """
    found = []
    tmp = np.copy(profile['mp'])
    n = len(tmp)
    
    # obtain indices in ascending order
    indices = np.argsort(tmp)
    
    # created flipped view for discords
    indices = indices[::-1]

    for idx in indices:
        if not np.isinf(tmp[idx]):
            found.append(idx)

            # apply exclusion zone
            if exclusion_zone > 0 and idx != indices[-1]:
                exclusion_zone_start = np.max([0, idx - exclusion_zone])
                exclusion_zone_end = np.min([n, idx + exclusion_zone])
                tmp[exclusion_zone_start:exclusion_zone_end] = np.inf

        if len(found) >= k:
            break

    return found