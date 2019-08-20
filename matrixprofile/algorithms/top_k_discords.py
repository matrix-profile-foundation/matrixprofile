# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate


import numpy as np


def top_k_discords(profile, exclusion_zone, k=3):
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
        exclusion_zone_start = np.max([0, idx - exclusion_zone])
        exclusion_zone_end = np.min([n, idx + exclusion_zone])
        tmp[exclusion_zone_start:exclusion_zone_end] = np.inf

        if len(found) >= k:
            break

    return found