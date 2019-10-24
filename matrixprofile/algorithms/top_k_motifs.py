# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile import core
from matrixprofile.algorithms.mass2 import mass2


def top_k_motifs(obj, exclusion_zone=None, k=3, max_neighbors=10, radius=3):
    """
    Find the top K number of motifs (patterns) given a matrix profile. By
    default the algorithm will find up to 3 motifs (k) and up to 10 of their
    neighbors with a radius of 3 * min_dist.

    Parameters
    ----------
    obj : dict
        The output from one of the matrix profile algorithms.
    exclusion_zone : int, Default to algorithm ez
        Desired number of values to exclude on both sides of the motif. This
        avoids trivial matches. It defaults to half of the computed window
        size. Setting the exclusion zone to 0 makes it not apply.
    k : int, Default = 3
        Desired number of motifs to find.
    neighbor_count : int, Default = 10
        The maximum number of neighbors to include for a given motif.
    radius : int, Default = 3
        The radius is used to associate a neighbor by checking if the
        neighbor's distance is less than or equal to dist * radius

    Returns
    -------
    The original input obj with the addition of the "motifs" key. The motifs
    key consists of the following structure.

    A list of dicts containing motif indices and their corresponding neighbor
    indices.

    [
        {
            'motifs': [first_index, second_index],
            'neighbors': [index, index, index ...max_neighbors]
        }
    ]
    """
    window_size = obj['w']
    data = obj.get('data', None)
    if data:
        ts = data.get('ts', None)

    data_len = len(ts)
    motifs = []
    mp = np.copy(obj['mp'])
    mpi = obj['pi']

    # TODO: this is based on STOMP standards when this motif finding algorithm
    # originally came out. Should we default this to 4.0 instead? That seems
    # to be the common value now per new research.
    if exclusion_zone is None:
        exclusion_zone = obj.get('ez', None)

    for i in range(k):
        min_idx = np.argmin(mp)
        min_dist = mp[min_idx]

        # we no longer have any motifs to find as all values are nan/inf
        if core.is_nan_inf(min_dist):
            break

        # create a motif pair corresponding to the first appearance and
        # second appearance
        first_idx = np.min([min_idx, mpi[min_idx]])
        second_idx = np.max([min_idx, mpi[min_idx]])

        # compute distance profile using mass2 for first appearance
        query = ts[first_idx:first_idx + window_size]
        distance_profile = mass2(ts, query)

        # exclude already picked motifs and neighbors
        mask = core.nan_inf_indices(mp)    
        distance_profile[mask] = np.inf

        # apply exclusion zone for motif pair
        for j in (first_idx, second_idx):
            distance_profile = core.apply_exclusion_zone(
                exclusion_zone,
                False,
                window_size,
                data_len,
                j,
                distance_profile
            )
            mp = core.apply_exclusion_zone(
                exclusion_zone,
                False,
                window_size,
                data_len,
                j,
                mp
            )

        # find up to max_neighbors
        neighbors = []
        for j in range(max_neighbors):
            neighbor_idx = np.argmin(distance_profile)
            neighbor_dist = distance_profile[neighbor_idx]
            not_in_radius = not ((radius * min_dist) >= neighbor_dist)

            # no more neighbors exist based on radius
            if core.is_nan_inf(neighbor_dist) or not_in_radius:
                break;

            # add neighbor and apply exclusion zone
            neighbors.append(neighbor_idx)
            distance_profile = core.apply_exclusion_zone(
                exclusion_zone,
                False,
                window_size,
                data_len,
                neighbor_idx,
                distance_profile
            )
            mp = core.apply_exclusion_zone(
                exclusion_zone,
                False,
                window_size,
                data_len,
                neighbor_idx,
                mp
            )

        # add motifs and neighbors to results
        motifs.append({
            'motifs': [first_idx, second_idx],
            'neighbors': neighbors
        })

        obj['motifs'] = motifs

    return obj