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


def pmp_top_k_motifs(profile, exclusion_zone=None, k=3, max_neighbors=10, radius=3):
    """
    Find the top K number of motifs (patterns) given a pan matrix profile. By
    default the algorithm will find up to 3 motifs (k) and up to 10 of their
    neighbors with a radius of 3 * min_dist.

    Parameters
    ----------
    profile : dict
        The output from one of the pan matrix profile algorithms.
    exclusion_zone : int, Default to algorithm ez
        Desired number of values to exclude on both sides of the motif. This
        avoids trivial matches. It defaults to half of the computed window
        size. Setting the exclusion zone to 0 makes it not apply.
    k : int, Default = 3
        Desired number of motifs to find.
    max_neighbors : int, Default = 10
        The maximum number of neighbors to include for a given motif.
    radius : int, Default = 3
        The radius is used to associate a neighbor by checking if the
        neighbor's distance is less than or equal to dist * radius

    Returns
    -------
    profile : dict
        The original input obj with the addition of the "motifs" key. The
        motifs key consists of the following structure.

        A list of dicts containing motif indices and their corresponding
        neighbor indices. Note that each index is a (row, col) index
        corresponding to the pan matrix profile.
        >>> [
        >>>     {
        >>>         'motifs': [first_index, second_index],
        >>>         'neighbors': [index, index, index ...max_neighbors]
        >>>     }
        >>> ]

    """
    if not core.is_pmp_obj(profile):
        raise ValueError('Expecting PMP data structure!')

    data = profile.get('data', None)
    ts = data.get('ts', None)
    data_len = len(ts)

    pmp = profile.get('pmp', None)
    profile_len = pmp.shape[1]
    pmpi = profile.get('pmpi', None)
    windows = profile.get('windows', None)

    # make sure we are working with Euclidean distances
    tmp = None
    if core.is_pearson_array(pmp):
        tmp = core.pearson_to_euclidean(pmp, windows)
    else:
        tmp = np.copy(pmp).astype('d')

    # replace nan and infs with infinity
    tmp[core.nan_inf_indices(tmp)] = np.inf

    motifs = []
    for _ in range(k):
        min_idx = np.unravel_index(np.argmin(tmp), tmp.shape)
        min_dist = tmp[min_idx]

        # nothing else to find...
        if core.is_nan_inf(min_dist):
            break

        # create the motif pair
        min_row_idx = min_idx[0]
        min_col_idx = min_idx[1]

        # motif pairs are respective to the column of the matching row
        first_idx = np.min([min_col_idx, pmpi[min_row_idx][min_col_idx]])
        second_idx = np.max([min_col_idx, pmpi[min_row_idx][min_col_idx]])

        # compute distance profile for first appearance
        window_size = windows[min_row_idx]
        query = ts[first_idx:first_idx + window_size]
        distance_profile = mass2(ts, query)

        # extend the distance profile to be as long as the original
        infs = np.full(profile_len - len(distance_profile), np.inf)
        distance_profile = np.append(distance_profile, infs)

        # exclude already picked motifs and neighbors
        mask = core.nan_inf_indices(pmp[min_row_idx])
        distance_profile[mask] = np.inf

        # determine the exclusion zone if not set
        if not exclusion_zone:
            exclusion_zone = int(np.floor(window_size / 2))

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
            tmp2 = core.apply_exclusion_zone(
                exclusion_zone,
                False,
                window_size,
                data_len,
                j,
                tmp[min_row_idx]
            )
            tmp[min_row_idx] = tmp2

        # find up to max_neighbors
        neighbors = []
        for j in range(max_neighbors):
            neighbor_idx = np.argmin(distance_profile)
            neighbor_dist = np.real(distance_profile[neighbor_idx])
            not_in_radius = not ((radius * min_dist) >= neighbor_dist)

            # no more neighbors exist based on radius
            if core.is_nan_inf(neighbor_dist) or not_in_radius:
                break

            # add neighbor and apply exclusion zone
            neighbors.append((min_row_idx, neighbor_idx))
            distance_profile = core.apply_exclusion_zone(
                exclusion_zone,
                False,
                window_size,
                data_len,
                neighbor_idx,
                distance_profile
            )
            tmp2 = core.apply_exclusion_zone(
                exclusion_zone,
                False,
                window_size,
                data_len,
                neighbor_idx,
                tmp[min_row_idx]
            )
            tmp[min_row_idx] = tmp2

        # add the motifs and neighbors
        # note that they are (row, col) indices
        motifs.append({
            'motifs': [(min_row_idx, first_idx), (min_row_idx, second_idx)],
            'neighbors': neighbors
        })

    profile['motifs'] = motifs

    return profile


def mp_top_k_motifs(profile, exclusion_zone=None, k=3, max_neighbors=10, radius=3, use_cmp=False):
    """
    Find the top K number of motifs (patterns) given a matrix profile. By
    default the algorithm will find up to 3 motifs (k) and up to 10 of their
    neighbors with a radius of 3 * min_dist using the regular matrix profile.

    Parameters
    ----------
    profile : dict
        The output from one of the matrix profile algorithms.
    exclusion_zone : int, Default to algorithm ez
        Desired number of values to exclude on both sides of the motif. This
        avoids trivial matches. It defaults to half of the computed window
        size. Setting the exclusion zone to 0 makes it not apply.
    k : int, Default = 3
        Desired number of motifs to find.
    max_neighbors : int, Default = 10
        The maximum number of neighbors to include for a given motif.
    radius : int, Default = 3
        The radius is used to associate a neighbor by checking if the
        neighbor's distance is less than or equal to dist * radius
    use_cmp : bool, Default = False
        Use the Corrected Matrix Profile to compute the motifs.

    Returns
    -------
    dict : profile
        The original input obj with the addition of the "motifs" key. The
        motifs key consists of the following structure.

        A list of dicts containing motif indices and their corresponding
        neighbor indices.

        >>> [
        >>>    {
        >>>        'motifs': [first_index, second_index],
        >>>        'neighbors': [index, index, index ...max_neighbors]
        >>>    }
        >>> ]

    """
    if not core.is_mp_obj(profile):
        raise ValueError('Expecting MP data structure!')

    window_size = profile['w']
    data = profile.get('data', None)
    if data:
        ts = data.get('ts', None)

    data_len = len(ts)
    motifs = []
    mp = np.copy(profile['mp'])
    if use_cmp:
        mp = np.copy(profile['cmp'])
    mpi = profile['pi']

    # TODO: this is based on STOMP standards when this motif finding algorithm
    # originally came out. Should we default this to 4.0 instead? That seems
    # to be the common value now per new research.
    if exclusion_zone is None:
        exclusion_zone = profile.get('ez', None)

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
                break

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

    profile['motifs'] = motifs

    return profile


def top_k_motifs(profile, exclusion_zone=None, k=3, max_neighbors=10, radius=3, use_cmp=False):
    """
    Find the top K number of motifs (patterns) given a matrix profile or a pan
    matrix profile. By default the algorithm will find up to 3 motifs (k) and
    up to 10 of their neighbors with a radius of 3 * min_dist using the
    regular matrix profile. If the profile is a Matrix Profile data structure,
    you can also use a Corrected Matrix Profile to compute the motifs.

    Parameters
    ----------
    profile : dict
        The output from one of the matrix profile algorithms.
    exclusion_zone : int, Default to algorithm ez
        Desired number of values to exclude on both sides of the motif. This
        avoids trivial matches. It defaults to half of the computed window
        size. Setting the exclusion zone to 0 makes it not apply.
    k : int, Default = 3
        Desired number of motifs to find.
    max_neighbors : int, Default = 10
        The maximum number of neighbors to include for a given motif.
    radius : int, Default = 3
        The radius is used to associate a neighbor by checking if the
        neighbor's distance is less than or equal to dist * radius
    use_cmp : bool, Default = False
        Use the Corrected Matrix Profile to compute the motifs (only for
        a Matrix Profile data structure).

    Returns
    -------
    dict : profile
        The original input profile with the addition of the "motifs" key. The
        motifs key consists of the following structure.

        A list of dicts containing motif indices and their corresponding
        neighbor indices.

        >>> [
        >>>     {
        >>>         'motifs': [first_index, second_index],
        >>>         'neighbors': [index, index, index ...max_neighbors]
        >>>     }
        >>> ]

        The index is a single value when a MatrixProfile is passed in otherwise
        the index contains a row and column index for Pan-MatrixProfile.

    """
    if not core.is_mp_or_pmp_obj(profile):
        raise ValueError('Expecting MP or PMP data structure!')

    cls = profile.get('class', None)
    func = None

    if cls == 'MatrixProfile':
        func = mp_top_k_motifs
    elif cls == 'PMP':
        func = pmp_top_k_motifs
    else:
        raise ValueError('Unsupported data structure!')

    if cls == 'PMP':
        return func(
            profile,
            exclusion_zone=exclusion_zone,
            k=k,
            max_neighbors=max_neighbors,
            radius=radius
        )
        
    return func(
        profile,
        exclusion_zone=exclusion_zone,
        k=k,
        max_neighbors=max_neighbors,
        radius=radius,
        use_cmp=use_cmp
    )
