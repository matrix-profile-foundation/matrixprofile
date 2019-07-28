# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile import core, mass2


def stomp(ts, window_size, query=None):
    """
    Compute the matrix profile and profile index for a one dimensional time
    series. When a query is provided, it is assumed to be a join. When one is
    not provided, a self-join occurs. Essentially, the time series is
    duplicated as the query.

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix profile for.
    window_size: int
        The size of the window to compute the matrix profile over.
    query : array_like
        Optionally, a query can be provided to perform a similarity join.

    Returns
    -------
    A dict of key data points computed.
    {
        'mp': The matrix profile,
        'pi': The matrix profile 1NN indices,
        'rmp': The right matrix profile,
        'rpi': The right matrix profile 1NN indices,
        'lmp': The left matrix profile,
        'lpi': The left matrix profile 1NN indices,
        'w': The window size used to compute the matrix profile,
        'ez': The exclusion zone used,
        'join': Flag indicating if a similarity join was computed
    }

    Raises
    ------
    ValueError
        If window_size < 4.
        If window_size > query length / 2.
        If ts is not a list or np.array.
        If query is not a list or np.array.
        If ts or query is not one dimensional.
    """    
    is_join = core.is_similarity_join(ts, query)
    if not is_join:
        query = ts

    # data conversion to np.array
    ts = core.to_np_array(ts)
    query = core.to_np_array(query)

    if window_size < 4:
        error = "m, window size, must be at least 4."
        raise ValueError(error)

    if window_size > len(query) / 2:
        error = "Time series is too short relative to desired window size"
        raise ValueError(error)
    
    # precompute some common values - profile length, query length etc.
    profile_length = core.get_profile_length(ts, query, window_size)
    data_length = len(ts)
    query_length = len(query)
    exclusion_zone = int(1 / 2)
    num_queries = query_length - window_size + 1

    # do not use exclusion zone for join
    if is_join:
        exclusion_zone = 0

    # determine what locations need skipped based on nan and infinite values
    skip_loc = core.find_skip_locations(ts, window_size, profile_length)
    
    # clean up nan and inf in the ts_a and query
    search = (np.isinf(ts) | np.isnan(ts))
    ts[search] = 0

    search = (np.isinf(query) | np.isnan(search))
    query[search] = 0

    first_product = np.zeros(num_queries)

    # forward distance profile
    nn = mass2(ts, query, extras=True)

    # reverse
    # required for similarity join
    rnn = None
    if is_join:
        rnn = mass2(query, ts, extras=True)
        first_product[:] = rnn['product']

    matrix_profile = np.full(profile_length, np.nan)
    profile_index = np.full(profile_length, -np.inf)

    left_matrix_profile = np.copy(matrix_profile)
    right_matrix_profile = np.copy(matrix_profile)
    left_profile_index = np.copy(profile_index)
    right_profile_index = np.copy(profile_index)

    distance_profile = np.zeros(profile_length)
    last_product = np.copy(distance_profile)
    drop_value = np.full((1, 1), 0)

    # iteratively compute distance profile and update with element-wise mins
    for i in range(num_queries):
        query_window = query[i:i + window_size]

        if i == 0:            
            distance_profile[:] = nn['distance_profile']
            last_product[:] = nn['product'][window_size - 1:]
        else:
            last_product[1:(data_length - window_size - 1)] = last_product[0:(data_length - window_size - 1)] \
                - ts[0:(data_length - window_size - 1)] * drop_value \
                + ts[(window_size - 1):data_length] * query_window[window_size - 1]
            
            last_product[0] = first_product[i]
            distance_profile = 2 * (window_size - (last_product - window_size * nn['data_mean'] * nn['query_mean']) / 
                (nn['data_std'] * nn['query_std']))


        distance_profile = np.real(np.sqrt(distance_profile))
        drop_value = query_window[0]

        # apply the exclusion zone
        if exclusion_zone > 0:
            ez_start = np.max(0, i - exclusion_zone)
            ez_end = np.min(profile_length, i + exclusion_zone)
            distance_profile[ez_start:ez_end] = np.inf
        
        # TODO: ask franz about the skip loc logic being applied twice?
        if skip_loc[i]:
            distance_profile[:] = np.inf
        
        # update left and right matrix profile looking at element-wise minimums
        if not is_join:
            # left mp
            indices = (distance_profile[i:profile_length] < left_matrix_profile[i:profile_length])

            # pad to left
            if i > 0:
                indices = np.append(np.zeros(i - 1).astype('bool'), indices)

            left_matrix_profile[indices] = distance_profile[indices]
            left_profile_index[np.argwhere(indices)] = i

            # right mp
            indices = (distance_profile[0:i] < right_matrix_profile[0:i])

            # pad to right
            if i > 0:
                indices = np.append(np.zeros(profile_length - i).astype('bool'), indices)

            right_matrix_profile[indices] = distance_profile[indices]
            right_profile_index[np.argwhere(indices)] = i
        
        # update the matrix profile
        indices = (distance_profile < matrix_profile)
        matrix_profile[indices] = distance_profile[indices]
        profile_index[np.argwhere(indices)] = i

    return {
        'mp': matrix_profile,
        'pi': profile_index,
        'rmp': right_matrix_profile,
        'rpi': right_profile_index,
        'lmp': left_matrix_profile,
        'lpi': left_profile_index,
        'w': window_size,
        'ez': exclusion_zone,
        'join': is_join
    }