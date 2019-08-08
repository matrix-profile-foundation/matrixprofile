# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile import core


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
        'class': "MatrixProfile"
        'algorithm': "stomp"
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
    exclusion_zone = int(np.ceil(window_size / 2.0))

    # do not use exclusion zone for join
    if is_join:
        exclusion_zone = 0

    # find skip locations, clean up nan and inf in the ts and query
    skip_locs = core.find_skip_locations(ts, profile_length, window_size)
    ts = core.clean_nan_inf(ts)
    query = core.clean_nan_inf(query)

    # initialize matrices
    matrix_profile = np.full(profile_length, np.inf)
    profile_index = np.full(profile_length, -np.inf)

    # compute left and right matrix profile when similarity join does not happen
    left_matrix_profile = None
    right_matrix_profile = None
    left_profile_index = None
    right_profile_index = None

    if not is_join:
        left_matrix_profile = np.copy(matrix_profile)
        right_matrix_profile = np.copy(matrix_profile)
        left_profile_index = np.copy(profile_index)
        right_profile_index = np.copy(profile_index)

    # precompute some statistics on ts
    data_mu, data_sig = core.moving_avg_std(ts, window_size)
    data_freq = np.fft.fft(ts)

    # here we pull out the mass_post to make the loop easier to read
    # compute query stats
    first_window = query[0:window_size]
    last_product = core.sliding_dot_product(ts, first_window)

    query_sum = np.sum(first_window)
    query_2sum = np.sum(first_window ** 2)
    query_mu, query_sig = core.moving_avg_std(first_window, window_size)

    first_product = np.copy(last_product)
    drop_value = first_window[0]

    distance_profile = core.distance_profile(
            last_product, window_size, data_mu, data_sig, query_mu, query_sig)
       
    # update the matrix profile
    indices = (distance_profile < matrix_profile)
    matrix_profile[indices] = distance_profile[indices]
    profile_index[np.argwhere(indices)] = 0

    # iteratively compute distance profile and update with element-wise mins
    for i in range(1, profile_length):
        if skip_locs[i]:
            continue

        query_window = query[i:i + window_size]
        query_sum = query_sum - drop_value + query_window[-1]
        query_2sum = query_2sum - drop_value ** 2 + query_window[-1] ** 2
        query_mu = query_sum / window_size
        query_sig2 = query_2sum / window_size - query_mu ** 2
        query_sig = np.sqrt(query_sig2)
        last_product[1:] = last_product[0:data_length - window_size] \
            - ts[0:data_length - window_size] * drop_value \
            + ts[window_size:] * query_window[-1]
        last_product[0] = first_product[i]
        drop_value = query_window[0]

        distance_profile = core.distance_profile(
            last_product, window_size, data_mu, data_sig, query_mu, query_sig)

        # apply the exclusion zone
        # for similarity join we do not apply exclusion zone
        if exclusion_zone > 0 and not is_join:
            ez_start = np.max([0, i - exclusion_zone])
            ez_end = np.min([profile_length, i + exclusion_zone])
            distance_profile[ez_start:ez_end] = np.inf
        
        # update the left and right matrix profiles
        if not is_join:
            # find differences, shift left and update
            indices = distance_profile[i:] < left_matrix_profile[i:]
            falses = np.zeros(i).astype('bool')
            indices = np.append(falses, indices)
            left_matrix_profile[indices] = distance_profile[indices]
            left_profile_index[np.argwhere(indices)] = i

            # find differences, shift right and update
            indices = distance_profile[0:i] < right_matrix_profile[0:i]
            falses = np.zeros(profile_length - i).astype('bool')
            indices = np.append(indices, falses)
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
        'join': is_join,
        'class': "MatrixProfile",
        'algorithm': "stomp"
    }