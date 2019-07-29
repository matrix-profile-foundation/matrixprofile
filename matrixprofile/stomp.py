# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile import core, mass2

def sliding_dot_product(ts, query):
    m = len(query)
    n = len(ts)

    #If length is odd, zero-pad time time series
    ts_add = 0
    if n%2 ==1:
        ts = np.insert(ts,0,0)
        ts_add = 1

    q_add = 0
    #If length is odd, zero-pad query
    if m%2 == 1:
        query = np.insert(query,0,0)
        q_add = 1

    #This reverses the array
    query = query[::-1]
    query = np.pad(query,(0,n-m+ts_add-q_add),'constant')

    # Determine trim length for dot product. Note that zero-padding of the 
    # query has no effect on array length, which is solely determined by 
    # the longest vector
    trim = m-1+ts_add

    dot_product = np.fft.irfft(np.fft.rfft(ts)*np.fft.rfft(query))

    return dot_product[trim:]


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
    exclusion_zone = int(np.ceil(window_size / 2.0))
    num_queries = query_length - window_size + 1

    # do not use exclusion zone for join
    if is_join:
        exclusion_zone = 0

    # clean up nan and inf in the ts_a and query
    search = (np.isinf(ts) | np.isnan(ts))
    ts[search] = 0

    search = (np.isinf(query) | np.isnan(search))
    query[search] = 0

    # initialize matrices
    matrix_profile = np.full(profile_length, np.inf)
    profile_index = np.full(profile_length, -np.inf)

    distance_profile = np.zeros(profile_length)
    drop_value = np.full((1, 1), 0)

    last_product = np.copy(distance_profile)
    first_product = None    

    # reverse
    # required for similarity join
    # note that we simply use sliding_dot_product in the case of a self join
    # to save some computation
    rnn = None
    if is_join:
        rnn = mass2(query, ts, extras=True)
        first_product = rnn['product'][window_size - 1:]
    else:
        first_product = sliding_dot_product(ts, ts[0:window_size])

    # TODO: these are not being used right now
    left_matrix_profile = np.copy(matrix_profile)
    right_matrix_profile = np.copy(matrix_profile)
    left_profile_index = np.copy(profile_index)
    right_profile_index = np.copy(profile_index)

    # iteratively compute distance profile and update with element-wise mins
    for i in range(num_queries - 1):

        # check for nan or inf and skip
        segment = ts[i:i + window_size]
        search = (np.isinf(segment) | np.isnan(segment))
        
        if np.any(search):
            continue

        query_window = query[i:i + window_size]
        nn = mass2(ts, query_window, extras=True)

        if i == 0:
            distance_profile[:] = np.real(nn['distance_profile'])
            last_product[:] = np.real(nn['product'][window_size - 1:])
        else:
            prod = nn['product'][window_size:]
            last_product[1:(data_length - window_size + 1)] = prod - \
                ts[0:(data_length - window_size)] * drop_value + \
                ts[(window_size):data_length] * query_window[window_size - 1]
            
            last_product[0] = first_product[i]

            data_mu = nn['data_mean'][window_size - 1:]
            data_sig = nn['data_std'][window_size - 1:]
            q_mu = nn['query_mean']
            q_sig = nn['query_std']

            distance_profile = 2 * (window_size - (last_product - window_size * data_mu * q_mu) / \
                               (data_sig * q_sig))

        distance_profile = np.real(np.sqrt(distance_profile))
        drop_value = query_window[0]

        # apply the exclusion zone
        if exclusion_zone > 0:
            ez_start = np.max([0, i - exclusion_zone])
            ez_end = np.min([profile_length, i + exclusion_zone])
            distance_profile[ez_start:ez_end] = np.inf
        
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