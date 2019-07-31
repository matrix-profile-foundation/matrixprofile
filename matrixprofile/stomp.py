# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile import core, mass2

def mass_pre(ts, window_size):
    """
    Precomputes some statistics used in stomp. It is essentially reworked
    MASS algorithm.

    Parameters
    ----------
    ts : array_like
        The time series to compute statistics for.
    window_size: int
        The size of the window to compute statistics over.

    Returns
    -------
    Statistics used for later computations.
    """
    n = len(ts)
    mu, sig = core.moving_avg_std(ts, window_size)
    data_freq = np.fft.fft(ts)

    return (data_freq, n, mu, sig)


def mass_post(data_freq, query, n, m, data_mu, data_sig):
    """
    Compute the distance profile as later stage for stomp.
    
    Parameters
    ----------
    data_freq : array_like
        The time series to compute statistics for.
    query: array_like
        The query to compute statistics for.
    n: int
        The length of the time series.
    m: int
        The size of the window to compute statistics over.
    data_mu: array_like
        The rolling mean for the time series.
    data_sig: array_like
        The rolling standard deviation for the time series..

    Returns
    -------
    The distance profile for the given query and statistics that are later 
    used for further computation.
    """
    # flip query and append zeros
    query = np.append(np.flip(query), np.zeros(n - m))

    # transform query to frequency domain and compute dot product
    query_freq = np.fft.fft(query)
    product_freq = data_freq * query_freq
    product = np.fft.ifft(product_freq)

    # compute query stats
    query_sum = np.sum(query)
    query_2sum = np.sum(query ** 2)
    query_mu, query_sig = core.moving_avg_std(query, m)

    distance_profile = 2 * (m - (product[m - 1:] - m * data_mu * query_mu) / \
                       (data_sig * query_sig))
    last_product = np.real(product[m - 1:])

    return (distance_profile, last_product, query_sum, query_2sum, query_sig)


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
    exclusion_zone = int(np.ceil(window_size / 2.0))

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

    # precompute mass_pre
    data_freq, data_length, data_mu, data_sig = mass_pre(ts, window_size)

    # iteratively compute distance profile and update with element-wise mins
    for i in range(profile_length):

        # check for nan or inf and skip
        segment = ts[i:i + window_size]
        search = (np.isinf(segment) | np.isnan(segment))
        
        if np.any(search):
            continue

        query_window = query[i:i + window_size]
        if i == 0:
            tmp = mass_post(
                data_freq, query_window, data_length, window_size, data_mu, data_sig
            )
            distance_profile, last_product, query_sum, query_2sum, query_sig = tmp
            distance_profile = np.real(distance_profile)
            first_product = np.copy(last_product)
        else:
            query_sum = query_sum - drop_value + query_window[-1]
            query_2sum = query_2sum - drop_value ** 2 + query_window[-1] ** 2
            query_mu = query_sum / window_size
            query_sig2 = query_2sum / window_size - query_mu ** 2
            query_sig = np.sqrt(query_sig2)
            last_product[1:] = last_product[0:data_length - window_size] \
                - ts[0:data_length - window_size] * drop_value \
                + ts[window_size:] * query_window[-1]
            last_product[0] = first_product[i]
            distance_profile = 2 * (window_size - (last_product - window_size \
                 * data_mu * query_mu) / (data_sig * query_sig))

        drop_value = query_window[0]

        # apply the exclusion zone
        # for similarity join we do not apply exclusion zone
        if exclusion_zone > 0 and not is_join:
            ez_start = np.max([0, i - exclusion_zone])
            ez_end = np.min([profile_length, i + exclusion_zone])
            distance_profile[ez_start:ez_end] = np.inf
        
        # update the left and right matrix profiles
        if not is_join and i > 0:
            # find differences, shift left and update
            indices = distance_profile[i:] < left_matrix_profile[i:]
            indices = np.append(np.zeros(i).astype('bool'), indices)
            left_matrix_profile[indices] = distance_profile[indices]
            left_profile_index[np.argwhere(indices)] = i

            # find differences, shift right and update
            indices = distance_profile[0:i] < right_matrix_profile[0:i]
            indices = np.append(indices, np.zeros(profile_length -i).astype('bool'))
            right_matrix_profile[indices] = distance_profile[indices]
            right_profile_index[np.argwhere(indices)] = i
        
        # update the matrix profile
        indices = (distance_profile < matrix_profile)
        matrix_profile[indices] = distance_profile[indices]
        profile_index[np.argwhere(indices)] = i
    
    matrix_profile = np.real(np.sqrt(matrix_profile))

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