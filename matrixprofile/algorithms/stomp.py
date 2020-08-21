# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import logging

import numpy as np

from matrixprofile import core

logger = logging.getLogger(__name__)


def _batch_compute(args):
    """
    Internal function to compute a batch of the time series in parallel.

    Parameters
    ----------
    args : tuple
        Various attributes used for computing the batch.
        (
            batch_start : int
                The starting index for this batch.
            batch_end : int
                The ending index for this batch.
            ts : array_like
                The time series to compute the matrix profile for.
            query : array_like
                The query.
            window_size : int
                The size of the window to compute the profile over.
            data_length : int
                The number of elements in the time series.
            profile_length : int
                The number of elements that will be in the final matrix
                profile.
            exclusion_zone : int
                Used to exclude trivial matches.
            is_join : bool
                Flag to indicate if an AB join or self join is occuring.
            data_mu : array_like
                The moving average over the time series for the given window
                size.
            data_sig : array_like
                The moving standard deviation over the time series for the
                given window size.
            first_product : array_like
                The first sliding dot product for the time series over index
                0 to window_size.
            skip_locs : array_like
                Indices that should be skipped for distance profile calculation
                due to a nan or inf.
        )

    Returns
    -------
    dict : profile
        The matrix profile, left and right matrix profiles and their respective
        profile indices.

        >>> {
        >>>     'mp': The matrix profile,
        >>>     'pi': The matrix profile 1NN indices,
        >>>     'rmp': The right matrix profile,
        >>>     'rpi': The right matrix profile 1NN indices,
        >>>     'lmp': The left matrix profile,
        >>>     'lpi': The left matrix profile 1NN indices,
        >>> }

    """
    batch_start, batch_end, ts, query, window_size, data_length, \
    profile_length, exclusion_zone, is_join, data_mu, data_sig, \
    first_product, skip_locs = args

     # initialize matrices
    matrix_profile = np.full(profile_length, np.inf)
    profile_index = np.full(profile_length, 0)

    left_matrix_profile = None
    right_matrix_profile = None
    left_profile_index = None
    right_profile_index = None

    if not is_join:
        left_matrix_profile = np.copy(matrix_profile)
        right_matrix_profile = np.copy(matrix_profile)
        left_profile_index = np.copy(profile_index)
        right_profile_index = np.copy(profile_index)

    # with batch 0 we do not need to recompute the dot product
    # however with other batch windows, we need the previous iterations sliding
    # dot product
    last_product = None
    if batch_start is 0:
        first_window = query[batch_start:batch_start + window_size]
        last_product = np.copy(first_product)
    else:
        first_window = query[batch_start - 1:batch_start + window_size - 1]        
        last_product = core.fft_convolve(ts, first_window)

    query_sum = np.sum(first_window)
    query_2sum = np.sum(first_window ** 2)
    query_mu, query_sig = core.moving_avg_std(first_window, window_size)

    drop_value = first_window[0]

    # only compute the distance profile for index 0 and update
    if batch_start is 0:
        distance_profile = core.distance_profile(last_product, window_size,
         data_mu, data_sig, query_mu, query_sig)

        # apply exclusion zone
        distance_profile = core.apply_exclusion_zone(exclusion_zone, is_join,
            window_size, data_length, 0, distance_profile)
           
        # update the matrix profile
        indices = (distance_profile < matrix_profile)
        matrix_profile[indices] = distance_profile[indices]
        profile_index[indices] = 0

        # update the left matrix profile
        if not is_join:
            left_matrix_profile[indices] = distance_profile[indices]
            left_profile_index[np.argwhere(indices)] = 0

        batch_start += 1

    # make sure to compute inclusively from batch start to batch end
    # otherwise there are gaps in the profile
    if batch_end < profile_length:
        batch_end += 1

    # iteratively compute distance profile and update with element-wise mins
    for i in range(batch_start, batch_end):

        # check for nan or inf and skip
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
        distance_profile = core.apply_exclusion_zone(exclusion_zone, is_join, 
            window_size, data_length, i, distance_profile)
            
        # update the matrix profile
        indices = (distance_profile < matrix_profile)
        matrix_profile[indices] = distance_profile[indices]
        profile_index[indices] = i

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

    return {
        'mp': matrix_profile,
        'pi': profile_index,
        'rmp': right_matrix_profile,
        'rpi': right_profile_index,
        'lmp': left_matrix_profile,
        'lpi': left_profile_index,
    }


def stomp(ts, window_size, query=None, n_jobs=1):
    """
    Computes matrix profiles for a single dimensional time series using the 
    parallelized STOMP algorithm (by default). Ray or Python's multiprocessing
    library may be used. When you have initialized Ray on your machine, 
    it takes priority over using Python's multiprocessing.

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix profile for.
    window_size: int
        The size of the window to compute the matrix profile over.
    query : array_like
        Optionally, a query can be provided to perform a similarity join.
    n_jobs : int, Default = 1
        Number of cpu cores to use.

    Returns
    -------
    dict : profile
        A MatrixProfile data structure.
        
        >>> {
        >>>     'mp': The matrix profile,
        >>>     'pi': The matrix profile 1NN indices,
        >>>     'rmp': The right matrix profile,
        >>>     'rpi': The right matrix profile 1NN indices,
        >>>     'lmp': The left matrix profile,
        >>>     'lpi': The left matrix profile 1NN indices,
        >>>     'metric': The distance metric computed for the mp,
        >>>     'w': The window size used to compute the matrix profile,
        >>>     'ez': The exclusion zone used,
        >>>     'join': Flag indicating if a similarity join was computed,
        >>>     'sample_pct': Percentage of samples used in computing the MP,
        >>>     'data': {
        >>>         'ts': Time series data,
        >>>         'query': Query data if supplied
        >>>     }
        >>>     'class': "MatrixProfile"
        >>>     'algorithm': "stomp_parallel"
        >>> }

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
        error = "window size must be at least 4."
        raise ValueError(error)

    if window_size > len(query) / 2:
        error = "Time series is too short relative to desired window size"
        raise ValueError(error)

    # multiprocessing or single threaded approach
    if n_jobs == 1:
        pass
    else:
        n_jobs = core.valid_n_jobs(n_jobs)

    # precompute some common values - profile length, query length etc.
    profile_length = core.get_profile_length(ts, query, window_size)
    data_length = len(ts)
    query_length = len(query)
    num_queries = query_length - window_size + 1
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
    profile_index = np.full(profile_length, 0)

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
    first_window = query[0:window_size]
    first_product = core.fft_convolve(ts, first_window)

    batch_windows = []
    results = []
    
    # batch compute with multiprocessing
    args = []
    for start, end in core.generate_batch_jobs(num_queries, n_jobs):
        args.append((
            start, end, ts, query, window_size, data_length,
            profile_length, exclusion_zone, is_join, data_mu, data_sig,
            first_product, skip_locs
        ))
        batch_windows.append((start, end))

    # we are running single threaded stomp - no need to initialize any
    # parallel environments.
    if n_jobs == 1 or len(args) == 1:
        results.append(_batch_compute(args[0]))
    else:
        # parallelize
        with core.mp_pool()(n_jobs) as pool:
            results = pool.map(_batch_compute, args)

    # now we combine the batch results
    if len(results) == 1:
        result = results[0]
        matrix_profile = result['mp']
        profile_index = result['pi']
        left_matrix_profile = result['lmp']
        left_profile_index = result['lpi']
        right_matrix_profile = result['rmp']
        right_profile_index = result['rpi']
    else:
        for index, result in enumerate(results):
            start = batch_windows[index][0]
            end = batch_windows[index][1]        
            
            # update the matrix profile
            indices = result['mp'] < matrix_profile
            matrix_profile[indices] = result['mp'][indices]
            profile_index[indices] = result['pi'][indices]

            # update the left and right matrix profiles
            if not is_join:
                indices = result['lmp'] < left_matrix_profile
                left_matrix_profile[indices] = result['lmp'][indices]
                left_profile_index[indices] = result['lpi'][indices]

                indices = result['rmp'] < right_matrix_profile
                right_matrix_profile[indices] = result['rmp'][indices]
                right_profile_index[indices] = result['rpi'][indices]           

    return {
        'mp': matrix_profile,
        'pi': profile_index,
        'rmp': right_matrix_profile,
        'rpi': right_profile_index,
        'lmp': left_matrix_profile,
        'lpi': left_profile_index,
        'metric': 'euclidean',
        'w': window_size,
        'ez': exclusion_zone,
        'join': is_join,
        'sample_pct': 1,
        'data': {
            'ts': ts,
            'query': query
        },
        'class': "MatrixProfile",
        'algorithm': "stomp"
    }
