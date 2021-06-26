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

_EPS = 1e-14


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
    num_dim, batch_start, batch_end, ts, query, window_size, data_length, \
    profile_length, exclusion_zone, data_mu, data_sig, \
    first_product, skip_locs, profile_dimension, return_dimension = args

    # initialize matrices
    matrix_profile = np.full((num_dim, profile_length), np.inf)
    profile_index = np.full((num_dim, profile_length), 0)

    left_matrix_profile = None
    right_matrix_profile = None
    left_profile_index = None
    right_profile_index = None

    left_matrix_profile = np.copy(matrix_profile)
    right_matrix_profile = np.copy(matrix_profile)
    left_profile_index = np.copy(profile_index)
    right_profile_index = np.copy(profile_index)

    # with batch 0 we do not need to recompute the dot product
    # however with other batch windows, we need the previous iterations sliding
    # dot product
    last_product = np.copy(first_product)
    if batch_start is 0:
        first_window = query[:, batch_start:batch_start + window_size]
    else:
        first_window = query[:, batch_start - 1:batch_start + window_size - 1]
        for i in range(num_dim):
            last_product[i, :] = core.fft_convolve(ts[i, :], first_window[i, :])

    query_sum = np.sum(first_window, axis=1)
    query_2sum = np.sum(first_window**2, axis=1)
    query_mu, query_sig = np.empty(num_dim), np.empty(num_dim)
    for i in range(num_dim):
        query_mu[i], query_sig[i] = core.moving_avg_std(first_window[i, :], window_size)

    drop_value = np.empty(num_dim)
    for i in range(num_dim):
        drop_value[i] = first_window[i, 0]
    distance_profile = np.empty((num_dim, profile_length))

    # make sure to compute inclusively from batch start to batch end
    # otherwise there are gaps in the profile
    if batch_end < profile_length:
        batch_end += 1

    # iteratively compute distance profile and update with element-wise mins
    for i in range(batch_start, batch_end):
        # check for nan or inf and skip
        if skip_locs[i]:
            continue
        for j in range(num_dim):
            if i == 0:
                query_window = query[j, i:i + window_size]
                distance_profile[j, :] = core.distance_profile(last_product[j, :], window_size, data_mu[j, :],
                                                               data_sig[j, :], query_mu[j], query_sig[j])

                # apply exclusion zone
                distance_profile[j, :] = core.apply_exclusion_zone(exclusion_zone, 0, window_size, data_length, 0,
                                                                   distance_profile[j, :])
            else:
                query_window = query[j, i:i + window_size]
                query_sum[j] = query_sum[j] - drop_value[j] + query_window[-1]
                query_2sum[j] = query_2sum[j] - drop_value[j]**2 + query_window[-1]**2
                query_mu[j] = query_sum[j] / window_size
                query_sig2 = query_2sum[j] / window_size - query_mu[j]**2
                if query_sig2 < _EPS:
                    query_sig2 = _EPS
                query_sig[j] = np.sqrt(query_sig2)
                last_product[j, 1:] = last_product[j, 0:data_length - window_size] \
                - ts[j, 0:data_length - window_size] * drop_value[j] \
                + ts[j, window_size:] * query_window[-1]
                last_product[j, 0] = first_product[j, i]

                distance_profile[j, :] = core.distance_profile(last_product[j, :], window_size, data_mu[j, :],
                                                               data_sig[j, :], query_mu[j], query_sig[j])

                # apply the exclusion zone
                distance_profile[j, :] = core.apply_exclusion_zone(exclusion_zone, 0, window_size, data_length, i,
                                                                   distance_profile[j, :])
            distance_profile[j, distance_profile[j, :] < _EPS] = 0
            drop_value[j] = query_window[0]
        if np.any(query_sig < _EPS):
            continue
        distance_profile[:, skip_locs] = np.inf
        distance_profile[data_sig < np.sqrt(_EPS)] = np.inf

        distance_profile_dim = np.argsort(distance_profile, axis=0)
        distance_profile_sort = np.sort(distance_profile, axis=0)
        distance_profile_cumsum = np.zeros(profile_length)
        for j in range(num_dim):
            distance_profile_cumsum += distance_profile_sort[j, :]
            distance_profile_mean = distance_profile_cumsum / (j + 1)

            # update the matrix profile
            indices = (distance_profile_mean < matrix_profile[j, :])
            matrix_profile[j, indices] = distance_profile_mean[indices]
            profile_index[j, indices] = i
            if return_dimension:
                profile_dimension[j][:, indices] = distance_profile_dim[:j + 1, indices]

            # update the left and right matrix profiles
            # find differences, shift left and update
            indices = distance_profile_mean[i:] < left_matrix_profile[j, i:]
            falses = np.zeros(i).astype('bool')
            indices = np.append(falses, indices)
            left_matrix_profile[j, indices] = distance_profile_mean[indices]
            left_profile_index[j, np.argwhere(indices)] = i

            # find differences, shift right and update
            indices = distance_profile_mean[0:i] < right_matrix_profile[j, 0:i]
            falses = np.zeros(profile_length - i).astype('bool')
            indices = np.append(indices, falses)
            right_matrix_profile[j, indices] = distance_profile_mean[indices]
            right_profile_index[j, np.argwhere(indices)] = i
    return {
        'mp': matrix_profile,
        'pi': profile_index,
        'pd': profile_dimension,
        'rmp': right_matrix_profile,
        'rpi': right_profile_index,
        'lmp': left_matrix_profile,
        'lpi': left_profile_index,
    }


def mstomp(ts, window_size, return_dimension=False, n_jobs=1):
    """
    Computes multidimensional matrix profile with mSTAMP (stomp based). Ray or Python's multiprocessing library may be used. When you have initialized Ray on your machine, it takes priority over using Python's multiprocessing.

    Parameters
    ----------
    ts : array_like, shape (n_dim, seq_len)
        The multidimensional time series to compute the multidimensional matrix profile for.
    window_size: int
        The size of the window to compute the matrix profile over.
    return_dimension : bool
        if True, also return the matrix profile dimension. It takses O(d^2 n)
        to store and O(d^2 n^2) to compute. (default is False)
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
        >>>     'sample_pct': Percentage of samples used in computing the MP,
        >>>     'data': {
        >>>         'ts': Time series data,
        >>>         'query': Query data if supplied
        >>>     }
        >>>     'class': "MatrixProfile"
        >>>     'algorithm': "stomp_based_mstamp"
        >>> }

    Raises
    ------
    ValueError
        If window_size < 4.
        If window_size > time series length / 2.
        If ts is not a list or np.array.

    """

    query = ts

    # data conversion to np.array
    ts = core.to_np_array(ts)
    query = core.to_np_array(query)

    if window_size < 4:
        error = "window size must be at least 4."
        raise ValueError(error)

    if ts.ndim == 1:
        ts = np.expand_dims(ts, axis=0)
        query = np.expand_dims(query, axis=0)

    if window_size > query.shape[1] / 2:
        error = "Time series is too short relative to desired window size"
        raise ValueError(error)

    # multiprocessing or single threaded approach
    if n_jobs == 1:
        pass
    else:
        n_jobs = core.valid_n_jobs(n_jobs)

    # precompute some common values - profile length, query length etc.
    profile_length = core.get_profile_length(ts, query, window_size)
    data_length = ts.shape[1]
    query_length = query.shape[1]
    num_queries = query_length - window_size + 1
    exclusion_zone = int(np.ceil(window_size / 2.0))
    num_dim = ts.shape[0]

    # find skip locations, clean up nan and inf in the ts and query
    skip_locs = core.find_multid_skip_locations(ts, profile_length, window_size)
    ts = core.clean_nan_inf(ts)
    query = core.clean_nan_inf(query)

    # initialize matrices
    matrix_profile = np.full((num_dim, profile_length), np.inf)
    profile_index = np.full((num_dim, profile_length), 0)
    # profile_index = np.full((num_dim, profile_length), -1)

    # compute left and right matrix profile when similarity join does not happen
    left_matrix_profile = np.copy(matrix_profile)
    right_matrix_profile = np.copy(matrix_profile)
    left_profile_index = np.copy(profile_index)
    right_profile_index = np.copy(profile_index)

    profile_dimension = []
    if return_dimension:
        n_jobs = 1
        for i in range(num_dim):
            profile_dimension.append(np.empty((i + 1, profile_length), dtype=int))

    # precompute some statistics on ts
    data_mu, data_sig, first_product = np.empty((num_dim, profile_length)), np.empty(
        (num_dim, profile_length)), np.empty((num_dim, profile_length))
    for i in range(num_dim):
        data_mu[i, :], data_sig[i, :] = core.moving_avg_std(ts[i, :], window_size)
        first_window = query[i, 0:window_size]
        first_product[i, :] = core.fft_convolve(ts[i, :], first_window)

    batch_windows = []
    results = []

    # batch compute with multiprocessing
    args = []
    for start, end in core.generate_batch_jobs(num_queries, n_jobs):
        args.append((num_dim, start, end, ts, query, window_size, data_length, profile_length, exclusion_zone, data_mu,
                     data_sig, first_product, skip_locs, profile_dimension, return_dimension))
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
        profile_dimension = result['pd']
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
            indices = result['lmp'] < left_matrix_profile
            left_matrix_profile[indices] = result['lmp'][indices]
            left_profile_index[indices] = result['lpi'][indices]

            indices = result['rmp'] < right_matrix_profile
            right_matrix_profile[indices] = result['rmp'][indices]
            right_profile_index[indices] = result['rpi'][indices]

    return {
        'mp': matrix_profile,
        'pi': profile_index,
        'pd': profile_dimension,
        'rmp': right_matrix_profile,
        'rpi': right_profile_index,
        'lmp': left_matrix_profile,
        'lpi': left_profile_index,
        'metric': 'euclidean',
        'w': window_size,
        'ez': exclusion_zone,
        'sample_pct': 1,
        'data': {
            'ts': ts,
            'query': query
        },
        'class': "MatrixProfile",
        'algorithm': "stomp_based_mstamp"
    }