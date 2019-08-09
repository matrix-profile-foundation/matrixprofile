# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import math
from multiprocessing import cpu_count, Pool
import logging

import numpy as np
import ray

from matrixprofile import core

logger = logging.getLogger(__name__)

def _batch_compute(args):
    batch_start, batch_end, ts, query, window_size, data_length, \
    profile_length, exclusion_zone, is_join, data_mu, data_sig, \
    first_product = args

     # initialize matrices
    matrix_profile = np.full(profile_length, np.inf)
    profile_index = np.full(profile_length, -np.inf)

    left_matrix_profile = None
    right_matrix_profile = None
    left_profile_index = None
    right_profile_index = None

    if not is_join:
        left_matrix_profile = np.copy(matrix_profile)
        right_matrix_profile = np.copy(matrix_profile)
        left_profile_index = np.copy(profile_index)
        right_profile_index = np.copy(profile_index)

    # here we pull out the mass_post to make the loop easier to read
    # compute query stats
    last_product = None
    if batch_start is 0:
        first_window = query[batch_start:batch_start + window_size]
        last_product = np.copy(first_product)
    else:
        first_window = query[batch_start - 1:batch_start + window_size - 1]        
        last_product = core.sliding_dot_product(ts, first_window)

    query_sum = np.sum(first_window)
    query_2sum = np.sum(first_window ** 2)
    query_mu, query_sig = core.moving_avg_std(first_window, window_size)

    drop_value = first_window[0]

    if batch_start is 0:
        distance_profile = core.distance_profile(last_product, window_size,
         data_mu, data_sig, query_mu, query_sig)
           
        # update the matrix profile
        index = np.argmin(distance_profile)
        matrix_profile[0] = distance_profile[index]
        profile_index[index] = 0

        batch_start += 1

    if batch_end < profile_length:
        batch_end += 1

    # iteratively compute distance profile and update with element-wise mins
    for i in range(batch_start, batch_end):

        # check for nan or inf and skip
        segment = ts[i:i + window_size]
        search = (np.isinf(segment) | np.isnan(segment))
        
        if np.any(search):
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
            ez_end = np.min([len(ts) - window_size + 1, i + exclusion_zone])
            distance_profile[ez_start:ez_end] = np.inf
            
        # update the matrix profile
        # indices = (distance_profile < matrix_profile)
        index = np.argmin(distance_profile)
        matrix_profile[i] = distance_profile[index]
        profile_index[index] = i

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


@ray.remote
def _batch_compute_ray(batch_start, batch_end, ts, query, window_size, 
                       data_length, profile_length, exclusion_zone, 
                       is_join, data_mu, data_sig, first_product):
    args = (batch_start, batch_end, ts, query, window_size, data_length, 
            profile_length, exclusion_zone, is_join, data_mu, data_sig,
            first_product)
    return _batch_compute(args)


def stomp_parallel(ts, window_size, query=None, n_jobs=-1):
    """
    Computes matrix profiles for a single dimensional time series using the 
    parallelized STOMP algorithm. Ray or Python's multiprocessing library may
    be used. When you have initialized Ray on your machine, it takes priority
    over using Python's multiprocessing.

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix profile for.
    window_size: int
        The size of the window to compute the matrix profile over.
    query : array_like
        Optionally, a query can be provided to perform a similarity join.
    n_jobs : int, default All
        The number of batch jobs to compute at once. Note that when ray is
        initialized we cannot tell how many cores are available. You must
        explicitly state how many jobs you want. In the case of
        multiprocessing, n_jobs is a 1:1 relationship with number of cpu cores.

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
        'algorithm': "stomp_parallel"
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

    # use ray or multiprocessing
    if ray.is_initialized():
        cpus = int(ray.available_resources()['CPU'])
        logger.warn('Using Ray with {} cpus'.format(cpus))
    else:
        n_jobs = core.valid_n_jobs(n_jobs)

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
    first_product = core.sliding_dot_product(ts, first_window)

    batch_windows = []
    if ray.is_initialized():

        # ray likes to have globally accessible data
        # here it makes sense for these variables so we put them in the object
        # store
        ts_id = ray.put(ts)
        query_id = ray.put(query)
        window_size_id = ray.put(window_size)
        data_length_id = ray.put(data_length)
        profile_length_id = ray.put(profile_length)
        exclusion_zone_id = ray.put(exclusion_zone)
        is_join_id = ray.put(is_join)
        data_mu_id = ray.put(data_mu)
        data_sig_id = ray.put(data_sig)
        first_product_id = ray.put(first_product)

        # batch compute with ray
        batches = []        
        for start, end in core.generate_batch_jobs(profile_length, n_jobs):
            batches.append(_batch_compute_ray.remote(
                start, end, ts_id, query_id, window_size_id, data_length_id,
                profile_length_id, exclusion_zone_id, is_join_id, data_mu_id,
                data_sig_id, first_product_id
            ))
            batch_windows.append((start, end))
        
        results = ray.get(batches)
    else:
        # batch compute with multiprocessing
        args = []
        for start, end in core.generate_batch_jobs(profile_length, n_jobs):
            args.append((
                start, end, ts, query, window_size, data_length,
                profile_length, exclusion_zone, is_join, data_mu, data_sig,
                first_product
            ))
            batch_windows.append((start, end))

        with core.mp_pool()(n_jobs) as pool:
            results = pool.map(_batch_compute, args)

    # now we combine the batch results
    for index, result in enumerate(results):
        start = batch_windows[index][0]
        end = batch_windows[index][1]        
        
        # update the matrix profile
        matrix_profile[start:end] = result['mp'][start:end]
        profile_index[start:end] = result['pi'][start:end]

        # # update the left and right matrix profiles
        if not is_join:
            left_matrix_profile[start:end] = result['lmp'][start:end]
            left_profile_index[start:end] = result['lpi'][start:end] 
            right_matrix_profile[start:end] = result['rmp'][start:end]
            right_profile_index[start:end] = result['rpi'][start:end]             

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
        'algorithm': "stomp_parallel"
    }
