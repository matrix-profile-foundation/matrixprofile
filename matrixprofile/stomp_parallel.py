# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import math

from multiprocessing import cpu_count

import numpy as np

from matrixprofile import core

import ray


@ray.remote
class StompParamActor(object):
    def __init__(self, ts, query, window_size, data_length, profile_length,
                 exclusion_zone, is_join):
        self.ts = ts
        self.query = query
        self.window_size = window_size
        self.data_length = data_length
        self.profile_length = profile_length
        self.exclusion_zone = exclusion_zone
        self.is_join = is_join

        # compute some stats
        self.data_mu, self.data_sig = core.moving_avg_std(ts, window_size)
        self.data_freq = np.fft.fft(ts)

        first_window = query[0:window_size]
        self.first_product = core.sliding_dot_product(ts, first_window)


    def get_attr(self, attr):
        return getattr(self, attr)



@ray.remote
def batch_compute2(batch_start, batch_end, ts, query, window_size, data_length,
                   profile_length, exclusion_zone, is_join):
    
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

     # initialize matrices
    matrix_profile = np.full(profile_length, np.inf)
    profile_index = np.full(profile_length, -np.inf)

    # here we pull out the mass_post to make the loop easier to read
    # compute query stats
    first_window = query[batch_start:batch_start + window_size]
    last_product = core.sliding_dot_product(ts, first_window)

    query_sum = np.sum(first_window)
    query_2sum = np.sum(first_window ** 2)
    query_mu, query_sig = core.moving_avg_std(first_window, window_size)

    drop_value = first_window[0]

    distance_profile = core.distance_profile(
            last_product, window_size, data_mu, data_sig, query_mu, query_sig)
       
    # update the matrix profile
    indices = (distance_profile < matrix_profile)
    matrix_profile[indices] = distance_profile[indices]
    profile_index[np.argwhere(indices)] = batch_start

    # iteratively compute distance profile and update with element-wise mins
    for i in range(batch_start + 1, batch_end):

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
        index = np.argmin(distance_profile)
        matrix_profile[i] = distance_profile[index]
        # profile_index[index] = i

    return matrix_profile


@ray.remote
def batch_compute3(batch_start, batch_end, actor):
    # grab data so it is local again
    ts = ray.get(actor.get_attr.remote('ts'))
    window_size = ray.get(actor.get_attr.remote('window_size'))
    query = ray.get(actor.get_attr.remote('query'))
    data_length = ray.get(actor.get_attr.remote('data_length'))
    profile_length = ray.get(actor.get_attr.remote('profile_length'))
    is_join = ray.get(actor.get_attr.remote('is_join'))
    exclusion_zone = ray.get(actor.get_attr.remote('exclusion_zone'))

    first_product = ray.get(actor.get_attr.remote('first_product'))
    data_freq = ray.get(actor.get_attr.remote('data_freq'))
    data_mu = ray.get(actor.get_attr.remote('data_mu'))
    data_sig = ray.get(actor.get_attr.remote('data_sig'))

     # initialize matrices
    matrix_profile = np.full(profile_length, np.inf)
    profile_index = np.full(profile_length, -np.inf)

    # iteratively compute distance profile and update with element-wise mins
    for i in range(batch_start, batch_end):

        # check for nan or inf and skip
        segment = ts[i:i + window_size]
        search = (np.isinf(segment) | np.isnan(segment))
        
        if np.any(search):
            continue

        query_window = query[i:i + window_size]
        last_product = core.sliding_dot_product(ts, query_window)
        query_mu, query_sig = core.moving_avg_std(query_window, window_size)

        distance_profile = core.distance_profile(
            last_product, window_size, data_mu, data_sig, query_mu, query_sig)

        # apply the exclusion zone
        # for similarity join we do not apply exclusion zone
        if exclusion_zone > 0 and not is_join:
            ez_start = np.max([0, i - exclusion_zone])
            ez_end = np.min([len(ts) - window_size + 1, i + exclusion_zone])
            distance_profile[ez_start:ez_end] = np.inf
            
        # update the matrix profile
        index = np.argmin(distance_profile)
        matrix_profile[i] = distance_profile[index]
        # profile_index[index] = i

    return matrix_profile


@ray.remote
def batch_compute(batch_start, batch_end, actor):
    # grab data so it is local again
    ts = ray.get(actor.get_attr.remote('ts'))
    window_size = ray.get(actor.get_attr.remote('window_size'))
    query = ray.get(actor.get_attr.remote('query'))
    data_length = ray.get(actor.get_attr.remote('data_length'))
    profile_length = ray.get(actor.get_attr.remote('profile_length'))
    is_join = ray.get(actor.get_attr.remote('is_join'))
    exclusion_zone = ray.get(actor.get_attr.remote('exclusion_zone'))

    first_product = ray.get(actor.get_attr.remote('first_product'))
    data_freq = ray.get(actor.get_attr.remote('data_freq'))
    data_mu = ray.get(actor.get_attr.remote('data_mu'))
    data_sig = ray.get(actor.get_attr.remote('data_sig'))

     # initialize matrices
    matrix_profile = np.full(profile_length, np.inf)
    profile_index = np.full(profile_length, -np.inf)

    # here we pull out the mass_post to make the loop easier to read
    # compute query stats
    first_window = query[batch_start:batch_start + window_size]
    last_product = core.sliding_dot_product(ts, first_window)

    query_sum = np.sum(first_window)
    query_2sum = np.sum(first_window ** 2)
    query_mu, query_sig = core.moving_avg_std(first_window, window_size)

    drop_value = first_window[0]

    distance_profile = core.distance_profile(
            last_product, window_size, data_mu, data_sig, query_mu, query_sig)
       
    # update the matrix profile
    indices = (distance_profile < matrix_profile)
    matrix_profile[indices] = distance_profile[indices]
    profile_index[np.argwhere(indices)] = batch_start

    # iteratively compute distance profile and update with element-wise mins
    for i in range(batch_start + 1, batch_end):

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
        index = np.argmin(distance_profile)
        matrix_profile[i] = distance_profile[index]
        # profile_index[index] = i

    return matrix_profile


def get_batch_windows(profile_length, n_jobs):
    batch_size = int(math.ceil(profile_length / n_jobs))

    if batch_size == profile_length:
        yield (0, profile_length)
    else:
        for i in range(n_jobs):
            start = i * batch_size        
            end = (i + 1) * batch_size
            
            if end > profile_length:
                end = profile_length

            yield (start, end)


def stomp_parallel(ts, window_size, query=None, n_jobs=-1):
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
    n_jobs : int, default All
        The number of cpu cores to use for processing. It defaults to all
        available cores.

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

    # set the n_jobs appropriately
    if n_jobs < 1:
        n_jobs = cpu_count()

    if n_jobs > cpu_count():
        n_jobs = cpu_count()
    
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

    # start ray
    # we can optionally set the memory usage, but I don't know if I want to
    # expose this to users
    #
    # convert GB to bytes:
    # int(1e+9 * memory)
    #
    # object_store_memory – The amount of memory (in bytes) to start the object 
    # store with. By default, this is capped at 20GB but can be set higher.
    #
    # redis_max_memory – The max amount of memory (in bytes) to allow each 
    # edis shard to use. Once the limit is exceeded, redis will start LRU 
    # eviction of entries. This only applies to the sharded redis tables 
    # (task, object, and profile tables). By default, this is capped at 10GB 
    # but can be set higher.
    ray.init(num_cpus=n_jobs, ignore_reinit_error=True, logging_level=40,) #local_mode=True)

    # TODO: create batch jobs and submit to ray

    #####################
    # test computebatch
    #####################

    # actor = StompParamActor.remote(
    #     ts,
    #     query,
    #     window_size,
    #     data_length,
    #     profile_length,
    #     exclusion_zone,
    #     is_join
    # )
    # batches = []
    # batch_windows = []
    # for start, end in get_batch_windows(profile_length, n_jobs):
    #     batches.append(batch_compute.remote(start, end, actor))
    #     batch_windows.append((start, end))

    #####################
    # test computebatch2
    #####################
    # batches = []
    # batch_windows = []
    # for start, end in get_batch_windows(profile_length, n_jobs):
    #     batches.append(batch_compute2.remote(
    #         start, end, ts, query, window_size, data_length,
    #         profile_length, exclusion_zone, is_join
    #     ))
    #     batch_windows.append((start, end))

    #####################
    # test computebatch3
    #####################
    actor = StompParamActor.remote(
        ts,
        query,
        window_size,
        data_length,
        profile_length,
        exclusion_zone,
        is_join
    )
    batches = []
    batch_windows = []
    for start, end in get_batch_windows(profile_length, n_jobs):
        batches.append(batch_compute3.remote(start, end, actor))
        batch_windows.append((start, end))  

    results = ray.get(batches)
    for index, result in enumerate(results):
        batch_start = batch_windows[index][0]
        batch_end = batch_windows[index][1]        
        
        # update the matrix profile
        matrix_profile[batch_start:batch_end] = result[batch_start:batch_end]
        # indices = (result < matrix_profile)
        # matrix_profile[indices] = result[indices]
        # profile_index[indices] = 0

        # # update the left and right matrix profiles
        # if not is_join:
        #     # find differences, shift left and update
        #     indices = result['mp'][i:] < left_matrix_profile[batch_start + i:]
        #     falses = np.zeros(batch_start + i).astype('bool')
        #     indices = np.append(falses, indices)
        #     left_matrix_profile[indices] = distance_profile[indices]
        #     left_profile_index[np.argwhere(indices)] = i

        #     # find differences, shift right and update
        #     indices = distance_profile[0:i] < right_matrix_profile[0:i]
        #     falses = np.zeros(profile_length - i).astype('bool')
        #     indices = np.append(indices, falses)
        #     right_matrix_profile[indices] = distance_profile[indices]
        #     right_profile_index[np.argwhere(indices)] = i


    #results = batch_process(ts, window_size, n_jobs, query=None)

    # TODO: re-assemble full matrix profiles
    
    ray.shutdown()

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
