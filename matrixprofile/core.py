# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import logging
import math
import multiprocessing
import sys

import numpy as np

from matrixprofile import cycore

logger = logging.getLogger(__name__)


def mp_pool():
    """
    Utility function to get the appropriate multiprocessing
    handler for Python 2 and 3.
    """
    ctxt = None
    if sys.version_info[0] == 2:
        from contextlib import contextmanager

        @contextmanager
        def multiprocessing_context(*args, **kwargs):
            pool = multiprocessing.Pool(*args, **kwargs)
            yield pool
            pool.terminate()

        ctxt = multiprocessing_context
    else:
        ctxt = multiprocessing.Pool

    return ctxt


def is_array_like(a):
    """
    Helper function to determine if a value is array like.

    Parameters
    ----------
    a : obj
        Object to test.

    Returns
    -------
    True or false respectively.
    """
    return isinstance(a, (list, tuple, np.ndarray))


def is_similarity_join(ts_a, ts_b):
    """
    Helper function to determine if a similarity join is occuring or not.

    Parameters
    ----------
    ts_a : array_like
        Time series A.
    ts_b : array_like, None
        Time series B.

    Returns
    -------
    True or false respectively.
    """
    return is_array_like(ts_a) and is_array_like(ts_b)


def to_np_array(a):
    """
    Helper function to convert tuple or list to np.ndarray.

    Parameters
    ----------
    a : Tuple, list or np.ndarray
        The object to transform.

    Returns
    -------
    The np.ndarray.

    Raises
    ------
    ValueError
        If a is not a valid type.
    """
    if isinstance(a, np.ndarray):
        return a

    if not is_array_like(a):
        raise ValueError('Unable to convert to np.ndarray!')    

    return np.array(a)


def is_one_dimensional(a):
    """
    Helper function to determine if value is one dimensional.

    Parameters
    ----------
    a : array_like
        Object to test.

    Returns
    -------
    True or false respectively.
    """
    return a.ndim == 1


def get_profile_length(ts_a, ts_b, m):
    """
    Determines the profile length based on the provided inputs.

    Parameters
    ----------
    ts_a : array_like
        Time series containing the queries for which to calculate the Matrix Profile.
    ts_b : array_line
        Time series containing the queries for which to calculate the Matrix Profile.
    m : int
        Length of subsequence to compare.
    
    Returns
    -------
    int - the length of the matrix profile.
    """
    return len(ts_a) - m + 1


def find_skip_locations(ts, profile_length, window_size):
    """
    Determines which locations should be skipped based on nan or inf values.

    Parameters
    ----------
    ts : array_like
        Time series containing the queries for which to calculate the Matrix Profile.
    query : array_line
        Time series containing the queries for which to calculate the Matrix Profile.
    window_size : int
        Length of subsequence to compare.
    
    Returns
    -------
    int - the length of the matrix profile.
    """
    skip_loc = np.zeros(profile_length).astype(bool)

    for i in range(profile_length):
        segment = ts[i:i + window_size]
        search = (np.isinf(segment) | np.isnan(segment))
        
        if np.any(search):
            skip_loc[i] = True
        
    return skip_loc
    

def clean_nan_inf(ts):
    """
    Replaces nan and inf values with zeros per matrix profile algorithms.

    Parameters
    ----------
    ts: array_like
        Time series to clean.
    
    Returns
    -------
    np.ndarray - The cleaned time series.

    Raises
    ------
    ValueError
        When the ts is not array like.
    """
    ts = to_np_array(ts)
    search = (np.isinf(ts) | np.isnan(ts))
    ts[search] = 0

    return ts


def is_nan_inf(val):
    """
    Helper function to determine if a given value is NaN or infinite.

    Parameters
    ----------
    val : obj
        The value to test against.

    Returns
    -------
    boolean - True or False respectively.
    """
    return np.isnan(val) or np.isinf(val)


def is_not_nan_inf(val):
    """
    Helper function to determine if a given value is not NaN or infinite.

    Parameters
    ----------
    val : obj
        The value to test against.

    Returns
    -------
    boolean - True or False respectively.
    """
    not_nan = not np.isnan(val)
    not_inf = not np.isinf(val)
    return not_nan and not_inf


def nan_inf_indices(a):
    """
    Helper method to obtain the nan/inf indices of an array.

    Parameters
    ----------
    a : array_like
        The array to test.

    Returns
    -------
    Masked array of indices containing nan/inf.
    """
    return (np.isnan(a)) | (np.isinf(a))


def not_nan_inf_indices(a):
    """
    Helper method to obtain the non-nan/inf indices of an array.

    Parameters
    ----------
    a : array_like
        The array to test.

    Returns
    -------
    Masked array of indices containing not nan/inf.
    """
    return (~nan_inf_indices(a))


def rolling_window(a, window):
    """
    Provides a rolling window on a numpy array given an array and window size.

    Parameters
    ----------
    a : array_like
        The array to create a rolling window on.
    window : int
        The window size.

    Returns
    -------
    Strided array for computation.
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def moving_average(a, window=3):
    """
    Computes the moving average over an array given a window size.

    Parameters
    ----------
    a : array_like
        The array to compute the moving average on.
    window : int
        The window size.

    Returns
    -------
    The moving average over the array.
    """
    return np.mean(rolling_window(a, window), -1)


def moving_std(a, window=3):
    """
    Computes the moving std. over an array given a window size.

    Parameters
    ----------
    a : array_like
        The array to compute the moving std. on.
    window : int
        The window size.

    Returns
    -------
    The moving std. over the array.
    """
    return np.std(rolling_window(a, window), -1)


def moving_avg_std(a, window=3):
    """
    Computes the moving avg and std. over an array given a window size.

    Parameters
    ----------
    a : array_like
        The array to compute the moving std. on.
    window : int
        The window size.

    Returns
    -------
    The moving avg and std. over the array as a tuple.
    (avg, std)
    """
    a = a.astype('d')
    mu, sig = cycore.moving_avg_std(a, window)

    return (np.asarray(mu), np.asarray(sig))


def moving_min(a, window=3):
    """
    Computes the moving minimum over an array given a window size.

    Parameters
    ----------
    a : array_like
        The array to compute the moving min on.
    window : int
        The window size.

    Returns
    -------
    array_like :
        The array of moving minimums.
    """
    return np.min(rolling_window(a, window), axis=1)


def moving_max(a, window=3):
    """
    Computes the moving maximum over an array given a window size.

    Parameters
    ----------
    a : array_like
        The array to compute the moving max on.
    window : int
        The window size.

    Returns
    -------
    array_like :
        The array of moving maximums.
    """
    return np.max(rolling_window(a, window), axis=1)


def moving_median(a, window=3):
    """
    Computes the moving median over an array given a window size.

    Parameters
    ----------
    a : array_like
        The array to compute the moving median on.
    window : int
        The window size.

    Returns
    -------
    array_like :
        The array of moving medians.
    """
    return np.median(rolling_window(a, window), axis=1)


def fft_convolve(ts, query):
    """
    Computes the sliding dot product for query over the time series using
    the quicker FFT convolution approach.

    Parameters
    ----------
    ts : array_like
        The time series.
    query : array_like
        The query.

    Returns
    -------
    array_like - The sliding dot product.
    """
    n = len(ts)
    m = len(query)
    x = np.fft.fft(ts)
    y = np.append(np.flipud(query), np.zeros([1, n - m]))
    y = np.fft.fft(y)
    z = np.fft.ifft(x * y)

    return np.real(z[m - 1:n])


def sliding_dot_product(ts, query):
    """
    Computes the sliding dot product for query over the time series using
    convolution. Note that the result is trimmed due to the computations
    being invalid; the len(query) to len(ts) is kept.

    Parameters
    ----------
    ts : array_like
        The time series.
    query : array_like
        The query.

    Returns
    -------
    array_like - The sliding dot product.
    """
    m = len(query)
    n = len(ts)
    dp = np.convolve(ts, np.flipud(query), mode='full')

    return np.real(dp[m - 1:n])


def distance_profile(prod, ws, data_mu, data_sig, query_mu, query_sig):
    """
    Computes the distance profile for the given statistics.

    Parameters
    ----------
    prod : array_like
        The sliding dot product between the time series and query.
    ws : int
        The window size.
    data_mu : array_like
        The time series moving average.
    data_sig : array_like
        The time series moving standard deviation.
    query_mu : array_like
        The querys moving average.
    query_sig : array_like
        The querys moving standard deviation.


    Returns
    -------
    array_like - The distance profile.
    """
    distance_profile = (
        2 * (ws - (prod - ws * data_mu * query_mu) / (data_sig * query_sig))
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        distance_profile = np.sqrt(np.real(distance_profile))

    return distance_profile


def precheck_series_and_query_1d(ts, query):
    """
    Helper function to ensure we have 1d time series and query.

    Parameters
    ----------
    ts : array_like
        The array to create a rolling window on.
    query : array_like
        The query.

    Returns
    -------
    (np.array, np.array) - The ts and query respectively.

    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If query is not a list or np.array.
        If ts or query is not one dimensional.
    """
    try:
        ts = to_np_array(ts)
    except ValueError:
        raise ValueError('Invalid ts value given. Must be array_like!')

    try:
        query = to_np_array(query)
    except ValueError:
        raise ValueError('Invalid query value given. Must be array_like!')

    if not is_one_dimensional(ts):
        raise ValueError('ts must be one dimensional!')

    if not is_one_dimensional(query):
        raise ValueError('query must be one dimensional!')

    return (ts, query)


def valid_n_jobs(n_jobs):
    """
    Validates and assigns correct number of cpu cores.

    Parameters
    ----------
    n_jobs : int
        Number of desired cpu cores.
    
    Returns
    -------
    Valid number of cpu cores.
    """
    max_cpus = multiprocessing.cpu_count()
    if n_jobs < 1:
        n_jobs = max_cpus

    if n_jobs > max_cpus:
        n_jobs = max_cpus
    
    return n_jobs


def generate_batch_jobs(profile_length, n_jobs):
    """
    Generates start and end positions for a matrix profile length and number
    of jobs.

    Parameters
    ----------
    profile_length : int
        The length of the matrix profile to compute.
    n_jobs : int
        The number of jobs (cpu cores).
    

    Returns
    -------
    Yielded start and end index for each job.
    """
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

            if end == profile_length:
                break


def apply_exclusion_zone(exclusion_zone, is_join, window_size, data_length,
    index, distance_profile):
    if exclusion_zone > 0 and not is_join:
        ez_start = np.max([0, index - exclusion_zone])
        ez_end = np.min([data_length - window_size + 1, index + exclusion_zone + 1])
        distance_profile[ez_start:ez_end] = np.inf

    return distance_profile


def pearson_to_euclidean(a, windows):
    """
    Converts an array of Pearson metrics to Euclidean. The array and windows
    should both be row-wise aligned.

    Parameters
    ----------
    a : array_like
        The array of Pearson metrics.
    windows : int or array_like
        The window(s) used to compute the Pearson metrics.

    Returns
    -------
    New array of same dimensions as input array, but with Euclidean distance.
    """
    euc_a = np.full(a.shape, np.inf, dtype='d')
    
    if is_one_dimensional(a):

        window = windows
        if is_array_like(windows):
            window = windows[0]

        is_inf = np.isinf(a)
        euc_a = np.sqrt(2 * window * (1 - a))
    else:
        for window, idx in zip(windows, range(a.shape[0])):
            is_inf = np.isinf(a[idx])
            euc_a[idx] = np.sqrt(2 * window * (1 - a[idx]))
            euc_a[idx][is_inf] = np.inf
    
    return euc_a


def is_pearson_array(a):
    """
    Helper function to determine if an array contains pearson metrics only. It
    is assumed that if the min is >= 0 and max is <= 1, then it is pearson.

    Parameters
    ----------
    a : array_like
        The array to test.

    Returns
    -------
    True or false respectively.
    """
    mask = ~nan_inf_indices(a)
    min_val = a[mask].min()
    max_val = a[mask].max()

    return min_val >= 0 and max_val <= 1


def is_stats_obj(obj):
    """
    Helper function to determine if the current object matches the structure
    that the library enforices for Statistics.

    Parameters
    ----------
    obj : object
        The object to test.
    
    Returns
    -------
    bool :
        True or false respectively.
    """
    return isinstance(obj, dict) and obj.get('class') == 'Statistics'


def is_mp_obj(obj):
    """
    Helper function to determine if the current object matches the structure
    that the library enforices for MatrixProfiles.

    Parameters
    ----------
    obj : object
        The object to test.
    
    Returns
    -------
    bool :
        True or false respectively.
    """
    return isinstance(obj, dict) and obj.get('class') == 'MatrixProfile'


def is_pmp_obj(obj):
    """
    Helper function to determine if the current object matches the structure
    that the library enforices for Pan-MatrixProfiles.

    Parameters
    ----------
    obj : object
        The object to test.
    
    Returns
    -------
    bool :
        True or false respectively.
    """
    return isinstance(obj, dict) and obj.get('class') == 'PMP'


def is_mp_or_pmp_obj(obj):
    """
    Helper function to determine if the current object matches the structure
    that the library enforices for MatrixProfile and Pan-MatrixProfiles.

    Parameters
    ----------
    obj : object
        The object to test.
    
    Returns
    -------
    bool :
        True or false respectively.
    """
    return is_pmp_obj(obj) or is_mp_obj(obj)