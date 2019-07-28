# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import multiprocessing
import sys

import numpy as np


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
    if not is_array_like(a):
        raise ValueError('Unable to convert to np.ndarray!')

    return np.array(a)


def is_one_dimensional(a):
    """
    Helper function to determine if value is one dimensional.
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


def find_skip_locations(ts, query, window_size):
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
    

def self_join_or_not_preprocess(ts_a, ts_b, m):
    """
    Determines if a self join is occuring and returns appropriate
    profile and index numpy arrays with correct dimensions as all np.nan values.
    
    Parameters
    ----------
    ts_a : array_like
        Time series containing the queries for which to calculate the Matrix Profile.
    ts_b : array_line
        Time series containing the queries for which to calculate the Matrix Profile.
    m : int
        Length of subsequence to compare.
    """
    n = len(ts_a)
    if ts_b is not None:
        n = len(ts_b)
    
    shape = n - m + 1
    
    return (np.full(shape, np.inf), np.full(shape, np.inf))


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
    windowed = rolling_window(a, window)
    mu = np.avg(windowed, -1)
    sig = np.std(windowed, -1)


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