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