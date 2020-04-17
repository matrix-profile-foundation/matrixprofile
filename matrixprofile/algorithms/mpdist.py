#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import math

import numpy as np

from matrixprofile import core
from matrixprofile.algorithms.cympx import mpx_ab_parallel as cympx_ab_parallel
from matrixprofile.algorithms.mass2 import mass2


def mpdist(ts, ts_b, w, threshold=0.05, n_jobs=1):
    """
    Computes the MPDist between the two series ts and ts_b. For more details
    refer to the paper:

    Matrix Proﬁle XII: MPdist: A Novel Time Series Distance Measure to Allow 
    Data Mining in More Challenging Scenarios. Shaghayegh Gharghabi, 
    Shima Imani, Anthony Bagnall, Amirali Darvishzadeh, Eamonn Keogh. ICDM 2018

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix profile for.
    ts_b : array_like
        The time series to compare against.
    w : int
        The window size.
    threshold : float, Default 0.05
        The percentile in which the distance is taken from. By default it is
        set to 0.05 based on empircal research results from the paper. 
        Generally, you should not change this unless you know what you are
        doing! This value must be a float greater than 0 and less than 1.
    n_jobs : int, Default = 1
        Number of cpu cores to use.
    
    Returns
    -------
    float : mpdist
        The MPDist.

    """
    ts = core.to_np_array(ts).astype('d')
    ts_b = core.to_np_array(ts_b).astype('d')
    n_jobs = core.valid_n_jobs(n_jobs)

    if not core.is_one_dimensional(ts):
        raise ValueError('ts must be one dimensional!')

    if not core.is_one_dimensional(ts_b):
        raise ValueError('ts_b must be one dimensional!')

    if not isinstance(threshold, float) or threshold <= 0 or threshold >= 1:
        raise ValueError('threshold must be a float greater than 0 and less'\
            ' than 1')

    mp, mpi, mpb, mpib = cympx_ab_parallel(ts, ts_b, w, 0, n_jobs)

    mp_abba = np.append(mp, mpb)
    data_len = len(ts) + len(ts_b)
    abba_sorted = np.sort(mp_abba[~core.nan_inf_indices(mp_abba)])

    distance = np.inf
    if len(abba_sorted) > 0:
        upper_idx = int(np.ceil(threshold * data_len)) - 1
        idx = np.min([len(abba_sorted) - 1, upper_idx])
        distance = abba_sorted[idx]

    return distance


def mass_distance_matrix(ts, query, w):
    """
    Computes a distance matrix using mass that is used in mpdist_vector
    algorithm.

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix for.
    query : array_like
        The time series to compare against.
    w : int
        The window size.
    
    Returns
    -------
    array_like : dist_matrix
        The MASS distance matrix.

    """
    subseq_num = len(query) - w + 1
    distances = []
    
    for i in range(subseq_num):
        distances.append(np.real(mass2(ts, query[i:i + w])))
    
    return np.array(distances)


def calculate_mpdist(profile, threshold, data_length):
    """
    Computes the MPDist given a profile, threshold and data length. This is
    primarily used for MPDist Vector algorithm.

    Parameters
    ----------
    profile : array_like
        The profile to calculate the mpdist for.
    threshold : float
        The threshold to use in computing the distance.
    data_length : int
        The length of the original data.

    Returns
    -------
    float : mpdist
        The MPDist.

    """
    dist_loc = int(np.ceil(threshold * data_length))
    profile_sorted = np.sort(profile)
    mask = core.not_nan_inf_indices(profile_sorted)
    
    profile_clean = profile_sorted[mask]
    
    if len(profile_clean) < 1:
        distance = np.inf
    elif len(profile_clean) >= dist_loc:
        distance = profile_clean[dist_loc]
    else:
        distance = np.max(profile_clean)
    
    return distance


def mpdist_vector(ts, ts_b, w):
    """
    Computes a vector of MPDist measures.

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix for.
    ts_b : array_like
        The time series to compare against.
    w : int
        The window size.
    
    Returns
    -------
    array_like : mpdist_vector
        The MPDist vector.

    """
    matrix = mass_distance_matrix(ts, ts_b, w)
    rows, cols = matrix.shape

    # compute row and column minimums
    all_right_hist = matrix.min(axis=0)
    mass_minimums = np.apply_along_axis(core.moving_min, 1, matrix, window=rows)

    # recreate the matrix profile and compute MPDist
    mpdist_length = len(ts) - len(ts_b) + 1
    right_hist_length = len(ts_b) - w + 1
    mpdist_array = np.zeros(mpdist_length)
    left_hist = np.zeros(right_hist_length)
    
    mpdist_array = []
    for i in range(mpdist_length):
        right_hist = all_right_hist[i:right_hist_length + i]
        left_hist = mass_minimums[:, i]
        profile = np.append(left_hist, right_hist)
        mpdist_array.append(calculate_mpdist(profile, 0.05, 2 * len(ts_b)))
    
    return np.array(mpdist_array)
