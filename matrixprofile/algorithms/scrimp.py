# -*- coding: utf-8 -*-

"""
This module consists of all code to implement the SCRIMP++ algorithm. SCRIMP++ 
is an anytime algorithm that computes the matrix profile for a given time 
series (ts) over a given window size (m).

This algorithm was originally created at the University of California 
Riverside. For further academic understanding, please review this paper:

Matrix Proﬁle XI: SCRIMP++: Time Series Motif Discovery at Interactive
Speed. Yan Zhu, Chin-Chia Michael Yeh, Zachary Zimmerman, Kaveh Kamgar
Eamonn Keogh, ICDM 2018.

https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import math

import warnings

import numpy as np

from matrixprofile import core


def calc_distance_profile(X, y, n, m, meanx, sigmax):
    """
    Computes the distance profile.

    Parameters
    ----------
    X : array_like
        The FFT transformed time series.
    y : array_like
        The query.
    n : int
        The length of the time series.
    m : int
        The window size.
    meanx : array_like
        The moving mean of the time series.
    sigmax : array_like
        The moving standard deviation of the time series.

    Returns
    -------
    array_like :
        The distance profile.

    """
    # reverse the query
    y = np.flip(y, 0)
   
    # make y same size as ts with zero fill
    y = np.concatenate([y, np.zeros(n-m)])

    # main trick of getting dot product in O(n log n) time
    Y = np.fft.fft(y)
    Z = X * Y
    z = np.fft.ifft(Z)

    # compute y stats in O(n)
    sumy = np.sum(y)
    sumy2 = np.sum(np.power(y, 2))
    meany = sumy / m
    sigmay2 = sumy2 / m - meany ** 2
    sigmay = np.sqrt(sigmay2)

    dist = (z[m - 1:n] - m * meanx * meany) / (sigmax * sigmay)
    dist = m - dist
    dist = np.real(2 * dist)

    return np.sqrt(np.absolute(dist))


def calc_dotproduct_idx(dotproduct, m, mp, idx, sigmax, idx_nn, meanx):
    dotproduct[idx] = (m - mp[idx] ** 2 / 2) * \
        sigmax[idx] * sigmax[idx_nn] + m * meanx[idx] * meanx[idx_nn]

    return dotproduct


def calc_dotproduct_end_idx(ts, dp, idx, m, endidx, idx_nn, idx_diff):
    tmp_a = ts[idx+m:endidx+m]
    tmp_b = ts[idx_nn+m:endidx+m+idx_diff]
    tmp_c = ts[idx:endidx]
    tmp_d = ts[idx_nn:endidx+idx_diff]
    tmp_f = tmp_a * tmp_b - tmp_c * tmp_d

    dp[idx+1:endidx+1] = dp[idx] + np.cumsum(tmp_f)

    return dp


def calc_refine_distance_end_idx(refine_distance, dp, idx, endidx, meanx, 
                                 sigmax, idx_nn, idx_diff, m):
    tmp_a = dp[idx+1:endidx+1]
    tmp_b = meanx[idx+1:endidx+1]
    tmp_c = meanx[idx_nn+1:endidx+idx_diff+1]
    tmp_d = sigmax[idx+1:endidx+1]
    tmp_e = sigmax[idx_nn+1:endidx+idx_diff+1]
    tmp_f = tmp_b * tmp_c
    tmp_g = tmp_d * tmp_e
    tmp_h = (m-(tmp_a - m * tmp_f) / (tmp_g))
    refine_distance[idx+1:endidx+1] = np.sqrt(np.abs(2 * tmp_h))    

    return refine_distance


def calc_dotproduct_begin_idx(ts, dp, beginidx, idx, idx_diff, m, 
                              idx_nn):
    indices = list(range(idx - 1, beginidx - 1, -1))    

    if not indices:
        return dp

    tmp_a = ts[indices]
    indices_b = list(range(idx_nn - 1, beginidx + idx_diff - 1, -1))
    tmp_b = ts[indices_b]
    indices_c = list(range(idx + m - 1, beginidx + m - 1, -1))
    tmp_c = ts[indices_c]
    indices_d = list(range(idx_nn - 1 + m, beginidx + idx_diff + m - 1, -1))
    tmp_d = ts[indices_d]

    dp[indices] = dp[idx] + \
        np.cumsum((tmp_a * tmp_b) - (tmp_c * tmp_d))

    return dp


def calc_refine_distance_begin_idx(refine_distance, dp, beginidx, idx, 
                                   idx_diff, idx_nn, sigmax, meanx, m):
    if not (beginidx < idx):
        return refine_distance

    tmp_a = dp[beginidx:idx]
    tmp_b = meanx[beginidx:idx]
    tmp_c = meanx[beginidx+idx_diff:idx_nn]
    tmp_d = sigmax[beginidx:idx]
    tmp_e = sigmax[beginidx+idx_diff:idx_nn]
    tmp_f = tmp_b * tmp_c
    tmp_g = tmp_d * tmp_e
    tmp_h = (m-(tmp_a - m * tmp_f) / (tmp_g))

    refine_distance[beginidx:idx] = np.sqrt(np.abs(2 * tmp_h))

    return refine_distance


def apply_update_positions(matrix_profile, mp_index, refine_distance, beginidx,
                           endidx, orig_index, idx_diff):
    tmp_a = refine_distance[beginidx:endidx+1]
    tmp_b = matrix_profile[beginidx:endidx+1]
    update_pos1 = np.argwhere(tmp_a < tmp_b).flatten()    

    if len(update_pos1) > 0:        
        update_pos1 = update_pos1 + beginidx
        matrix_profile[update_pos1] = refine_distance[update_pos1]
        mp_index[update_pos1] = orig_index[update_pos1] + idx_diff

    tmp_a = refine_distance[beginidx:endidx + 1]
    tmp_b = matrix_profile[beginidx + idx_diff:endidx + idx_diff + 1]
    update_pos2 = np.argwhere(tmp_a < tmp_b).flatten()

    if len(update_pos2) > 0:
        update_pos2 = update_pos2 + beginidx
        matrix_profile[update_pos2 + idx_diff] = refine_distance[update_pos2]
        mp_index[update_pos2 + idx_diff] = orig_index[update_pos2]

    return (matrix_profile, mp_index)


def compute_indices(profile_len, step_size, sample_pct):
    """
    Computes the indices used for profile index iteration based on the
    number of samples and step size.

    Parameters
    ----------
    profile_len : int
        The length of the profile to be computed.
    step_size : float
        step_size : float, default 0.25
        The sampling interval for the window. The paper suggest 0.25 is the
        most practical. It should be a float value between 0 and 1.
    sample_pct : float, default = 0.1 (10%)
        Number of samples to compute distances for in the MP.

    Returns
    -------
    array_like :
        The indices to compute.

    """
    compute_order = np.arange(0, profile_len, step=step_size)
    sample_size = int(np.ceil(len(compute_order) * sample_pct))
    samples = np.random.choice(compute_order, size=sample_size, replace=False)

    return samples


def prescrimp(ts, window_size, query=None, step_size=0.25, sample_pct=0.1,
                     random_state=None, n_jobs=1):
    """
    This is the PreScrimp algorithm from the SCRIMP++ paper. It is primarly
    used to compute the approximate matrix profile. In this case we use
    a sample percentage to mock "the anytime/approximate nature".

    Parameters
    ----------
    ts : np.ndarray
        The time series to compute the matrix profile for.
    window_size : int
        The window size.
    query : array_like
        Optionally, a query can be provided to perform a similarity join.
    step_size : float, default 0.25
        The sampling interval for the window. The paper suggest 0.25 is the
        most practical. It should be a float value between 0 and 1.
    sample_pct : float, default = 0.1 (10%)
        Number of samples to compute distances for in the MP.
    random_state : int, default None
        Set the random seed generator for reproducible results.
    n_jobs : int, Default = 1
        Number of cpu cores to use.

    Note
    ----
    The matrix profiles computed from prescrimp will always be the approximate
    solution.

    Returns
    -------
    dict : profile
        A MatrixProfile data structure.
        
        >>> {
        >>>    'mp': The matrix profile,
        >>>    'pi': The matrix profile 1NN indices,
        >>>    'rmp': The right matrix profile,
        >>>    'rpi': The right matrix profile 1NN indices,
        >>>    'lmp': The left matrix profile,
        >>>    'lpi': The left matrix profile 1NN indices,
        >>>    'metric': The distance metric computed for the mp,
        >>>    'w': The window size used to compute the matrix profile,
        >>>    'ez': The exclusion zone used,
        >>>    'join': Flag indicating if a similarity join was computed,
        >>>    'sample_pct': Percentage of samples used in computing the MP,
        >>>    'data': {
        >>>        'ts': Time series data,
        >>>        'query': Query data if supplied
        >>>    }
        >>>    'class': "MatrixProfile"
        >>>    'algorithm': "prescrimp"
        >>>}

    Raises
    ------
    ValueError
        If window_size < 4.
        If window_size > query length / 2.
        If ts is not a list or np.array.
        If query is not a list or np.array.
        If ts or query is not one dimensional.
        If sample_pct is not between 0 and 1.

    """
    is_join = core.is_similarity_join(ts, query)
    if not is_join:
        query = ts

    # data conversion to np.array
    ts = core.to_np_array(ts)
    query = core.to_np_array(query)

    # validate step_size
    if not isinstance(step_size, float) or step_size > 1 or step_size < 0:
        raise ValueError('step_size should be a float between 0 and 1.')

    # validate sample_pct
    if not isinstance(sample_pct, float) or sample_pct > 1 or sample_pct < 0:
        raise ValueError('sample_pct should be a float between 0 and 1.')

    # validate random_state
    if random_state is not None:
        try:
            np.random.seed(random_state)
        except:
            raise ValueError('Invalid random_state value given.')

    if window_size < 4:
        error = "window size must be at least 4."
        raise ValueError(error)

    if window_size > len(query) / 2:
        error = "Time series is too short relative to desired window size"
        raise ValueError(error)

    # precompute some common values - profile length, query length etc.
    step_size = int(math.floor(window_size * step_size))
    profile_length = core.get_profile_length(ts, query, window_size)
    data_length = len(ts)
    exclusion_zone = int(np.ceil(window_size / 4.0))

    matrix_profile = np.zeros(profile_length)
    mp_index = np.zeros(profile_length, dtype='int')

    X = np.fft.fft(ts)
    mux, sigx = core.moving_avg_std(ts, window_size)

    dotproduct = np.zeros(profile_length)
    refine_distance = np.full(profile_length, np.inf)
    orig_index = np.arange(profile_length)

    # iterate over sampled indices and update the matrix profile
    # compute_order = compute_indices(profile_length, step_size, sample_pct)
    compute_order = np.arange(0, profile_length, step=step_size)

    for iteration, idx in enumerate(compute_order):
        subsequence = ts[idx:idx + window_size]

        # compute distance profile
        distance_profile = calc_distance_profile(X, subsequence, data_length,
            window_size, mux, sigx)
        
        # apply exclusion zone
        distance_profile = core.apply_exclusion_zone(exclusion_zone, is_join,
            window_size, data_length, idx, distance_profile)

        # find and store nearest neighbor
        if iteration == 0:
            matrix_profile = distance_profile
            mp_index[:] = idx
        else:
            update_pos = distance_profile < matrix_profile
            mp_index[update_pos] = idx
            matrix_profile[update_pos] = distance_profile[update_pos]

        idx_min = np.argmin(distance_profile)
        matrix_profile[idx] = distance_profile[idx_min]
        mp_index[idx] = idx_min
        idx_nn = mp_index[idx]

        # compute the target indices
        idx_diff = idx_nn - idx
        endidx = np.min([
            profile_length - 1,
            idx + step_size - 1,
            profile_length - idx_diff - 1
        ])
        beginidx = np.max([0, idx - step_size + 1, 2 - idx_diff])

        # compute dot product and refine distance for the idx, begin idx 
        # and end idx
        dotproduct = calc_dotproduct_idx(dotproduct, window_size, 
            matrix_profile, idx, sigx, idx_nn, mux)

        dotproduct = calc_dotproduct_end_idx(ts, dotproduct, idx, window_size,
                                             endidx, idx_nn, idx_diff)

        refine_distance = calc_refine_distance_end_idx(
            refine_distance, dotproduct, idx, endidx, mux, sigx, idx_nn,
            idx_diff, window_size)
        
        dotproduct = calc_dotproduct_begin_idx(
            ts, dotproduct, beginidx, idx, idx_diff, window_size, idx_nn)

        refine_distance = calc_refine_distance_begin_idx(
            refine_distance, dotproduct, beginidx, idx, idx_diff, idx_nn, 
            sigx, mux, window_size)

        matrix_profile, mp_index = apply_update_positions(matrix_profile, 
                                                          mp_index, 
                                                          refine_distance, 
                                                          beginidx, 
                                                          endidx, 
                                                          orig_index, idx_diff)

    return {
        'mp': matrix_profile,
        'pi': mp_index,
        'rmp': None,
        'rpi': None,
        'lmp': None,
        'lpi': None,
        'w': window_size,
        'ez': exclusion_zone,
        'join': is_join,
        'sample_pct': sample_pct,
        'metric': 'euclidean',
        'data': {
            'ts': ts,
            'query': query if is_join else None
        },
        'class': 'MatrixProfile',
        'algorithm': 'prescrimp',
    }


def scrimp_plus_plus(ts, window_size, query=None, step_size=0.25, sample_pct=0.1,
                     random_state=None, n_jobs=1):
    """SCRIMP++ is an anytime algorithm that computes the matrix profile for a 
    given time series (ts) over a given window size (m). Essentially, it allows
    for an approximate solution to be provided for quicker analysis. In the 
    case of this implementation, sample percentage is used. An approximate
    solution is given based a sample percentage from 0 to 1. The default sample
    percentage is currently 10%.

    This algorithm was created at the University of California Riverside. For
    further academic understanding, please review this paper:

    Matrix Proﬁle XI: SCRIMP++: Time Series Motif Discovery at Interactive
    Speed. Yan Zhu, Chin-Chia Michael Yeh, Zachary Zimmerman, Kaveh Kamgar
    Eamonn Keogh, ICDM 2018.

    https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf

    Parameters
    ----------
    ts : np.ndarray
        The time series to compute the matrix profile for.
    window_size : int
        The window size.
    query : array_like
        Optionally, a query can be provided to perform a similarity join.
    step_size : float, default 0.25
        The sampling interval for the window. The paper suggest 0.25 is the
        most practical. It should be a float value between 0 and 1.
    sample_pct : float, default = 0.1 (10%)
        Number of samples to compute distances for in the MP.
    random_state : int, default None
        Set the random seed generator for reproducible results.
    n_jobs : int, Default = 1
        Number of cpu cores to use.

    Returns
    -------
    dict : profile
        A MatrixProfile data structure.

        >>> {
        >>>    'mp': The matrix profile,
        >>>    'pi': The matrix profile 1NN indices,
        >>>    'rmp': The right matrix profile,
        >>>    'rpi': The right matrix profile 1NN indices,
        >>>    'lmp': The left matrix profile,
        >>>    'lpi': The left matrix profile 1NN indices,
        >>>    'metric': The distance metric computed for the mp,
        >>>    'w': The window size used to compute the matrix profile,
        >>>    'ez': The exclusion zone used,
        >>>    'join': Flag indicating if a similarity join was computed,
        >>>    'sample_pct': Percentage of samples used in computing the MP,
        >>>    'data': {
        >>>        'ts': Time series data,
        >>>        'query': Query data if supplied
        >>>    }
        >>>    'class': "MatrixProfile"
        >>>    'algorithm': "scrimp++"
        >>> }

    Raises
    ------
    ValueError
        If window_size < 4.
        If window_size > query length / 2.
        If ts is not a list or np.array.
        If query is not a list or np.array.
        If ts or query is not one dimensional.
        If sample_pct is not between 0 and 1.

    """
    # validate random_state
    if random_state is not None:
        try:
            np.random.seed(random_state)
        except:
            raise ValueError('Invalid random_state value given.')

    ###########################
    # PreSCRIMP
    ###########################
    profile = prescrimp(ts, window_size, query=query, step_size=step_size,
        sample_pct=sample_pct, random_state=random_state, n_jobs=n_jobs)

    # data conversion to np.array
    ts = profile['data']['ts']
    query = profile['data']['query']
    if isinstance(query, type(None)):
        query = ts

    # precompute some common values - profile length, query length etc.
    step_size = int(math.floor(window_size * step_size))
    profile_length = core.get_profile_length(ts, query, window_size)
    data_length = len(ts)
    exclusion_zone = profile['ez']
    window_size = profile['w']

    # precompute some statistics on ts
    data_mu, data_sig = core.moving_avg_std(ts, window_size)

    ###########################
    # SCRIMP
    ###########################

    # randomly sort indices for compute order
    orig_index = np.arange(profile_length)
    compute_order = np.copy(orig_index[orig_index > exclusion_zone])
    #np.random.shuffle(compute_order)

    # Only refine to provided sample_pct
    sample_size = int(np.ceil(len(compute_order) * sample_pct))
    compute_order = np.random.choice(compute_order, size=sample_size, 
        replace=False)

    # initialize some values
    curlastz = np.zeros(profile_length)
    curdistance = np.zeros(profile_length)
    dist1 = np.full(profile_length, np.inf)
    dist2 = np.full(profile_length, np.inf)

    for idx in compute_order:
        # compute last z
        curlastz[idx] = np.sum(ts[0:window_size] * ts[idx:idx + window_size])
        curlastz[idx+1:] = curlastz[idx] + np.cumsum(
            (ts[window_size:data_length - idx] * ts[idx + window_size:data_length]) -\
            (ts[0:profile_length - idx - 1] * ts[idx:profile_length - 1])
        )

        # compute distances
        curdistance[idx:] = np.sqrt(np.abs(
            2 * (window_size - (curlastz[idx:profile_length + 1] -\
                window_size * (data_mu[idx:] * data_mu[0:profile_length - idx])) /\
                (data_sig[idx:] * data_sig[0:profile_length - idx]))
        ))

        dist1[0:idx - 1] = np.inf
        dist1[idx:] = curdistance[idx:]

        dist2[0:profile_length - idx] = curdistance[idx:]
        dist2[profile_length - idx + 2:] = np.inf

        loc1 = dist1 < profile['mp']
        if loc1.any():
            profile['mp'][loc1] = dist1[loc1]
            profile['pi'][loc1] = orig_index[loc1] - idx

        loc2 = dist2 < profile['mp']
        if loc2.any():
            profile['mp'][loc2] = dist2[loc2]
            profile['pi'][loc2] = orig_index[loc2] + idx


    profile['algorithm'] = 'scrimp++'
    profile['sample_pct'] = sample_pct

    return profile
