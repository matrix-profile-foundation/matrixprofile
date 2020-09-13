# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

# Third-party imports
import numpy as np

# Project imports
from matrixprofile import core


def is_subsequence_constant(subsequence):
    """
    Determines whether the given time series subsequence is an array of constants.

    Parameters
    ----------
    subsequence : array_like
        The time series subsequence to analyze.

    Returns
    -------
    is_constant : bool
        A boolean value indicating whether the given subsequence is an array of constants.

    """
    if not core.is_array_like(subsequence):
        raise ValueError('subsequence is not array like!')

    temp = core.to_np_array(subsequence)
    is_constant = np.all(temp == temp[0])

    return is_constant


def add_noise_to_series(series):
    """
    Adds noise to the given time series.

    Parameters
    ----------
    series : array_like
        The time series subsequence to be added noise.

    Returns
    -------
    temp : array_like
        The time series subsequence after being added noise.

    """
    if not core.is_array_like(series):
        raise ValueError('series is not array like!')

    temp = np.copy(core.to_np_array(series))
    noise = np.random.uniform(0, 0.0000009, size=len(temp))
    temp = temp + noise

    return temp


def impute_missing(ts, window, method='mean', direction='forward'):
    """
    Imputes missing data in time series.

    Parameters
    ----------
    ts : array_like
        The time series to be handled.
    window : int
        The window size to compute the mean/median/minimum value/maximum
        value.
    method : string, Default = 'mean'
        A string indicating the data imputation method, which should be
        'mean', 'median', 'min' or 'max'.
    direction : string, Default = 'forward'
        A string indicating the data imputation direction, which should be
        'forward', 'fwd', 'f', 'backward', 'bwd', 'b'. If the direction is
        forward, we use previous data for imputation; if the direction is
        backward, we use subsequent data for imputation.

    Returns
    -------
    temp : array_like
        The time series after being imputed missing data.

    """
    method_map = {
        'mean': np.mean,
        'median': np.median,
        'min': np.min,
        'max': np.max
    }

    directions = ['forward', 'fwd', 'f', 'backward', 'bwd', 'b']

    if not core.is_array_like(ts):
        raise ValueError('ts is not array like!')

    if method not in method_map:
        raise ValueError('invalid imputation method! valid include options: {}'.format(', '.join(method_map.keys())))

    if direction not in directions:
        raise ValueError('invalid imputation direction! valid include options: ' + ', '.join(directions))

    temp = np.copy(core.to_np_array(ts))
    nan_infs = core.nan_inf_indices(temp)
    func = method_map[method]

    # Deal with missing data at the beginning and end of time series
    if np.isnan(temp[0]) or np.isinf(temp[0]):
        temp[0] = temp[~nan_infs][0]
        nan_infs = core.nan_inf_indices(temp)

    if np.isnan(temp[-1]) or np.isinf(temp[-1]):
        temp[-1] = temp[~nan_infs][-1]
        nan_infs = core.nan_inf_indices(temp)

    # Use previous data for imputation / fills in data in a forward direction
    if direction in directions[:3]:
        for index in range(len(temp) - window + 1):
            start = index
            end = index + window
            has_missing = np.any(nan_infs[index:index + window])

            if has_missing:
                subseq = temp[start:end]
                nan_infs_subseq = nan_infs[start:end]
                stat = func(temp[start:end][~nan_infs_subseq])
                temp[start:end][nan_infs_subseq] = stat
                # Update nan_infs after array 'temp' is changed
                nan_infs = core.nan_inf_indices(temp)

    # Use subsequent data for imputation / fills in data in a backward direction
    elif direction in directions[3:]:
        for index in range(len(temp) - window + 1, 0, -1):
            start = index
            end = index + window
            has_missing = np.any(nan_infs[index:index + window])

            if has_missing:
                subseq = temp[start:end]
                nan_infs_subseq = nan_infs[start:end]
                stat = func(temp[start:end][~nan_infs_subseq])
                temp[start:end][nan_infs_subseq] = stat
                # Update nan_infs after array 'temp' is changed
                nan_infs = core.nan_inf_indices(temp)

    return temp


def preprocess(ts, window, impute_method='mean', impute_direction='forward', add_noise=True):
    """
    Preprocesses the given time series by adding noise and imputing missing data.

    Parameters
    ----------
    ts : array_like
        The time series to be preprocessed.
    window : int
        The window size to compute the mean/median/minimum value/maximum
        value.
    method : string, Default = 'mean'
        A string indicating the data imputation method, which should be
        'mean', 'median', 'min' or 'max'.
    direction : string, Default = 'forward'
        A string indicating the data imputation direction, which should be
        'forward', 'fwd', 'f', 'backward', 'bwd', 'b'. If the direction is
        forward, we use previous data for imputation; if the direction is
        backward, we use subsequent data for imputation.
    add_noise : bool, Default = True
        A boolean value indicating whether noise needs to be added into the time series.

    Returns
    -------
    temp : array_like
        The time series after being preprocessed.

    """
    if not core.is_array_like(ts):
        raise ValueError('ts is not array like!')

    temp = np.copy(core.to_np_array(ts))

    # impute missing
    temp = impute_missing(temp, window, method=impute_method, direction=impute_direction)

    # handle constant values
    if add_noise:
        for index in range(len(temp) - window + 1):
            start = index
            end = index + window
            subseq = temp[start:end]

            if is_subsequence_constant(subseq):
                temp[start:end] = add_noise_to_series(subseq)

    return temp