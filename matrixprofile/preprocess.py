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


def validate_preprocess_kwargs(preprocessing_kwargs):
    """
    Tests the arguments of preprocess function and raises errors for invalid arguments.

    Parameters
    ----------
    preprocessing_kwargs : dict-like or None or False
        A dictionary object to store keyword arguments for the preprocess function.
        It can also be None/False/{}/"".

    Returns
    -------
    valid_kwargs : dict-like or None
        The valid keyword arguments for the preprocess function.
        Returns None if the input preprocessing_kwargs is None/False/{}/"".

    Raises
    ------
    ValueError
        If preprocessing_kwargs is not dict-like or None.
        If gets invalid key(s) for preprocessing_kwargs.
        If gets invalid value(s) for preprocessing_kwargs['window'], preprocessing_kwargs['impute_method']
        preprocessing_kwargs['impute_direction'] and preprocessing_kwargs['add_noise'].

    """
    if preprocessing_kwargs:

        valid_preprocessing_kwargs_keys = {'window', 'impute_method', 'impute_direction', 'add_noise'}

        if not isinstance(preprocessing_kwargs,dict):
            raise ValueError("The parameter 'preprocessing_kwargs' is not dict like!")

        elif set(preprocessing_kwargs.keys()).issubset(valid_preprocessing_kwargs_keys):
            window = 4
            impute_method = 'mean'
            impute_direction = 'forward'
            add_noise = True
            methods = ['mean', 'median', 'min', 'max']
            directions = ['forward', 'fwd', 'f', 'backward', 'bwd', 'b']

            if 'window' in preprocessing_kwargs.keys():
                if not isinstance(preprocessing_kwargs['window'],int):
                    raise ValueError("The value for preprocessing_kwargs['window'] is not an integer!")
                window = preprocessing_kwargs['window']


            if 'impute_method' in preprocessing_kwargs.keys():
                if preprocessing_kwargs['impute_method'] not in methods:
                    raise ValueError('invalid imputation method! valid include options: ' + ', '.join(methods))
                impute_method = preprocessing_kwargs['impute_method']

            if 'impute_direction' in preprocessing_kwargs.keys():
                if preprocessing_kwargs['impute_direction'] not in directions:
                    raise ValueError('invalid imputation direction! valid include options: ' + ', '.join(directions))
                impute_direction = preprocessing_kwargs['impute_direction']

            if 'add_noise' in preprocessing_kwargs.keys():
                if not isinstance(preprocessing_kwargs['add_noise'],bool):
                    raise ValueError("The value for preprocessing_kwargs['add_noise'] is not a boolean value!")
                add_noise = preprocessing_kwargs['add_noise']

            valid_kwargs =  { 'window': window,
                              'impute_method': impute_method,
                              'impute_direction': impute_direction,
                              'add_noise': add_noise }
        else:
            raise ValueError('invalid key(s) for preprocessing_kwargs! '
                             'valid key(s) should include '+ str(valid_preprocessing_kwargs_keys))
    else:
        valid_kwargs = None

    return valid_kwargs


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

    if not isinstance(window, int):
        raise ValueError("window is not an integer!")

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

    index_order = None

    if direction.startswith('f'):
        # Use previous data for imputation / fills in data in a forward direction
        index_order = range(len(temp) - window + 1)
    elif direction.startswith('b'):
        # Use subsequent data for imputation / fills in data in a backward direction
        index_order = range(len(temp) - window + 1, 0, -1)

    for index in index_order:
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