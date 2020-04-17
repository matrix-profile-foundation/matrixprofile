# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile import core


def apply_av(profile, av="default", custom_av=None):
    """
    Utility function that returns a MatrixProfile data structure
    with a calculated annotation vector that has been applied
    to correct the matrix profile.

    Parameters
    ----------
    profile : dict
        A MatrixProfile structure.
    av : str, Default = "default"
        The type of annotation vector to apply.
    custom_av : array_like, Default = None
        Custom annotation vector (will only be applied if av is "custom").

    Returns
    -------
    dict : profile
        A MatrixProfile data structure with a calculated annotation vector
        and a corrected matrix profile.

    Raises
    ------
    ValueError
        If profile is not a MatrixProfile data structure.
        If custom_av parameter is not array-like when using a custom av.
        If av paramter is invalid.
        If lengths of annotation vector and matrix profile are different.
        If values in annotation vector are outside [0.0, 1.0].

    """
    if not core.is_mp_obj(profile):
        raise ValueError('apply_av expects profile as an MP data structure')

    temp_av = None
    av_type = None

    if av == "default":
        temp_av = make_default_av(profile['data']['ts'], profile['w'])
        av_type = av
    elif av == "complexity":
        temp_av = make_complexity_av(profile['data']['ts'], profile['w'])
        av_type = av
    elif av == "meanstd":
        temp_av = make_meanstd_av(profile['data']['ts'], profile['w'])
        av_type = av
    elif av == "clipping":
        temp_av = make_clipping_av(profile['data']['ts'], profile['w'])
        av_type = av
    elif av == "custom":
        try:
            temp_av = core.to_np_array(custom_av)
        except ValueError:
            raise ValueError('apply_av expects custom_av to be array-like')

        av_type = av
    else:
        raise ValueError("av parameter is invalid")

    if len(temp_av) != len(profile['mp']):
        raise ValueError("Lengths of annotation vector and mp are different")

    if (temp_av < 0.0).any() or (temp_av > 1.0).any():
        raise ValueError("Annotation vector values must be between 0 and 1")

    max_val = np.max(profile['mp'])
    temp_cmp = profile['mp'] + (np.ones(len(temp_av)) - temp_av) * max_val

    profile['cmp'] = temp_cmp
    profile['av'] = temp_av
    profile['av_type'] = av_type

    return profile


def make_default_av(ts, window):
    """
    Utility function that returns an annotation vector filled with 1s
    (should not change the matrix profile).

    Parameters
    ----------
    ts : array_like
        The time series.
    window : int
        The specific window size used to compute the MatrixProfile.

    Returns
    -------
    np.array : av
        An annotation vector.

    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If ts is not one-dimensional.
        If window is not an integer.

    """
    try:
        ts = core.to_np_array(ts)
    except ValueError:
        raise ValueError('make_default_av expects ts to be array-like')

    if not core.is_one_dimensional(ts):
        raise ValueError('make_default_av expects ts to be one-dimensional')

    if not isinstance(window, int):
        raise ValueError('make_default_av expects window to be an integer')

    av = np.ones(len(ts) - window + 1)

    return av


def make_complexity_av(ts, window):
    """
    Utility function that returns an annotation vector where values are based
    on the complexity estimation of the signal.

    Parameters
    ----------
    ts : array_like
        The time series.
    window : int
        The specific window size used to compute the MatrixProfile.

    Returns
    -------
    np.array : av
        An annotation vector.

    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If ts is not one-dimensional.
        If window is not an integer.

    """
    try:
        ts = core.to_np_array(ts)
    except ValueError:
        raise ValueError('make_complexity_av expects ts to be array-like')

    if not core.is_one_dimensional(ts):
        raise ValueError('make_complexity_av expects ts to be one-dimensional')

    if not isinstance(window, int):
        raise ValueError('make_complexity_av expects window to be an integer')

    av = np.zeros(len(ts) - window + 1)

    for i in range(len(av)):
        ce = np.sum(np.diff(ts[i: i + window]) ** 2)
        av[i] = np.sqrt(ce)

    max_val, min_val = np.max(av), np.min(av)
    if max_val == 0:
        av = np.zeros(len(av))
    else:
        av = (av - min_val) / max_val

    return av


def make_meanstd_av(ts, window):
    """
    Utility function that returns an annotation vector where values are set to
    1 if the standard deviation is less than the mean of standard deviation.
    Otherwise, the values are set to 0.

    Parameters
    ----------
    ts : array_like
        The time series.
    window : int
        The specific window size used to compute the MatrixProfile.

    Returns
    -------
    np.array : av
        An annotation vector.

    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If ts is not one-dimensional.
        If window is not an integer.

    """
    try:
        ts = core.to_np_array(ts)
    except ValueError:
        raise ValueError('make_meanstd_av expects ts to be array-like')

    if not core.is_one_dimensional(ts):
        raise ValueError('make_meanstd_av expects ts to be one-dimensional')

    if not isinstance(window, int):
        raise ValueError('make_meanstd_av expects window to be an integer')

    av = np.zeros(len(ts) - window + 1)

    std = core.moving_std(ts, window)
    mu = np.mean(std)
    for i in range(len(av)):
        if std[i] < mu:
            av[i] = 1

    return av


def make_clipping_av(ts, window):
    """
    Utility function that returns an annotation vector such that
    subsequences that have more clipping have less importance.

    Parameters
    ----------
    ts : array_like
        The time series.
    window : int
        The specific window size used to compute the MatrixProfile.

    Returns
    -------
    np.array : av
        An annotation vector.

    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If ts is not one-dimensional.
        If window is not an integer.

    """
    try:
        ts = core.to_np_array(ts)
    except ValueError:
        raise ValueError('make_clipping_av expects ts to be array-like')

    if not core.is_one_dimensional(ts):
        raise ValueError('make_clipping_av expects ts to be one-dimensional')

    if not isinstance(window, int):
        raise ValueError('make_clipping_av expects window to be an integer')

    av = np.zeros(len(ts) - window + 1)

    max_val, min_val = np.max(ts), np.min(ts)
    for i in range(len(av)):
        num_clip = 0.0
        for j in range(window):
            if ts[i + j] == max_val or ts[i + j] == min_val:
                num_clip += 1
        av[i] = num_clip

    min_val = np.min(av)
    av -= min_val

    max_val = np.max(av)
    if max_val == 0:
        av = np.zeros(len(av))
    else:
        av = 1 - av / max_val

    return av
