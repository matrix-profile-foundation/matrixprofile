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
    av : str
        The type of annotation vector to apply.
    custom_av : array_like, Optional
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

    if av == "default":
        profile['av'] = make_default_av(profile['data']['ts'], profile['w'])
        profile['av_cmp'] = av
    elif av == "complexity":
        profile['av'] = make_complexity_av(profile['data']['ts'], profile['w'])
        profile['av_cmp'] = av
    elif av == "mean_std":
        profile['av'] = make_meanstd_av(profile['data']['ts'], profile['w'])
        profile['av_cmp'] = av
    elif av == "clipping":
        profile['av'] = make_clipping_av(profile['data']['ts'], profile['w'])
        profile['av_cmp'] = av
    elif av == "custom":
        try:
            profile['av'] = core.to_np_array(custom_av)
        except ValueError:
            raise ValueError('apply_av expects custom_av to be array-like')

        profile['av'] = custom_av
        profile['av_cmp'] = av
    else:
        raise ValueError("av parameter is invalid")

    if len(profile['av']) != len(profile['mp']):
        raise ValueError("Lengths of annotation vector and mp are different")

    if (profile['av'] < 0.0).any() or (profile['av'] > 1.0).any():
        raise ValueError("Annotation vector values must be between 0 and 1")

    max_val = np.max(profile['mp'])
    profile['mp'] += (np.ones(len(profile['av'])) - profile['av']) * max_val

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
        raise ValueError('make_default_av expects ts to be array-like')

    if not core.is_one_dimensional(ts):
        raise ValueError('make_default_av expects ts to be one-dimensional')

    if not isinstance(window, int):
        raise ValueError('make_default_av expects window to be an integer')

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
        raise ValueError('make_default_av expects ts to be array-like')

    if not core.is_one_dimensional(ts):
        raise ValueError('make_default_av expects ts to be one-dimensional')

    if not isinstance(window, int):
        raise ValueError('make_default_av expects window to be an integer')

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
        raise ValueError('make_default_av expects ts to be array-like')

    if not core.is_one_dimensional(ts):
        raise ValueError('make_default_av expects ts to be one-dimensional')

    if not isinstance(window, int):
        raise ValueError('make_default_av expects window to be an integer')

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


def empty_mp():
    """
    Utility function that provides an empty MatrixProfile data structure.

    Returns
    -------
    dict : profile
        An empty MatrixProfile data structure.

    """
    return {
        'mp': None,
        'pi': None,
        'rmp': None,
        'rpi': None,
        'lmp': None,
        'lpi': None,
        'metric': None,
        'w': None,
        'ez': None,
        'join': None,
        'data': {
            'ts': None,
            'query': None
        },
        'class': 'MatrixProfile',
        'algorithm': None
    }


def pick_mp(profile, window):
    """
    Utility function that extracts a MatrixProfile from a Pan-MatrixProfile
    placing it into the MatrixProfile data structure.

    Parameters
    ----------
    profile : dict
        A Pan-MatrixProfile data structure.
    window : int
        The specific window size used to compute the desired MatrixProfile.

    Returns
    -------
    dict : profile
        A MatrixProfile data structure.

    Raises
    ------
    ValueError
        If profile is not a Pan-MatrixProfile data structure.
        If window is not an integer.
        If desired MatrixProfile is not found based on window.

    """

    if not core.is_pmp_obj(profile):
        raise ValueError('pluck_mp expects profile as a PMP data structure!')

    if not isinstance(window, int):
        raise ValueError('pluck_mp expects window to be an int!')

    mp_profile = empty_mp()

    # find the window index
    windows = profile.get('windows')
    window_index = np.argwindowhere(windows == window)

    if len(window_index) < 1:
        raise RuntimeError('Unable to find window {} in the provided PMP!'.format(window))

    window_index = window_index.flatten()[0]

    window = windows[window_index]
    mp = profile['pmp'][window_index]
    n = len(mp)
    mp_profile['mp'] = mp[0:n-window+1]
    mp_profile['pi'] = profile['pmpi'][window_index][0:n-window+1]
    mp_profile['metric'] = profile['metric']
    mp_profile['data']['ts'] = profile['data']['ts']
    mp_profile['join'] = False
    mp_profile['w'] = int(window)
    mp_profile['ez'] = int(np.floor(windows[window_index] / 4))
    mp_profile['algorithm'] = 'mpx'

    return mp_profile
