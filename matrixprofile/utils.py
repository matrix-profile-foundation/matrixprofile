# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile import core


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
    window_index = np.argwhere(windows == window)

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
