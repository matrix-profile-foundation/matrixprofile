# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile import core


def statistics(ts, window_size):
    """
    Compute global and moving statistics for the provided 1D time
    series. The statistics computed include the min, max, mean, std. and median
    over the window specified and globally.

    Parameters
    ----------
    ts : array_like
        The time series.
    window_size: int
        The size of the window to compute moving statistics over.

    Returns
    -------
    dict : statistics
        The global and rolling window statistics.
        
        >>> {
        >>>     ts: the original time series,
        >>>     min: the global minimum,
        >>>     max: the global maximum,
        >>>     mean: the global mean,
        >>>     std: the global standard deviation,
        >>>     median: the global median,
        >>>     moving_min: the moving minimum,
        >>>     moving_max: the moving maximum,
        >>>     moving_mean: the moving mean,
        >>>     moving_std: the moving standard deviation,
        >>>     moving_median: the moving median,
        >>>     window_size: the window size provided,
        >>>     class: Statistics
        >>> }

    Raises
    ------
    ValueError
        If window_size is not an int.
        If window_size > len(ts)
        If ts is not a list or np.array.
        If ts is not 1D.

    """
    if not core.is_array_like(ts):
        raise ValueError('ts must be array like')	

    if not core.is_one_dimensional(ts):
        raise ValueError('The time series must be 1D')

    if not isinstance(window_size, int):
        raise ValueError('Expecting int for window_size')

    if window_size > len(ts):
        raise ValueError('Window size cannot be greater than len(ts)')

    if window_size < 3:
        raise ValueError('Window size cannot be less than 3')

    moving_mu, moving_sigma = core.moving_avg_std(ts, window_size)
    rolling_ts = core.rolling_window(ts, window_size)

    return {
        'ts': ts,
        'min': np.min(ts),
        'max': np.max(ts),
        'mean': np.mean(ts),
        'std': np.std(ts),
        'median': np.median(ts),
        'moving_min': np.min(rolling_ts, axis=1),
        'moving_max': np.max(rolling_ts, axis=1),
        'moving_mean': moving_mu,
        'moving_std': moving_sigma,
        'moving_median': np.median(rolling_ts, axis=1),
        'window_size': window_size,
        'class': 'Statistics'
    }
