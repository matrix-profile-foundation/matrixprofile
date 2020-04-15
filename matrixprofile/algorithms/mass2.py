# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile import core

def mass2(ts, query, extras=False, threshold=1e-10):
    """
    Compute the distance profile for the given query over the given time 
    series.

    Parameters
    ----------
    ts : array_like
        The time series to search.
    query : array_like
        The query.
    extras : boolean, default False
        Optionally return additional data used to compute the matrix profile.

    Returns
    -------
    np.array, dict : distance_profile
        An array of distances np.array() or dict with extras.

        With extras:
        
        >>> {
        >>>     'distance_profile': The distance profile,
        >>>     'product': The FFT product between ts and query,
        >>>     'data_mean': The moving average of the ts over len(query),
        >>>     'query_mean': The mean of the query,
        >>>     'data_std': The moving std. of the ts over len(query),
        >>>     'query_std': The std. of the query
        >>> }

    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If query is not a list or np.array.
        If ts or query is not one dimensional.

    """
    ts, query = core.precheck_series_and_query_1d(ts, query)

    n = len(ts)
    m = len(query)
    x = ts
    y = query

    meany = np.mean(y)
    sigmay = np.std(y)
    
    meanx, sigmax = core.moving_avg_std(x, m)
    meanx = np.append(np.ones([1, len(x) - len(meanx)]), meanx)    
    sigmax = np.append(np.zeros([1, len(x) - len(sigmax)]), sigmax)

    
    y = np.append(np.flip(y), np.zeros([1, n - m]))
    
    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    Y.resize(X.shape)
    Z = X * Y
    z = np.fft.ifft(Z)
    
    # do not allow divide by zero
    tmp = (sigmax[m - 1:n] * sigmay)
    tmp[tmp == 0] = 1e-12

    dist = 2 * (m - (z[m - 1:n] - m * meanx[m - 1:n] * meany) / tmp)

    # fix to handle constant values
    dist[sigmax[m - 1:n] < threshold] = m
    dist[(sigmax[m - 1:n] < threshold) & (sigmay < threshold)] = 0
    dist = np.sqrt(dist)
    
    if extras:
        return {
            'distance_profile': dist,
            'product': z,
            'data_mean': meanx,
            'query_mean': meany,
            'data_std': sigmax,
            'query_std': sigmay
        }

    return dist
