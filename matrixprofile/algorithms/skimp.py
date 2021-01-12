# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate


# handle Python 2/3 Iterable import
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import math

import warnings

from matplotlib import pyplot as plt
import numpy as np

from matrixprofile import core
from matrixprofile.algorithms.mpx import mpx
from matrixprofile.exceptions import NoSolutionPossible


def split(lower_bound, upper_bound, middle):
    """
    Helper function to split the indices for BFS.
    """
    if lower_bound == middle:
        L = None
        R = [middle + 1, upper_bound]
    elif upper_bound == middle:
        L = [lower_bound, middle - 1]
        R = None
    else:
        L = [lower_bound, middle - 1]
        R = [middle + 1, upper_bound]
    
    return (L, R)


def binary_split(n):
    """
    Create a breadth first search for indices 0..n.

    Parameters
    ----------
    n : int
        The length of indices.

    Returns
    -------
    array_like :
        The indices to iterate to perform BFS.

    """
    # having length of 1 causes infinite loop and it just doesn't really
    # make much sense.
    # just return it
    if n < 2:
        return [0,]

    index = []
    intervals = []
    
    # always begin by exploring the first integer
    index.append(0)
    
    # after exploring first integer, split interval 2:n
    intervals.append([1, n - 1])
    
    while len(intervals) > 0:
        interval = intervals.pop(0)
        lower_bound = interval[0]
        upper_bound = interval[1]
        middle = int(math.floor((lower_bound + upper_bound) / 2))
        index.append(middle)
        
        if lower_bound == upper_bound:
            continue
        else:
            L, R = split(lower_bound, upper_bound, middle)
            
            if L is not None:
                intervals.append(L)
            
            if R is not None:
                intervals.append(R)
    
    return index


def skimp(ts, windows=None, show_progress=False, cross_correlation=False,
          pmp_obj=None, sample_pct=0.1, n_jobs=1):
    """
    Computes the Pan Matrix Profile (PMP) for the given time series. When the
    time series is only passed, windows start from 8 and increase by increments
    of 2 up to length(ts) / 2. Also, the PMP is only computed using 10% of the
    windows unless sample_pct is set to a different value.

    Note
    ----
    When windows is explicitly provided, sample_pct no longer takes affect. The
    MP for all windows provided will be computed.

    Parameters
    ----------
    ts : array_like
        The time series.
    show_progress: bool, default = False
        Show the progress in percent complete in increments of 5% by printing
        it out to the console.
    cross_correlation : bool, default = False
        Return the MP values as Pearson Correlation instead of Euclidean
        distance.
    pmp_obj : dict, default = None
        Repurpose already computed window sizes with this provided PMP. It
        should be the output of a PMP algorithm such as skimp or maximum
        subsequence.
    sample_pct : float, default = 0.1 (10%)
        Number of window sizes to compute MPs for. Decimal percent between
        0 and 1.
    n_jobs : int, Default = 1
        Number of cpu cores to use.

    Returns
    -------
    dict : profile
        A Pan-MatrixProfile data structure.
        
        >>> {
        >>>     'pmp': the pan matrix profile as a 2D array,
        >>>     'pmpi': the pmp indices,
        >>>     'data': {
        >>>         'ts': time series used,
        >>>     },
        >>>     'windows': the windows used to compute the pmp,
        >>>     'sample_pct': the sample percent used,
        >>>     'metric':The distance metric computed for the pmp,
        >>>     'algorithm': the algorithm used,
        >>>     'class': PMP
        >>> }

    Raises
    ------
    ValueError :
        1. ts is not array_like.
        2. windows is not an iterable
        3. show_progress is not a boolean.
        4. cross_correlation is not a boolean.
        5. sample_pct is not between 0 and 1.

    """
    ts = core.to_np_array(ts)
    n = len(ts)
    
    # Argument validation
    if isinstance(windows, type(None)):
        start = 8
        end = int(math.floor(len(ts) / 2))
        windows = range(start, end + 1)
        
    if not isinstance(show_progress, bool):
        raise ValueError('show_progress must be a boolean!')
    
    if not isinstance(cross_correlation, bool):
        raise ValueError('cross_correlation must be a boolean!')
    
    if not isinstance(sample_pct, (int, float)) or sample_pct > 1 or sample_pct < 0:
        raise ValueError('sample_pct must be a decimal between 0 and 1')
    
    # create a breath first search index list of our window sizes
    split_index = binary_split(len(windows))
    pmp = np.full((len(split_index), n), np.inf)
    pmpi = np.full((len(split_index), n), np.nan, dtype='int')
    idx = np.full(len(split_index), -1)
    
    # compute the sample pct index
    last_index = len(split_index)
    if sample_pct < 1:
        last_index = int(np.floor(len(split_index) * sample_pct))
        last_index = np.minimum(len(split_index), last_index)
    
    pct_shown = {}

    # compute all matrix profiles for each window size
    for i in range(last_index):
        window_size = windows[split_index[i]]

        # check if we already computed this MP given a passed in PMP
        if isinstance(pmp_obj, dict):
            cw = pmp_obj.get('windows', None)
            w_idx = np.argwhere(cw == window_size)
            
            # having the window provided, we simply copy over the data instead
            # of recomputing it
            if len(w_idx) == 1:
                w_idx = w_idx[0][0]
                pmp[split_index[i], :] = pmp_obj['pmp'][w_idx, :]
                pmpi[split_index[i], :] = pmp_obj['pmpi'][w_idx, :]

                continue

        profile = mpx(ts, window_size, cross_correlation=cross_correlation,
            n_jobs=n_jobs)
        mp = profile.get('mp')
        pi = profile.get('pi')
        pmp[split_index[i], 0:len(mp)] = mp
        pmpi[split_index[i], 0:len(pi)] = pi
        
        j = split_index[i]
        while j < last_index and idx[j] != j:
            idx[j] = split_index[i]
            j = j + 1
        
        # output the progress
        if show_progress:
            pct_complete = round((i / (last_index - 1)) * 100, 2)
            int_pct = math.floor(pct_complete)
            if int_pct % 5 == 0 and int_pct not in pct_shown:
                print('{}% complete'.format(int_pct))
                pct_shown[int_pct] = 1

    metric = 'euclidean'
    if cross_correlation:
        metric = 'pearson'

    return {
        'pmp': pmp,
        'pmpi': pmpi,
        'data': {
            'ts': ts,
        },
        'windows': np.array(windows),
        'sample_pct': sample_pct,
        'metric': metric,
        'algorithm': 'skimp',
        'class': 'PMP'
    }
    

def maximum_subsequence(ts, threshold=0.95, refine_stepsize=0.05, n_jobs=1,
    include_pmp=False, lower_window=8):
    """
    Finds the maximum subsequence length based on the threshold provided. Note
    that this threshold is domain specific requiring some knowledge about the
    underyling time series in question.

    The subsequence length starts at 8 and iteratively doubles until the
    maximum correlation coefficient is no longer met. When no solution is
    possible given the threshold, a matrixprofile.exceptions.NoSolutionPossible
    exception is raised.

    Parameters
    ----------
    ts : array_like
        The time series to analyze.
    threshold : float, Default 0.95
        The correlation coefficient used as the threshold. It should be between
        0 and 1.
    refine_stepsize : float, Default 0.05
        Used in the refinement step to find a more precise upper window. It
        should be a percentage between 0.01 and 0.99.
    n_jobs : int, Default = 1
        Number of cpu cores to use.
    include_pmp : bool, default False
        Include the PanMatrixProfile for the computed windows.
    lower_window : int, default 8
        Lower bound of subsequence length that can be altered if required.
    
    Returns
    -------
    obj :
        With include_pmp=False (default)
        int : The maximum subsequence length based on the threshold provided.
        
        With include_pmp=True
        dict : A dict containing the upper window, windows and pmp.

        >>> {
        >>>     'upper_window': The upper window,
        >>>     'windows': array_like windows used to compute the pmp,
        >>>     'pmp': the pan matrix profile as a 2D array,
        >>>     'pmpi': the pmp indices,
        >>> }

    """
    windows = np.array([], dtype='int')
    pearson = np.array([], dtype='d')
    pmp = []
    pmpi = []

    ts = core.to_np_array(ts)
    n = len(ts)
    correlation_max = np.inf
    window_size = lower_window
    max_window = int(np.floor(len(ts) / 2))
    
    def resize(mp, pi, n):
        """Helper function to resize mp and pi to be aligned with the
        PMP. Also convert pearson to euclidean."""
        mp = core.pearson_to_euclidean(profile['mp'], window_size)
        infs = np.full(n - mp.shape[0], np.inf)
        nans = np.full(n - mp.shape[0], np.nan)
        mp = np.append(mp, infs)
        pi = np.append(profile['pi'], nans)

        return (mp, pi)

    # first perform a wide search by increasing window by 2 in
    # each iteration
    while window_size <= max_window:
        profile = mpx(ts, window_size, cross_correlation=True)
        mask = ~np.isinf(profile['mp'])
        correlation_max = np.max(profile['mp'][mask])

        windows = np.append(windows, window_size)
        pearson = np.append(pearson, correlation_max)

        if include_pmp:
            mp, pi = resize(profile['mp'], profile['pi'], n)
            pmp.append(mp)
            pmpi.append(pi)

        if correlation_max < threshold:
            break

        window_size = window_size * 2

    # find last window within threshold and throw away
    # computations outside of the threshold
    mask = pearson > threshold
    pearson = pearson[mask]
    windows = windows[mask]

    if len(windows) < 1:
        raise NoSolutionPossible('Given the threshold {:.2f}, no window was ' \
                                 'found. Please try increasing your ' \
                                 'threshold.')

    window_size = windows[-1]

    if include_pmp:
        pmp = np.vstack(pmp)[mask]
        pmpi = np.vstack(pmpi)[mask]

    # refine the upper u by increase by + X% increments
    test_windows = np.arange(refine_stepsize, 1, step=refine_stepsize) + 1
    test_windows = np.append(test_windows, 2)
    test_windows = np.floor(test_windows * window_size).astype('int')

    # keep windows divisible by 2
    mask = test_windows % 2 == 1
    test_windows[mask] = test_windows[mask] + 1

    for window_size in test_windows:
        profile = mpx(ts, window_size, cross_correlation=True)
        mask = ~np.isinf(profile['mp'])
        correlation_max = np.max(profile['mp'][mask])

        windows = np.append(windows, window_size)
        pearson = np.append(pearson, correlation_max)

        if include_pmp:
            mp, pi = resize(profile['mp'], profile['pi'], n)
            pmp = np.append(pmp, [mp,], axis=0)
            pmpi = np.append(pmpi, [pi,], axis=0)

        if correlation_max < threshold:
            break

    if include_pmp:
        return {
            'upper_window': window_size,
            'windows': windows,
            'pmp': pmp,
            'pmpi': pmpi
        }

    return window_size
