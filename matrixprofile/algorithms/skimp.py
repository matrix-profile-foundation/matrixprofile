# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate


from collections.abc import Iterable
import math

from matplotlib import pyplot as plt
import numpy as np

from matrixprofile import core
from matrixprofile.algorithms.mpx import mpx


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


def plot_pmp(pmp, cmap=None):
    """
    Plots the PMP. Right now it assumes you are using a Jupyter or Ipython
    notebook.

    Parameters
    ----------
    pmp : array_like
        The Pan Matrix Profile to plot.
    cmap: str
        A valid Matplotlib color map.
    """
    plt.figure(figsize = (10,10))
    depth = 256
    test = np.ceil(pmp * depth) / depth
    test[test > 1] = 1
    plt.imshow(test, cmap=cmap, interpolation=None, aspect='auto')
    plt.gca().invert_yaxis()
    plt.title('PMP')
    plt.xlabel('Profile Index')
    plt.ylabel('Window Size')
    plt.show()


def skimp(ts, windows=None, show_progress=False, cross_correlation=False,
          sample_pct=0.1, n_jobs=-1):
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
    sample_pct : float, default = 0.1 (10%)
        Number of window sizes to compute MPs for. Decimal percent between
        0 and 1.
    n_jobs : int, default all
        The number of cpu cores to use.

    Returns
    -------
    (array_like, array_like, array_like) :
        The (PMP, PMPI, Windows).

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
    if not isinstance(windows, (Iterable, np.ndarray)):
        start = 8
        end = int(math.floor(len(ts) / 2))
        windows = range(start, end + 1)
    else:
        sample_pct = 1
        
    if not isinstance(show_progress, bool):
        raise ValueError('show_progress must be a boolean!')
    
    if not isinstance(cross_correlation, bool):
        raise ValueError('cross_correlation must be a boolean!')
    
    if not isinstance(sample_pct, (int, float)) or sample_pct > 1 or sample_pct < 0:
        raise ValueError('sample_pct must be a decimal between 0 and 1')
    
    # create a breath first search index list of our window sizes
    split_index = binary_split(len(windows))
    pmp = np.full((len(split_index), n), np.inf)
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
        profile = mpx(ts, window_size, cross_correlation=cross_correlation,
            n_jobs=n_jobs)
        mp = profile.get('mp')
        pmp[split_index[i], 0:len(mp)] = mp
        
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
    
    return (pmp, idx, np.array(windows))


def maximum_subsequence(ts, threshold):
    """
    Finds the maximum subsequence length based on the threshold provided. Note
    that this threshold is domain specific requiring some knowledge about the
    underyling time series in question.

    The subsequence length starts at 8 and iteratively doubles until the
    maximum correlation coefficient is no longer met.

    Parameters
    ----------
    ts : array_like
        The time series to analyze.
    threshold : float
        The correlation coefficient used as the threshold. It should be between
        0 and 1.
    
    Returns
    -------
    int :
        The maximum subsequence length based on the threshold provided.
    """
    ts = core.to_np_array(ts)
    correlation_max = np.inf
    window_size = 8
    max_window = int(np.floor(len(ts) / 2))
    cross_correlation = 1

    while window_size <= max_window:
        pmp, pmpi = mpx(ts, window_size, cross_correlation)
        mask = ~np.isinf(pmp)
        correlation_max = np.max(pmp[mask])

        if correlation_max < threshold:
            break

        window_size = window_size * 2
    
    return window_size