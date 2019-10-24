# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate


from collections.abc import Iterable
import math

import warnings

from matplotlib import pyplot as plt
import numpy as np

from matrixprofile import core
from matrixprofile.algorithms.mpx import mpx
from matrixprofile.algorithms.mass2 import mass2


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
    
    return (pmp, pmpi, np.array(windows))


def maximum_subsequence(ts, threshold, n_jobs=-1):
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
    n_jobs : int, default all
        The number of cpu cores to use.
    
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
        mp = mpx(ts, window_size, cross_correlation=True, n_jobs=n_jobs)['mp']
        mask = ~np.isinf(mp)
        correlation_max = np.max(mp[mask])

        if correlation_max < threshold:
            break

        window_size = window_size * 2
    
    return window_size


def top_k_discords(pmp, windows, exclusion_zone=None, k=3):
    """
    Computes the top K discords for the given Pan-MatrixProfile. The return
    values is a list of row by col indices.

    Note
    ----
    This algorithm is written to work with Euclidean distance. If you submit
    a PMP of Pearson metrics, then it is first converted to Euclidean.

    Parameters
    ----------
    pmp : array_like
        The Pan-MatrixProfile.
    windows : array_like
        The windows used to compute the respective matrix profiles. They should
        match row wise with the PMP.
    exclusion_zone : int, Default window / 2
        The zone to exclude around the found discords to reduce trivial
        findings. By default we use the row-wise window / 2.
    k : int
        Maximum number of discords to find.

    Returns
    -------
    A 2D array of indices. The first column corresponds to the row index and 
    the second column corresponds to the column index of the submitted PMP.
    """

    # this function requires euclidean distance
    # we assume that it is pearson when min max is between 0 and 1
    mask = ~core.nan_inf_indices(pmp)
    min_val = pmp[mask].min()
    max_val = pmp[mask].max()
    
    tmp = None
    if min_val >= 0 and max_val <= 1:
        msg = """
        min and max values appear to be Pearson metric.
        This function requires Euclidean distance. Converting...
        """
        warnings.warn(msg)
        tmp = core.pearson_to_euclidean(pmp, windows)
    else:
        tmp = np.copy(pmp).astype('d')        
    
    # replace nan and infs with -infinity
    # for whatever reason numpy argmax finds infinity as max so
    # this is a way to get around it by converting to -infinity
    tmp[core.nan_inf_indices(tmp)] = -np.inf
            
    # iterate finding the max value k times or until negative
    # infinity is obtained
    found = []
    
    for _ in range(k):
        max_idx = np.unravel_index(np.argmax(tmp), tmp.shape)
        window = windows[max_idx[0]]
        
        if tmp[max_idx] == -np.inf:
            break
        
        found.append(max_idx)
        
        # apply exclusion zone
        # the exclusion zone is based on 1/2 of the window size
        # used to compute that specific matrix profile
        n = tmp[max_idx[0]].shape[0]
        if exclusion_zone is None:
            exclusion_zone = int(np.floor(window / 2))

        ez_start = np.max([0, max_idx[1] - exclusion_zone])
        ez_stop = np.min([n, max_idx[1] + exclusion_zone])
        tmp[max_idx[0]][ez_start:ez_stop] = -np.inf
        
    return np.array(found)


def top_k_motifs(obj, exclusion_zone=None, k=3, max_neighbors=10, radius=3):
    """
    Find the top K number of motifs (patterns) given a pan matrix profile. By
    default the algorithm will find up to 3 motifs (k) and up to 10 of their
    neighbors with a radius of 3 * min_dist.

    Parameters
    ----------
    obj : dict
        The output from one of the pan matrix profile algorithms.
    exclusion_zone : int, Default to algorithm ez
        Desired number of values to exclude on both sides of the motif. This
        avoids trivial matches. It defaults to half of the computed window
        size. Setting the exclusion zone to 0 makes it not apply.
    k : int, Default = 3
        Desired number of motifs to find.
    neighbor_count : int, Default = 10
        The maximum number of neighbors to include for a given motif.
    radius : int, Default = 3
        The radius is used to associate a neighbor by checking if the
        neighbor's distance is less than or equal to dist * radius

    Returns
    -------
    The original input obj with the addition of the "motifs" key. The motifs
    key consists of the following structure.

    A list of dicts containing motif indices and their corresponding neighbor
    indices. Note that each index is a (row, col) index corresponding to the
    pan matrix profile.

    [
        {
            'motifs': [first_index, second_index],
            'neighbors': [index, index, index ...max_neighbors]
        }
    ]
    """
    data = obj.get('data', None)
    ts = data.get('ts', None)
    data_len = len(ts)
    
    pmp = obj.get('pmp', None)
    profile_len = pmp.shape[1]   
    pmpi = obj.get('pmpi', None)
    windows = obj.get('windows', None)
    
    # make sure we are working with Euclidean distances
    tmp = None
    if core.is_pearson_array(pmp):
        msg = """
        min and max values appear to be Pearson metric.
        This function requires Euclidean distance. Converting...
        """
        warnings.warn(msg)
        tmp = core.pearson_to_euclidean(pmp, windows)
    else:
        tmp = np.copy(pmp).astype('d')
    
    # replace nan and infs with infinity
    tmp[core.nan_inf_indices(tmp)] = np.inf
    
    motifs = []
    for _ in range(k):
        min_idx = np.unravel_index(np.argmin(tmp), tmp.shape)
        min_dist = tmp[min_idx]
        
        # nothing else to find...
        if core.is_nan_inf(min_dist):
            break
        
        # create the motif pair
        min_row_idx = min_idx[0]
        min_col_idx = min_idx[1]
        
        # motif pairs are respective to the column of the matching row
        first_idx = np.min([min_col_idx, pmpi[min_row_idx][min_col_idx]])
        second_idx = np.max([min_col_idx, pmpi[min_row_idx][min_col_idx]])
        
        # compute distance profile for first appearance
        window_size = windows[min_row_idx]
        query = ts[first_idx:first_idx + window_size]
        distance_profile = mass2(ts, query)
        
        # extend the distance profile to be as long as the original
        infs = np.full(profile_len - len(distance_profile), np.inf)
        distance_profile = np.append(distance_profile, infs)
        
        # exclude already picked motifs and neighbors
        mask = core.nan_inf_indices(pmp[min_row_idx])        
        distance_profile[mask] = np.inf
        
        # determine the exclusion zone if not set
        if not exclusion_zone:
            exclusion_zone = int(np.floor(window_size / 2))
        
        # apply exclusion zone for motif pair
        for j in (first_idx, second_idx):
            distance_profile = core.apply_exclusion_zone(
                exclusion_zone,
                False,
                window_size,
                data_len,
                j,
                distance_profile
            )
            tmp2 = core.apply_exclusion_zone(
                exclusion_zone,
                False,
                window_size,
                data_len,
                j,
                tmp[min_row_idx]
            )
            tmp[min_row_idx] = tmp2
        
        # find up to max_neighbors
        neighbors = []
        for j in range(max_neighbors):
            neighbor_idx = np.argmin(distance_profile)
            neighbor_dist = np.real(distance_profile[neighbor_idx])
            not_in_radius = not ((radius * min_dist) >= neighbor_dist)

            # no more neighbors exist based on radius
            if core.is_nan_inf(neighbor_dist) or not_in_radius:
                break;

            # add neighbor and apply exclusion zone
            neighbors.append((min_row_idx, neighbor_idx))
            distance_profile = core.apply_exclusion_zone(
                exclusion_zone,
                False,
                window_size,
                data_len,
                neighbor_idx,
                distance_profile
            )
            tmp2 = core.apply_exclusion_zone(
                exclusion_zone,
                False,
                window_size,
                data_len,
                neighbor_idx,
                tmp[min_row_idx]
            )
            tmp[min_row_idx] = tmp2
        
        # add the motifs and neighbors
        # note that they are (row, col) indices
        motifs.append({
            'motifs': [(min_row_idx, first_idx), (min_row_idx, second_idx)],
            'neighbors': neighbors
        })
    
    return motifs
