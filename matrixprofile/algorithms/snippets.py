#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile import core
from matrixprofile.algorithms.mpdist import mpdist_vector


def snippets(ts, snippet_size, num_snippets=2, window_size=None):
    """
    The snippets algorithm is used to summarize your time series by
    identifying N number of representative subsequences. If you want to
    identify typical patterns in your time series, then this is the algorithm
    to use.

    Parameters
    ----------
    ts : array_like
        The time series.
    snippet_size : int
        The size of snippet desired.
    num_snippets : int, Default 2
        The number of snippets you would like to find.
    window_size : int, Default (snippet_size / 2)
        The window size.

    Returns
    -------
    list : snippets
        A list of snippets as dictionary objects with the following structure.

        >>> {
        >>> 	index: the index of the snippet,
        >>> 	snippet: the snippet values,
        >>>     neighbors: the starting indices of all subsequences similar to the current snippet
        >>>     fraction: fraction of the snippet
        >>> }

    """
    ts = core.to_np_array(ts).astype('d')
    time_series_len = len(ts)
    n = len(ts)

    if not isinstance(snippet_size, int) or snippet_size < 4:
        raise ValueError('snippet_size must be an integer >= 4')

    if n < (2 * snippet_size):
        raise ValueError('Time series is too short relative to snippet length')

    if not window_size:
        window_size = int(np.floor(snippet_size / 2))

    if window_size >= snippet_size:
        raise ValueError('window_size must be smaller than snippet_size')

    # pad end of time series with zeros
    num_zeros = int(snippet_size * np.ceil(n / snippet_size) - n)
    ts = np.append(ts, np.zeros(num_zeros))

    # compute all profiles
    indices = np.arange(0, len(ts) - snippet_size, snippet_size)
    distances = []

    for j, i in enumerate(indices):
        distance = mpdist_vector(ts, ts[i:(i + snippet_size - 1)], int(window_size))
        distances.append(distance)

    distances = np.array(distances)

    # find N snippets
    snippets = []
    minis = np.inf
    total_min = None
    for n in range(num_snippets):
        minims = np.inf

        for i in range(len(indices)):
            s = np.sum(np.minimum(distances[i, :], minis))

            if minims > s:
                minims = s
                index = i

        minis = np.minimum(distances[index, :], minis)
        actual_index = indices[index]
        snippet = ts[actual_index:actual_index + snippet_size]
        snippet_distance = distances[index]
        snippets.append({
            'index': actual_index,
            'snippet': snippet,
            'distance': snippet_distance
        })

        if isinstance(total_min, type(None)):
            total_min = snippet_distance
        else:
            total_min = np.minimum(total_min, snippet_distance)

    # compute the fraction of each snippet
    for snippet in snippets:
        mask = (snippet['distance'] <= total_min)
        # create a key "neighbors" for the snippet dict,
        # and store all the time series indices for the data represented by a snippet (arr[mask])
        arr = np.arange(len(mask))
        # max_index indicates the length of a profile, which is (n-m) in the Snippets paper)
        max_index = time_series_len - snippet_size
        # since 'ts' is padded with 0 before calculate the MPdist profile
        # all parts of the profile that are out of range [0, n-m] cannot be used as neighboring snippet indices
        snippet['neighbors'] = list(filter(lambda x : x <= max_index, arr[mask]))
        # Add the last m time series indices into the neighboring snippet indices
        if max_index in snippet['neighbors']:
            last_m_indices = list(range(max_index+1, time_series_len))
            snippet['neighbors'].extend(last_m_indices)
        snippet['fraction'] = mask.sum() / (len(ts) - snippet_size)
        total_min = total_min - mask
        del snippet['distance']

    return snippets
