# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

import mass_ts as mts

from matrixprofile import core


def stomp(ts, window_size, query=None):
    if window_size < 4:
        error = "m, window size, must be at least 4."
        raise ValueError(error)

    if window_size > len(query) / 2:
        error = "Time series is too short relative to desired window size"
        raise ValueError(error)

    is_join = core.is_similarity_join(ts, query)
    if not is_join:
        query = ts
    
    profile_length = core.get_profile_length(ts, query, window_size)
    query_length = len(query)
    exclusion_zone = 1 / 2
    num_queries = query_size - window_size + 1

    # do not use exclusion zone for join
    if is_join:
        exclusion_zone = 0

    # determine what locations need skipped based on nan and infinite values
    skip_loc = core.find_skip_locations(ts, window_size, profile_length)
    
    # clean up nan and inf in the ts_a and query
    search = (np.isinf(ts) | np.isnan(ts))
    ts[search] = 0

    search = (np.isinf(query) | np.isnan(search))
    query[search] = 0

    first_product = np.zeros(num_queries)

    # forward distance profile
    nn = mts.mass3(ts, query, window_size)

    # reverse
    # required for similarity join
    rnn = None
    if is_join:
        rnn = mts.mass3(query, ts, window_size)

    #TODO: last_product
    # this is when mass computes distances

    matrix_profile = np.full(profile_length, np.nan)
    profile_index = np.full(profile_length, np.inf)

    # assume no joins
    left_matrix_profile = None
    right_matrix_profile = None
    left_profile_index = None
    right_profile_index = None

    # otherwise left and right MP and MPI are same as MP initially
    if is_join:
        left_matrix_profile = np.copy(matrix_profile)
        right_matrix_profile = np.copy(matrix_profile)
        left_profile_index = np.copy(profile_index)
        right_profile_index = np.copy(profile_index)

    distance_profile = np.zeros(profile_length)
    last_product = np.copy(distance_profile)
    # TODO: drop value????????

    for i in range(num_queries):
        query_window = query[i:i + window_size]

        if i == 0:
            #TODO
            pass
        else:
            #TODO

        # apply the exclusion zone
        if exclusion_zone > 0:
            ez_start = np.max(0, i - exclusion_zone)
            ez_end = np.min(profile_length, i + exclusion_zone)
            distance_profile[ez_start:ez_end] = np.inf
        
        # 
