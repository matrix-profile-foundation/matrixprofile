# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

from matrixprofile import core
from matrixprofile.algorithms.mass2 import mass2


def top_k_motifs(ts, profile, exclusion_zone, k=3, max_neighbors=10, radius=3):
    """
    Find the top K number of motifs (patterns) given a matrix profile. By
    default the algorithm will find up to 3 motifs (k) and up to 10 of their
    neighbors with a radius of 3.

    Parameters
    ----------
    ts : array_like
    	The original data used to compute the matrix profile.
    profile : dict
        The matrix profile computed from the compute function.
    exclusion_zone : int
        Desired number of values to exclude on both sides of the motif. This
        avoid trivial matches.
    k : int, Default = 3
        Desired number of motifs to find.
    neighbor_count : int, Default = 10
    	The maximum number of neighbors to include for a given motif.
    radius : int, Default = 3
    	The radius is used to associate a neighbor by checking if the
    	neighbor's distance is less than or equal to dist * radius

    Returns
    -------
    A list of dicts containing motif indices and their corresponding neighbor
    indices.

    [
    	{
    		'motifs': [first_index, second_index],
    		'neighbors': [index, index, index ...max_neighbors]
    	}
    ]
    """
    window_size = profile['w']
    data_len = len(ts)
    mu, sig = core.moving_avg_std(ts, window_size)
    profile_len = len(profile['mp'])
    motifs = []
    mp = np.copy(profile['mp'])
    mpi = profile['pi']

    for i in range(k):
    	min_idx = np.argmin(mp)
    	min_dist = mp[min_idx]

    	# we no longer have any motifs to find as all values are nan/inf
    	if core.is_nan_inf(min_dist):
    		break

    	# create a motif pair corresponding to the first appearance and
    	# second appearance
    	first_idx = np.min(min_idx, mpi[min_idx])
    	second_idx = np.max(min_idx, mpi[min_idx])

    	# compute distance profile using mass2 for first appearance
    	query = ts[first_idx:]
    	distance_profile = mass2(ts, query)

    	# exclude already picked motifs and neighbors
    	distance_profile[nan_inf_indices(mp)] = np.inf

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
			mp = core.apply_exclusion_zone(
				exclusion_zone,
				False,
				window_size,
				data_len,
				j,
				mp
			)

    	# find up to max_neighbors
    	neighbors = []
    	for j in range(max_neighbors):
    		neighbor_idx = np.argmin(distance_profile)
    		neighbor_dist = distance_profile[neighbor_idx]
    		not_in_radius = not ((radius * min_dist) < neighbor_dist)

    		# no more neighbors exist based on radius
    		if core.is_nan_inf(neighbor_val) or not_in_radius:
    			break;

    		# add neighbor and apply exclusion zone
    		neighbors.append(neighbor_idx)
    		distance_profile = core.apply_exclusion_zone(
				exclusion_zone,
				False,
				window_size,
				data_len,
				neighbor_idx,
				distance_profile
			)
			mp = core.apply_exclusion_zone(
				exclusion_zone,
				False,
				window_size,
				data_len,
				neighbor_idx,
				mp
			)

    	# add motifs and neighbors to results
    	motifs.append({
    		'motifs': [first_idx, second_idx],
    		'neighbors': neighbors
    	})

    return motifs