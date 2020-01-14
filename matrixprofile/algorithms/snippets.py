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


def snippets(ts, snippet_size, num_snippets=2, window_size=None, n_jobs=1):
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
	n_jobs : int, Default 1
		The number of cpu cores to use.

	Returns
	-------
	list :
		A list of snippets as dictionary objects with the following structure.
		{
			fraction: fraction of the snippet,
			index: the index of the snippet,
		}
	"""
	ts = core.to_np_array(ts).astype('d')
	n = len(ts)
	n_jobs = core.valid_n_jobs(n_jobs)

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
	indices = np.arange(0, n, snippet_size, dtype=int)
	distances = np.full((len(indices), n - snippet_size + 1), np.nan)
	distances_snippet = np.full((len(indices), n - snippet_size + 1), np.nan)

	for j, i in enumerate(indices):
		distance = mpdist(ts, ts[i:(i + snippet_size)], window_size, n_jobs=n_jobs)
		distances[j] = distance

	print(distances)

	# # compute the Nth snippets
	# minis = np.inf
	# snippet_index = np.nan

	# for z in range(num_snippets):
	# 	minims = np.inf
	# 	index = np.nan

	# 	for i in range(distances.shape[0]):
	# 		# compute area under the profile (maximize converage)
	# 		mask = distances[i] < minis
	# 		s = np.sum(distances[i][mask])

	# 		if minims > s:
	# 			minims = s
	# 			index = i

	# 	mask = distances[index] < minis
	# 	minis = distances[index][mask]
	# 	if np.isnan(snippet_index):
	# 		snippet_index = np.array([snippet_index])
	# 	else:
	# 		snippet_index = np.append(snippet_index, [indices[index],])
	# 	distances_snippet[z] = distances[index]

	# return snippet_index
