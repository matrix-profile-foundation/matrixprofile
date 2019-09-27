# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import math

from matrixprofile.algorithms.stomp import stomp
from matrixprofile import core


def brute_force_pmp(ts, windows=None, n_jobs=-1):
	"""
	Computes the PMP data structure using the brute force approach.

	Parameters
	----------
	ts : array_like
		The time series to compute the PMP for.
	windows : iterable(int), Optional
		An iterable composed of integer values for window sizes to compute the
		matrix profile for. It defaults to half of the time series when it is
		not provided.
	n_jobs : int
		Number of cpu cores to allocate. Defaults to all.

	NOTE
	----
	You can use Python's range() function to generate many windows to test.

	Returns
	-------
	(array_like, array_like) :
		The Matrix Profiles and Matrix Profile Indices for all window sizes
		provided - PMP.
	"""
	n = len(ts)

	# default windows to all applicable
	if windows is None:
		start_window = 4
		end_window = math.floor(n / 2)
		windows = range(start_window, end_window)

	pmp = []
	pmpi = []
	for window in windows:
		result = stomp(ts, window, n_jobs=n_jobs)
		pmp.append(result['mp'])
		pmpi.append(result['pi'])

	return (pmp, pmpi)