#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import math

import numpy as np

from matrixprofile import core
from matrixprofile.algorithms.cympx_ab import mpx_ab as cympx_ab


def mpdist(ts, ts_b, w, n_jobs=-1):
	"""
	Computes the MPDist between the two series ts and ts_b. For more details
	refer to the paper:

	Matrix Proﬁle XII: MPdist: A Novel Time Series Distance Measure to Allow 
	Data Mining in More Challenging Scenarios. Shaghayegh Gharghabi, 
	Shima Imani, Anthony Bagnall, Amirali Darvishzadeh, Eamonn Keogh. ICDM 2018

	Parameters
    ----------
    ts : array_like
        The time series to compute the matrix profile for.
    ts_b : array_like
        The time series to compare against.
    w : int
        The window size.
    n_jobs : int, Default all
        Number of cpu cores to use.
    
    Returns
    -------
    float :
    	The MPDist.
    """
	ts = core.to_np_array(ts).astype('d')
	ts_b = core.to_np_array(ts_b).astype('d')

	if not core.is_one_dimensional(ts):
		raise ValueError('ts must be one dimensional!')

	if not core.is_one_dimensional(ts_b):
		raise ValueError('ts_b must be one dimensional!')

	mp, mpi, mpb, mpib = cympx_ab(ts, ts_b, w, 0, n_jobs)

	mp_abba = np.append(mp, mpb)
	data_len = len(ts) + len(ts_b)
	abba_sorted = np.sort(mp_abba[~core.nan_inf_indices(mp_abba)])

	print(abba_sorted)

	distance = np.inf
	if len(abba_sorted) > 0:
		idx = np.min([len(abba_sorted) - 1, int(np.ceil(0.05 * data_len)) - 1])
		distance = abba_sorted[idx]

	return distance