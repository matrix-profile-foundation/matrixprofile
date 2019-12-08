# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import logging

logger = logging.getLogger(__name__)


from matrixprofile.algorithms.mpx import mpx
from matrixprofile.algorithms.scrimp import scrimp_plus_plus


def compute(ts, window_size, query=None, sample_pct=1, n_jobs=-1):
	"""
	Computes the exact or approximate MatrixProfile based on the sample percent
	specified. Currently, MPX and SCRIMP++ is used for the exact and
	approximate algorithms respectively.

	Parameters
    ----------
    ts : array_like
        The time series to analyze.
	window_size : int
        The window size to compute the MatrixProfile.
    query : array_like, Optional
        The query to analyze.
	sample_pct : float, default = 1
        A float between 0 and 1 representing how many samples to compute for
        the MP. When it is 1, the exact algorithm is used.
    n_jobs : int, default -1 (all cpu cores)
        The number of cpu cores to use when computing the MP.
	
	Returns
	-------
	dict : profile
		The profile computed.
	"""

	result = None
	if sample_pct >= 1:
		result = mpx(ts, window_size, query=query, n_jobs=n_jobs)
	else:
		result = scrimp_plus_plus(ts, window_size, query=query, n_jobs=n_jobs)

	return result