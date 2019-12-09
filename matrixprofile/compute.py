# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import logging

logger = logging.getLogger(__name__)

from matrixprofile import core
from matrixprofile.algorithms.mpx import mpx
from matrixprofile.algorithms.scrimp import scrimp_plus_plus
from matrixprofile.algorithms.skimp import skimp


def compute(ts, windows, query=None, sample_pct=1, n_jobs=-1):
	"""
	Computes the exact or approximate MatrixProfile based on the sample percent
	specified. Currently, MPX and SCRIMP++ is used for the exact and
	approximate algorithms respectively. When multiple windows are passed, the
	Pan-MatrixProfile is computed and returned.

	Note
	----
	When multiple windows are passed and the Pan-MatrixProfile is computed, the
	query is ignored!

	Parameters
    ----------
    ts : array_like
        The time series to analyze.
	windows : int or array_like
        The window(s) to compute the MatrixProfile. Note that it may be an int
		for a single matrix profile computation or an array of ints for
		computing the pan matrix profile.
    query : array_like, Optional
        The query to analyze. Note that when computing the PMP the query is
		ignored!
	sample_pct : float, default = 1
        A float between 0 and 1 representing how many samples to compute for
        the MP or PMP. When it is 1, the exact algorithm is used.
    n_jobs : int, default -1 (all cpu cores)
        The number of cpu cores to use when computing the MP.
	
	Returns
	-------
	dict : profile
		The profile computed.
	"""

	result = None
	multiple_windows = core.is_array_like(windows) and len(windows) > 1
	
	if core.is_array_like(windows) and len(windows) == 1:
		windows = windows[0]

	if multiple_windows:
		if core.is_array_like(query):
			logger.warn('Computing PMP - query is ignored!')

		result = skimp(ts, windows=windows, sample_pct=sample_pct,
			n_jobs=n_jobs)
	elif sample_pct >= 1:
		result = mpx(ts, windows, query=query, n_jobs=n_jobs)
	else:
		result = scrimp_plus_plus(ts, windows, query=query, n_jobs=n_jobs)

	return result