# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import logging

logger = logging.getLogger(__name__)


from matrixprofile.algorithms.stomp import stomp
from matrixprofile.algorithms.scrimp import scrimp_plus_plus


#TODO: method could be changed to "sample" or "runtime"
def compute(ts, window_size, query=None, n_jobs=-1, method='exact'):
	if method not in ('exact', 'approximate'):
		raise ValueError('method expects "exact" or "approximate".')

	# run single threaded if series is small
	if len(ts) < 1000:
		logger.warning('ts is small, running in single threaded mode.')
		n_jobs = 1

	result = None
	if method is 'exact':
		result = stomp(ts, window_size, query=query, n_jobs=n_jobs)
	else:
		result = scrimp_plus_plus(ts, window_size, query=query, n_jobs=n_jobs)

	return result