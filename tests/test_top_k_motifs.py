#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import os

import pytest

import numpy as np

from matrixprofile.algorithms.top_k_motifs import top_k_motifs

def test_all_inf():
	obj = {
		'mp': np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),
		'pi': np.array([0, 0, 0, 0, 0]),
		'w': 4,
		'data': {
			'ts': np.array([1, 1, 1, 1, 1, 1, 1, 1])
		},
		'class': 'MatrixProfile'
	}

	motifs = top_k_motifs(obj)
	desired = np.array([])

	np.testing.assert_equal(motifs, desired)