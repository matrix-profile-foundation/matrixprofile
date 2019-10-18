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

from matrixprofile.algorithms.top_k_discords import top_k_discords


def test_mp_all_same():
	"""In this case it should only find one entry - the last as the view
	gets flipped.
	"""
	profile = {
		'mp': np.ones(10),
        'w': 4
	}

	discords = top_k_discords(profile)
	desired = np.array([9, 6, 3])
	np.testing.assert_almost_equal(discords, desired)


def test_discords_no_exclusion():
    raise ValueError('NEEDS MORE TESTING!!!!!!!')
