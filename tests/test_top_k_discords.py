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
	profile = {
		'mp': np.ones(10),
		'ez': 2,
        'w': 4,
        'class': 'MatrixProfile'
	}

	discords = top_k_discords(profile)['discords']
	desired = np.array([9, 6, 3])
	np.testing.assert_almost_equal(discords, desired)


def test_discords_no_exclusion():
	profile = {
		'mp': np.array([1, 2, 3, 4]),
		'w': 4,
		'class': 'MatrixProfile'
	}
	desired = np.array([3, 2, 1])
	discords = top_k_discords(profile, k=3, exclusion_zone=0)['discords']
	np.testing.assert_almost_equal(discords, desired)


def test_discords_no_exclusion_all():
	profile = {
		'mp': np.array([1, 2, 3, 4]),
		'w': 4,
		'class': 'MatrixProfile'
	}
	desired = np.array([3, 2, 1, 0])
	discords = top_k_discords(profile, k=4, exclusion_zone=0)['discords']
	np.testing.assert_almost_equal(discords, desired)


def test_discords_exclude_one():
	profile = {
		'mp': np.array([1, 2, 3, 4]),
		'w': 4,
		'class': 'MatrixProfile'
	}
	desired = np.array([3, 1])
	discords = top_k_discords(profile, k=4, exclusion_zone=1)['discords']
	np.testing.assert_almost_equal(discords, desired)