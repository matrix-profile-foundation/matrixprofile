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
		'mp': np.ones(10)
	}

	discords = top_k_discords(profile, 4)
	desired = np.array([9, 4])
	np.testing.assert_almost_equal(discords, desired)


def test_discords_no_exclusion():
    mp = np.array([1.0, 2.0, 3.0, 4.0])
    discords = top_k_discords({'mp': mp}, 0, k=4)
    desired = np.array([3, 2, 1, 0])

    np.testing.assert_almost_equal(discords, desired)


def test_discords_exclude_one():
    mp = np.array([1.0, 2.0, 3.0, 4.0])
    discords = top_k_discords({'mp': mp}, 1, 4)
    desired = np.array([3, 1])

    np.testing.assert_almost_equal(discords, desired)


def test_discords_exclude_big():
    mp = np.array([1.0, 2.0, 3.0, 4.0])
    discords = top_k_discords({'mp': mp}, 10, 4)
    desired = np.array([3,])

    np.testing.assert_almost_equal(discords, desired)


def test_discords_empty_mp():
    mp = np.array([])
    discords = top_k_discords({'mp': mp}, 10, 4)
    desired = np.array([])

    np.testing.assert_almost_equal(discords, desired)


def test_discords_k_larger_than_mp():
    mp = np.array([1.0, 2.0, 3.0, 4.0])
    discords = top_k_discords({'mp': mp}, 1, 10)
    desired = np.array([3, 1,])

    np.testing.assert_almost_equal(discords, desired)
