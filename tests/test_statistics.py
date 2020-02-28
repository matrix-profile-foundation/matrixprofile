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

from matrixprofile.algorithms.statistics import statistics


def test_invalid_ts_not_1d():
    ts = np.array([[1, 1], [1, 1]])
    w = 2

    with pytest.raises(ValueError) as excinfo:
        statistics(ts, w)
        assert 'The time series must be 1D' in str(excinfo.value)


def test_invalid_ts_not_array():
    ts = None
    w = 2

    with pytest.raises(ValueError) as excinfo:
        statistics(ts, w)
        assert 'ts must be array like' in str(excinfo.value)


def test_invalid_window_size_not_int():
    ts = np.arange(10)
    w = 's'

    with pytest.raises(ValueError) as excinfo:
        statistics(ts, w)
        assert 'Expecting int for window_size' in str(excinfo.value)


def test_invalid_window_size_too_large():
    ts = np.arange(10)
    w = 11

    with pytest.raises(ValueError) as excinfo:
        statistics(ts, w)
        assert 'Window size cannot be greater than len(ts)' in str(excinfo.value)


def test_invalid_window_size_too_small():
    ts = np.arange(10)
    w = 2

    with pytest.raises(ValueError) as excinfo:
        statistics(ts, w)
        assert 'Window size cannot be less than 3' in str(excinfo.value)


def test_valid():
	ts = np.array([1, 3, 2, 4, 5, 1, 1, 1, 2, 4, 9, 7])
	w = 4
	ts_stats = statistics(ts, w)

	assert(ts_stats['min'] == 1)
	assert(ts_stats['max'] == 9)
	np.testing.assert_almost_equal(ts_stats['mean'], 3.3333333)
	np.testing.assert_almost_equal(ts_stats['std'], 2.494438257)
	assert(ts_stats['median'] == 2.5)
	np.testing.assert_almost_equal(ts_stats['moving_min'], np.array([1, 2, 1, 1, 1, 1, 1, 1, 2]))
	np.testing.assert_almost_equal(ts_stats['moving_max'], np.array([4, 5, 5, 5, 5, 2, 4, 9, 9]))
	np.testing.assert_almost_equal(ts_stats['moving_mean'], np.array([2.5, 3.5, 3.0, 2.75, 2.0, 1.25, 2.0, 4.0, 5.5]))
	np.testing.assert_almost_equal(ts_stats['moving_std'], np.array([1.11803399, 1.11803399, 1.58113883, 1.78535711, 1.73205081, 0.4330127, 1.22474487, 3.082207, 2.6925824]))
	np.testing.assert_almost_equal(ts_stats['moving_median'], np.array([2.5, 3.5, 3.0, 2.5, 1.0, 1.0, 1.5, 3.0, 5.5]))
	np.testing.assert_equal(ts_stats['ts'], ts)
	assert(ts_stats['window_size'] == w)
	assert(ts_stats['class'] == 'Statistics')
