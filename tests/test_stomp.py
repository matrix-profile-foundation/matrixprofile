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

from matrixprofile.algorithms.stomp import stomp


def test_stomp_window_size_less_than_4():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    w = 2

    with pytest.raises(ValueError) as excinfo:
        stomp(ts, w)
        assert 'window size must be at least 4.' in str(excinfo.value)


def test_stomp_window_size_too_small():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    w = 8

    with pytest.raises(ValueError) as excinfo:
        stomp(ts, w)
        assert 'Time series is too short' in str(excinfo.value)


def test_stomp_small_series_self_join_single_threaded():
    ts = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    w = 4
    desired = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    desired_pi = np.array([4, 5, 6, 7, 0, 1, 2, 3, 0])

    desired_lmp = np.array([np.inf, np.inf, np.inf, 2.82842712, 0, 0, 0, 0, 0])
    desired_lpi = np.array([0, 0, 0, 0, 0, 1, 2, 3, 0])

    desired_rmp = np.array([0, 0, 0, 0, 0, 2.82842712, np.inf, np.inf, np.inf])
    desired_rpi = np.array([4, 5, 6, 7, 8, 8, 0, 0, 0])

    profile = stomp(ts, w, n_jobs=1)
    np.testing.assert_almost_equal(profile['mp'], desired)
    np.testing.assert_almost_equal(profile['pi'], desired_pi)

    np.testing.assert_almost_equal(profile['lmp'], desired_lmp)
    np.testing.assert_almost_equal(profile['lpi'], desired_lpi)

    np.testing.assert_almost_equal(profile['rmp'], desired_rmp)
    np.testing.assert_almost_equal(profile['rpi'], desired_rpi)


def test_stomp_small_series_self_join_multi_threaded():
    ts = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    w = 4
    desired = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    desired_pi = np.array([4, 5, 6, 7, 0, 1, 2, 3, 0])

    desired_lmp = np.array([np.inf, np.inf, np.inf, 2.82842712, 0, 0, 0, 0, 0])
    desired_lpi = np.array([0, 0, 0, 0, 0, 1, 2, 3, 0])

    desired_rmp = np.array([0, 0, 0, 0, 0, 2.82842712, np.inf, np.inf, np.inf])
    desired_rpi = np.array([4, 5, 6, 7, 8, 8, 0, 0, 0])

    profile = stomp(ts, w, n_jobs=-1)
    np.testing.assert_almost_equal(profile['mp'], desired)
    np.testing.assert_almost_equal(profile['pi'], desired_pi)

    np.testing.assert_almost_equal(profile['lmp'], desired_lmp)
    np.testing.assert_almost_equal(profile['lpi'], desired_lpi)

    np.testing.assert_almost_equal(profile['rmp'], desired_rmp)
    np.testing.assert_almost_equal(profile['rpi'], desired_rpi)

