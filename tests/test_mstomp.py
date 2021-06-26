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

from matrixprofile.algorithms.mstomp import mstomp


def test_mstomp_window_size_less_than_4():
    ts = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    w = 2

    with pytest.raises(ValueError) as excinfo:
        mstomp(ts, w)
        assert 'window size must be at least 4.' in str(excinfo.value)


def test_mstomp_time_series_too_small():
    ts = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    w = 8

    with pytest.raises(ValueError) as excinfo:
        mstomp(ts, w)
        assert 'Time series is too short' in str(excinfo.value)


def test_mstomp_single_dimension():
    ts = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    w = 4
    desired_mp = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
    desired_pi = np.array([[4, 5, 6, 7, 0, 1, 2, 3, 0]])

    desired_lmp = np.array([[np.inf, np.inf, np.inf, 2.82842712, 0, 0, 0, 0, 0]])
    desired_lpi = np.array([[0, 0, 0, 0, 0, 1, 2, 3, 0]])

    desired_rmp = np.array([[0, 0, 0, 0, 0, 2.82842712, np.inf, np.inf, np.inf]])
    desired_rpi = np.array([[4, 5, 6, 7, 8, 8, 0, 0, 0]])

    profile = mstomp(ts, w, n_jobs=1)
    np.testing.assert_almost_equal(profile['mp'], desired_mp)
    np.testing.assert_almost_equal(profile['pi'], desired_pi)

    np.testing.assert_almost_equal(profile['lmp'], desired_lmp)
    np.testing.assert_almost_equal(profile['lpi'], desired_lpi)

    np.testing.assert_almost_equal(profile['rmp'], desired_rmp)
    np.testing.assert_almost_equal(profile['rpi'], desired_rpi)


def test_mstomp_multi_dimension():
    ts = np.array([[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]])
    w = 4
    desired_mp = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 9.19401687e-01, 9.19401687e-01, 2.98023224e-08, 0, 9.19401687e-01, 9.19401687e-01, 9.19401687e-01]])
    desired_pi = np.array([[4, 5, 6, 7, 0, 1, 2, 3, 0], [4, 5, 6, 7, 0, 1, 2, 3, 0]])
    desired_pd = [
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
    ]

    profile = mstomp(ts, w, return_dimension=True, n_jobs=1)
    np.testing.assert_almost_equal(profile['mp'], desired_mp)
    np.testing.assert_almost_equal(profile['pi'], desired_pi)
    for i in range(len(ts)):
        np.testing.assert_almost_equal(profile['pd'][i], desired_pd[i])


def test_mstomp_single_dimension_multi_threaded():
    ts = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    w = 4
    desired_mp = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
    desired_pi = np.array([[4, 5, 6, 7, 0, 1, 2, 3, 0]])

    desired_lmp = np.array([[np.inf, np.inf, np.inf, 2.82842712, 0, 0, 0, 0, 0]])
    desired_lpi = np.array([[0, 0, 0, 0, 0, 1, 2, 3, 0]])

    desired_rmp = np.array([[0, 0, 0, 0, 0, 2.82842712, np.inf, np.inf, np.inf]])
    desired_rpi = np.array([[4, 5, 6, 7, 8, 8, 0, 0, 0]])

    profile = mstomp(ts, w, n_jobs=-1)
    np.testing.assert_almost_equal(profile['mp'], desired_mp)
    np.testing.assert_almost_equal(profile['pi'], desired_pi)

    np.testing.assert_almost_equal(profile['lmp'], desired_lmp)
    np.testing.assert_almost_equal(profile['lpi'], desired_lpi)

    np.testing.assert_almost_equal(profile['rmp'], desired_rmp)
    np.testing.assert_almost_equal(profile['rpi'], desired_rpi)


def test_mstomp_multi_dimension_multi_threaded():
    ts = np.array([[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]])
    w = 4
    desired_mp = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 9.19401687e-01, 9.19401687e-01, 2.98023224e-08, 0, 9.19401687e-01, 9.19401687e-01, 9.19401687e-01]])
    desired_pi = np.array([[4, 5, 6, 7, 0, 1, 2, 3, 0], [4, 5, 6, 7, 0, 1, 2, 3, 0]])
    desired_pd = [
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
    ]

    profile = mstomp(ts, w, return_dimension=True, n_jobs=-1)
    np.testing.assert_almost_equal(profile['mp'], desired_mp)
    np.testing.assert_almost_equal(profile['pi'], desired_pi)
    for i in range(len(ts)):
        np.testing.assert_almost_equal(profile['pd'][i], desired_pd[i])