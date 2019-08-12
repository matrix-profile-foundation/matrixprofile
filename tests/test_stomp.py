#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

"""Tests for `mass_ts` package."""

import os

import pytest

import numpy as np
import ray

from matrixprofile import stomp


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

    profile = stomp(ts, w, n_jobs=1)
    np.testing.assert_almost_equal(profile['mp'], desired)


def test_stomp_small_series_self_join_single_threaded_pi():
    ts = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    w = 4
    desired = np.array([4, 5, 6, 7, 0, 1, 2, 3, 0])

    profile = stomp(ts, w, n_jobs=1)
    np.testing.assert_almost_equal(profile['pi'], desired)


def test_stomp_small_series_self_join_multi_threaded():
    ts = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    w = 4
    desired = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    desired_pi = np.array([4, 5, 6, 7, 0, 1, 2, 3, 0])

    profile = stomp(ts, w)
    np.testing.assert_almost_equal(profile['mp'], desired)
    np.testing.assert_almost_equal(profile['pi'], desired_pi)


def test_stomp_small_series_self_join_multi_threaded_pi():
    ts = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    w = 4
    desired = np.array([4, 5, 6, 7, 0, 1, 2, 3, 0])

    profile = stomp(ts, w)
    np.testing.assert_almost_equal(profile['pi'], desired)


def test_stomp_small_series_self_join_ray():
    ts = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    w = 4
    desired = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    ray.init()
    profile = stomp(ts, w)
    ray.shutdown()

    np.testing.assert_almost_equal(profile['mp'], desired)


def test_stomp_small_series_self_join_ray_pi():
    ts = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    w = 4
    desired = np.array([4, 5, 6, 7, 0, 1, 2, 3, 0])

    ray.init()
    profile = stomp(ts, w)
    ray.shutdown()

    np.testing.assert_almost_equal(profile['pi'], desired)