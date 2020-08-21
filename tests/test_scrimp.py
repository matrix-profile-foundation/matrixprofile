#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

"""Tests for `scrimp` package."""

import os
import pytest

import numpy as np

import matrixprofile
from matrixprofile.algorithms import scrimp

MODULE_PATH = matrixprofile.__path__[0]


def test_time_series_too_short_exception():
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 4, 0.25)
        assert 'Time series is too short' in str(excinfo.value)


def test_window_size_minimum_exception():
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 2, 0.25)
        assert 'Window size must be at least 4' in str(excinfo.value)


def test_invalid_step_size_negative():
    exc = 'step_size should be a float between 0 and 1.'
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 2, -1)
        assert exc in str(excinfo.value)


def test_invalid_step_size_str():
    exc = 'step_size should be a float between 0 and 1.'
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 2, 'a')
        assert exc in str(excinfo.value)


def test_invalid_step_size_greater():
    exc = 'step_size should be a float between 0 and 1.'
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 2, 2)
        assert exc in str(excinfo.value)


def test_invalid_random_state_exception():
    exc = 'Invalid random_state value given.'
    with pytest.raises(ValueError) as excinfo:
        scrimp.scrimp_plus_plus([1, 2, 3, 4, 5], 2, random_state='adsf')
        assert exc in str(excinfo.value)


def test_scrimp_plus_plus():
    ts = np.array([0, 0, 1, 0, 0, 0, 1, 0])
    m = 4
    step_size = 0.25
    profile = scrimp.scrimp_plus_plus(ts, m, step_size=step_size, sample_pct=1.0)

    expected_mp = np.array([
        0,
        3.2660,
        3.2660,
        3.2660,
        0
    ])
    expected_mpidx = np.array([
        4,
        3,
        0,
        0,
        0,
    ])

    np.testing.assert_almost_equal(profile['mp'], expected_mp, decimal=4)
    np.testing.assert_equal(profile['pi'], expected_mpidx)

    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    m = 32
    step_size = 0.25
    profile = scrimp.scrimp_plus_plus(ts, m, step_size=step_size, sample_pct=1.0)
    expected_mp = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'scrimp.mp.txt'))
    expected_mpi = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'scrimp.mpi.txt')).astype('int') - 1

    np.testing.assert_almost_equal(profile['mp'], expected_mp)
    np.testing.assert_equal(profile['pi'], expected_mpi)
