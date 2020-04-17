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

from matrixprofile.algorithms.mpdist import mpdist
import matrixprofile

MODULE_PATH = matrixprofile.__path__[0]


def test_small_series_single_threaded():
    ts = np.array([
        1, 2, 3, 1, 2, 3, 4, 5, 6, 0, 0, 1, 
        1, 2, 2, 4, 5, 1, 1, 9
    ]).astype('d')
    query = np.array([
        0.23595094, 0.9865171, 0.1934413, 0.60880883,
        0.55174926, 0.77139988, 0.33529215, 0.63215848
    ]).astype('d')
    w = 4

    desired = 0.437690617625298
    actual = mpdist(ts, query, w, n_jobs=1)

    np.testing.assert_almost_equal(actual, desired)


def test_small_series_multi_threaded():
    ts = np.array([
        1, 2, 3, 1, 2, 3, 4, 5, 6, 0, 0, 1, 
        1, 2, 2, 4, 5, 1, 1, 9
    ]).astype('d')
    query = np.array([
        0.23595094, 0.9865171, 0.1934413, 0.60880883,
        0.55174926, 0.77139988, 0.33529215, 0.63215848
    ]).astype('d')
    w = 4

    desired = 0.437690617625298
    actual = mpdist(ts, query, w, n_jobs=-1)

    np.testing.assert_almost_equal(actual, desired)


def test_against_matlab():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    tsb = ts[199:300]
    w = 32

    desired = 1.460009659995543e-07
    actual = mpdist(ts, tsb, w, n_jobs=1)

    np.testing.assert_almost_equal(actual, desired)


def test_against_matlab_parallel():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    tsb = ts[199:300]
    w = 32

    desired = 1.460009659995543e-07
    actual = mpdist(ts, tsb, w, n_jobs=-1)

    np.testing.assert_almost_equal(actual, desired)


def test_ts_not_one_dimensional():
    ts = np.array([[1, 1], [2, 2]])
    tsb = np.arange(10)
    w = 32

    with pytest.raises(ValueError) as excinfo:
        mpdist(ts, tsb, w)
        assert('ts must be one dimensional!' == str(excinfo.value))


def test_tsb_not_one_dimensional():
    tsb = np.array([[1, 1], [2, 2]])
    ts = np.arange(10)
    w = 32

    with pytest.raises(ValueError) as excinfo:
        mpdist(ts, tsb, w)
        assert('ts_b must be one dimensional!' == str(excinfo.value))


def test_invalid_threshold():
    ts = np.arange(100)
    tsb = np.arange(100)
    w = 32
    threshold = -1
    error = 'threshold must be a float greater than 0 and less than 1'

    with pytest.raises(ValueError) as excinfo:
        mpdist(ts, tsb, w, threshold=threshold)
        assert(error == str(excinfo.value))

    threshold = 'str'
    with pytest.raises(ValueError) as excinfo:
        mpdist(ts, tsb, w, threshold=threshold)
        assert(error == str(excinfo.value))

    threshold = 1
    with pytest.raises(ValueError) as excinfo:
        mpdist(ts, tsb, w, threshold=threshold)
        assert(error == str(excinfo.value))
