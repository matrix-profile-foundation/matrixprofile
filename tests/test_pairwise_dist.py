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

from matrixprofile.algorithms.pairwise_dist import (
    compute_dist,
    pairwise_dist,
)


def test_pairwise_dist_valid_simple():
    X = [
        np.arange(100),
        np.arange(100),
        np.ones(100),
        np.zeros(100)
    ]
    w = 8
    dists = pairwise_dist(X, w)
    expected = np.array([ 0, 4, 4, 4, 4, 4])
    np.testing.assert_equal(dists, expected)

    # test with MxN np.ndarray
    X = np.array(X)
    dists = pairwise_dist(X, w)
    expected = np.array([ 0, 4, 4, 4, 4, 4])
    np.testing.assert_equal(dists, expected)


def test_pairwise_dist_invalid_params():
    X = [
        np.arange(10),
        np.arange(20)
    ]
    w = 4
    threshold = 0.05
    n_jobs = 1
    with pytest.raises(ValueError) as excinfo:
        pairwise_dist('', w, threshold=threshold, n_jobs=n_jobs)
        assert('X must be array_like!' == str(excinfo.value))

    error = 'threshold must be a float greater than 0 and less'\
                ' than 1'
    with pytest.raises(ValueError) as excinfo:
        pairwise_dist(X, w, threshold=1, n_jobs=n_jobs)
        assert(error == str(excinfo.value))


def test_compute_dist_valid():
    ts = np.arange(100)
    w = 8
    k = 0
    threshold = 0.05
    args = (k, ts, ts, w, threshold)
    result = compute_dist(args)

    assert(result[0] == k)
    assert(result[0] == 0)