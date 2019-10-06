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

from matrixprofile.algorithms.mpx import mpx


def test_mpx_small_series_self_join_euclidean_single_threaded():
    ts = np.array([0, 1, 1, 1, 0, 0, 2, 1, 0, 0, 2, 1])
    w = 4
    desired = np.array([1.9550, 1.8388, 0.8739, 0, 0, 1.9550, 0.8739, 0, 0])
    desired_pi = np.array([4, 2, 6, 7, 8, 1, 2, 3, 4])

    mp, pi = mpx(ts, w, cross_correlation=False)
    np.testing.assert_almost_equal(mp, desired, decimal=4)
    np.testing.assert_almost_equal(pi, desired_pi)


def test_mpx_small_series_self_join_pearson_single_threaded():
    ts = np.array([0, 1, 1, 1, 0, 0, 2, 1, 0, 0, 2, 1])
    w = 4
    desired = np.array([0.522232967867094, 0.577350269189626, 0.904534033733291, 1, 1, 0.522232967867094, 0.904534033733291, 1, 1])
    desired_pi = np.array([4, 2, 6, 7, 8, 1, 2, 3, 4])

    mp, pi = mpx(ts, w, cross_correlation=True)
    np.testing.assert_almost_equal(mp, desired)
    np.testing.assert_almost_equal(pi, desired_pi)