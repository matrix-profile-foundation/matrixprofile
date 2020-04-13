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

from matrixprofile import utils
from matrixprofile import compute

import matrixprofile


def test_default():
    ts_arr = [0, 1, 2, 3, 4, 5]
    expect = [1, 1, 1, 1]
    w = 3

    av = utils.make_default_av(ts_arr, w)

    assert(len(av) == len(expect))

    np.testing.assert_almost_equal(av, expect)


def test_complexity():
    ts_arr = [[3., 3., 3., 3., 3., 3.],
              [0., 1., 2., 3., 4., 5.],
              [0., 3., 0., 2., 0., 1.]]
    expect = [[0., 0., 0., 0.],
              [0., 0., 0., 0.],
              [0.47295372330527, 0.32279030890406757, 0.13962038997193682, 0.]]
    w = 3

    for i in range(len(ts_arr)):
        av = utils.make_complexity_av(ts_arr[i], w)

        assert(len(av) == len(expect[i]))

        np.testing.assert_almost_equal(av, expect[i])


def test_meanstd():
    ts_arr = [[3., 3., 3., 3., 3., 3.],
              [-10., 10., -10., 1., -1., 1.],
              [0., 3., 0., 2., 0., 1.]]
    expect = [[0., 0., 0., 0.],
              [0., 0., 1., 1.],
              [0., 0., 1., 1.]]
    w = 3

    for i in range(len(ts_arr)):
        av = utils.make_meanstd_av(ts_arr[i], w)

        assert(len(av) == len(expect[i]))

        np.testing.assert_almost_equal(av, expect[i])

        
def test_clipping():
    ts_arr = [[3., 3., 3., 3., 3., 3.],
              [0., 1., 2., 3., 4., 5.],
              [0., 3., 0., 2., 0., 1.]]
    expect = [[0., 0., 0., 0.],
              [0., 1., 1., 0.],
              [0., 0.5, 0.5, 1.]]
    w = 3

    for i in range(len(ts_arr)):
        av = utils.make_clipping_av(ts_arr[i], w)

        assert(len(av) == len(expect[i]))

        np.testing.assert_almost_equal(av, expect[i])


def test_apply_av():
    ts = [3., 3., 3., 3., 3., 3.]
    expect = [6.92820324, 6.92820324, 6.92820324, 6.92820324]
    w = 3

    profile = compute(ts, windows=w)

    av = np.array([0., 0., 0., 0.])
    profile = utils.apply_av(profile, "custom", av)

    np.testing.assert_almost_equal(profile['mp'], expect)

