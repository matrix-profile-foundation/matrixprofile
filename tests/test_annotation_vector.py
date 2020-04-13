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


def test_default_valid():
    ts_arr = [0, 1, 2, 3, 4, 5]
    expect = [1, 1, 1, 1]
    w = 3

    av = utils.make_default_av(ts_arr, w)

    assert(len(av) == len(expect))

    np.testing.assert_almost_equal(av, expect)


def test_complexity_valid():
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


def test_meanstd_valid():
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


def test_clipping_valid():
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


def test_default_invalid():
    with pytest.raises(ValueError) as excinfo:
        utils.make_default_av("array", 3)
        assert 'make_default_av expects ts to be array-like' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        utils.make_default_av([[1, 2, 3], [1, 2, 3]], 3)
        assert 'make_default_av expects ts to be one-dimensional' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        utils.make_default_av([1, 2, 3], "window")
        assert 'make_default_av expects window to be an integer' \
            in str(excinfo.value)


def test_complexity_invalid():
    with pytest.raises(ValueError) as excinfo:
        utils.make_complexity_av("array", 3)
        assert 'make_complexity_av expects ts to be array-like' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        utils.make_complexity_av([[1, 2, 3], [1, 2, 3]], 3)
        assert 'make_complexity_av expects ts to be one-dimensional' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        utils.make_complexity_av([1, 2, 3], "window")
        assert 'make_complexity_av expects window to be an integer' \
            in str(excinfo.value)


def test_meanstd_invalid():
    with pytest.raises(ValueError) as excinfo:
        utils.make_meanstd_av("array", 3)
        assert 'make_meanstd_av expects ts to be array-like' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        utils.make_meanstd_av([[1, 2, 3], [1, 2, 3]], 3)
        assert 'make_meanstd_av expects ts to be one-dimensional' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        utils.make_meanstd_av([1, 2, 3], "window")
        assert 'make_meanstd_av expects window to be an integer' \
            in str(excinfo.value)


def test_clipping_invalid():
    with pytest.raises(ValueError) as excinfo:
        utils.make_clipping_av("array", 3)
        assert 'make_clipping_av expects ts to be array-like' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        utils.make_clipping_av([[1, 2, 3], [1, 2, 3]], 3)
        assert 'make_clipping_av expects ts to be one-dimensional' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        utils.make_clipping_av([1, 2, 3], "window")
        assert 'make_clipping_av expects window to be an integer' \
            in str(excinfo.value)


def test_apply_default_av_valid():
    ts = [3., 3., 3., 3., 3., 3.]
    w = 3

    profile = compute(ts, windows=w)
    expect = profile['mp']

    profile = utils.apply_av(profile, "default")

    np.testing.assert_almost_equal(profile['mp'], expect)


def test_apply_complexity_av_valid():
    ts = [3., 3., 3., 3., 3., 3.]
    w = 3

    profile = compute(ts, windows=w)
    expect = profile['mp'] * 2

    profile = utils.apply_av(profile, "complexity")

    np.testing.assert_almost_equal(profile['mp'], expect)


def test_apply_meanstd_av_valid():
    ts = [3., 3., 3., 3., 3., 3.]
    w = 3

    profile = compute(ts, windows=w)
    expect = profile['mp'] * 2

    profile = utils.apply_av(profile, "meanstd")

    np.testing.assert_almost_equal(profile['mp'], expect)


def test_apply_clipping_av_valid():
    ts = [3., 3., 3., 3., 3., 3.]
    w = 3

    profile = compute(ts, windows=w)
    expect = profile['mp'] * 2

    profile = utils.apply_av(profile, "clipping")

    np.testing.assert_almost_equal(profile['mp'], expect)


def test_apply_custom_av_valid():
    ts = [3., 3., 3., 3., 3., 3.]
    w = 3

    profile = compute(ts, windows=w)
    expect = profile['mp'] * 2

    av = [0., 0., 0., 0.]
    profile = utils.apply_av(profile, "custom", av)

    np.testing.assert_almost_equal(profile['mp'], expect)


def test_apply_av_invalid():
    ts = [3., 3., 3., 3., 3., 3.]
    w = 3

    with pytest.raises(ValueError) as excinfo:
        utils.apply_av("profile", "default")
        assert 'apply_av expects profile as an MP data structure' \
            in str(excinfo.value)

    profile = compute(ts, windows=w)

    with pytest.raises(ValueError) as excinfo:
        utils.apply_av(profile, "custom", "av")
        assert 'apply_av expects custom_av to be array-like' \
            in str(excinfo.value)

    profile = compute(ts, windows=w)

    with pytest.raises(ValueError) as excinfo:
        utils.apply_av(profile, "not a parameter")
        assert 'av parameter is invalid' \
            in str(excinfo.value)

    profile = compute(ts, windows=w)

    with pytest.raises(ValueError) as excinfo:
        utils.apply_av(profile, "custom", [0.9, 0.9, 0.9])
        assert 'Lengths of annotation vector and mp are different' \
            in str(excinfo.value)

    profile = compute(ts, windows=w)

    with pytest.raises(ValueError) as excinfo:
        utils.apply_av(profile, "custom", [0.5, 0.5, 0.6, 1.2, -0.4])
        assert 'Annotation vector values must be between 0 and 1' \
            in str(excinfo.value)
