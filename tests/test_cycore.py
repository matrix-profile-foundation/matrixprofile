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

from matrixprofile import cycore
import matrixprofile

MODULE_PATH = matrixprofile.__path__[0]


def test_moving_avg_std():
    a = np.array([1, 2, 3, 4, 5, 6], dtype='d')
    mu, std = cycore.moving_avg_std(a, 3)
    mu_desired = np.array([2., 3., 4., 5.])
    std_desired = np.array([0.81649658, 0.81649658, 0.81649658, 0.81649658])

    np.testing.assert_almost_equal(mu, mu_desired)
    np.testing.assert_almost_equal(std, std_desired)


def test_it_should_not_produce_nan_values_when_std_is_almost_zero():
    a = np.array([10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1], dtype='d')
    mu, std = cycore.moving_avg_std(a, 3)
    mu_muinvn, std_muinvn = cycore.muinvn(a, 3)

    mu_desired = np.array([10.1, 10.1, 10.1, 10.1, 10.1])
    std_desired = np.array([0, 0, 0, 0, 0])

    np.testing.assert_almost_equal(mu, mu_desired)
    np.testing.assert_almost_equal(std, std_desired)

    np.testing.assert_almost_equal(mu_muinvn, mu_desired)
    np.testing.assert_almost_equal(std_muinvn, std_desired)


def test_moving_muinvn():
    a = np.array([1, 2, 3, 4, 5, 6], dtype='d')
    mu, std = cycore.muinvn(a, 3)
    mu_desired = np.array([2., 3., 4., 5.])
    std_desired = np.array([0.7071068, 0.7071068, 0.7071068, 0.7071068])

    np.testing.assert_almost_equal(mu, mu_desired)
    np.testing.assert_almost_equal(std, std_desired)


def test_muinvn_vs_matlab():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    w = 32

    ml_mu = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'muinvn_mua.txt'))
    ml_std = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'muinvn_stda.txt'))

    mu, std = cycore.muinvn(ts, w)

    np.testing.assert_almost_equal(ml_mu, mu, decimal=4)
    np.testing.assert_almost_equal(ml_std, std, decimal=4)