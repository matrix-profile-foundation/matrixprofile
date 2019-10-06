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


def test_moving_avg_std():
    a = np.array([1, 2, 3, 4, 5, 6], dtype='d')
    mu, std = cycore.moving_avg_std(a, 3)
    mu_desired = np.array([2., 3., 4., 5.])
    std_desired = np.array([0.81649658, 0.81649658, 0.81649658, 0.81649658])

    np.testing.assert_almost_equal(mu, mu_desired)
    np.testing.assert_almost_equal(std, std_desired)


def test_moving_muinvn():
    a = np.array([1, 2, 3, 4, 5, 6], dtype='d')
    mu, std = cycore.muinvn(a, 3)
    mu_desired = np.array([2., 3., 4., 5.])
    std_desired = np.array([0.7071068, 0.7071068, 0.7071068, 0.7071068])

    np.testing.assert_almost_equal(mu, mu_desired)
    np.testing.assert_almost_equal(std, std_desired)