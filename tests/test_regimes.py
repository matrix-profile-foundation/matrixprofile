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

from matrixprofile.algorithms.regimes import extract_regimes
from matrixprofile.algorithms.mpx import mpx

import matrixprofile

MODULE_PATH = matrixprofile.__path__[0]


def test_regimes():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    w = 32
    profile = mpx(ts, w)

    # test extract 3 regimes (default)
    profile = extract_regimes(profile)
    actual = profile['regimes']
    desired = np.array([759, 423, 583])

    np.testing.assert_array_equal(actual, desired)

    # test extract 2 regimes
    profile = extract_regimes(profile, num_regimes=2)
    actual = profile['regimes']
    desired = np.array([759, 423])

    np.testing.assert_array_equal(actual, desired)