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

from matrixprofile.algorithms import skimp
from matrixprofile.algorithms.skimp import binary_split
from matrixprofile.algorithms.skimp import maximum_subsequence
from matrixprofile.exceptions import NoSolutionPossible


def test_binary_split_1():
    desired = [0]
    actual = binary_split(1)

    np.testing.assert_equal(actual, desired)


def test_binary_split_many():
    desired = [0, 5, 2, 7, 1, 3, 6, 8, 4, 9]
    actual = binary_split(10)

    np.testing.assert_equal(actual, desired)


def test_maximum_subsequence_36():
    np.random.seed(9999)
    ts = np.random.uniform(size=2**10)
    w = 2**5
    subq = ts[0:w]
    ts[0:w] = subq
    ts[w+100:w+100+w] = subq

    upper = maximum_subsequence(ts, 0.98)

    assert(upper == 36)


def test_maximum_subsequence_68():
    np.random.seed(9999)
    ts = np.random.uniform(size=2**10)
    w = 2**6
    subq = ts[0:w]
    ts[0:w] = subq
    ts[w+100:w+100+w] = subq

    upper = maximum_subsequence(ts, 0.98)

    assert(upper == 68)

def test_maximum_subsequence_no_windows():
    np.random.seed(9999)
    ts = np.random.uniform(size=2**10)
    w = 2**6
    subq = ts[0:w]
    ts[0:w] = subq
    ts[w+100:w+100+w] = subq

    with pytest.raises(NoSolutionPossible) as excinfo:
        upper = maximum_subsequence(ts, 1.0)
        assert 'no windows' in str(excinfo.value)

