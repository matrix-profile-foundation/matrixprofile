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

from matrixprofile.algorithms.mass2 import mass2

def test_mass2():
    ts = np.array([1, 1, 1, 2, 1, 1, 4, 5])
    query = np.array([2, 1, 1, 4])
    actual = mass2(ts, query)
    desired = np.array([
        0.67640791-1.37044402e-16j,
        3.43092352+0.00000000e+00j,
        3.43092352+1.02889035e-17j,
        0.+0.00000000e+00j,
        1.85113597+1.21452707e-17j
    ])
    
    np.testing.assert_almost_equal(actual, desired)