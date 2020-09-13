#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile.preprocess import preprocess

def test_preprocess():
    ts = np.array([np.nan, np.inf, np.inf, np.nan, np.inf, 2, 3, 2, 3, 1, 2, 3, 4, 2,
                   np.nan, np.inf, 4, 2, 3, 4, 5, 6, 7, 8, 3, 4, 2, 3, 4, 5, 6, 7, 6,
                   5, 4, 3, np.nan, np.nan, np.inf, np.nan, np.inf, np.nan])

    ts = preprocess(ts, 4)
    assert(np.any(np.isnan(ts)) == False)
    assert(np.any(np.isinf(ts)) == False)