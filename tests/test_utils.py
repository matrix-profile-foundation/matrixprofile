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
MODULE_PATH = matrixprofile.__path__[0]

def test_empty_mp():
    keys = [
        'mp',
        'pi',
        'rmp',
        'rpi',
        'lmp',
        'lpi',
        'metric',
        'w',
        'ez',
        'join',
        'data',
        'class',
        'algorithm',
    ]

    empty = utils.empty_mp()

    for key in keys:
        assert(key in empty)
    
    assert('ts' in empty['data'])
    assert('query' in empty['data'])


def test_pick_mp():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    n = len(ts)
    pmp = compute(ts)
    mp = utils.pick_mp(pmp, 32)

    assert(mp['w'] == 32)
    assert(mp['algorithm'] == 'mpx')
    assert(len(mp['mp']) == n - mp['w'] + 1)
    np.testing.assert_equal(mp['data']['ts'], ts)