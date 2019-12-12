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

from matrixprofile import compute

import matrixprofile
MODULE_PATH = matrixprofile.__path__[0]


def test_compute_mp_exact_no_query():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    m = 32

    profile = compute(ts, windows=m)
    assert(profile['algorithm'] == 'mpx')
    assert(profile['w'] == 32)
    assert(profile['data']['query'] == None)
    assert(profile['join'] == False)
    assert(profile['sample_pct'] == 1)
    assert(profile['class'] == 'MatrixProfile')


def test_compute_mp_exact_with_query():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    query = ts[100:200]
    m = 32

    profile = compute(ts, windows=m, query=query)
    assert(profile['algorithm'] == 'mpx')
    assert(profile['w'] == 32)
    np.testing.assert_equal(profile['data']['query'], query)
    assert(profile['join'] == True)
    assert(profile['sample_pct'] == 1)
    assert(profile['class'] == 'MatrixProfile')


def test_compute_mp_approximate():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    m = 32

    profile = compute(ts, windows=m, sample_pct=0.5)
    assert(profile['algorithm'] == 'scrimp++')
    assert(profile['w'] == 32)
    assert(profile['data']['query'] == None)
    assert(profile['join'] == False)
    assert(profile['sample_pct'] == 0.5)
    assert(profile['class'] == 'MatrixProfile')


def test_compute_pmp_no_sample_pct_windows():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    windows = np.arange(8, 32)

    profile = compute(ts, windows=windows)
    assert(profile['algorithm'] == 'skimp')
    assert(profile['class'] == 'PMP')
    assert(profile['sample_pct'] == 1)
    np.testing.assert_equal(profile['windows'], windows)


def test_compute_pmp_sample_pct_windows():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    windows = np.arange(8, 32)

    profile = compute(ts, windows=windows, sample_pct=1)
    assert(profile['algorithm'] == 'skimp')
    assert(profile['class'] == 'PMP')
    assert(profile['sample_pct'] == 1)
    np.testing.assert_equal(profile['windows'], windows)


def test_compute_pmp_no_windows():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))

    profile = compute(ts)
    assert(profile['algorithm'] == 'skimp')
    assert(profile['class'] == 'PMP')

    # sample pct is ignored when windows are provided and defaults to 1
    assert(profile['sample_pct'] == 1)


def test_compute_pmp_no_windows_sample_pct():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))

    profile = compute(ts, sample_pct=0.1)
    assert(profile['algorithm'] == 'skimp')
    assert(profile['class'] == 'PMP')

    # sample pct is ignored when windows are provided and defaults to 1
    assert(profile['sample_pct'] == 0.1)
