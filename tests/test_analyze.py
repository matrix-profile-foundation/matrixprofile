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

from matrixprofile import analyze

import matrixprofile
MODULE_PATH = matrixprofile.__path__[0]


def test_analyze_mp_exact_no_query():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    m = 32

    profile, figures = analyze(ts, windows=m)
    assert(profile['algorithm'] == 'mpx')
    assert(profile['w'] == 32)
    assert(profile['data']['query'] == None)
    assert(profile['join'] == False)
    assert(profile['sample_pct'] == 1)
    assert(profile['class'] == 'MatrixProfile')
    assert('motifs' in profile)
    assert('discords' in profile)
    assert(len(figures) == 4)


def test_analyze_mp_exact_with_query():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    query = ts[100:200]
    m = 32

    profile, figures = analyze(ts, windows=m, query=query)
    assert(profile['algorithm'] == 'mpx')
    assert(profile['w'] == 32)
    np.testing.assert_equal(profile['data']['query'], query)
    assert(profile['join'] == True)
    assert(profile['sample_pct'] == 1)
    assert(profile['class'] == 'MatrixProfile')
    assert('motifs' in profile)
    assert('discords' in profile)
    assert(len(figures) == 4)


def test_analyze_mp_approximate():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    m = 32

    profile, figures = analyze(ts, windows=m, sample_pct=0.5)
    assert(profile['algorithm'] == 'scrimp++')
    assert(profile['w'] == 32)
    assert(profile['data']['query'] == None)
    assert(profile['join'] == False)
    assert(profile['sample_pct'] == 0.5)
    assert(profile['class'] == 'MatrixProfile')
    assert(len(figures) == 4)


def test_analyze_pmp_no_sample_pct():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))

    profile, figures = analyze(ts)
    assert(profile['algorithm'] == 'skimp')
    assert(profile['class'] == 'PMP')
    assert(profile['sample_pct'] == 1)
    assert(len(figures) == 6)


def test_analyze_pmp_sample_pct():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))

    profile, figures = analyze(ts, sample_pct=0.1)
    assert(profile['algorithm'] == 'skimp')
    assert(profile['class'] == 'PMP')
    assert(profile['sample_pct'] == 0.1)
    assert(len(figures) == 6)


def test_analyze_pmp_windows():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    windows = np.arange(8, 32)

    profile, figures = analyze(ts, windows=windows, sample_pct=1)
    assert(profile['algorithm'] == 'skimp')
    assert(profile['class'] == 'PMP')
    assert(profile['sample_pct'] == 1)
    np.testing.assert_equal(profile['windows'], windows)
    assert(len(figures) == 6)