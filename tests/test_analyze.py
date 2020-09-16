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


def test_preprocess():
    ts = np.array([2, 3, 2, 3, 1, 2, 3, 4, 2, np.nan, np.inf, 4, 2, 3, 4, 5,
                   6, 7, 8, 3, 4, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, np.nan, np.nan,
                   np.inf, np.nan, np.inf, np.nan, np.inf, np.nan, np.inf])
    m = 6
    preprocessing_kwargs = {
        'window': 5,
        'impute_method': 'median',
        'impute_direction': 'backward',
        'add_noise': False
    }

    result = analyze(ts, windows=m, preprocessing_kwargs=preprocessing_kwargs)
    preprocessed_ts = result[0]['data']['ts']
    assert (np.any(np.isnan(preprocessed_ts)) == False)
    assert (np.any(np.isinf(preprocessed_ts)) == False)

    # if preprocessing_kwargs=None, we disable the preprocessing procedure.
    result = analyze(ts, windows=m, preprocessing_kwargs=None)
    unprocessed_ts = result[0]['data']['ts']
    assert (np.any(np.isnan(unprocessed_ts)) == True)
    assert (np.any(np.isinf(unprocessed_ts)) == True)

    # check if preprocessing_kwargs is None by default.
    result = analyze(ts, windows=m)
    unprocessed_ts = result[0]['data']['ts']
    assert(np.any(np.isnan(unprocessed_ts)) == True)
    assert(np.any(np.isinf(unprocessed_ts)) == True)

    with pytest.raises(ValueError) as excinfo:
        analyze(ts, windows=m, preprocessing_kwargs=1)
        assert "The parameter 'preprocessing_kwargs' is not dict like!" \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        preprocessing_kwargs = {
            'win': 5,
            'impute_dir': 'backward',
        }
        analyze(ts, windows=m, preprocessing_kwargs=preprocessing_kwargs)
        assert "invalid key(s) for preprocessing_kwargs! valid key(s) should include " \
               "{'impute_direction', 'add_noise', 'impute_method', 'window'}" \
            in str(excinfo.value)